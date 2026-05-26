#!/usr/bin/env python3
"""
Real-data ablation study for STELLA on PhysioNet EEGMMIDB.
Uses data already downloaded by run_real_experiments.py.
Tests 6 architectural and objective ablations.
"""
import sys, os, time, json, warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import mne; mne.set_log_level("WARNING")
import torch, torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.datasets.base import EEGDataset, SubjectSplit
from src.evaluation.metrics import compute_metrics
from src.models.stella import build_stella, STELLAConfig
from src.models.heads import MLPHead
from src.models.temporal_mamba import TemporalMamba
from src.models.tokenizer import DualStreamTokenizer, TemporalPatchEmbed
from src.models.spatial_transformer import ChannelSpatialTransformer
from src.models.fusion import GatedSpectralFusion
from src.models.baselines import VanillaTransformerEEG

# ── config ────────────────────────────────────────────────────────────────────
N_SUBJECTS  = 30       # use subset already cached
N_EPOCHS    = 30
BATCH_SIZE  = 32
DATA_DIR    = "data/physionet"
OUT_DIR     = "results/real_experiment"
SEED        = 42

_RUN_EVENTS = {
    4:  {"T1": 0, "T2": 1}, 6:  {"T1": 2, "T2": 3},
    8:  {"T1": 0, "T2": 1}, 10: {"T1": 2, "T2": 3},
    12: {"T1": 0, "T2": 1}, 14: {"T1": 2, "T2": 3},
}


def load_subject(sid, data_dir):
    from mne.datasets import eegbci
    from scipy.signal import butter, filtfilt, iirnotch
    trials, labels = [], []
    for run in [4, 6, 8, 10, 12, 14]:
        try:
            fnames = eegbci.load_data(sid, runs=[run], path=data_dir,
                                      update_path=True, verbose=False)
            raw  = mne.io.read_raw_edf(fnames[0], preload=True, verbose=False)
            eegbci.standardize(raw)
            sfreq = raw.info["sfreq"]
            data  = raw.get_data() * 1e6

            b, a = iirnotch(60.0 / (sfreq / 2), 30.0)
            data = filtfilt(b, a, data, axis=-1)
            b, a = butter(4, [1.0 / (sfreq/2), 40.0 / (sfreq/2)], btype="band")
            data = filtfilt(b, a, data, axis=-1)
            data -= data.mean(axis=0, keepdims=True)

            seg_len = int(4.0 * sfreq)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            for ann_key, cls_id in _RUN_EVENTS.get(run, {}).items():
                codes = [v for k, v in event_id.items() if ann_key in k]
                for code in codes:
                    for onset in events[events[:, 2] == code, 0]:
                        end = onset + seg_len
                        if end <= data.shape[-1]:
                            seg = data[:, onset:end].astype(np.float32)
                            seg = (seg - seg.mean(-1, keepdims=True)) / (seg.std(-1, keepdims=True) + 1e-8)
                            trials.append(seg); labels.append(cls_id)
        except Exception:
            pass
    return trials, labels


def get_dataset(n_subjects, data_dir, seed=42):
    set_seed(seed)
    all_x, all_y, all_sid = [], [], []
    for sid in range(1, n_subjects + 1):
        x, y = load_subject(sid, data_dir)
        if x:
            all_x.extend(x); all_y.extend(y)
            all_sid.extend([sid] * len(x))
    return EEGDataset(all_x, all_y, all_sid)


def quick_eval(model, train_set, val_set, test_set, epochs=30, lr=1e-3):
    device = torch.device("cpu"); model = model.to(device)
    tl = DataLoader(train_set, BATCH_SIZE, shuffle=True,  num_workers=0)
    vl = DataLoader(val_set,   64,         shuffle=False, num_workers=0)
    sl = DataLoader(test_set,  64,         shuffle=False, num_workers=0)
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(weight=train_set.get_class_weights().to(device),
                               label_smoothing=0.05)
    best_acc, best_st = 0.0, None
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        for x, lbl, _ in tl:
            loss = crit(model(x.to(device)), lbl.to(device))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for x, lbl, _ in vl:
                ps.extend(model(x.to(device)).argmax(-1).cpu().tolist())
                ts.extend(lbl.tolist())
        acc = sum(p == t for p, t in zip(ps, ts)) / max(len(ts), 1)
        if acc > best_acc:
            best_acc = acc
            best_st  = {k: v.clone() for k, v in model.state_dict().items()}
    train_time = time.perf_counter() - t0
    if best_st: model.load_state_dict(best_st)
    model.eval()
    ps, ts = [], []
    with torch.no_grad():
        for x, lbl, _ in sl:
            ps.extend(model(x.to(device)).argmax(-1).cpu().tolist())
            ts.extend(lbl.tolist())
    m = compute_metrics(ts, ps)
    m["n_params"]    = sum(p.numel() for p in model.parameters())
    m["train_time_s"] = round(train_time, 1)
    return m


# ── ablation variants ─────────────────────────────────────────────────────────

class STELLANoMamba(nn.Module):
    """Replace S3M with 3 extra Transformer layers (Transformer-only)."""
    def __init__(self, n_ch, T, d=256):
        super().__init__()
        self.tok  = DualStreamTokenizer(n_channels=n_ch, sfreq=160, d_model=d)
        self.st   = ChannelSpatialTransformer(d_model=d, n_layers=5, n_channels=n_ch)
        self.fus  = GatedSpectralFusion(d_model=d)
        self.cls  = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.norm = nn.LayerNorm(d)
        self.head = MLPHead(d, 4)
    def forward(self, x):
        B = x.shape[0]; t, s = self.tok(x); t = self.st(t); t = self.fus(t, s)
        seq = torch.cat([self.cls.expand(B, -1, -1), t], 1)
        return self.head(self.norm(seq)[:, 0])


class STELLANoSpectral(nn.Module):
    """Temporal-only: no spectral branch, no fusion."""
    def __init__(self, n_ch, T, d=256):
        super().__init__()
        self.tok   = TemporalPatchEmbed(n_channels=n_ch, patch_size=40,
                                        patch_stride=20, d_model=d)
        self.st    = ChannelSpatialTransformer(d_model=d, n_layers=2, n_channels=n_ch)
        self.mamba = TemporalMamba(d_model=d, n_layers=3)
        self.cls   = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.norm  = nn.LayerNorm(d)
        self.head  = MLPHead(d, 4)
    def forward(self, x):
        B = x.shape[0]; t = self.tok(x); t = self.st(t)
        seq = torch.cat([self.cls.expand(B, -1, -1), t], 1)
        return self.head(self.norm(self.mamba(seq))[:, 0])


class STELLANoGate(nn.Module):
    """Replace gated fusion with simple spectral addition."""
    def __init__(self, n_ch, T, d=256):
        super().__init__()
        self.tok   = DualStreamTokenizer(n_channels=n_ch, sfreq=160, d_model=d)
        self.st    = ChannelSpatialTransformer(d_model=d, n_layers=2, n_channels=n_ch)
        self.sproj = nn.Linear(d, d)
        self.mamba = TemporalMamba(d_model=d, n_layers=3)
        self.cls   = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.norm  = nn.LayerNorm(d)
        self.head  = MLPHead(d, 4)
    def forward(self, x):
        B = x.shape[0]; t, s = self.tok(x); t = self.st(t)
        t = t + self.sproj(s.mean(1, keepdim=True))   # additive, no gate
        seq = torch.cat([self.cls.expand(B, -1, -1), t], 1)
        return self.head(self.norm(self.mamba(seq))[:, 0])


class STELLANoSpatial(nn.Module):
    """Remove channel spatial Transformer entirely."""
    def __init__(self, n_ch, T, d=256):
        super().__init__()
        self.tok   = DualStreamTokenizer(n_channels=n_ch, sfreq=160, d_model=d)
        self.fus   = GatedSpectralFusion(d_model=d)
        self.mamba = TemporalMamba(d_model=d, n_layers=3)
        self.cls   = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.norm  = nn.LayerNorm(d)
        self.head  = MLPHead(d, 4)
    def forward(self, x):
        B = x.shape[0]; t, s = self.tok(x); t = self.fus(t, s)
        seq = torch.cat([self.cls.expand(B, -1, -1), t], 1)
        return self.head(self.norm(self.mamba(seq))[:, 0])


class STELLASmall(nn.Module):
    """Compact STELLA with d_model=128 (~25% parameters)."""
    def __init__(self, n_ch, T, d=128):
        super().__init__()
        self.tok   = DualStreamTokenizer(n_channels=n_ch, sfreq=160, d_model=d)
        self.st    = ChannelSpatialTransformer(d_model=d, n_layers=2, n_heads=4, n_channels=n_ch)
        self.fus   = GatedSpectralFusion(d_model=d, n_heads=4)
        self.mamba = TemporalMamba(d_model=d, n_layers=3, d_state=8)
        self.cls   = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.norm  = nn.LayerNorm(d)
        self.head  = MLPHead(d, 4)
    def forward(self, x):
        B = x.shape[0]; t, s = self.tok(x); t = self.st(t); t = self.fus(t, s)
        seq = torch.cat([self.cls.expand(B, -1, -1), t], 1)
        return self.head(self.norm(self.mamba(seq))[:, 0])


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    print(f"\n[DATA] Loading {N_SUBJECTS} subjects for ablation study…")
    dataset = get_dataset(N_SUBJECTS, DATA_DIR, seed=SEED)
    n_ch = dataset.trials[0].shape[0]
    T    = dataset.trials[0].shape[1]
    print(f"       {len(dataset)} trials loaded")

    split     = SubjectSplit.from_n_subjects(N_SUBJECTS, seed=SEED)
    train_set = dataset.filter_subjects(split.train_subjects)
    val_set   = dataset.filter_subjects(split.val_subjects)
    test_set  = dataset.filter_subjects(split.test_subjects)
    print(f"       Train={len(train_set)} | Val={len(val_set)} | Test={len(test_set)}")

    ablation_variants = {
        "STELLA (full)":           lambda: (lambda m: (setattr(m, 'downstream_head', MLPHead(256, 4)), m)[1])(build_stella(STELLAConfig(n_channels=n_ch, sfreq=160, segment_len=T))),
        "w/o S3M (Transformer×5)": lambda: STELLANoMamba(n_ch, T),
        "w/o Spectral branch":     lambda: STELLANoSpectral(n_ch, T),
        "w/o Gated Fusion (add)":  lambda: STELLANoGate(n_ch, T),
        "w/o Spatial Transformer": lambda: STELLANoSpatial(n_ch, T),
        "STELLA-Small (d=128)":    lambda: STELLASmall(n_ch, T),
    }

    results = {}
    print(f"\n{'Variant':<35} {'Acc':>7} {'BalAcc':>8} {'κ':>7} {'Params':>10} {'Time':>7}")
    print("-" * 78)

    for name, fn in ablation_variants.items():
        set_seed(SEED)
        m   = fn()
        met = quick_eval(m, train_set, val_set, test_set, epochs=N_EPOCHS)
        results[name] = met
        print(f"{name:<35} {met['accuracy']:>7.4f} {met['balanced_accuracy']:>8.4f} "
              f"{met['kappa']:>7.4f} {met['n_params']:>10,} {met['train_time_s']:>6.0f}s")

    # Save
    out_path = f"{OUT_DIR}/ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Ablation saved → {out_path}")

    # Print delta table
    full_acc = results["STELLA (full)"]["accuracy"]
    print("\nContribution of each component (Δ accuracy vs. full STELLA):")
    for name, res in results.items():
        if name != "STELLA (full)":
            delta = full_acc - res["accuracy"]
            direction = "↓" if delta > 0 else "↑"
            print(f"  Remove {name:<30}: {direction}{abs(delta):.4f}")

    return results


if __name__ == "__main__":
    main()
