#!/usr/bin/env python3
"""
Full real-data experiment pipeline for STELLA paper results.

Dataset: PhysioNet EEG Motor Movement/Imagery (EEGMMIDB)
  - Real public dataset, standard MI benchmark
  - 109 subjects, 64 channels, 160 Hz
  - 4-class MI: left fist / right fist / both fists / both feet
  - Runs 4,6,8,10,12,14 used for imagery epochs

Evaluation: Subject-independent (train/val/test subject split)
  - Models trained on train_subjects, evaluated on held-out test_subjects
  - 3 random seeds, results averaged

All outputs go to results/real_experiment/
"""

import sys, os, time, json, csv, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import mne
mne.set_log_level("WARNING")
from mne.datasets import eegbci

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.datasets.base import EEGDataset, SubjectSplit
from src.evaluation.metrics import compute_metrics
from src.models.stella import build_stella, STELLAConfig
from src.models.baselines import (
    EEGNet, ShallowConvNet, DeepConvNet,
    VanillaTransformerEEG, CNNTransformerEEG, MIRepNetBaseline,
)
from src.models.heads import MLPHead

# ── run config ────────────────────────────────────────────────────────────────
N_SUBJECTS   = 50          # download & use this many subjects
N_EPOCHS     = 40          # training epochs per model
BATCH_SIZE   = 32
LR           = 1e-3
DATA_DIR     = "data/physionet"
OUT_DIR      = "results/real_experiment"
SEEDS        = [42, 7, 123]
T_SFREQ      = 160.0       # target sampling frequency
SEG_SEC      = 4.0         # epoch length in seconds

# MI run → event mapping (PhysioNet convention)
_RUN_EVENTS = {
    4:  {"T1": 0, "T2": 1},   # T1=left fist, T2=right fist
    6:  {"T1": 2, "T2": 3},   # T1=both fists, T2=both feet
    8:  {"T1": 0, "T2": 1},
    10: {"T1": 2, "T2": 3},
    12: {"T1": 0, "T2": 1},
    14: {"T1": 2, "T2": 3},
}
CLASS_NAMES = ["Left fist", "Right fist", "Both fists", "Both feet"]


# ── data loading ──────────────────────────────────────────────────────────────

def load_subject(subj_id: int, data_dir: str) -> tuple[list, list]:
    """Load and preprocess one PhysioNet subject. Returns (trials, labels)."""
    from scipy.signal import butter, filtfilt, iirnotch

    trials, labels = [], []
    runs = [4, 6, 8, 10, 12, 14]

    for run in runs:
        try:
            fnames = eegbci.load_data(subj_id, runs=[run], path=data_dir,
                                      update_path=True, verbose=False)
            raw = mne.io.read_raw_edf(fnames[0], preload=True, verbose=False)
            eegbci.standardize(raw)    # standardise channel names to 10-20

            sfreq = raw.info["sfreq"]
            data  = raw.get_data() * 1e6    # V → µV

            # ── preprocessing ──────────────────────────────────────────────
            # 1. Notch 60 Hz
            b, a = iirnotch(60.0 / (sfreq / 2.0), 30.0)
            data = filtfilt(b, a, data, axis=-1)

            # 2. Bandpass 1–40 Hz
            nyq = sfreq / 2.0
            b, a = butter(4, [1.0 / nyq, 40.0 / nyq], btype="band")
            data = filtfilt(b, a, data, axis=-1)

            # 3. Common Average Reference
            data -= data.mean(axis=0, keepdims=True)

            # 4. Resample to 160 Hz if needed
            if abs(sfreq - T_SFREQ) > 1.0:
                from math import gcd
                from scipy.signal import resample_poly
                g    = gcd(int(T_SFREQ), int(sfreq))
                up   = int(T_SFREQ) // g
                down = int(sfreq) // g
                data = resample_poly(data, up, down, axis=-1)
                sfreq = T_SFREQ

            seg_len = int(SEG_SEC * sfreq)

            # 5. Extract epochs from annotations
            events, event_id = mne.events_from_annotations(
                raw, verbose=False
            )
            event_map = _RUN_EVENTS.get(run, {})

            for ann_key, cls_id in event_map.items():
                code_list = [v for k, v in event_id.items() if ann_key in k]
                for code in code_list:
                    onsets = events[events[:, 2] == code, 0]
                    for onset in onsets:
                        end = onset + seg_len
                        if end <= data.shape[-1]:
                            seg = data[:, onset:end].astype(np.float32)
                            # z-score per channel
                            mu  = seg.mean(-1, keepdims=True)
                            std = seg.std(-1,  keepdims=True) + 1e-8
                            seg = (seg - mu) / std
                            trials.append(seg)
                            labels.append(cls_id)

        except Exception as e:
            pass    # skip failed runs silently

    return trials, labels


def download_and_load(n_subjects: int, data_dir: str):
    """Download + preprocess all subjects. Returns EEGDataset."""
    os.makedirs(data_dir, exist_ok=True)
    all_trials, all_labels, all_sids = [], [], []
    failed = []

    print(f"\n[DATA] Loading {n_subjects} subjects from PhysioNet EEGMMIDB…")
    for sid in range(1, n_subjects + 1):
        t0 = time.perf_counter()
        trials, labels = load_subject(sid, data_dir)
        elapsed = time.perf_counter() - t0
        if trials:
            all_trials.extend(trials)
            all_labels.extend(labels)
            all_sids.extend([sid] * len(trials))
            print(f"  S{sid:03d}: {len(trials):3d} trials  ({elapsed:.1f}s)")
        else:
            failed.append(sid)
            print(f"  S{sid:03d}: FAILED")

    if not all_trials:
        raise RuntimeError("No trials loaded — check internet access and MNE install.")

    from collections import Counter
    print(f"\n[DATA] Loaded {len(all_trials)} trials from {len(set(all_sids))} subjects")
    print(f"       Class distribution: {dict(Counter(all_labels))}")
    if failed:
        print(f"       Failed subjects: {failed}")

    return EEGDataset(all_trials, all_labels, all_sids)


# ── training ──────────────────────────────────────────────────────────────────

def train_and_eval(model, train_set, val_set, test_set,
                   n_epochs=40, lr=1e-3, name=""):
    """Supervised training loop → test metrics."""
    device = torch.device("cpu")
    model  = model.to(device)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, drop_last=False)
    val_loader   = DataLoader(val_set,   batch_size=64,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=64,
                              shuffle=False, num_workers=0)

    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion  = nn.CrossEntropyLoss(
        weight=train_set.get_class_weights().to(device),
        label_smoothing=0.05,
    )

    best_val_acc = 0.0
    best_state   = None
    t0_train     = time.perf_counter()

    for ep in range(n_epochs):
        # ── train ──
        model.train()
        for x, lbl, _ in train_loader:
            x, lbl = x.to(device), lbl.to(device)
            loss = criterion(model(x), lbl)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # ── val ──
        model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for x, lbl, _ in val_loader:
                ps.extend(model(x.to(device)).argmax(-1).cpu().tolist())
                ts.extend(lbl.tolist())
        acc = sum(p == t for p, t in zip(ps, ts)) / max(len(ts), 1)
        if acc > best_val_acc:
            best_val_acc = acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    train_time = time.perf_counter() - t0_train
    if best_state:
        model.load_state_dict(best_state)

    # ── test ──
    model.eval()
    ps, ts = [], []
    with torch.no_grad():
        for x, lbl, _ in test_loader:
            ps.extend(model(x.to(device)).argmax(-1).cpu().tolist())
            ts.extend(lbl.tolist())

    # ── inference latency (single trial) ──
    model.eval()
    x_one = torch.randn(1, 64, int(SEG_SEC * T_SFREQ))
    with torch.no_grad():
        for _ in range(5):          # warm up
            _ = model(x_one)
    t_inf = time.perf_counter()
    with torch.no_grad():
        for _ in range(50):
            _ = model(x_one)
    latency_ms = (time.perf_counter() - t_inf) * 1000 / 50

    metrics = compute_metrics(ts, ps)
    metrics["train_time_s"] = round(train_time, 1)
    metrics["latency_ms"]   = round(latency_ms, 2)
    metrics["n_params"]     = sum(p.numel() for p in model.parameters())
    return metrics


# ── model factory ─────────────────────────────────────────────────────────────

def make_models(n_ch: int, T: int, n_classes: int, sfreq: float):
    def stella():
        cfg = STELLAConfig(
            n_channels=n_ch, sfreq=sfreq, segment_len=T, n_classes_aux=n_classes
        )
        m   = build_stella(cfg)
        m.downstream_head = MLPHead(cfg.d_model, n_classes)
        return m

    return {
        "EEGNet":            lambda: EEGNet(n_ch, n_classes, sfreq, T),
        "ShallowConvNet":    lambda: ShallowConvNet(n_ch, n_classes, T),
        "DeepConvNet":       lambda: DeepConvNet(n_ch, n_classes, T),
        "VanillaTransformer":lambda: VanillaTransformerEEG(n_ch, n_classes, T,
                                        d_model=256, n_heads=4, n_layers=4),
        "CNNTransformer":    lambda: CNNTransformerEEG(n_ch, n_classes, sfreq, T,
                                        d_model=128),
        "MIRepNet-style":    lambda: MIRepNetBaseline(n_ch, n_classes, T),
        "STELLA":            stella,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # ── 1. Load real data ──────────────────────────────────────────────────
    dataset = download_and_load(N_SUBJECTS, DATA_DIR)
    n_ch    = dataset.trials[0].shape[0]
    T       = dataset.trials[0].shape[1]
    n_cl    = len(set(dataset.labels.tolist()))
    print(f"\n[INFO] Shape per trial: ({n_ch}, {T})  |  Classes: {n_cl}")

    # ── 2. Multi-seed experiment ───────────────────────────────────────────
    all_seed_results = defaultdict(list)    # model → [metrics_seed1, …]

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        set_seed(seed)

        split     = SubjectSplit.from_n_subjects(N_SUBJECTS,
                                                 val_frac=0.10,
                                                 test_frac=0.20,
                                                 seed=seed)
        train_set = dataset.filter_subjects(split.train_subjects)
        val_set   = dataset.filter_subjects(split.val_subjects)
        test_set  = dataset.filter_subjects(split.test_subjects)

        print(f"  Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)} trials")

        model_fns = make_models(n_ch, T, n_cl, T_SFREQ)

        for name, fn in model_fns.items():
            print(f"\n  ── {name} ──")
            set_seed(seed)
            model   = fn()
            metrics = train_and_eval(model, train_set, val_set, test_set,
                                     n_epochs=N_EPOCHS, lr=LR, name=name)
            all_seed_results[name].append(metrics)

            print(f"     Acc={metrics['accuracy']:.4f}  "
                  f"BalAcc={metrics['balanced_accuracy']:.4f}  "
                  f"F1={metrics['f1']:.4f}  "
                  f"κ={metrics['kappa']:.4f}  "
                  f"Params={metrics['n_params']:,}  "
                  f"Train={metrics['train_time_s']:.0f}s  "
                  f"Latency={metrics['latency_ms']:.1f}ms")

    # ── 3. Aggregate results ───────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("FINAL RESULTS  (mean ± std across 3 seeds)")
    print(f"{'='*70}")

    header = f"{'Model':<22} {'Acc':>9} {'BalAcc':>9} {'F1':>9} {'κ':>9} {'Params':>10} {'Train(s)':>9}"
    print(header)
    print("-" * len(header))

    aggregated = {}
    for name, runs in all_seed_results.items():
        def m(key):
            vals = [r[key] for r in runs]
            return np.mean(vals), np.std(vals)

        acc_m,  acc_s  = m("accuracy")
        bal_m,  bal_s  = m("balanced_accuracy")
        f1_m,   f1_s   = m("f1")
        kap_m,  kap_s  = m("kappa")
        par_m          = runs[0]["n_params"]
        trn_m          = np.mean([r["train_time_s"] for r in runs])
        lat_m          = np.mean([r["latency_ms"]   for r in runs])

        aggregated[name] = {
            "accuracy_mean": round(acc_m, 4), "accuracy_std": round(acc_s, 4),
            "balanced_accuracy_mean": round(bal_m, 4), "balanced_accuracy_std": round(bal_s, 4),
            "f1_mean": round(f1_m, 4), "f1_std": round(f1_s, 4),
            "kappa_mean": round(kap_m, 4), "kappa_std": round(kap_s, 4),
            "n_params": par_m,
            "train_time_s_mean": round(trn_m, 1),
            "latency_ms_mean": round(lat_m, 2),
        }

        row = (f"{name:<22} "
               f"{acc_m:.4f}±{acc_s:.4f}  "
               f"{bal_m:.4f}±{bal_s:.4f}  "
               f"{f1_m:.4f}±{f1_s:.4f}  "
               f"{kap_m:.4f}±{kap_s:.4f}  "
               f"{par_m:>10,}  "
               f"{trn_m:>7.0f}s")
        print(row)

    # ── 4. Save outputs ────────────────────────────────────────────────────
    json_path = f"{OUT_DIR}/aggregated_results.json"
    with open(json_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    csv_path = f"{OUT_DIR}/results_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregated["STELLA"].keys()))
        writer.writeheader()
        for name, res in aggregated.items():
            writer.writerow({"model_name": name, **res}
                            if "model_name" not in res else res)

    # ── also save per-seed raw results ────────────────────────────────────
    raw_path = f"{OUT_DIR}/per_seed_results.json"
    with open(raw_path, "w") as f:
        json.dump({k: v for k, v in all_seed_results.items()}, f, indent=2, default=str)

    print(f"\n[DONE] Results saved:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {raw_path}")

    return aggregated


if __name__ == "__main__":
    main()
