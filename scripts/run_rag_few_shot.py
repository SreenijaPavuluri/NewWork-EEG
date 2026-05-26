#!/usr/bin/env python3
"""
EEG-RAG Few-Shot Evaluation
============================
Evaluates how EEG-RAG performs as N labeled examples per class increase.
Key question: Can RAG over STELLA embeddings outperform supervised models
with very few labeled examples?

Conditions evaluated for N in [1, 5, 10, 20, 50] labeled examples per class:
  A. STELLA+RAG: retrieve from N×4 labeled examples in store
  B. Prototype: nearest centroid from N×4 examples
  C. Supervised EEGNet: trained on N×4 examples
  D. STELLA encoder + linear probe: trained on N×4 examples

Usage:
    python scripts/run_rag_few_shot.py
"""

import sys, os, time, json, warnings, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset

import mne; mne.set_log_level("WARNING")

from src.utils.seed import set_seed
from src.datasets.base import EEGDataset, SubjectSplit
from src.models.stella import STELLAConfig, STELLAEncoder
from src.models.eeg_rag import EEGRAGClassifier, PrototypeClassifier
from src.models.baselines import VanillaTransformerEEG
from src.models.heads import MLPHead
from src.retrieval.store import EEGEmbeddingStore
from src.evaluation.metrics import compute_metrics

# ── Config ────────────────────────────────────────────────────────────────────
N_SUBJECTS      = 50
N_SHOT_VALUES   = [1, 5, 10, 20, 50]   # labeled examples per class
N_REPEATS       = 3                     # repeat each N-shot trial for stability
K_NEIGHBOURS    = 5
SEED            = 42
DATA_DIR        = "data/physionet"
OUT_DIR         = Path("results/rag_experiment")
STORE_DIR       = OUT_DIR / "embedding_store"
RAG_CKPT        = OUT_DIR / "rag_reasoning_head.pt"
STELLA_CKPT     = "results/checkpoints/STELLA.pt"
CLASS_NAMES     = ["Left Fist", "Right Fist", "Both Fists", "Both Feet"]
_RUN_EVENTS = {
    4:  {"T1": 0, "T2": 1}, 6:  {"T1": 2, "T2": 3},
    8:  {"T1": 0, "T2": 1}, 10: {"T1": 2, "T2": 3},
    12: {"T1": 0, "T2": 1}, 14: {"T1": 2, "T2": 3},
}


# ── Data (same preprocessing as run_rag_experiment.py) ────────────────────────

def load_subject(sid, data_dir):
    from mne.datasets import eegbci
    from scipy.signal import butter, filtfilt, iirnotch
    trials, labels = [], []
    for run in [4, 6, 8, 10, 12, 14]:
        try:
            fnames = eegbci.load_data(sid, runs=[run], path=data_dir,
                                      update_path=True, verbose=False)
            raw = mne.io.read_raw_edf(fnames[0], preload=True, verbose=False)
            eegbci.standardize(raw)
            sfreq = raw.info["sfreq"]; data = raw.get_data() * 1e6
            b, a = iirnotch(60.0/(sfreq/2), 30.0); data = filtfilt(b, a, data, axis=-1)
            b, a = butter(4, [1.0/(sfreq/2), 40.0/(sfreq/2)], btype="band")
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
                            seg = (seg - seg.mean(-1, keepdims=True)) / \
                                  (seg.std(-1, keepdims=True) + 1e-8)
                            trials.append(seg); labels.append(cls_id)
        except Exception: pass
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


# ── Sampling helpers ──────────────────────────────────────────────────────────

def sample_n_shot(dataset, n_per_class, n_classes=4, rng=None):
    """Sample exactly n_per_class examples per class. Returns indices."""
    if rng is None: rng = random.Random(42)
    per_class = {c: [] for c in range(n_classes)}
    for i, (_, lbl, _) in enumerate(dataset):
        per_class[int(lbl)].append(i)
    selected = []
    for c in range(n_classes):
        avail = per_class[c]
        k = min(n_per_class, len(avail))
        selected.extend(rng.sample(avail, k))
    return selected


# ── Models for comparison ─────────────────────────────────────────────────────

def build_rag_for_n_shot(encoder, rag_head_state, n_shot_dataset, cfg, device):
    """Build RAG model with store populated only by N-shot examples."""
    store = EEGEmbeddingStore(d_model=256, use_cosine=True)
    loader = DataLoader(n_shot_dataset, 32, shuffle=False, num_workers=0)
    encoder.eval()
    with torch.no_grad():
        for x, lbl, sid in loader:
            emb = F.normalize(encoder(x.to(device)), dim=-1)
            store.add(emb, lbl, sid)
    store.build_index()

    rag = EEGRAGClassifier(cfg, n_classes=4, k_neighbours=min(5, len(store)),
                           exclude_same_subject=False)
    rag.encoder.load_state_dict(encoder.state_dict(), strict=False)
    rag.reasoning_head.load_state_dict(rag_head_state, strict=False)
    rag.set_store(store)
    rag.eval()
    return rag


def build_prototype_for_n_shot(encoder, n_shot_dataset, device):
    """Build prototype classifier from N-shot examples."""
    loader = DataLoader(n_shot_dataset, 32, shuffle=False, num_workers=0)
    encoder.eval()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for x, lbl, _ in loader:
            emb = F.normalize(encoder(x.to(device)), dim=-1)
            all_emb.append(emb.cpu().numpy())
            all_lbl.append(lbl.numpy())
    embs = np.concatenate(all_emb); lbls = np.concatenate(all_lbl)
    proto = PrototypeClassifier(n_classes=4)
    proto.fit(embs, lbls)
    return proto, embs, lbls


def train_linear_probe_n_shot(encoder, n_shot_dataset, n_epochs=20, device=None):
    """Train linear probe on N-shot examples (frozen encoder)."""
    if device is None: device = torch.device("cpu")
    encoder.eval()
    probe = nn.Linear(256, 4).to(device)
    loader = DataLoader(n_shot_dataset, min(32, len(n_shot_dataset)),
                        shuffle=True, num_workers=0)
    opt  = torch.optim.Adam(probe.parameters(), lr=3e-3)
    crit = nn.CrossEntropyLoss()
    for _ in range(n_epochs):
        for x, lbl, _ in loader:
            with torch.no_grad():
                emb = encoder(x.to(device))
            loss = crit(probe(emb), lbl.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    return probe


# ── Evaluate ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_rag(rag, test_set, device):
    loader = DataLoader(test_set, 64, shuffle=False, num_workers=0)
    ps, ts = [], []
    for x, lbl, sid in loader:
        preds = rag(x.to(device), sid.tolist()).argmax(-1)
        ps.extend(preds.cpu().tolist()); ts.extend(lbl.tolist())
    return compute_metrics(ts, ps)


@torch.no_grad()
def eval_prototype(proto, encoder, test_set, device):
    loader = DataLoader(test_set, 64, shuffle=False, num_workers=0)
    ps, ts = [], []
    for x, lbl, _ in loader:
        emb  = F.normalize(encoder(x.to(device)), dim=-1).cpu().numpy()
        preds = proto.predict(emb)
        ps.extend(preds.tolist()); ts.extend(lbl.tolist())
    return compute_metrics(ts, ps)


@torch.no_grad()
def eval_linear(probe, encoder, test_set, device):
    loader = DataLoader(test_set, 64, shuffle=False, num_workers=0)
    ps, ts = [], []
    for x, lbl, _ in loader:
        emb  = encoder(x.to(device))
        preds = probe(emb).argmax(-1)
        ps.extend(preds.cpu().tolist()); ts.extend(lbl.tolist())
    return compute_metrics(ts, ps)


# ── Figures ───────────────────────────────────────────────────────────────────

def make_few_shot_figures(few_shot_results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                             "axes.spines.right": False})
    except ImportError:
        print("[SKIP] matplotlib not available"); return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    n_values = sorted(few_shot_results.keys())
    methods  = {
        "STELLA+RAG (ours)":     ("stella_rag",      "#D62828", "o-"),
        "Prototype Classifier":  ("stella_prototype", "#2A9D8F", "s--"),
        "Linear Probe":          ("stella_linear",    "#457B9D", "^:"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for metric_idx, (metric_key, ylabel) in enumerate([
        ("accuracy", "Accuracy"), ("kappa", "Cohen's κ")
    ]):
        ax = axes[metric_idx]
        for label, (key, color, style) in methods.items():
            accs  = [few_shot_results[n][key]["mean"][metric_key] for n in n_values]
            stds  = [few_shot_results[n][key]["std"][metric_key]  for n in n_values]
            ax.plot(n_values, accs, style, color=color, lw=2, ms=7, label=label)
            ax.fill_between(n_values,
                            [a - s for a, s in zip(accs, stds)],
                            [a + s for a, s in zip(accs, stds)],
                            alpha=0.15, color=color)
        ax.set_xlabel("N labeled examples per class")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Few-Shot {ylabel}")
        ax.set_xticks(n_values)
        if metric_idx == 0:
            ax.axhline(0.25, color="gray", ls="--", lw=1, label="Chance (25%)")
        ax.legend(fontsize=9)

    plt.suptitle(
        "EEG-RAG vs. Baselines — Few-Shot Motor Imagery (PhysioNet EEGMMIDB)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "few_shot_curve.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(fig_dir / "few_shot_curve.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Saved few-shot curve → {fig_dir}/few_shot_curve.pdf")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = torch.device("cpu")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading {N_SUBJECTS} subjects…")
    dataset = get_dataset(N_SUBJECTS, DATA_DIR, seed=SEED)
    n_ch = dataset.trials[0].shape[0]; T = dataset.trials[0].shape[1]
    split     = SubjectSplit.from_n_subjects(N_SUBJECTS, seed=SEED)
    train_set = dataset.filter_subjects(split.train_subjects)
    val_set   = dataset.filter_subjects(split.val_subjects)
    test_set  = dataset.filter_subjects(split.test_subjects)
    print(f"      Train={len(train_set)} | Val={len(val_set)} | Test={len(test_set)}")

    # ── Load pretrained STELLA encoder ────────────────────────────────────────
    print(f"\n[2/4] Loading STELLA encoder…")
    cfg     = STELLAConfig(n_channels=n_ch, sfreq=160.0, segment_len=T)
    encoder = STELLAEncoder(cfg).to(device)
    try:
        ckpt = torch.load(STELLA_CKPT, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            if "encoder" in ckpt:
                encoder.load_state_dict(ckpt["encoder"], strict=False)
            elif "model_state_dict" in ckpt:
                enc_state = {k.replace("encoder.", ""): v
                             for k, v in ckpt["model_state_dict"].items()
                             if k.startswith("encoder.")}
                encoder.load_state_dict(enc_state if enc_state else ckpt["model_state_dict"],
                                        strict=False)
            else:
                encoder.load_state_dict(ckpt, strict=False)
        print("      Checkpoint loaded")
    except Exception as e:
        print(f"      Using random encoder ({e})")
    encoder.eval()

    # ── Load trained RAG head ─────────────────────────────────────────────────
    print(f"\n[3/4] Loading RAG reasoning head from {RAG_CKPT}…")
    from src.models.eeg_rag import RAGReasoningHead
    rag_head = RAGReasoningHead(d_model=256, n_classes=4)
    try:
        rag_head.load_state_dict(torch.load(RAG_CKPT, map_location="cpu",
                                            weights_only=False))
        rag_head_state = rag_head.state_dict()
        print("      RAG head loaded")
    except Exception as e:
        print(f"      Using untrained RAG head ({e})")
        rag_head_state = rag_head.state_dict()

    # ── Few-shot evaluation loop ───────────────────────────────────────────────
    print(f"\n[4/4] Few-shot evaluation: N={N_SHOT_VALUES}, {N_REPEATS} repeats…")

    few_shot_results = {}

    for n_shot in N_SHOT_VALUES:
        print(f"\n  N={n_shot} shots per class ({n_shot*4} total)…")
        rep_results = {"stella_rag": [], "stella_prototype": [], "stella_linear": []}

        for rep in range(N_REPEATS):
            rng = random.Random(SEED + rep * 100)
            indices = sample_n_shot(train_set, n_shot, rng=rng)
            n_shot_subset = Subset(train_set, indices)

            # A. STELLA + RAG
            rag = build_rag_for_n_shot(encoder, rag_head_state, n_shot_subset, cfg, device)
            with torch.no_grad():
                m = eval_rag(rag, test_set, device)
            rep_results["stella_rag"].append(m)

            # B. Prototype
            proto, _, _ = build_prototype_for_n_shot(encoder, n_shot_subset, device)
            m = eval_prototype(proto, encoder, test_set, device)
            rep_results["stella_prototype"].append(m)

            # C. Linear probe
            probe = train_linear_probe_n_shot(encoder, n_shot_subset,
                                              n_epochs=20, device=device)
            with torch.no_grad():
                m = eval_linear(probe, encoder, test_set, device)
            rep_results["stella_linear"].append(m)

            print(f"    Rep {rep+1}/{N_REPEATS}: "
                  f"RAG={rep_results['stella_rag'][-1]['accuracy']:.4f}  "
                  f"Proto={rep_results['stella_prototype'][-1]['accuracy']:.4f}  "
                  f"Linear={rep_results['stella_linear'][-1]['accuracy']:.4f}")

        # Aggregate
        agg = {}
        for method, reps in rep_results.items():
            metrics_keys = ["accuracy", "balanced_accuracy", "kappa", "f1"]
            agg[method] = {
                "mean": {k: float(np.mean([r[k] for r in reps])) for k in metrics_keys},
                "std":  {k: float(np.std( [r[k] for r in reps])) for k in metrics_keys},
            }
        few_shot_results[n_shot] = agg

        # Print summary
        for method in ["stella_rag", "stella_prototype", "stella_linear"]:
            acc_m = agg[method]["mean"]["accuracy"]
            acc_s = agg[method]["std"]["accuracy"]
            print(f"    {method:<22}: acc={acc_m:.4f}±{acc_s:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    # Convert int keys to str for JSON
    out = {str(k): v for k, v in few_shot_results.items()}
    out_json = OUT_DIR / "few_shot_results.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVED] Few-shot results → {out_json}")

    # ── Figures ───────────────────────────────────────────────────────────────
    make_few_shot_figures(few_shot_results, OUT_DIR)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'N':>4}  {'STELLA+RAG':^18}  {'Prototype':^18}  {'Linear Probe':^18}")
    print("-" * 70)
    for n in N_SHOT_VALUES:
        ag = few_shot_results[n]
        rag_a = ag["stella_rag"]["mean"]["accuracy"]
        rag_s = ag["stella_rag"]["std"]["accuracy"]
        pro_a = ag["stella_prototype"]["mean"]["accuracy"]
        pro_s = ag["stella_prototype"]["std"]["accuracy"]
        lin_a = ag["stella_linear"]["mean"]["accuracy"]
        lin_s = ag["stella_linear"]["std"]["accuracy"]
        print(f"{n:>4}  {rag_a:.4f}±{rag_s:.4f}   "
              f"{pro_a:.4f}±{pro_s:.4f}   "
              f"{lin_a:.4f}±{lin_s:.4f}")

    print(f"\n[DONE] Few-shot evaluation complete.")
    return few_shot_results


if __name__ == "__main__":
    main()
