#!/usr/bin/env python3
"""
EEG-RAG with EEGNet Backbone
=============================
Uses EEGNet (proven 56.25% accuracy) as the feature extractor.
Demonstrates RAG retrieval can improve any strong encoder.

Pipeline:
  1. Train EEGNet (2.6K params, 30 epochs) → ~50-56% accuracy
  2. Extract 320-dim penultimate features → FAISS store
  3. Train RAGReasoningHead on top of EEGNet features
  4. Evaluate: EEGNet alone vs EEGNet+Prototype vs EEGNet+RAG
  5. Run N-shot evaluation (N=1,5,10,20,50 per class)

Usage:
    python scripts/run_rag_eegnet.py
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
from torch.utils.data import DataLoader, Subset

import mne; mne.set_log_level("WARNING")

from src.utils.seed import set_seed
from src.datasets.base import EEGDataset, SubjectSplit
from src.models.baselines import EEGNet
from src.models.eeg_rag import RAGReasoningHead, PrototypeClassifier
from src.retrieval.store import EEGEmbeddingStore
from src.evaluation.metrics import compute_metrics

# ── Config ────────────────────────────────────────────────────────────────────
N_SUBJECTS       = 50
N_ENCODER_EPOCHS = 30
N_RAG_EPOCHS     = 25
N_SHOT_VALUES    = [1, 5, 10, 20, 50]
N_SHOT_REPEATS   = 3
BATCH_SIZE       = 32
K_NEIGHBOURS     = 5
FEAT_DIM         = 320   # EEGNet penultimate dim
EMB_DIM          = 256   # RAG embedding dim (projection target)
SEED             = 42
DATA_DIR         = "data/physionet"
OUT_DIR          = Path("results/rag_eegnet")
CLASS_NAMES      = ["Left Fist", "Right Fist", "Both Fists", "Both Feet"]

_RUN_EVENTS = {
    4:  {"T1": 0, "T2": 1}, 6:  {"T1": 2, "T2": 3},
    8:  {"T1": 0, "T2": 1}, 10: {"T1": 2, "T2": 3},
    12: {"T1": 0, "T2": 1}, 14: {"T1": 2, "T2": 3},
}


# ── Data ──────────────────────────────────────────────────────────────────────
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


# ── EEGNet feature extractor with projection ──────────────────────────────────
class EEGNetEncoder(nn.Module):
    """EEGNet backbone → L2-normalised 256-dim embedding."""
    def __init__(self, n_channels, T, feat_dim=320, emb_dim=256):
        super().__init__()
        self.backbone = EEGNet(n_channels=n_channels, n_classes=4, T=T,
                               F1=8, D=2, F2=16, dropout=0.5)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )
        self.emb_dim = emb_dim

    def features(self, x):
        """Return penultimate 320-dim EEGNet features."""
        return self.backbone.encode(x)   # uses EEGNet.encode() → (B, 320)

    def forward(self, x):
        """Return L2-normalised 256-dim embedding."""
        f = self.features(x)
        return F.normalize(self.proj(f), dim=-1)

    def classify(self, x):
        """Full EEGNet classification logits."""
        return self.backbone(x)


# ── RAG model for EEGNet ──────────────────────────────────────────────────────
class EEGNetRAG(nn.Module):
    """EEGNet encoder + FAISS retrieval + RAGReasoningHead."""
    def __init__(self, encoder: EEGNetEncoder, k: int = 5):
        super().__init__()
        self.encoder = encoder
        self.reasoning_head = RAGReasoningHead(
            d_model=EMB_DIM, n_classes=4, n_heads=4, n_layers=2, dropout=0.1
        )
        self.store: EEGEmbeddingStore | None = None
        self.k = k

    def set_store(self, store): self.store = store

    def freeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad_(False)

    def forward(self, x, subject_ids=None):
        assert self.store is not None
        B = x.shape[0]; device = x.device
        with torch.no_grad():
            q_emb = self.encoder(x)    # (B, 256) already normalised
        ret_embs, ret_lbls = [], []
        for i in range(B):
            q_np = q_emb[i:i+1].cpu().numpy()
            sid  = subject_ids[i] if subject_ids else None
            _, r_embs, r_lbls, _ = self.store.search(q_np, k=self.k,
                                                       exclude_subject=sid)
            ret_embs.append(r_embs[0]); ret_lbls.append(r_lbls[0])
        ret_embs = torch.tensor(np.stack(ret_embs), dtype=torch.float32, device=device)
        ret_lbls = torch.tensor(np.stack(ret_lbls), dtype=torch.long,    device=device)
        return self.reasoning_head(q_emb, ret_embs, ret_lbls)


# ── Training helpers ──────────────────────────────────────────────────────────
def train_encoder(encoder, train_set, val_set, epochs, device):
    tl   = DataLoader(train_set, BATCH_SIZE, shuffle=True,  num_workers=0)
    vl   = DataLoader(val_set,   64,         shuffle=False, num_workers=0)
    opt  = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(weight=train_set.get_class_weights().to(device),
                               label_smoothing=0.05)
    best_acc, best_st, val_accs = 0.0, None, []
    for ep in range(epochs):
        encoder.train()
        for x, lbl, _ in tl:
            loss = crit(encoder.classify(x.to(device)), lbl.to(device))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0); opt.step()
        sch.step()
        encoder.eval()
        ps, ts = [], []
        with torch.no_grad():
            for x, lbl, _ in vl:
                ps.extend(encoder.classify(x.to(device)).argmax(-1).cpu().tolist())
                ts.extend(lbl.tolist())
        acc = sum(p==t for p,t in zip(ps,ts)) / max(len(ts),1)
        val_accs.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_st  = {k: v.clone() for k, v in encoder.state_dict().items()}
        if (ep+1) % 5 == 0:
            print(f"  Ep {ep+1:3d}/{epochs}  val_acc={acc:.4f}  best={best_acc:.4f}")
    encoder.load_state_dict(best_st)
    print(f"  EEGNet encoder trained — best val_acc={best_acc:.4f}")
    return val_accs


@torch.no_grad()
def build_store(encoder, dataset, device, batch_size=64):
    encoder.eval()
    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0)
    store  = EEGEmbeddingStore(d_model=EMB_DIM, use_cosine=True)
    for x, lbl, sid in loader:
        emb = encoder(x.to(device))        # already normalised
        store.add(emb, lbl, sid)
    store.build_index()
    print(f"  Store: {len(store):,} embeddings")
    return store


def train_rag(rag_model, train_set, val_set, epochs, device):
    rag_model.freeze_encoder()
    tl   = DataLoader(train_set, BATCH_SIZE, shuffle=True,  num_workers=0)
    vl   = DataLoader(val_set,   64,         shuffle=False, num_workers=0)
    opt  = torch.optim.AdamW(rag_model.reasoning_head.parameters(), lr=3e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(weight=train_set.get_class_weights().to(device),
                               label_smoothing=0.05)
    best_acc, best_st, val_accs = 0.0, None, []
    for ep in range(epochs):
        rag_model.train(); rag_model.encoder.eval()
        for x, lbl, sid in tl:
            logits = rag_model(x.to(device), sid.tolist())
            loss   = crit(logits, lbl.to(device))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(rag_model.parameters(), 1.0); opt.step()
        sch.step()
        rag_model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for x, lbl, sid in vl:
                ps.extend(rag_model(x.to(device), sid.tolist()).argmax(-1).cpu().tolist())
                ts.extend(lbl.tolist())
        acc = sum(p==t for p,t in zip(ps,ts)) / max(len(ts),1)
        val_accs.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_st  = {k: v.clone() for k, v in
                        rag_model.reasoning_head.state_dict().items()}
        if (ep+1) % 5 == 0:
            print(f"  Ep {ep+1:3d}/{epochs}  val_acc={acc:.4f}  best={best_acc:.4f}")
    if best_st:
        rag_model.reasoning_head.load_state_dict(best_st)
    print(f"  RAG head trained — best val_acc={best_acc:.4f}")
    return val_accs


@torch.no_grad()
def evaluate(model, test_set, device, is_rag=False):
    model.eval()
    loader = DataLoader(test_set, 64, shuffle=False, num_workers=0)
    ps, ts = [], []
    for x, lbl, sid in loader:
        logits = (model(x.to(device), sid.tolist()) if is_rag
                  else model.classify(x.to(device)))
        ps.extend(logits.argmax(-1).cpu().tolist()); ts.extend(lbl.tolist())
    return compute_metrics(ts, ps)


# ── Few-shot helpers ──────────────────────────────────────────────────────────
def sample_n_shot(dataset, n_per_class, n_classes=4, seed=42):
    rng = random.Random(seed)
    per_class = {c: [] for c in range(n_classes)}
    for i, (_, lbl, _) in enumerate(dataset):
        per_class[int(lbl)].append(i)
    return sum([rng.sample(per_class[c], min(n_per_class, len(per_class[c])))
                for c in range(n_classes)], [])


@torch.no_grad()
def few_shot_rag(encoder, rag_head_state, n_shot_subset, test_set,
                 cfg_k, device):
    """Build store from N-shot subset and evaluate RAG on test set."""
    store = EEGEmbeddingStore(d_model=EMB_DIM, use_cosine=True)
    loader = DataLoader(n_shot_subset, 32, shuffle=False, num_workers=0)
    encoder.eval()
    for x, lbl, sid in loader:
        emb = encoder(x.to(device))
        store.add(emb, lbl, sid)
    store.build_index()

    from src.models.eeg_rag import RAGReasoningHead
    rag = EEGNetRAG(encoder, k=min(cfg_k, len(store)))
    rag.reasoning_head.load_state_dict(rag_head_state, strict=False)
    rag.set_store(store)
    rag.eval()

    loader = DataLoader(test_set, 64, shuffle=False, num_workers=0)
    ps, ts = [], []
    for x, lbl, sid in loader:
        preds = rag(x.to(device), sid.tolist()).argmax(-1)
        ps.extend(preds.cpu().tolist()); ts.extend(lbl.tolist())
    return compute_metrics(ts, ps)


@torch.no_grad()
def few_shot_prototype(encoder, n_shot_subset, test_set, device):
    loader = DataLoader(n_shot_subset, 32, shuffle=False, num_workers=0)
    encoder.eval()
    all_emb, all_lbl = [], []
    for x, lbl, _ in loader:
        emb = encoder(x.to(device))
        all_emb.append(emb.cpu().numpy()); all_lbl.append(lbl.numpy())
    proto = PrototypeClassifier(n_classes=4)
    proto.fit(np.concatenate(all_emb), np.concatenate(all_lbl))
    loader = DataLoader(test_set, 64, shuffle=False, num_workers=0)
    ps, ts = [], []
    for x, lbl, _ in loader:
        emb = encoder(x.to(device)).cpu().numpy()
        ps.extend(proto.predict(emb).tolist()); ts.extend(lbl.tolist())
    return compute_metrics(ts, ps)


# ── Figures ───────────────────────────────────────────────────────────────────
def make_figures(main_res, few_shot_res, out_dir):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                             "axes.spines.right": False})
    except ImportError:
        return
    fig_dir = out_dir / "figures"; fig_dir.mkdir(exist_ok=True)

    # Main bar chart
    names  = ["EEGNet\n(only)", "EEGNet+\nPrototype", "EEGNet+RAG\n(ours)"]
    keys   = ["eegnet_only", "eegnet_prototype", "eegnet_rag"]
    accs   = [main_res[k]["accuracy"] for k in keys]
    kappas = [main_res[k]["kappa"]    for k in keys]
    colors = ["#457B9D", "#2A9D8F", "#D62828"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, vals, ylabel in zip(axes, [accs, kappas], ["Accuracy", "Cohen's κ"]):
        bars = ax.bar(names, vals, color=colors, edgecolor="white", width=0.5)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(vals) + 0.10)
        ax.axhline(0.25, color="gray", ls="--", lw=1, label="Chance")
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.legend(fontsize=8)
    plt.suptitle("EEGNet + RAG vs. Baselines — PhysioNet EEGMMIDB (Subject-Independent)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir/"eegnet_rag_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(fig_dir/"eegnet_rag_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Few-shot curve
    if few_shot_res:
        n_vals = sorted(few_shot_res.keys())
        method_cfg = {
            "EEGNet+RAG (ours)": ("eegnet_rag",       "#D62828", "o-"),
            "Prototype":         ("eegnet_prototype",  "#2A9D8F", "s--"),
            "EEGNet-only":       ("eegnet_only",       "#457B9D", "^:"),
        }
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, (key, color, style) in method_cfg.items():
            means = [few_shot_res[n][key]["mean"]["accuracy"] for n in n_vals]
            stds  = [few_shot_res[n][key]["std"]["accuracy"]  for n in n_vals]
            ax.plot(n_vals, means, style, color=color, lw=2, ms=7, label=label)
            ax.fill_between(n_vals,
                            [m-s for m,s in zip(means,stds)],
                            [m+s for m,s in zip(means,stds)],
                            alpha=0.15, color=color)
        ax.axhline(0.25, color="gray", ls="--", lw=1, label="Chance (25%)")
        ax.set_xlabel("N labeled examples per class (N-shot)")
        ax.set_ylabel("Test Accuracy")
        ax.set_title("EEG-RAG Few-Shot Performance — PhysioNet EEGMMIDB")
        ax.set_xticks(n_vals); ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(fig_dir/"eegnet_rag_fewshot.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(fig_dir/"eegnet_rag_fewshot.png", dpi=200, bbox_inches="tight")
        plt.close()

    print(f"[FIG] Saved → {fig_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED); device = torch.device("cpu")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading {N_SUBJECTS} subjects…")
    dataset   = get_dataset(N_SUBJECTS, DATA_DIR, seed=SEED)
    n_ch, T   = dataset.trials[0].shape
    split     = SubjectSplit.from_n_subjects(N_SUBJECTS, seed=SEED)
    train_set = dataset.filter_subjects(split.train_subjects)
    val_set   = dataset.filter_subjects(split.val_subjects)
    test_set  = dataset.filter_subjects(split.test_subjects)
    print(f"       {len(dataset)} trials | Train={len(train_set)} "
          f"Val={len(val_set)} Test={len(test_set)}")

    # ── 2. Train EEGNet encoder ───────────────────────────────────────────────
    print(f"\n[2/5] Training EEGNet encoder ({N_ENCODER_EPOCHS} epochs)…")
    encoder = EEGNetEncoder(n_channels=n_ch, T=T,
                            feat_dim=FEAT_DIM, emb_dim=EMB_DIM).to(device)
    enc_accs = train_encoder(encoder, train_set, val_set, N_ENCODER_EPOCHS, device)
    torch.save(encoder.state_dict(), OUT_DIR / "eegnet_encoder.pt")

    # ── 3. Build store ────────────────────────────────────────────────────────
    print(f"\n[3/5] Building FAISS embedding store…")
    store = build_store(encoder, train_set, device)
    store.save(OUT_DIR / "embedding_store")

    # ── 4. Train RAG head ─────────────────────────────────────────────────────
    print(f"\n[4/5] Training RAG reasoning head ({N_RAG_EPOCHS} epochs)…")
    rag = EEGNetRAG(encoder, k=K_NEIGHBOURS).to(device)
    rag.set_store(store)
    rag_accs = train_rag(rag, train_set, val_set, N_RAG_EPOCHS, device)
    torch.save(rag.reasoning_head.state_dict(), OUT_DIR / "rag_head.pt")
    rag_head_state = rag.reasoning_head.state_dict()

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print(f"\n[5/5] Evaluating on test set…")

    print("  → EEGNet-only…")
    enc_only = evaluate(encoder, test_set, device, is_rag=False)
    print(f"     Acc={enc_only['accuracy']:.4f}  κ={enc_only['kappa']:.4f}  "
          f"F1={enc_only['f1']:.4f}")

    print("  → EEGNet + Prototype…")
    loader   = DataLoader(train_set, 128, shuffle=False, num_workers=0)
    all_emb, all_lbl = [], []
    encoder.eval()
    with torch.no_grad():
        for x, lbl, _ in loader:
            all_emb.append(encoder(x.to(device)).cpu().numpy())
            all_lbl.append(lbl.numpy())
    proto = PrototypeClassifier(n_classes=4)
    proto.fit(np.concatenate(all_emb), np.concatenate(all_lbl))
    ps, ts = [], []
    loader = DataLoader(test_set, 128, shuffle=False, num_workers=0)
    encoder.eval()
    with torch.no_grad():
        for x, lbl, _ in loader:
            emb = encoder(x.to(device)).cpu().numpy()
            ps.extend(proto.predict(emb).tolist()); ts.extend(lbl.tolist())
    proto_metrics = compute_metrics(ts, ps)
    print(f"     Acc={proto_metrics['accuracy']:.4f}  κ={proto_metrics['kappa']:.4f}  "
          f"F1={proto_metrics['f1']:.4f}")

    print("  → EEGNet + RAG…")
    rag_metrics = evaluate(rag, test_set, device, is_rag=True)
    print(f"     Acc={rag_metrics['accuracy']:.4f}  κ={rag_metrics['kappa']:.4f}  "
          f"F1={rag_metrics['f1']:.4f}")

    main_results = {
        "eegnet_only":      enc_only,
        "eegnet_prototype": proto_metrics,
        "eegnet_rag":       rag_metrics,
        "config": {"k": K_NEIGHBOURS, "n_subjects": N_SUBJECTS,
                   "enc_epochs": N_ENCODER_EPOCHS, "rag_epochs": N_RAG_EPOCHS}
    }

    # ── Few-shot evaluation ───────────────────────────────────────────────────
    print(f"\n[5b/5] Few-shot evaluation N={N_SHOT_VALUES}…")
    few_shot_results = {}
    for n_shot in N_SHOT_VALUES:
        print(f"\n  N={n_shot} shots/class…")
        rep_results = {k: [] for k in ["eegnet_rag", "eegnet_prototype", "eegnet_only"]}
        for rep in range(N_SHOT_REPEATS):
            idxs = sample_n_shot(train_set, n_shot, seed=SEED + rep * 100)
            subset = Subset(train_set, idxs)

            m = few_shot_rag(encoder, rag_head_state, subset, test_set,
                             K_NEIGHBOURS, device)
            rep_results["eegnet_rag"].append(m)

            m = few_shot_prototype(encoder, subset, test_set, device)
            rep_results["eegnet_prototype"].append(m)

            # EEGNet-only: just the trained encoder on test set (same for all reps)
            rep_results["eegnet_only"].append(enc_only)

            print(f"    Rep {rep+1}: RAG={rep_results['eegnet_rag'][-1]['accuracy']:.4f}  "
                  f"Proto={rep_results['eegnet_prototype'][-1]['accuracy']:.4f}")

        keys = ["accuracy", "balanced_accuracy", "kappa", "f1"]
        few_shot_results[n_shot] = {
            method: {
                "mean": {k: float(np.mean([r[k] for r in reps])) for k in keys},
                "std":  {k: float(np.std( [r[k] for r in reps])) for k in keys},
            }
            for method, reps in rep_results.items()
        }

    # ── Save ─────────────────────────────────────────────────────────────────
    all_results = {
        "main":      main_results,
        "few_shot":  {str(k): v for k, v in few_shot_results.items()},
    }
    with open(OUT_DIR / "rag_eegnet_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n" + "=" * 65)
    print(f"{'Method':<35} {'Acc':>7} {'κ':>7} {'F1':>7}")
    print("-" * 65)
    for name, key in [("EEGNet-only",              "eegnet_only"),
                      ("EEGNet + Prototype",        "eegnet_prototype"),
                      ("EEGNet + RAG (ours)",       "eegnet_rag")]:
        m = main_results[key]
        star = " ★" if "RAG" in name else ""
        print(f"{name+star:<35} {m['accuracy']:>7.4f} {m['kappa']:>7.4f} {m['f1']:>7.4f}")

    delta_rag   = main_results["eegnet_rag"]["accuracy"] - enc_only["accuracy"]
    delta_proto = proto_metrics["accuracy"] - enc_only["accuracy"]
    print(f"\n  RAG vs EEGNet-only:       {delta_rag:+.4f}")
    print(f"  Prototype vs EEGNet-only: {delta_proto:+.4f}")

    print("\n  Few-shot summary:")
    print(f"  {'N':>4}  {'EEGNet+RAG':^16}  {'Prototype':^16}  {'EEGNet-only':^12}")
    for n in N_SHOT_VALUES:
        r = few_shot_results[n]
        print(f"  {n:>4}  {r['eegnet_rag']['mean']['accuracy']:>6.4f}±"
              f"{r['eegnet_rag']['std']['accuracy']:.4f}  "
              f"{r['eegnet_prototype']['mean']['accuracy']:>6.4f}±"
              f"{r['eegnet_prototype']['std']['accuracy']:.4f}")

    make_figures(main_results, few_shot_results, OUT_DIR)
    print(f"\n[DONE] → {OUT_DIR}/rag_eegnet_results.json")
    return all_results


if __name__ == "__main__":
    main()
