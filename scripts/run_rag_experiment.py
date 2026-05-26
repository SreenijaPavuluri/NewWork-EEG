#!/usr/bin/env python3
"""
EEG-RAG: Retrieval-Augmented Motor Imagery Classification
==========================================================
Complete pipeline:
  1. Load 50 PhysioNet subjects (already cached)
  2. Train STELLA encoder with supervised CE loss (20 epochs)
  3. Build FAISS embedding store from training subjects
  4. Train RAGReasoningHead (cross-attention, 20 epochs)
  5. Evaluate: STELLA-only vs STELLA+RAG vs supervised baselines

Usage:
    python scripts/run_rag_experiment.py
"""

import sys, os, time, json, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import mne; mne.set_log_level("WARNING")

from src.utils.seed import set_seed
from src.datasets.base import EEGDataset, SubjectSplit
from src.models.stella import STELLAConfig, STELLAEncoder
from src.models.heads import MLPHead
from src.models.eeg_rag import EEGRAGClassifier, PrototypeClassifier
from src.retrieval.store import EEGEmbeddingStore
from src.evaluation.metrics import compute_metrics

# ── Config ────────────────────────────────────────────────────────────────────
N_SUBJECTS       = 50
N_ENCODER_EPOCHS = 20    # supervised encoder training
N_RAG_EPOCHS     = 20    # reasoning head training
BATCH_SIZE       = 32
K_NEIGHBOURS     = 5
SEED             = 42
DATA_DIR         = "data/physionet"
OUT_DIR          = Path("results/rag_experiment")
STORE_DIR        = OUT_DIR / "embedding_store"

_RUN_EVENTS = {
    4:  {"T1": 0, "T2": 1}, 6:  {"T1": 2, "T2": 3},
    8:  {"T1": 0, "T2": 1}, 10: {"T1": 2, "T2": 3},
    12: {"T1": 0, "T2": 1}, 14: {"T1": 2, "T2": 3},
}
CLASS_NAMES = ["Left Fist", "Right Fist", "Both Fists", "Both Feet"]


# ── Data loading ──────────────────────────────────────────────────────────────
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


# ── Step 1: Train STELLA encoder with supervised CE ───────────────────────────
def train_encoder(encoder, head, train_set, val_set, epochs, device):
    tl = DataLoader(train_set, BATCH_SIZE, shuffle=True,  num_workers=0)
    vl = DataLoader(val_set,   64,         shuffle=False, num_workers=0)
    params = list(encoder.parameters()) + list(head.parameters())
    opt  = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(
        weight=train_set.get_class_weights().to(device),
        label_smoothing=0.05
    )
    best_acc, best_enc, best_head = 0.0, None, None
    for ep in range(epochs):
        encoder.train(); head.train()
        for x, lbl, _ in tl:
            logits = head(encoder(x.to(device)))
            loss   = crit(logits, lbl.to(device))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sch.step()
        encoder.eval(); head.eval()
        ps, ts = [], []
        with torch.no_grad():
            for x, lbl, _ in vl:
                ps.extend(head(encoder(x.to(device))).argmax(-1).cpu().tolist())
                ts.extend(lbl.tolist())
        acc = sum(p == t for p, t in zip(ps, ts)) / max(len(ts), 1)
        if acc > best_acc:
            best_acc = acc
            best_enc  = {k: v.clone() for k, v in encoder.state_dict().items()}
            best_head = {k: v.clone() for k, v in head.state_dict().items()}
        if (ep + 1) % 5 == 0:
            print(f"    Ep {ep+1:3d}/{epochs}  val_acc={acc:.4f}  (best={best_acc:.4f})")
    encoder.load_state_dict(best_enc)
    head.load_state_dict(best_head)
    print(f"  Encoder trained — best val acc: {best_acc:.4f}")
    return best_acc


# ── Step 2: Build embedding store ─────────────────────────────────────────────
@torch.no_grad()
def build_store(encoder, dataset, device, batch_size=64):
    encoder.eval()
    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0)
    store  = EEGEmbeddingStore(d_model=256, use_cosine=True)
    for x, lbl, sid in loader:
        emb = F.normalize(encoder(x.to(device)), dim=-1)
        store.add(emb, lbl, sid)
    store.build_index()
    print(f"  Store: {len(store):,} embeddings indexed")
    return store


# ── Step 3: Train RAG reasoning head ──────────────────────────────────────────
def train_rag_head(rag_model, train_set, val_set, store, epochs, device):
    rag_model.freeze_encoder()
    rag_model.set_store(store)
    tl = DataLoader(train_set, BATCH_SIZE, shuffle=True,  num_workers=0)
    vl = DataLoader(val_set,   64,         shuffle=False, num_workers=0)
    opt  = torch.optim.AdamW(rag_model.reasoning_head.parameters(), lr=3e-4,
                              weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(
        weight=train_set.get_class_weights().to(device),
        label_smoothing=0.05
    )
    best_acc, best_st = 0.0, None
    val_accs, losses = [], []
    for ep in range(epochs):
        rag_model.train(); rag_model.encoder.eval()
        ep_loss = 0.0
        for x, lbl, sid in tl:
            logits = rag_model(x.to(device), sid.tolist())
            loss   = crit(logits, lbl.to(device))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(rag_model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        sch.step()
        rag_model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for x, lbl, sid in vl:
                ps.extend(rag_model(x.to(device), sid.tolist()).argmax(-1).cpu().tolist())
                ts.extend(lbl.tolist())
        acc = sum(p == t for p, t in zip(ps, ts)) / max(len(ts), 1)
        val_accs.append(acc); losses.append(ep_loss / len(tl))
        if acc > best_acc:
            best_acc = acc
            best_st  = {k: v.clone() for k, v in
                        rag_model.reasoning_head.state_dict().items()}
        if (ep + 1) % 5 == 0:
            print(f"    Ep {ep+1:3d}/{epochs}  loss={ep_loss/len(tl):.4f}"
                  f"  val_acc={acc:.4f}  (best={best_acc:.4f})")
    if best_st:
        rag_model.reasoning_head.load_state_dict(best_st)
    print(f"  RAG head trained — best val acc: {best_acc:.4f}")
    return val_accs, losses


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_model(model, test_set, device, is_rag=False):
    model.eval()
    loader = DataLoader(test_set, 64, shuffle=False, num_workers=0)
    ps, ts = [], []
    for x, lbl, sid in loader:
        if is_rag:
            logits = model(x.to(device), sid.tolist())
        else:
            logits = model(x.to(device))
        ps.extend(logits.argmax(-1).cpu().tolist())
        ts.extend(lbl.tolist())
    return compute_metrics(ts, ps)


# ── Figures ───────────────────────────────────────────────────────────────────
def make_figures(results, train_curves, out_dir):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                             "axes.spines.right": False})
    except ImportError:
        print("[SKIP] matplotlib not available"); return
    fig_dir = out_dir / "figures"; fig_dir.mkdir(exist_ok=True)

    # Comparison bar
    methods = {
        "STELLA-only\n(linear probe)": "stella_linear",
        "Prototype\nClassifier": "stella_prototype",
        "STELLA+RAG\n(ours)": "stella_rag",
    }
    accs   = [results[v]["accuracy"] for v in methods.values()]
    kappas = [results[v]["kappa"]    for v in methods.values()]
    colors = ["#457B9D", "#2A9D8F", "#D62828"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, vals, ylabel in zip(axes, [accs, kappas], ["Accuracy", "Cohen's κ"]):
        bars = ax.bar(list(methods.keys()), vals, color=colors, edgecolor="white", width=0.5)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(vals) + 0.12)
        ax.axhline(0.25, color="gray", ls="--", lw=1, label="Chance (25%)")
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.008,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.legend(fontsize=8)
    plt.suptitle("STELLA+RAG vs. Baselines — PhysioNet EEGMMIDB",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir/"rag_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(fig_dir/"rag_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Training curves
    if train_curves:
        enc_accs, rag_accs = train_curves
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(enc_accs, "#457B9D", lw=2)
        axes[0].set_title("STELLA Encoder Training"); axes[0].set_ylabel("Val Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[1].plot(rag_accs, "#D62828", lw=2)
        axes[1].set_title("RAG Reasoning Head Training"); axes[1].set_ylabel("Val Accuracy")
        axes[1].set_xlabel("Epoch")
        plt.suptitle("Training Curves — EEG-RAG", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(fig_dir/"rag_training_curves.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(fig_dir/"rag_training_curves.png", dpi=200, bbox_inches="tight")
        plt.close()

    # Confusion matrix
    rag_cm = results["stella_rag"].get("confusion_matrix", None)
    if rag_cm:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, key, title in zip(
            axes,
            ["stella_linear", "stella_rag"],
            ["STELLA-only (linear probe)", "STELLA+RAG (ours)"]
        ):
            cm = np.array(results[key].get("confusion_matrix", [[0]*4]*4))
            cm_n = cm.astype(float) / (cm.sum(1, keepdims=True) + 1e-8)
            im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(4)); ax.set_yticks(range(4))
            ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=8)
            ax.set_yticklabels(CLASS_NAMES, fontsize=8)
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, f"{cm_n[i,j]:.2f}", ha="center", va="center",
                            color="white" if cm_n[i,j]>0.5 else "black", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title(title)
        plt.suptitle("Confusion Matrices — PhysioNet EEGMMIDB (Subject-Independent)",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(fig_dir/"rag_confusion_matrices.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(fig_dir/"rag_confusion_matrices.png", dpi=200, bbox_inches="tight")
        plt.close()

    print(f"[FIG] Figures saved → {fig_dir}")


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

    # ── 2. Train STELLA encoder ───────────────────────────────────────────────
    print(f"\n[2/5] Training STELLA encoder ({N_ENCODER_EPOCHS} epochs)…")
    cfg     = STELLAConfig(n_channels=n_ch, sfreq=160.0, segment_len=T)
    encoder = STELLAEncoder(cfg).to(device)
    head    = MLPHead(256, 4, dropout=0.3).to(device)

    enc_val_accs = []
    enc_best_acc_track = [0.0]
    tl = DataLoader(train_set, BATCH_SIZE, shuffle=True,  num_workers=0)
    vl = DataLoader(val_set,   64,         shuffle=False, num_workers=0)
    params = list(encoder.parameters()) + list(head.parameters())
    opt  = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_ENCODER_EPOCHS)
    crit = nn.CrossEntropyLoss(weight=train_set.get_class_weights().to(device),
                               label_smoothing=0.05)
    best_enc_acc, best_enc_st, best_head_st = 0.0, None, None
    for ep in range(N_ENCODER_EPOCHS):
        encoder.train(); head.train()
        for x, lbl, _ in tl:
            logits = head(encoder(x.to(device)))
            loss = crit(logits, lbl.to(device))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        sch.step()
        encoder.eval(); head.eval()
        ps, ts = [], []
        with torch.no_grad():
            for x, lbl, _ in vl:
                ps.extend(head(encoder(x.to(device))).argmax(-1).cpu().tolist())
                ts.extend(lbl.tolist())
        acc = sum(p==t for p,t in zip(ps,ts)) / max(len(ts),1)
        enc_val_accs.append(acc)
        if acc > best_enc_acc:
            best_enc_acc = acc
            best_enc_st  = {k:v.clone() for k,v in encoder.state_dict().items()}
            best_head_st = {k:v.clone() for k,v in head.state_dict().items()}
        if (ep+1) % 5 == 0:
            print(f"  Ep {ep+1:3d}/{N_ENCODER_EPOCHS}  val_acc={acc:.4f}  best={best_enc_acc:.4f}")
    encoder.load_state_dict(best_enc_st); head.load_state_dict(best_head_st)
    print(f"  Encoder done — best val_acc={best_enc_acc:.4f}")

    # Save encoder
    torch.save(encoder.state_dict(), OUT_DIR / "stella_encoder.pt")
    torch.save(head.state_dict(),    OUT_DIR / "stella_head.pt")

    # ── 3. Build embedding store ──────────────────────────────────────────────
    print(f"\n[3/5] Building FAISS embedding store…")
    import shutil
    if STORE_DIR.exists(): shutil.rmtree(STORE_DIR)
    store = build_store(encoder, train_set, device)
    store.save(STORE_DIR)

    # ── 4. Train RAG reasoning head ───────────────────────────────────────────
    print(f"\n[4/5] Training RAG reasoning head ({N_RAG_EPOCHS} epochs)…")
    rag = EEGRAGClassifier(cfg, n_classes=4, k_neighbours=K_NEIGHBOURS,
                           exclude_same_subject=True).to(device)
    rag.encoder.load_state_dict(encoder.state_dict(), strict=False)
    rag_val_accs, rag_losses = train_rag_head(rag, train_set, val_set, store,
                                               N_RAG_EPOCHS, device)
    torch.save(rag.reasoning_head.state_dict(), OUT_DIR / "rag_reasoning_head.pt")

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print(f"\n[5/5] Evaluating on test set…")

    # STELLA + RAG
    print("  → STELLA+RAG…")
    rag_metrics = eval_model(rag, test_set, device, is_rag=True)
    print(f"     Acc={rag_metrics['accuracy']:.4f}  κ={rag_metrics['kappa']:.4f}  "
          f"F1={rag_metrics['f1']:.4f}")

    # STELLA + linear probe (already trained)
    print("  → STELLA-only (linear probe)…")
    class EncoderHead(nn.Module):
        def __init__(self, enc, h): super().__init__(); self.enc=enc; self.head=h
        def forward(self, x): return self.head(self.enc(x))
    enc_head_model = EncoderHead(encoder, head).to(device)
    linear_metrics = eval_model(enc_head_model, test_set, device)
    print(f"     Acc={linear_metrics['accuracy']:.4f}  κ={linear_metrics['kappa']:.4f}  "
          f"F1={linear_metrics['f1']:.4f}")

    # Prototype classifier
    print("  → Prototype classifier…")
    loader = DataLoader(train_set, 128, shuffle=False, num_workers=0)
    all_emb, all_lbl = [], []
    encoder.eval()
    with torch.no_grad():
        for x, lbl, _ in loader:
            emb = F.normalize(encoder(x.to(device)), dim=-1)
            all_emb.append(emb.cpu().numpy()); all_lbl.append(lbl.numpy())
    proto_clf = PrototypeClassifier(n_classes=4)
    proto_clf.fit(np.concatenate(all_emb), np.concatenate(all_lbl))
    ps, ts = [], []
    loader = DataLoader(test_set, 128, shuffle=False, num_workers=0)
    encoder.eval()
    with torch.no_grad():
        for x, lbl, _ in loader:
            emb  = F.normalize(encoder(x.to(device)), dim=-1).cpu().numpy()
            ps.extend(proto_clf.predict(emb).tolist()); ts.extend(lbl.tolist())
    proto_metrics = compute_metrics(ts, ps)
    print(f"     Acc={proto_metrics['accuracy']:.4f}  κ={proto_metrics['kappa']:.4f}  "
          f"F1={proto_metrics['f1']:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    results = {
        "stella_rag":       rag_metrics,
        "stella_linear":    linear_metrics,
        "stella_prototype": proto_metrics,
        "config": {
            "k_neighbours": K_NEIGHBOURS, "n_subjects": N_SUBJECTS,
            "n_encoder_epochs": N_ENCODER_EPOCHS, "n_rag_epochs": N_RAG_EPOCHS,
            "seed": SEED
        }
    }
    with open(OUT_DIR / "rag_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 65)
    print(f"{'Method':<35} {'Acc':>7} {'κ':>7} {'F1':>7}")
    print("-" * 65)
    for name, key in [("STELLA-only (linear probe)", "stella_linear"),
                      ("Prototype Classifier",        "stella_prototype"),
                      ("STELLA + RAG (ours)",         "stella_rag")]:
        m = results[key]
        marker = " ★" if key == "stella_rag" else ""
        print(f"{name+marker:<35} {m['accuracy']:>7.4f} {m['kappa']:>7.4f} {m['f1']:>7.4f}")
    delta = results["stella_rag"]["accuracy"] - results["stella_linear"]["accuracy"]
    print(f"\n  RAG improvement over STELLA-only: {delta:+.4f}")

    # Demo explanation
    print("\n[DEMO] Sample EEG-RAG explanation:")
    trial, label, sid = test_set[0]
    pred, conf, expl = rag.predict_with_explanation(trial, int(sid))
    print(expl)
    print(f"  True: {CLASS_NAMES[label]}  |  Pred: {CLASS_NAMES[pred]}"
          f"  |  Conf: {conf:.3f}")

    make_figures(results, (enc_val_accs, rag_val_accs), OUT_DIR)
    print(f"\n[DONE] → {OUT_DIR}/rag_results.json")
    return results


if __name__ == "__main__":
    main()
