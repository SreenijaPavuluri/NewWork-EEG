#!/usr/bin/env python3
"""
EEG-RAG Architecture Diagram
=============================
Generates a publication-quality diagram of the EEG-RAG system:
  - EEGNet encoder (backbone)
  - FAISS embedding store
  - K-NN retrieval
  - RAG Reasoning Head (cross-attention + soft vote)
  - Natural language explanation output

Usage:
    python3.11 scripts/generate_rag_diagram.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patheffects import withStroke
import numpy as np

# ── Colors ────────────────────────────────────────────────────────────────────
C_ENCODER   = "#2C7BB6"   # blue  — EEGNet encoder
C_STORE     = "#1A9641"   # green — FAISS store
C_RETRIEVAL = "#F4A736"   # orange — retrieval
C_RAG       = "#D62828"   # red   — RAG reasoning head
C_OUT       = "#7B2D8B"   # purple — outputs
C_BG        = "#F7F7F7"
C_ARROW     = "#444444"

def box(ax, x, y, w, h, label, sublabel=None, color="#457B9D", fontsize=9,
        radius=0.03, alpha=0.92):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={radius}",
                       facecolor=color, edgecolor="white", linewidth=1.5,
                       alpha=alpha, zorder=3)
    ax.add_patch(b)
    cy = y + h/2 + (0.025 if sublabel else 0)
    ax.text(x + w/2, cy, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white", zorder=4)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.035, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color="white", alpha=0.9, zorder=4)


def arrow(ax, x1, y1, x2, y2, color=C_ARROW, label=None, lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.01, my, label, fontsize=7, color=color,
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))


def curved_arrow(ax, x1, y1, x2, y2, color=C_ARROW, rad=0.25, label=None, lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        offset = 0.08 * rad / abs(rad)
        ax.text(mx, my + offset, label, fontsize=7, color=color,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))


def main():
    out_dir = ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(C_BG)
    fig.patch.set_facecolor(C_BG)

    # ── Title ────────────────────────────────────────────────────────────────
    ax.text(0.5, 0.97, "EEG-RAG: Retrieval-Augmented Motor Imagery Decoding",
            ha="center", va="top", fontsize=14, fontweight="bold", color="#222")
    ax.text(0.5, 0.93, "FAISS Embedding Store · K-NN Retrieval · Cross-Attention Reasoning",
            ha="center", va="top", fontsize=9.5, color="#555", style="italic")

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 1 — EEG INPUT + EEGNET ENCODER
    # ══════════════════════════════════════════════════════════════════════════
    ax.text(0.08, 0.88, "INPUT", fontsize=8, ha="center", color="#888", fontstyle="italic")

    # EEG Input
    box(ax, 0.01, 0.77, 0.13, 0.09, "EEG Trial x",
        "ℝ^{C×T}  (64×640)", color="#555", fontsize=8)

    # ── EEGNet Backbone stages ──────────────────────────────────────────────
    ax.text(0.38, 0.88, "EEGNet ENCODER  (2.6K params, frozen after Stage 1)",
            fontsize=8, ha="center", color=C_ENCODER, fontweight="bold")

    box(ax, 0.16, 0.77, 0.14, 0.09, "Temporal\nConv2d", "(B,8,C,T)", color=C_ENCODER, fontsize=7.5)
    box(ax, 0.32, 0.77, 0.14, 0.09, "DepthwiseConv\n+ BatchNorm", "(B,16,1,T)", color=C_ENCODER, fontsize=7.5)
    box(ax, 0.48, 0.77, 0.14, 0.09, "SepConv +\nAvgPool×2", "(B,16,1,T/32)", color=C_ENCODER, fontsize=7.5)
    box(ax, 0.64, 0.77, 0.13, 0.09, "Flatten → 320\ndim features", "(B, 320)", color=C_ENCODER, fontsize=7.5)
    box(ax, 0.79, 0.77, 0.13, 0.09, "Linear(320→256)\n+ L2Norm", "q ∈ ℝ²⁵⁶", color=C_ENCODER, fontsize=7.5)

    # Arrows through encoder
    for x1, x2 in [(0.14, 0.16), (0.30, 0.32), (0.46, 0.48), (0.62, 0.64), (0.77, 0.79)]:
        arrow(ax, x1, 0.815, x2, 0.815)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 2 — FAISS STORE (OFFLINE) + RETRIEVAL
    # ══════════════════════════════════════════════════════════════════════════
    ax.text(0.5, 0.71, "─" * 95, ha="center", color="#ccc", fontsize=6)

    # FAISS store (offline build, left side)
    ax.text(0.18, 0.68, "OFFLINE: Build Embedding Store (training set)",
            fontsize=8, ha="center", color=C_STORE, fontweight="bold")

    box(ax, 0.01, 0.54, 0.13, 0.12, "Training DB",
        "3,060 trials", color="#555", fontsize=8)
    box(ax, 0.17, 0.54, 0.13, 0.12, "EEGNet\nEncoder", "same weights", color=C_STORE, fontsize=8)
    box(ax, 0.33, 0.54, 0.15, 0.12, "FAISS Index\n(IndexFlatIP)",
        "3,060 × 256\ncosine sim", color=C_STORE, fontsize=8)

    arrow(ax, 0.14, 0.60, 0.17, 0.60)
    arrow(ax, 0.30, 0.60, 0.33, 0.60)

    # Retrieval (middle)
    ax.text(0.67, 0.68, "ONLINE: K-NN Retrieval at Inference",
            fontsize=8, ha="center", color=C_RETRIEVAL, fontweight="bold")

    box(ax, 0.50, 0.54, 0.15, 0.12, "FAISS Search\n(K=5 NN)",
        "exclude query\nsubject", color=C_RETRIEVAL, fontsize=8)
    box(ax, 0.67, 0.54, 0.15, 0.12, "Retrieved Set\n{(eᵢ, yᵢ)}ᵢ₌₁ᴷ",
        "K embeddings\n+ class labels", color=C_RETRIEVAL, fontsize=8)

    arrow(ax, 0.65, 0.60, 0.67, 0.60)

    # Query q → FAISS search
    curved_arrow(ax, 0.855, 0.77, 0.575, 0.66, color=C_ENCODER, rad=-0.2,
                 label="q ∈ ℝ²⁵⁶")

    # FAISS index → FAISS search (dashed, offline→online)
    ax.annotate("", xy=(0.50, 0.60), xytext=(0.48, 0.60),
                arrowprops=dict(arrowstyle="->", color=C_STORE, lw=1.5, ls="dashed"))

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 3 — RAG REASONING HEAD
    # ══════════════════════════════════════════════════════════════════════════
    ax.text(0.5, 0.49, "─" * 95, ha="center", color="#ccc", fontsize=6)
    ax.text(0.5, 0.46, "RAG REASONING HEAD  (~526K params)",
            fontsize=9, ha="center", color=C_RAG, fontweight="bold")

    # RAG steps
    box(ax, 0.05, 0.29, 0.14, 0.12, "Label\nEnrichment",
        "eᵢ + embed(yᵢ)\n→ C ∈ ℝ^{K×256}", color=C_RAG, fontsize=7.5)
    box(ax, 0.22, 0.29, 0.15, 0.12, "Cross-Attention\n(×2 layers)",
        "Q=q, KV=C\n4 heads, pre-LN", color=C_RAG, fontsize=7.5)
    box(ax, 0.40, 0.29, 0.14, 0.12, "MLP Head",
        "[q ‖ ctx] → 512\n→ 256 → 4 logits", color=C_RAG, fontsize=7.5)
    box(ax, 0.57, 0.29, 0.14, 0.12, "Soft Majority\nVote",
        "Σ sim(eᵢ,q)·𝟙[yᵢ=c]\n→ vote logits", color=C_RAG, fontsize=7.5)
    box(ax, 0.74, 0.29, 0.14, 0.12, "α-Mixture\n(learned)",
        "α·logits_mlp\n+(1-α)·vote", color=C_RAG, fontsize=7.5)

    arrow(ax, 0.19, 0.35, 0.22, 0.35)
    arrow(ax, 0.37, 0.35, 0.40, 0.35)
    arrow(ax, 0.54, 0.35, 0.57, 0.35)
    arrow(ax, 0.71, 0.35, 0.74, 0.35)

    # q → Label Enrichment
    curved_arrow(ax, 0.855, 0.77, 0.12, 0.41, color=C_ENCODER, rad=0.3, label="q")

    # Retrieved set → Label Enrichment
    arrow(ax, 0.745, 0.54, 0.12, 0.41)

    # ══════════════════════════════════════════════════════════════════════════
    # OUTPUT
    # ══════════════════════════════════════════════════════════════════════════
    box(ax, 0.30, 0.13, 0.20, 0.11, "Classification",
        "argmax → class\n(0–3)", color=C_OUT, fontsize=8)
    box(ax, 0.55, 0.13, 0.20, 0.11, "Confidence +\nExplanation",
        "retrieval evidence\n+ clinical text", color=C_OUT, fontsize=8)

    arrow(ax, 0.88, 0.35, 0.40, 0.24)
    arrow(ax, 0.50, 0.29, 0.40, 0.24)
    arrow(ax, 0.65, 0.24, 0.55, 0.24)

    # ══════════════════════════════════════════════════════════════════════════
    # LEGEND
    # ══════════════════════════════════════════════════════════════════════════
    legend_items = [
        mpatches.Patch(color=C_ENCODER,   label="EEGNet Encoder (Stage 1 training)"),
        mpatches.Patch(color=C_STORE,     label="FAISS Embedding Store (offline build)"),
        mpatches.Patch(color=C_RETRIEVAL, label="K-NN Retrieval (online inference)"),
        mpatches.Patch(color=C_RAG,       label="RAG Reasoning Head (Stage 2 training)"),
        mpatches.Patch(color=C_OUT,       label="Output: class + explanation"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=8,
              framealpha=0.9, edgecolor="#ccc",
              bbox_to_anchor=(0.01, 0.01))

    # ── Save ─────────────────────────────────────────────────────────────────
    plt.tight_layout(pad=0.5)
    for fmt in ["pdf", "png"]:
        path = out_dir / f"eeg_rag_architecture.{fmt}"
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=C_BG)
        print(f"  Saved → {path}")
    plt.close()
    print("EEG-RAG architecture diagram complete.")


if __name__ == "__main__":
    main()
