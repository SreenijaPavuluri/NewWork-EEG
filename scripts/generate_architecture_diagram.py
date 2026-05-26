#!/usr/bin/env python3
"""
Generate a publication-quality STELLA architecture diagram.
Output: results/figures/stella_architecture.pdf  +  .png

Run:
    python scripts/generate_architecture_diagram.py
"""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette ─────────────────────────────────────────────────────────────
C_INPUT   = "#2C3E50"   # dark navy
C_TEMP    = "#1A6B8A"   # teal-blue  (temporal stream)
C_SPEC    = "#B35C00"   # burnt-orange (spectral stream)
C_SPATIAL = "#4A7C59"   # forest green
C_FUSION  = "#6B3FA0"   # purple
C_MAMBA   = "#C0392B"   # crimson
C_HEAD    = "#34495E"   # slate
C_ARROW   = "#555555"
C_DIM     = "#888888"   # dimension annotation colour
C_BG      = "#FAFAFA"

def draw_box(ax, x, y, w, h, label, sublabel="", color="#2C3E50",
             fontsize=9, sub_fontsize=7.5, alpha=0.9, radius=0.3):
    """Draw a rounded box with title + optional dimension sublabel."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad=0.08,rounding_size={radius}",
                          linewidth=1.2, edgecolor=color,
                          facecolor=color + "22", zorder=3)
    ax.add_patch(box)
    ax.text(x, y + (0.10 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=color, zorder=4)
    if sublabel:
        ax.text(x, y - 0.20, sublabel,
                ha="center", va="center", fontsize=sub_fontsize,
                color=C_DIM, style="italic", zorder=4)

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.3, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=5)

def curved_arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.3, rad=0.25):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=lw, connectionstyle=f"arc3,rad={rad}"),
                zorder=5)

def dim_label(ax, x, y, text, color=C_DIM, fontsize=7):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=color, style="italic")

# ══════════════════════════════════════════════════════════════════════════════
# Build figure
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 22))
ax.set_xlim(0, 16); ax.set_ylim(0, 22)
ax.set_aspect("equal"); ax.axis("off")
ax.set_facecolor(C_BG); fig.patch.set_facecolor(C_BG)

ax.text(8, 21.4, "STELLA: Spectro-Temporal EEG Learning with Latent Alignment",
        ha="center", va="center", fontsize=13, fontweight="bold", color=C_INPUT)
ax.text(8, 21.0, "Architecture Overview — PhysioNet EEGMMIDB (64ch, 160 Hz, 4-class MI)",
        ha="center", va="center", fontsize=9, color=C_DIM)

# ── INPUT ─────────────────────────────────────────────────────────────────────
draw_box(ax, 8, 20.2, 6, 0.7, "EEG Input", "(B, C=64, T=640)", color=C_INPUT)

# ── SPLIT ARROW ───────────────────────────────────────────────────────────────
ax.annotate("", xy=(4.5, 18.9), xytext=(6.5, 19.85),
            arrowprops=dict(arrowstyle="->", color=C_TEMP, lw=1.5), zorder=5)
ax.annotate("", xy=(11.5, 18.9), xytext=(9.5, 19.85),
            arrowprops=dict(arrowstyle="->", color=C_SPEC, lw=1.5), zorder=5)
ax.text(4.0, 19.3, "Temporal Stream", fontsize=8, color=C_TEMP, fontweight="bold")
ax.text(10.5, 19.3, "Spectral Stream", fontsize=8, color=C_SPEC, fontweight="bold")

# ── TEMPORAL STREAM (left column, x≈4.5) ──────────────────────────────────────
TX = 4.5
# DW Conv
draw_box(ax, TX, 18.55, 4.8, 0.65,
         "Depthwise Conv1d",
         "k=40, stride=20, groups=64  →  (B, 64, 31)",
         color=C_TEMP, fontsize=8.5)
arrow(ax, TX, 18.22, TX, 17.65, color=C_TEMP)

# Channel Projection
draw_box(ax, TX, 17.35, 4.8, 0.55,
         "Channel Proj  Linear(64→64)  +  GELU",
         "(B, 31, 64)",
         color=C_TEMP, fontsize=8.5)
arrow(ax, TX, 17.07, TX, 16.55, color=C_TEMP)

# Temporal Projection
draw_box(ax, TX, 16.25, 4.8, 0.55,
         "Temporal Proj  Linear(64→256)  +  LayerNorm",
         "(B, 31, 256)",
         color=C_TEMP, fontsize=8.5)
arrow(ax, TX, 15.97, TX, 15.45, color=C_TEMP)

# Positional Embed
draw_box(ax, TX, 15.15, 4.8, 0.55,
         "Learnable Positional Embedding",
         "Embedding(65, 256)  →  (B, 31, 256)",
         color=C_TEMP, fontsize=8.5)

# ── SPECTRAL STREAM (right column, x≈11.5) ────────────────────────────────────
SX = 11.5
# FFT
draw_box(ax, SX, 18.55, 4.8, 0.65,
         "Real FFT  (n_fft=256)",
         "|X(f)|²  →  (B, 64, 129)",
         color=C_SPEC, fontsize=8.5)
arrow(ax, SX, 18.22, SX, 17.65, color=C_SPEC)

# Band Filter
draw_box(ax, SX, 17.35, 4.8, 0.55,
         "Gaussian Band Filters × 5 bands",
         "δ θ α β γ  (learnable mix)  →  (B, 64, 5)",
         color=C_SPEC, fontsize=8.5)
arrow(ax, SX, 17.07, SX, 16.55, color=C_SPEC)

# Band Projection
draw_box(ax, SX, 16.25, 4.8, 0.55,
         "Band Proj  Linear(64→128→256)  +  GELU",
         "log-power  →  (B, 5, 256)",
         color=C_SPEC, fontsize=8.5)
arrow(ax, SX, 15.97, SX, 15.45, color=C_SPEC)

# Band Positional Embed
draw_box(ax, SX, 15.15, 4.8, 0.55,
         "Band Positional Embedding",
         "Embedding(5, 256)  →  (B, 5, 256)",
         color=C_SPEC, fontsize=8.5)

# param labels
ax.text(TX, 13.92, "Tokenizer  params = 354,693", ha="center",
        fontsize=7.5, color=C_DIM, style="italic")

# ── MERGE ARROWS ──────────────────────────────────────────────────────────────
ax.annotate("", xy=(6.8, 14.05), xytext=(TX, 14.87),
            arrowprops=dict(arrowstyle="->", color=C_TEMP, lw=1.5), zorder=5)
ax.annotate("", xy=(9.2, 14.05), xytext=(SX, 14.87),
            arrowprops=dict(arrowstyle="->", color=C_SPEC, lw=1.5), zorder=5)

# ── SPATIAL TRANSFORMER ───────────────────────────────────────────────────────
ST_Y = 13.6
draw_box(ax, 8, ST_Y, 10.5, 0.75,
         "Channel Spatial Transformer  (2 layers, 4 heads, d=256)",
         "Pre-LN  →  MHA [Q,K,V: Linear(256→256)]  →  GLU-FFN [Linear(256→512→256)]  →  (B, 31, 256)",
         color=C_SPATIAL, fontsize=9)
ax.text(8, 13.10, "Topology-aware attention bias  (64×64 learnable)  ·  params = 1,348,608",
        ha="center", fontsize=7.5, color=C_DIM, style="italic")

arrow(ax, 8, 13.22, 8, 12.72, color=C_SPATIAL, lw=1.5)

# ── GATED SPECTRAL FUSION ─────────────────────────────────────────────────────
GF_Y = 12.38
draw_box(ax, 8, GF_Y, 10.5, 0.75,
         "Gated Spectral-Temporal Fusion",
         "Q=temporal(B,31,256)  ·  K,V=spectral(B,5,256)  →  Cross-Attn(4h)  →  gate=σ(Linear(512→256))  →  (B,31,256)",
         color=C_FUSION, fontsize=9)

# mini gate diagram
ax.text(8, 11.88,
        "gate = σ([temporal ‖ ctx])  ·  fused = temporal + gate ⊙ ctx   ·   params = 395,008",
        ha="center", fontsize=7.5, color=C_DIM, style="italic")

arrow(ax, 8, 12.00, 8, 11.52, color=C_FUSION, lw=1.5)

# ── PREPEND CLS ───────────────────────────────────────────────────────────────
draw_box(ax, 8, 11.22, 5.5, 0.55,
         "Prepend [CLS] token",
         "(B, 31, 256)  →  cat([CLS], seq)  →  (B, 32, 256)",
         color=C_MAMBA, fontsize=8.5)
arrow(ax, 8, 10.95, 8, 10.45, color=C_MAMBA, lw=1.5)

# ── S3M / TEMPORAL MAMBA ──────────────────────────────────────────────────────
MB_Y = 10.0
draw_box(ax, 8, MB_Y, 10.5, 0.80,
         "Temporal Mamba  —  3 × S3M Block  (CPU-native Selective SSM)",
         "in_proj(256→1024) → DW-Conv1d(k=4) → x_proj(512→48) → selective scan → out_proj(512→256)",
         color=C_MAMBA, fontsize=9)

# S3M detail box
s3m_detail = (
    "S3M state update:  h_t = diag(A_t) h_{t−1} + B_t x_t  ·  y_t = C_t h_t + D x_t\n"
    "d_inner=512  ·  d_state=16  ·  dt_rank=16  ·  per-block params≈438,443  ·  total params=1,315,328"
)
ax.text(8, 9.47, s3m_detail, ha="center", fontsize=7.5, color=C_DIM,
        style="italic", multialignment="center")

arrow(ax, 8, 9.59, 8, 9.10, color=C_MAMBA, lw=1.5)

# ── FINAL NORM ────────────────────────────────────────────────────────────────
draw_box(ax, 8, 8.80, 5.5, 0.55,
         "Final LayerNorm  +  CLS readout",
         "seq[:, 0]  →  (B, 256)  encoder representation",
         color=C_INPUT, fontsize=8.5)

arrow(ax, 8, 8.52, 8, 8.02, color=C_INPUT, lw=1.5)

# ── DOWNSTREAM HEAD ───────────────────────────────────────────────────────────
draw_box(ax, 8, 7.72, 6.5, 0.55,
         "Downstream MLP Head",
         "LayerNorm  →  Linear(256→128)  →  GELU  →  Dropout  →  Linear(128→4)",
         color=C_HEAD, fontsize=8.5)

arrow(ax, 8, 7.44, 8, 6.94, color=C_HEAD, lw=1.5)

draw_box(ax, 8, 6.64, 4.5, 0.55,
         "Output Logits  (B, 4)",
         "Left fist  ·  Right fist  ·  Both fists  ·  Both feet",
         color=C_HEAD, fontsize=8.5)

# ── PRETRAINING HEADS (side annotation) ───────────────────────────────────────
ax.text(15.4, 11.0, "Pretraining Heads\n(discarded after pretraining)",
        ha="center", va="center", fontsize=8, color=C_DIM,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0",
                  edgecolor=C_DIM, linewidth=0.8))
for yy, txt in zip([10.5, 9.9, 9.3, 8.7],
                   ["L₁: NT-Xent\n(contrastive)", "L₂: Aux Sup.\n(CE loss)",
                    "L₃: VICReg\n(spectral-temp)", "L₄: MMD\n(domain align)"]):
    draw_box(ax, 15.4, yy, 1.8, 0.48, txt, color=C_DIM,
             fontsize=7, sub_fontsize=6.5, radius=0.2)
    curved_arrow(ax, 14.4, 10.0, 14.52, yy, color=C_DIM, lw=1.0, rad=-0.3)

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────
summary_y = 5.9
ax.add_patch(FancyBboxPatch((0.5, 4.2), 15, 1.50,
                             boxstyle="round,pad=0.2,rounding_size=0.3",
                             linewidth=1.0, edgecolor="#CCCCCC",
                             facecolor="#F5F5F5", zorder=2))
ax.text(8, 5.80, "Model Summary", ha="center", fontsize=9.5,
        fontweight="bold", color=C_INPUT)

headers = ["Module", "Output Shape", "Params"]
cols    = [2.8, 8.5, 13.5]
rows_data = [
    ("Temporal Patch Embed",       "(B, 31, 256)",  "  88,512"),
    ("Spectral Band Encoder",      "(B,  5, 256)",  "  47,941"),
    ("Channel Spatial Transformer","(B, 31, 256)",  "1,348,608"),
    ("Gated Spectral Fusion",      "(B, 31, 256)",  "  395,008"),
    ("S3M Temporal Mamba ×3",      "(B, 32, 256)",  "1,315,328"),
    ("CLS + Final LayerNorm",      "(B,    256)",   "     256 (CLS param)"),
    ("Encoder Total",              "(B,    256)",   "3,414,405"),
]

for i, (h, c) in enumerate(zip(headers, cols)):
    ax.text(c, 5.52, h, ha="center", fontsize=8, fontweight="bold", color="#333333")

ax.axhline(5.42, xmin=0.04, xmax=0.96, color="#BBBBBB", lw=0.8)

row_y = [5.28, 5.08, 4.88, 4.68, 4.48, 4.38, 4.28]
for i, (name, shape, params) in enumerate(rows_data):
    y = 5.28 - i * 0.17
    c = C_MAMBA if "Total" in name else "#222222"
    fw = "bold" if "Total" in name else "normal"
    ax.text(cols[0], y, name, ha="center", fontsize=7.5, color=c, fontweight=fw)
    ax.text(cols[1], y, shape, ha="center", fontsize=7.5, color=c, fontweight=fw,
            family="monospace")
    ax.text(cols[2], y, params, ha="center", fontsize=7.5, color=c, fontweight=fw,
            family="monospace")

# ── LEGEND ────────────────────────────────────────────────────────────────────
legend_items = [
    (C_TEMP,    "Temporal Stream"),
    (C_SPEC,    "Spectral Stream"),
    (C_SPATIAL, "Spatial Transformer"),
    (C_FUSION,  "Gated Fusion"),
    (C_MAMBA,   "S3M / Mamba"),
    (C_HEAD,    "Classification Head"),
]
for i, (col, label) in enumerate(legend_items):
    x = 1.0 + i * 2.5
    ax.add_patch(mpatches.Rectangle((x, 3.65), 0.3, 0.22,
                                     facecolor=col+"44", edgecolor=col, lw=1.0))
    ax.text(x + 0.42, 3.76, label, va="center", fontsize=7.5, color=col)

ax.text(8, 3.35,
        "All operations are CPU-compatible  ·  No CUDA dependencies  ·  "
        "S3M: O(L·d·N) sequential scan  ·  Encoder: 3.41M params total",
        ha="center", fontsize=8, color=C_DIM, style="italic")

plt.tight_layout(pad=0.5)
plt.savefig(OUT_DIR / "stella_architecture.pdf", dpi=300, bbox_inches="tight",
            facecolor=C_BG)
plt.savefig(OUT_DIR / "stella_architecture.png", dpi=200, bbox_inches="tight",
            facecolor=C_BG)
plt.close()
print(f"[DONE] Architecture diagram saved to {OUT_DIR}/stella_architecture.pdf")
print(f"       High-res PNG also saved: {OUT_DIR}/stella_architecture.png")
