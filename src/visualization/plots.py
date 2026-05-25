"""
Publication-quality visualization for STELLA paper figures.

All figures use a consistent style suitable for IEEE TNSRE / NeurIPS.
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.5,
})


def plot_training_curves(
    history: dict[str, list],
    save_path: str = "results/figures/training_curves.pdf",
    title: str = "STELLA Pretraining Loss",
) -> None:
    """Plot pretraining loss curves (total + per-objective)."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    keys_train = [k for k in history if k.startswith("train_")]
    keys_val = [k.replace("train_", "val_") for k in keys_train]

    n_plots = min(len(keys_train), 5)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 3))
    if n_plots == 1:
        axes = [axes]

    for ax, kt, kv in zip(axes, keys_train[:n_plots], keys_val[:n_plots]):
        label = kt.replace("train_", "")
        epochs = range(1, len(history[kt]) + 1)
        ax.plot(epochs, history[kt], label="train", color="#2878B5")
        if kv in history:
            ax.plot(epochs, history[kv], label="val", color="#C82423", linestyle="--")
        ax.set_title(label.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | None = None,
    save_path: str = "results/figures/confusion_matrix.pdf",
    title: str = "Confusion Matrix",
) -> None:
    """Plot normalized confusion matrix."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    n = cm.shape[0]
    if class_names is None:
        class_names = [f"C{i}" for i in range(n)]

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(n + 1, n + 1))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.2f})",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_ablation_table(
    results: dict[str, dict],
    save_path: str = "results/figures/ablation_table.pdf",
    title: str = "Ablation Study",
) -> None:
    """
    Render ablation results as a table figure.

    Parameters
    ----------
    results : dict mapping model_name → {accuracy, balanced_accuracy, kappa, n_params}
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    models = list(results.keys())
    metrics = ["accuracy", "balanced_accuracy", "kappa"]

    fig, ax = plt.subplots(figsize=(len(models) * 1.2 + 2, 3))
    ax.axis("off")

    # Build table data
    rows = []
    for model in models:
        m = results[model]
        rows.append([
            model,
            f"{m.get('accuracy', 0):.3f}",
            f"{m.get('balanced_accuracy', 0):.3f}",
            f"{m.get('kappa', 0):.3f}",
            f"{m.get('n_params', 0)/1e6:.2f}M",
        ])

    col_labels = ["Model", "Acc", "BalAcc", "κ", "Params"]
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(True)
    table.scale(1.2, 1.5)
    ax.set_title(title, fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_spectral_weights(
    model,
    save_path: str = "results/figures/spectral_weights.pdf",
) -> None:
    """Visualize learned spectral band mixing weights."""
    import torch

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        band_mix = model.encoder.tokenizer.spectral.band_mix.detach().cpu()
        band_mix_sig = torch.sigmoid(band_mix).numpy()  # (n_bands, n_freqs)
        band_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    except AttributeError:
        return

    n_bands = band_mix_sig.shape[0]
    fig, axes = plt.subplots(1, n_bands, figsize=(12, 2.5))

    for i, (ax, name) in enumerate(zip(axes, band_names)):
        ax.plot(band_mix_sig[i], color=f"C{i}")
        ax.set_title(name)
        ax.set_xlabel("Freq bin")
        ax.set_ylabel("Weight")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Learned Spectral Band Mixing Weights", fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_mamba_attention(
    save_path: str = "results/figures/mamba_states.pdf",
) -> None:
    """Placeholder — visualize S3M state evolution."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.text(0.5, 0.5, "S3M state visualization\n(see notebooks/07_figure_generation.ipynb)",
            ha="center", va="center", transform=ax.transAxes)
    ax.set_title("S3M State Evolution")
    ax.axis("off")
    fig.savefig(save_path)
    plt.close(fig)


def plot_per_subject_boxplot(
    results_dict: dict[str, list[float]],
    metric: str = "accuracy",
    save_path: str = "results/figures/per_subject_boxplot.pdf",
) -> None:
    """
    Boxplot of per-subject metric for each model.

    Parameters
    ----------
    results_dict : {"ModelName": [per-subject accuracy values], ...}
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    models = list(results_dict.keys())
    data = [results_dict[m] for m in models]

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), 4))
    bp = ax.boxplot(data, labels=models, patch_artist=True, notch=False)

    colors = [f"C{i}" for i in range(len(models))]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Per-Subject {metric.replace('_', ' ').title()} Distribution")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
