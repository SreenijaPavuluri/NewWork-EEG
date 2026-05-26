#!/usr/bin/env python3
"""
Post-experiment analysis: load JSON results, generate paper figures, and print
LaTeX/Markdown tables for the STELLA manuscript.

Usage:
    python scripts/analyze_results.py

Outputs:
    results/figures/comparison_bar.pdf
    results/figures/ablation_bar.pdf
    results/figures/confusion_matrix.pdf   (if per-trial preds saved)
    results/tables/main_results.tex
    results/tables/ablation_results.tex
    results/tables/main_results.md
    results/tables/ablation_results.md
"""
import json, sys, os
from pathlib import Path
import numpy as np

RESULTS_DIR = Path("results/real_experiment")
FIG_DIR     = Path("results/figures")
TAB_DIR     = Path("results/tables")

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────

def _pm(mean, std, fmt=".4f"):
    return f"{mean:{fmt}} ± {std:{fmt}}"

BASELINE_PARAMS = {
    "EEGNet":             2_624,
    "ShallowConvNet":    40_890,
    "DeepConvNet":       93_514,
    "VanillaTransformer": 3_145_728,
    "CNNTransformer":    1_835_008,
    "MIRepNetBaseline":  1_212_416,
    "STELLA":            3_414_405,
}

DISPLAY_NAMES = {
    "EEGNet":             "EEGNet [14]",
    "ShallowConvNet":     "ShallowConvNet [15]",
    "DeepConvNet":        "DeepConvNet [15]",
    "VanillaTransformer": "Vanilla Transformer",
    "CNNTransformer":     "CNN + Transformer",
    "MIRepNetBaseline":   "MIRepNet-style [11]",
    "STELLA":             "\\textbf{STELLA (ours)}",
}

ORDER = ["EEGNet", "ShallowConvNet", "DeepConvNet",
         "VanillaTransformer", "CNNTransformer", "MIRepNetBaseline", "STELLA"]

# ── load main results ──────────────────────────────────────────────────────────

def load_main():
    p = RESULTS_DIR / "aggregated_results.json"
    if not p.exists():
        print(f"[WARN] {p} not found — run scripts/run_real_experiments.py first")
        return None
    with open(p) as f:
        return json.load(f)

def load_ablation():
    p = RESULTS_DIR / "ablation_results.json"
    if not p.exists():
        print(f"[WARN] {p} not found — run scripts/run_ablation_real.py first")
        return None
    with open(p) as f:
        return json.load(f)

# ── markdown tables ────────────────────────────────────────────────────────────

def print_main_md(data):
    rows = []
    for key in ORDER:
        if key not in data:
            continue
        d  = data[key]
        mn = d.get("accuracy_mean", d.get("accuracy", 0))
        sd = d.get("accuracy_std",  0)
        bm = d.get("balanced_accuracy_mean", d.get("balanced_accuracy", 0))
        bs = d.get("balanced_accuracy_std", 0)
        fm = d.get("f1_mean", d.get("f1", 0))
        fs = d.get("f1_std", 0)
        km = d.get("kappa_mean", d.get("kappa", 0))
        ks = d.get("kappa_std", 0)
        np_ = d.get("n_params", BASELINE_PARAMS.get(key, "?"))
        rows.append((key, mn, sd, bm, bs, fm, fs, km, ks, np_))

    print("\n### Main Results (PhysioNet EEGMMIDB, subject-independent, 3 seeds)\n")
    hdr = "| Model | Acc (mean±std) | BalAcc | F1-macro | κ | #Params |"
    sep = "|-------|----------------|--------|----------|---|---------|"
    print(hdr); print(sep)
    for (key, mn, sd, bm, bs, fm, fs, km, ks, np_) in rows:
        star = " **" if key == "STELLA" else ""
        name = key + star
        print(f"| {name:<30} | {mn:.4f} ± {sd:.4f} | {bm:.4f} ± {bs:.4f} | "
              f"{fm:.4f} ± {fs:.4f} | {km:.4f} ± {ks:.4f} | {np_:,} |")
    return rows

def print_ablation_md(data):
    full_acc = data.get("STELLA (full)", {}).get("accuracy", 0)
    print("\n### Ablation Study (PhysioNet EEGMMIDB, 30 subjects, seed=42)\n")
    hdr = "| Variant | Acc | BalAcc | κ | ΔAcc vs. full | #Params |"
    sep = "|---------|-----|--------|---|---------------|---------|"
    print(hdr); print(sep)
    for name, d in data.items():
        acc = d.get("accuracy", 0)
        bal = d.get("balanced_accuracy", 0)
        kap = d.get("kappa", 0)
        npa = d.get("n_params", "?")
        delta = full_acc - acc
        dir_  = "↓" if delta > 0 else ("↑" if delta < 0 else "–")
        dstr  = f"{dir_}{abs(delta):.4f}" if delta != 0 else "–"
        print(f"| {name:<35} | {acc:.4f} | {bal:.4f} | {kap:.4f} | {dstr:>13} | {npa:>10,} |")

# ── LaTeX tables ──────────────────────────────────────────────────────────────

def latex_main(data):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Subject-independent 4-class MI classification on PhysioNet EEGMMIDB"
                 r" (3 random seeds, mean $\pm$ std). Best result \textbf{bolded}.}")
    lines.append(r"\label{tab:main}")
    lines.append(r"\resizebox{\columnwidth}{!}{")
    lines.append(r"\begin{tabular}{lccccr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Accuracy & Bal.\ Acc. & F1-macro & $\kappa$ & \#Params \\")
    lines.append(r"\midrule")

    all_accs = []
    for key in ORDER:
        if key not in data:
            continue
        d = data[key]
        all_accs.append(d.get("accuracy_mean", d.get("accuracy", 0)))
    best_acc = max(all_accs) if all_accs else 0

    for key in ORDER:
        if key not in data:
            continue
        d  = data[key]
        mn = d.get("accuracy_mean", d.get("accuracy", 0))
        sd = d.get("accuracy_std",  0)
        bm = d.get("balanced_accuracy_mean", d.get("balanced_accuracy", 0))
        bs = d.get("balanced_accuracy_std", 0)
        fm = d.get("f1_mean", d.get("f1", 0))
        fs = d.get("f1_std", 0)
        km = d.get("kappa_mean", d.get("kappa", 0))
        ks = d.get("kappa_std", 0)
        np_ = d.get("n_params", BASELINE_PARAMS.get(key, 0))
        name = DISPLAY_NAMES.get(key, key)

        def fmt(m, s):
            v = f"{m:.4f}$\\pm${s:.4f}"
            if abs(m - best_acc) < 1e-6:
                v = f"\\textbf{{{m:.4f}}}$\\pm${s:.4f}"
            return v

        lines.append(f"{name} & {fmt(mn,sd)} & {bm:.4f}$\\pm${bs:.4f} & "
                     f"{fm:.4f}$\\pm${fs:.4f} & {km:.4f}$\\pm${ks:.4f} & {np_:,} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

def latex_ablation(data):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation study: contribution of each STELLA component"
                 r" on PhysioNet EEGMMIDB (30 subjects, seed=42).}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\begin{tabular}{lcccr}")
    lines.append(r"\toprule")
    lines.append(r"Variant & Acc & Bal.\ Acc. & $\kappa$ & $\Delta$Acc \\")
    lines.append(r"\midrule")

    full_acc = data.get("STELLA (full)", {}).get("accuracy", 0)
    for name, d in data.items():
        acc = d.get("accuracy", 0)
        bal = d.get("balanced_accuracy", 0)
        kap = d.get("kappa", 0)
        delta = full_acc - acc
        dstr = f"${'-' if delta>0 else '+'}{abs(delta):.4f}$" if delta != 0 else "–"
        bold = r"\textbf{" if name == "STELLA (full)" else ""
        endb = r"}" if name == "STELLA (full)" else ""
        lines.append(f"{bold}{name}{endb} & {acc:.4f} & {bal:.4f} & {kap:.4f} & {dstr} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

# ── figures ───────────────────────────────────────────────────────────────────

def make_figures(main_data, abl_data):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[WARN] matplotlib not available — skipping figures")
        return

    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    # ── 1. Main comparison bar chart ──────────────────────────────────────────
    if main_data:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        names, accs, errs, colors = [], [], [], []
        for key in ORDER:
            if key not in main_data:
                continue
            d = main_data[key]
            names.append(key.replace("VanillaTransformer","VanillaTx").replace("CNNTransformer","CNN+Tx")
                         .replace("MIRepNetBaseline","MIRepNet"))
            accs.append(d.get("accuracy_mean", d.get("accuracy", 0)))
            errs.append(d.get("accuracy_std", 0))
            colors.append("#D62828" if key == "STELLA" else "#457B9D")

        x = np.arange(len(names))
        bars = ax.bar(x, accs, yerr=errs, capsize=4, color=colors,
                      edgecolor="white", linewidth=0.8, error_kw=dict(elinewidth=1.2))
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel("Accuracy (subject-independent)")
        ax.set_title("STELLA vs. Baselines — PhysioNet EEGMMIDB (4-class MI)")
        ax.set_ylim(0, min(1.0, max(accs) + 0.15))
        ax.axhline(0.25, color="gray", ls="--", lw=1, label="Chance (4-class)")
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{acc:.3f}", ha="center", va="bottom", fontsize=9)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "comparison_bar.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(FIG_DIR / "comparison_bar.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[FIG] Saved comparison bar chart → {FIG_DIR}/comparison_bar.pdf")

    # ── 2. Ablation bar chart ─────────────────────────────────────────────────
    if abl_data:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        names = list(abl_data.keys())
        accs  = [abl_data[n].get("accuracy", 0) for n in names]
        full_acc = abl_data.get("STELLA (full)", {}).get("accuracy", 0)
        colors = ["#D62828" if n == "STELLA (full)" else "#457B9D" for n in names]

        x = np.arange(len(names))
        bars = ax.bar(x, accs, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("w/o ","–").replace("STELLA ","STELLA\n") for n in names],
                            rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Accuracy (subject-independent)")
        ax.set_title("Ablation Study — STELLA Component Contributions")
        ax.axhline(full_acc, color="#D62828", ls="--", lw=1.2,
                   label=f"Full STELLA ({full_acc:.4f})")
        ax.set_ylim(0, min(1.0, max(accs) + 0.12))
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{acc:.3f}", ha="center", va="bottom", fontsize=9)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "ablation_bar.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(FIG_DIR / "ablation_bar.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[FIG] Saved ablation bar chart → {FIG_DIR}/ablation_bar.pdf")

    # ── 3. Per-metric radar (optional) ───────────────────────────────────────
    if main_data and "STELLA" in main_data:
        metrics = ["accuracy", "balanced_accuracy", "f1", "kappa"]
        labels  = ["Accuracy", "Bal. Acc.", "F1-macro", "κ"]
        model_keys = ["EEGNet", "ShallowConvNet", "VanillaTransformer", "STELLA"]
        model_colors = ["#A8DADC", "#457B9D", "#E63946", "#D62828"]

        vals = {}
        for k in model_keys:
            if k not in main_data:
                continue
            d = main_data[k]
            vals[k] = [d.get(f"{m}_mean", d.get(m, 0)) for m in metrics]

        if len(vals) < 2:
            return

        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(metrics))
        width = 0.8 / len(vals)
        for i, (k, v) in enumerate(vals.items()):
            offset = (i - len(vals)/2 + 0.5) * width
            ax.bar(x + offset, v, width, label=k,
                   color=model_colors[list(vals.keys()).index(k)],
                   alpha=0.85, edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("Score"); ax.set_ylim(0, 1)
        ax.set_title("Per-Metric Comparison (Subject-Independent)")
        ax.legend(fontsize=8, loc="upper right")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "per_metric_bar.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(FIG_DIR / "per_metric_bar.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[FIG] Saved per-metric bar chart → {FIG_DIR}/per_metric_bar.pdf")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    main_data = load_main()
    abl_data  = load_ablation()

    if main_data is None and abl_data is None:
        print("No results found. Run the experiment scripts first.")
        sys.exit(1)

    print("=" * 72)
    print("STELLA Results Analysis")
    print("=" * 72)

    if main_data:
        rows = print_main_md(main_data)
        tex  = latex_main(main_data)
        (TAB_DIR / "main_results.tex").write_text(tex)
        print(f"\n[TEX] LaTeX main table → {TAB_DIR}/main_results.tex")
    else:
        print("\n[SKIP] Main results not available yet.")

    if abl_data:
        print_ablation_md(abl_data)
        tex = latex_ablation(abl_data)
        (TAB_DIR / "ablation_results.tex").write_text(tex)
        print(f"\n[TEX] LaTeX ablation table → {TAB_DIR}/ablation_results.tex")
    else:
        print("\n[SKIP] Ablation results not available yet.")

    make_figures(main_data, abl_data)
    print("\n[DONE] Analysis complete.")

if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    main()
