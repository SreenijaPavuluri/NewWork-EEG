#!/usr/bin/env python3
"""
Evaluate STELLA and all baselines on PhysioNet test subjects.

Usage:
  python scripts/evaluate.py [--n-subjects N] [--checkpoint PATH]

Outputs:
  results/tables/evaluation_results.csv
  results/figures/confusion_matrix_*.pdf
  results/figures/per_subject_boxplot.pdf
"""

import sys
import argparse
import csv
import json
import logging
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logging
from src.utils.seed import set_seed
from src.models.stella import STELLA, STELLAConfig, build_stella
from src.models.baselines import (
    EEGNet, ShallowConvNet, DeepConvNet, VanillaTransformerEEG,
    CNNTransformerEEG, MIRepNetBaseline,
)
from src.models.heads import LinearProbe
from src.datasets.physionet import load_physionet
from src.datasets.base import SubjectSplit
from src.preprocessing.pipeline import PreprocessingConfig
from src.training.finetune import finetune_stella, linear_probe_stella
from src.evaluation.metrics import evaluate_all_subjects, compute_metrics
from src.visualization.plots import (
    plot_confusion_matrix, plot_per_subject_boxplot, plot_ablation_table,
)

setup_logging()
logger = logging.getLogger(__name__)


def train_baseline(model_class, model_kwargs, train_set, val_set, n_epochs=50, lr=1e-3):
    """Generic supervised training for baseline models."""
    from torch.utils.data import DataLoader
    import torch.nn as nn

    device = torch.device("cpu")
    model = model_class(**model_kwargs).to(device)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            x, labels, _ = batch
            x, labels = x.to(device), labels.to(device)
            loss = criterion(model(x), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Val
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                x, labels, _ = batch
                preds = model(x.to(device)).argmax(-1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_true.extend(labels.tolist())

        acc = sum(p == t for p, t in zip(all_preds, all_true)) / len(all_true)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-subjects", type=int, default=30)
    parser.add_argument("--checkpoint", default=None,
                        help="Path to pretrained STELLA checkpoint")
    parser.add_argument("--data-dir", default="data/physionet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    # Dataset
    preprocess_cfg = PreprocessingConfig(
        target_sfreq=160.0, l_freq=1.0, h_freq=40.0, notch_freq=60.0,
        apply_car=True, normalization="zscore", segment_len_sec=4.0,
    )
    logger.info(f"Loading dataset: {args.n_subjects} subjects")
    dataset, meta = load_physionet(
        data_dir=args.data_dir,
        subjects=list(range(1, args.n_subjects + 1)),
        preprocess_cfg=preprocess_cfg,
    )

    # Subject-independent split
    split = SubjectSplit.from_n_subjects(
        args.n_subjects, val_frac=0.1, test_frac=0.2, seed=args.seed
    )
    train_set = dataset.filter_subjects(split.train_subjects)
    val_set = dataset.filter_subjects(split.val_subjects)
    test_set = dataset.filter_subjects(split.test_subjects)

    logger.info(
        f"Split: train={len(train_set)} | val={len(val_set)} | test={len(test_set)} trials"
    )

    n_ch = meta["n_channels"]
    n_classes = meta["n_classes"]
    seg_len = meta["trial_length"]
    sfreq = meta["sfreq"]

    all_results: dict[str, dict] = {}
    per_subject_accs: dict[str, list] = {}

    # ---------------------------------------------------------------- Baselines
    baseline_configs = {
        "EEGNet": (EEGNet, dict(n_channels=n_ch, n_classes=n_classes, sfreq=sfreq, T=seg_len)),
        "ShallowConvNet": (ShallowConvNet, dict(n_channels=n_ch, n_classes=n_classes, T=seg_len)),
        "DeepConvNet": (DeepConvNet, dict(n_channels=n_ch, n_classes=n_classes, T=seg_len)),
        "VanillaTransformer": (VanillaTransformerEEG, dict(n_channels=n_ch, n_classes=n_classes, T=seg_len)),
        "CNNTransformer": (CNNTransformerEEG, dict(n_channels=n_ch, n_classes=n_classes, sfreq=sfreq, T=seg_len)),
        "MIRepNetBaseline": (MIRepNetBaseline, dict(n_channels=n_ch, n_classes=n_classes, T=seg_len)),
    }

    for name, (cls, kwargs) in baseline_configs.items():
        logger.info(f"\n=== Baseline: {name} ===")
        try:
            model = train_baseline(cls, kwargs, train_set, val_set, n_epochs=30)
            n_params = sum(p.numel() for p in model.parameters())

            results = evaluate_all_subjects(
                model, test_set, split.test_subjects
            )
            logger.info(f"  {results.summary()}")
            all_results[name] = {
                "accuracy": results.accuracy,
                "balanced_accuracy": results.balanced_accuracy,
                "f1": results.f1_macro,
                "kappa": results.kappa,
                "n_params": n_params,
            }
            per_subject_accs[name] = results.per_subject_accuracy

            # Confusion matrix
            plot_confusion_matrix(
                results.confusion_matrix,
                class_names=["Left", "Right", "Fists", "Feet"],
                save_path=f"results/figures/cm_{name}.pdf",
                title=f"Confusion Matrix — {name}",
            )
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            all_results[name] = {"error": str(e)}

    # ---------------------------------------------------------------- STELLA
    logger.info("\n=== STELLA (Linear Probe) ===")
    try:
        stella_cfg = STELLAConfig(n_channels=n_ch, n_classes_aux=n_classes)
        stella = build_stella(stella_cfg)

        if args.checkpoint and Path(args.checkpoint).exists():
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            stella.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded checkpoint from {args.checkpoint}")
        else:
            logger.warning("No pretrained checkpoint — evaluating with random encoder")

        ft_cfg = {"n_epochs": 30, "lr": 1e-2, "batch_size": 64, "patience": 10}
        best_lp_metrics = linear_probe_stella(
            stella, train_set, val_set, n_classes, ft_cfg,
            save_dir="results/checkpoints/linear_probe",
        )

        # Final test evaluation
        stella.eval()
        results = evaluate_all_subjects(stella, test_set, split.test_subjects)
        n_params = stella.count_parameters()["encoder_total"]

        all_results["STELLA-LP"] = {
            "accuracy": results.accuracy,
            "balanced_accuracy": results.balanced_accuracy,
            "f1": results.f1_macro,
            "kappa": results.kappa,
            "n_params": n_params,
        }
        per_subject_accs["STELLA-LP"] = results.per_subject_accuracy
        logger.info(f"  {results.summary()}")

    except Exception as e:
        logger.error(f"  STELLA-LP FAILED: {e}")

    logger.info("\n=== STELLA (Full Finetune) ===")
    try:
        stella2 = build_stella(STELLAConfig(n_channels=n_ch, n_classes_aux=n_classes))
        if args.checkpoint and Path(args.checkpoint).exists():
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            stella2.load_state_dict(ckpt["model_state_dict"])

        ft_cfg = {"n_epochs": 50, "lr": 1e-4, "batch_size": 32, "patience": 15,
                  "weight_decay": 1e-4, "head_dropout": 0.3}
        finetune_stella(stella2, train_set, val_set, n_classes, ft_cfg,
                        save_dir="results/checkpoints/finetune")
        results = evaluate_all_subjects(stella2, test_set, split.test_subjects)
        all_results["STELLA-FT"] = {
            "accuracy": results.accuracy,
            "balanced_accuracy": results.balanced_accuracy,
            "f1": results.f1_macro,
            "kappa": results.kappa,
            "n_params": stella2.count_parameters()["encoder_total"],
        }
        per_subject_accs["STELLA-FT"] = results.per_subject_accuracy
        logger.info(f"  {results.summary()}")

    except Exception as e:
        logger.error(f"  STELLA-FT FAILED: {e}")

    # ---------------------------------------------------------------- Save results
    # CSV
    csv_path = "results/tables/evaluation_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "balanced_accuracy",
                                                "f1", "kappa", "n_params"])
        writer.writeheader()
        for name, res in all_results.items():
            if "error" not in res:
                writer.writerow({"model": name, **{k: round(v, 4) if isinstance(v, float) else v
                                                    for k, v in res.items()}})

    # JSON
    with open("results/tables/evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Figures
    plot_ablation_table(all_results, "results/figures/main_results_table.pdf",
                        "Main Results — Subject-Independent Evaluation")
    if per_subject_accs:
        plot_per_subject_boxplot(per_subject_accs, metric="accuracy",
                                 save_path="results/figures/per_subject_boxplot.pdf")

    logger.info(f"\nResults saved to {csv_path}")
    logger.info("\nSummary:")
    for name, res in all_results.items():
        if "error" not in res:
            logger.info(f"  {name:25s}: acc={res.get('accuracy', 0):.4f} | "
                       f"κ={res.get('kappa', 0):.4f} | "
                       f"params={res.get('n_params', 0)/1e6:.2f}M")


if __name__ == "__main__":
    main()
