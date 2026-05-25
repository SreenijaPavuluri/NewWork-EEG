#!/usr/bin/env python3
"""
STELLA pretraining script.

Usage:
  python scripts/pretrain.py [--n-subjects N] [--epochs N] [--sanity-check]

--sanity-check: overfit on 2 batches to verify training loop works
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logging
from src.utils.seed import set_seed
from src.utils.profiling import CPUProfiler
from src.models.stella import STELLA, STELLAConfig, build_stella
from src.datasets.physionet import load_physionet
from src.preprocessing.pipeline import PreprocessingConfig
from src.training.pretrain import pretrain_stella
from src.visualization.plots import plot_training_curves

setup_logging()
logger = logging.getLogger(__name__)


def sanity_check(model: STELLA) -> None:
    """Overfit on 2 batches to verify forward/backward pass."""
    import torch
    from src.losses.combined import STELLAPretrainLoss
    from src.augmentations.eeg_augmentations import AugmentationPipeline
    import numpy as np

    logger.info("=== SANITY CHECK: Overfit 2 batches ===")
    device = torch.device("cpu")
    model = model.to(device)

    cfg = model.cfg
    B, C, T = 4, cfg.n_channels, cfg.segment_len

    criterion = STELLAPretrainLoss()

    # Fixed dummy batch
    torch.manual_seed(0)
    x1 = torch.randn(B, C, T) * 0.1
    x2 = torch.randn(B, C, T) * 0.1
    labels = torch.randint(0, cfg.n_classes_aux, (B,))
    domain_ids = torch.randint(0, 109, (B,))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(20):
        model.train()
        out = model.pretrain_forward(x1, x2, labels=labels, domain_ids=domain_ids)
        loss, comps = criterion(out, labels=labels, domain_ids=domain_ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            logger.info(f"  Step {step:3d}: total={comps['total']:.4f} | "
                       f"contrast={comps.get('contrastive', 0):.4f} | "
                       f"aux={comps.get('auxiliary', 0):.4f}")

    assert comps["total"] < 5.0, f"Loss too high after 20 steps: {comps['total']}"
    logger.info("Sanity check PASSED — loss is decreasing")


def main():
    parser = argparse.ArgumentParser(description="Pretrain STELLA")
    parser.add_argument("--n-subjects", type=int, default=20,
                        help="Number of subjects for pretraining (default: 20 for CPU feasibility)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", default="data/physionet")
    parser.add_argument("--sanity-check", action="store_true",
                        help="Run sanity check before training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    logger.info(f"Seed: {args.seed}")

    # Build model
    with CPUProfiler("Model build"):
        cfg = STELLAConfig()
        model = build_stella(cfg)
        param_counts = model.count_parameters()
        logger.info(f"STELLA parameter counts:")
        for k, v in param_counts.items():
            logger.info(f"  {k}: {v:,}")

    if args.sanity_check:
        sanity_check(model)
        # Rebuild fresh model after sanity check
        model = build_stella(cfg)

    # Load dataset
    with CPUProfiler("Dataset loading"):
        preprocess_cfg = PreprocessingConfig(
            target_sfreq=160.0, l_freq=1.0, h_freq=40.0, notch_freq=60.0,
            apply_car=True, normalization="zscore", segment_len_sec=4.0,
        )
        dataset, meta = load_physionet(
            data_dir=args.data_dir,
            subjects=list(range(1, args.n_subjects + 1)),
            preprocess_cfg=preprocess_cfg,
        )

    logger.info(f"Dataset: {meta['n_trials']} trials, {meta['n_subjects']} subjects")

    # Training config
    train_cfg = {
        "seed": args.seed,
        "n_epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "grad_accumulation": 2,
        "val_frac": 0.1,
        "patience": 15,
        "sfreq": 160.0,
        "loss": {
            "lambda_contrastive": 1.0,
            "lambda_auxiliary": 0.5,
            "lambda_consistency": 0.3,
            "lambda_alignment": 0.1,
            "temperature": 0.07,
            "subject_weight": 0.5,
        },
    }

    # Pretrain
    with CPUProfiler("Pretraining"):
        history = pretrain_stella(
            model, dataset, train_cfg,
            save_dir="results/checkpoints/pretrain",
            log_dir="results/logs/pretrain",
        )

    # Plot curves
    plot_training_curves(history, "results/figures/pretrain_curves.pdf")
    logger.info("Training curves saved to results/figures/pretrain_curves.pdf")


if __name__ == "__main__":
    main()
