"""
STELLA pretraining loop.

Multi-objective pretraining with 4 loss terms:
  1. Subject-aware contrastive (NT-Xent)
  2. Auxiliary supervised classification
  3. Spectral-temporal consistency (VICReg)
  4. Domain-adaptive alignment (MMD)

CPU-optimized:
  - No AMP / mixed-precision (not needed for CPU)
  - Gradient accumulation for larger effective batch sizes
  - Periodic memory profiling
  - TensorBoard logging

Usage:
  metrics = pretrain_stella(model, dataset, cfg)
"""

from __future__ import annotations
import time
import logging
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from ..models.stella import STELLA
from ..losses.combined import STELLAPretrainLoss
from ..augmentations.eeg_augmentations import AugmentationPipeline
from ..datasets.base import EEGDataset
from ..utils.profiling import memory_mb
from .callbacks import EarlyStopping, CheckpointCallback

logger = logging.getLogger(__name__)


def pretrain_stella(
    model: STELLA,
    dataset: EEGDataset,
    cfg: dict,
    save_dir: str = "results/checkpoints/pretrain",
    log_dir: str = "results/logs/pretrain",
) -> dict[str, list]:
    """
    Run STELLA pretraining.

    Parameters
    ----------
    model : STELLA
    dataset : EEGDataset — full training dataset (will split internally)
    cfg : dict — training config (from configs/training/pretrain.yaml)
    save_dir : str
    log_dir : str

    Returns
    -------
    history : dict with per-epoch loss curves
    """
    # ------------------------------------------------------------------ setup
    torch.manual_seed(cfg.get("seed", 42))
    device = torch.device("cpu")
    model = model.to(device)

    n_epochs = cfg.get("n_epochs", 50)
    batch_size = cfg.get("batch_size", 32)
    lr = cfg.get("lr", 1e-3)
    weight_decay = cfg.get("weight_decay", 1e-4)
    grad_accum = cfg.get("grad_accumulation", 1)
    val_frac = cfg.get("val_frac", 0.1)
    patience = cfg.get("patience", 15)

    # ---------------------------------------------------------------- augmentation
    aug_cfg_dict = cfg.get("augmentation", {})
    from ..augmentations.eeg_augmentations import AugmentationConfig
    aug_cfg = AugmentationConfig(**aug_cfg_dict) if aug_cfg_dict else AugmentationConfig()
    augmentation = AugmentationPipeline(aug_cfg, sfreq=cfg.get("sfreq", 160.0))

    # Enable two-view mode on dataset copy
    dataset.augmentation = augmentation
    dataset.two_views = True

    # Train/val split
    n_val = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Val set: single view for loss tracking
    val_set.dataset = val_set.dataset.__class__(
        [val_set.dataset.trials[i] for i in val_set.indices],
        [val_set.dataset.labels[i] for i in val_set.indices],
        [val_set.dataset.subject_ids[i] for i in val_set.indices],
        augmentation=augmentation,
        two_views=True,
    )
    val_set = val_set.dataset   # unwrap

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True, pin_memory=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=0, drop_last=False,
    )

    # ---------------------------------------------------------------- loss
    loss_cfg = cfg.get("loss", {})
    criterion = STELLAPretrainLoss(
        lambda_contrastive=loss_cfg.get("lambda_contrastive", 1.0),
        lambda_auxiliary=loss_cfg.get("lambda_auxiliary", 0.5),
        lambda_consistency=loss_cfg.get("lambda_consistency", 0.3),
        lambda_alignment=loss_cfg.get("lambda_alignment", 0.1),
        temperature=loss_cfg.get("temperature", 0.07),
        subject_weight=loss_cfg.get("subject_weight", 0.5),
    )

    # ---------------------------------------------------------------- optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01,
    )

    # ---------------------------------------------------------------- callbacks
    checkpoint_cb = CheckpointCallback(save_dir, "stella_pretrain")
    early_stop = EarlyStopping(patience=patience)

    # ---------------------------------------------------------------- logging
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)

    history = defaultdict(list)

    # ---------------------------------------------------------------- training loop
    logger.info(
        f"Starting pretraining: {n_epochs} epochs, "
        f"batch={batch_size}, lr={lr}, device=cpu"
    )
    logger.info(f"Train: {len(train_set)} | Val: {len(val_set)} trials")
    logger.info(f"Memory: {memory_mb():.1f} MB")

    for epoch in range(n_epochs):
        t0 = time.perf_counter()
        model.train()

        epoch_losses = defaultdict(float)
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            x1, x2, labels, domain_ids = batch
            x1, x2 = x1.to(device), x2.to(device)
            labels = labels.to(device)
            domain_ids = domain_ids.to(device)

            # Forward pass
            model_out = model.pretrain_forward(x1, x2, labels=labels, domain_ids=domain_ids)
            loss, components = criterion(model_out, labels=labels, domain_ids=domain_ids)

            # Gradient accumulation
            (loss / grad_accum).backward()

            if (batch_idx + 1) % grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            for k, v in components.items():
                epoch_losses[k] += v
            n_batches += 1

        scheduler.step()

        # Average over batches
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        # --------------------------------------------------------- validation
        model.eval()
        val_losses = defaultdict(float)
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x1, x2, labels, domain_ids = batch
                x1, x2 = x1.to(device), x2.to(device)
                model_out = model.pretrain_forward(x1, x2, labels=labels, domain_ids=domain_ids)
                _, comps = criterion(model_out, labels=labels, domain_ids=domain_ids)
                for k, v in comps.items():
                    val_losses[k] += v
                n_val_batches += 1

        for k in val_losses:
            val_losses[k] /= max(n_val_batches, 1)

        # --------------------------------------------------------- logging
        elapsed = time.perf_counter() - t0
        train_total = epoch_losses["total"]
        val_total = val_losses["total"]

        # Use negative val loss as "metric" (lower loss = better)
        val_metric = -val_total

        for k, v in epoch_losses.items():
            history[f"train_{k}"].append(v)
            writer.add_scalar(f"pretrain/train_{k}", v, epoch)
        for k, v in val_losses.items():
            history[f"val_{k}"].append(v)
            writer.add_scalar(f"pretrain/val_{k}", v, epoch)

        writer.add_scalar("pretrain/lr", scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("system/memory_mb", memory_mb(), epoch)

        logger.info(
            f"Epoch {epoch+1:3d}/{n_epochs} | "
            f"Train={train_total:.4f} | Val={val_total:.4f} | "
            f"{elapsed:.1f}s | RAM={memory_mb():.0f}MB"
        )

        # --------------------------------------------------------- callbacks
        checkpoint_cb(model, optimizer, epoch, dict(epoch_losses), val_metric)
        if early_stop(val_metric):
            logger.info("Early stopping")
            break

    writer.close()
    logger.info(
        f"Pretraining complete. Best val total loss: {-early_stop.best:.4f}"
    )
    return dict(history)
