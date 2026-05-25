"""
STELLA finetuning and linear probing for downstream classification.

Two evaluation modes:
  1. linear_probe_stella — freeze encoder, train only classification head
  2. finetune_stella     — full end-to-end finetuning (or few-shot)

Both support:
  - Cross-subject evaluation (train on train_subjects, eval on test_subjects)
  - Class-weighted loss for imbalanced datasets
  - Early stopping + checkpointing
"""

from __future__ import annotations
import time
import logging
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.stella import STELLA
from ..models.heads import LinearProbe, MLPHead
from ..datasets.base import EEGDataset
from ..evaluation.metrics import compute_metrics
from .callbacks import EarlyStopping, CheckpointCallback

logger = logging.getLogger(__name__)


def _build_loaders(
    train_set: EEGDataset,
    val_set: EEGDataset,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


def linear_probe_stella(
    model: STELLA,
    train_set: EEGDataset,
    val_set: EEGDataset,
    n_classes: int,
    cfg: dict,
    save_dir: str = "results/checkpoints/linear_probe",
) -> dict:
    """
    Linear probing: freeze encoder, train linear classification head.

    Parameters
    ----------
    model : STELLA (pretrained)
    train_set, val_set : EEGDataset
    n_classes : int
    cfg : dict
    save_dir : str

    Returns
    -------
    best_metrics : dict with accuracy, balanced_acc, f1, kappa
    """
    device = torch.device("cpu")
    model = model.to(device)
    model.freeze_encoder()

    # Attach linear head
    head = LinearProbe(model.cfg.d_model, n_classes)
    model.attach_downstream_head(head)
    model = model.to(device)

    n_epochs = cfg.get("n_epochs", 30)
    lr = cfg.get("lr", 1e-2)
    batch_size = cfg.get("batch_size", 64)

    train_loader, val_loader = _build_loaders(train_set, val_set, batch_size)

    # Only optimize the head
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    class_weights = train_set.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    checkpoint_cb = CheckpointCallback(save_dir, "linear_probe")
    early_stop = EarlyStopping(patience=cfg.get("patience", 10))

    best_metrics = {}

    for epoch in range(n_epochs):
        model.train()
        model.encoder.eval()    # encoder stays frozen in eval mode

        train_loss = 0.0
        for x, labels, _ in train_loader:
            x, labels = x.to(device), labels.to(device)
            logits = model(x)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        metrics = _evaluate(model, val_loader, device)
        val_acc = metrics["accuracy"]

        logger.info(
            f"[LP] Epoch {epoch+1:3d}/{n_epochs} | "
            f"Train loss={train_loss/len(train_loader):.4f} | "
            f"Val acc={val_acc:.4f} | bal_acc={metrics['balanced_accuracy']:.4f}"
        )

        is_best = checkpoint_cb(model, optimizer, epoch, metrics, val_acc)
        if is_best:
            best_metrics = metrics.copy()

        if early_stop(val_acc):
            break

    return best_metrics


def finetune_stella(
    model: STELLA,
    train_set: EEGDataset,
    val_set: EEGDataset,
    n_classes: int,
    cfg: dict,
    save_dir: str = "results/checkpoints/finetune",
    mode: str = "full",   # "full" | "few_shot"
) -> dict:
    """
    Full finetuning or few-shot finetuning of STELLA.

    Parameters
    ----------
    mode : "full" | "few_shot"
        few_shot: use small lr, more regularization, fewer epochs
    """
    device = torch.device("cpu")
    model = model.to(device)
    model.unfreeze_encoder()

    head = MLPHead(model.cfg.d_model, n_classes, dropout=cfg.get("head_dropout", 0.3))
    model.attach_downstream_head(head)
    model = model.to(device)

    n_epochs = cfg.get("n_epochs", 50)
    batch_size = cfg.get("batch_size", 32)
    lr = cfg.get("lr", 1e-4) if mode == "full" else cfg.get("lr_few_shot", 5e-5)
    weight_decay = cfg.get("weight_decay", 1e-4)

    train_loader, val_loader = _build_loaders(train_set, val_set, batch_size)

    # Layer-wise LR: encoder gets smaller lr, head gets full lr
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": lr * 0.1},
        {"params": head.parameters(), "lr": lr},
    ], weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01,
    )

    class_weights = train_set.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    checkpoint_cb = CheckpointCallback(save_dir, "finetune")
    early_stop = EarlyStopping(patience=cfg.get("patience", 15))

    best_metrics = {}

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for x, labels, _ in train_loader:
            x, labels = x.to(device), labels.to(device)
            logits = model(x)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        metrics = _evaluate(model, val_loader, device)
        val_acc = metrics["accuracy"]

        logger.info(
            f"[FT] Epoch {epoch+1:3d}/{n_epochs} | "
            f"Train loss={train_loss/len(train_loader):.4f} | "
            f"Val acc={val_acc:.4f} | bal_acc={metrics['balanced_accuracy']:.4f} | "
            f"kappa={metrics['kappa']:.4f}"
        )

        is_best = checkpoint_cb(model, optimizer, epoch, metrics, val_acc)
        if is_best:
            best_metrics = metrics.copy()

        if early_stop(val_acc):
            break

    return best_metrics


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Run model on loader and return classification metrics."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            x, labels = batch[0], batch[1]
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    return compute_metrics(all_labels, all_preds)
