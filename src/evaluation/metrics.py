"""
Evaluation metrics for EEG classification.

All metrics are computed per-subject and aggregated (mean ± std).
Report: accuracy, balanced accuracy, F1 (macro), Cohen's kappa, confusion matrix.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
)
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for evaluation results across subjects."""
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    f1_macro: float = 0.0
    kappa: float = 0.0
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    per_subject_accuracy: list[float] = field(default_factory=list)
    per_subject_kappa: list[float] = field(default_factory=list)

    def summary(self) -> str:
        per_subj = self.per_subject_accuracy
        std = np.std(per_subj) if per_subj else 0.0
        return (
            f"Acc={self.accuracy:.4f} | BalAcc={self.balanced_accuracy:.4f} | "
            f"F1={self.f1_macro:.4f} | κ={self.kappa:.4f} | "
            f"Per-subj: {np.mean(per_subj):.4f}±{std:.4f}"
        )


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    average: str = "macro",
) -> dict:
    """
    Compute standard EEG BCI classification metrics.

    Parameters
    ----------
    y_true : ground truth labels
    y_pred : predicted labels
    average : sklearn average type for F1

    Returns
    -------
    dict with: accuracy, balanced_accuracy, f1, kappa, confusion_matrix
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "f1": float(f1),
        "kappa": float(kappa),
        "confusion_matrix": cm.tolist(),
    }


def evaluate_all_subjects(
    model,
    dataset,
    subject_ids: list[int],
    batch_size: int = 64,
    device_str: str = "cpu",
) -> EvaluationResults:
    """
    Evaluate model on each subject separately and aggregate.

    Parameters
    ----------
    model : nn.Module with forward() → logits
    dataset : EEGDataset
    subject_ids : list of subject IDs to evaluate
    batch_size : int
    device_str : str

    Returns
    -------
    EvaluationResults
    """
    import torch
    from torch.utils.data import DataLoader

    device = torch.device(device_str)
    model = model.to(device)
    model.eval()

    all_true, all_pred = [], []
    per_subj_acc = []
    per_subj_kappa = []

    for subj in subject_ids:
        subj_set = dataset.filter_subjects([subj])
        if len(subj_set) == 0:
            continue

        loader = DataLoader(subj_set, batch_size=batch_size, shuffle=False)
        s_true, s_pred = [], []

        with torch.no_grad():
            for batch in loader:
                x, labels = batch[0].to(device), batch[1]
                logits = model(x)
                preds = logits.argmax(dim=-1).cpu().numpy()
                s_pred.extend(preds.tolist())
                s_true.extend(labels.tolist())

        if len(s_true) > 0:
            m = compute_metrics(s_true, s_pred)
            per_subj_acc.append(m["accuracy"])
            per_subj_kappa.append(m["kappa"])
            all_true.extend(s_true)
            all_pred.extend(s_pred)

    if not all_true:
        logger.warning("No samples evaluated")
        return EvaluationResults()

    global_metrics = compute_metrics(all_true, all_pred)
    return EvaluationResults(
        accuracy=global_metrics["accuracy"],
        balanced_accuracy=global_metrics["balanced_accuracy"],
        f1_macro=global_metrics["f1"],
        kappa=global_metrics["kappa"],
        confusion_matrix=np.array(global_metrics["confusion_matrix"]),
        per_subject_accuracy=per_subj_acc,
        per_subject_kappa=per_subj_kappa,
    )
