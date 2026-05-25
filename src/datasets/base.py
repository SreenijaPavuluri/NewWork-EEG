"""
Base EEG dataset class and subject-split utilities.

Design:
  - EEGDataset wraps preprocessed (C, T) trials with labels and subject IDs
  - SubjectSplit provides subject-independent train/val/test splits
  - Supports two-view augmented batches for contrastive pretraining
"""

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass


@dataclass
class SubjectSplit:
    """Subject-independent train/val/test subject ID split."""
    train_subjects: list[int]
    val_subjects: list[int]
    test_subjects: list[int]

    @classmethod
    def from_n_subjects(
        cls,
        n_subjects: int,
        val_frac: float = 0.1,
        test_frac: float = 0.2,
        seed: int = 42,
    ) -> "SubjectSplit":
        rng = np.random.default_rng(seed)
        ids = rng.permutation(n_subjects).tolist()
        n_test = max(1, int(n_subjects * test_frac))
        n_val = max(1, int(n_subjects * val_frac))
        test = ids[:n_test]
        val = ids[n_test:n_test + n_val]
        train = ids[n_test + n_val:]
        return cls(train, val, test)


class EEGDataset(Dataset):
    """
    Generic EEG dataset for both pretraining and finetuning.

    Parameters
    ----------
    trials : list of (C, T) float32 arrays
    labels : list of int — class labels
    subject_ids : list of int
    augmentation : callable or None
        If provided, called on each trial for data augmentation.
    two_views : bool
        If True, returns (view1, view2, label, subject_id) for contrastive training.
    """

    def __init__(
        self,
        trials: list[np.ndarray],
        labels: list[int],
        subject_ids: list[int],
        augmentation=None,
        two_views: bool = False,
    ):
        assert len(trials) == len(labels) == len(subject_ids), \
            "Trials, labels, and subject_ids must have the same length"
        self.trials = trials
        self.labels = np.array(labels, dtype=np.int64)
        self.subject_ids = np.array(subject_ids, dtype=np.int64)
        self.augmentation = augmentation
        self.two_views = two_views

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, idx: int):
        trial = self.trials[idx].copy()
        label = int(self.labels[idx])
        subject_id = int(self.subject_ids[idx])

        if self.two_views and self.augmentation is not None:
            view1 = torch.from_numpy(self.augmentation(trial.copy()))
            view2 = torch.from_numpy(self.augmentation(trial.copy()))
            return view1, view2, label, subject_id

        if self.augmentation is not None:
            trial = self.augmentation(trial)

        return torch.from_numpy(trial), label, subject_id

    def filter_subjects(self, subject_ids: list[int]) -> "EEGDataset":
        """Return a new dataset containing only the specified subjects."""
        mask = np.isin(self.subject_ids, subject_ids)
        idx = np.where(mask)[0]
        return EEGDataset(
            trials=[self.trials[i] for i in idx],
            labels=self.labels[idx].tolist(),
            subject_ids=self.subject_ids[idx].tolist(),
            augmentation=self.augmentation,
            two_views=self.two_views,
        )

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced datasets."""
        from collections import Counter
        counts = Counter(self.labels.tolist())
        n_classes = max(counts.keys()) + 1
        weights = torch.zeros(n_classes)
        total = len(self.labels)
        for c, cnt in counts.items():
            weights[c] = total / (n_classes * cnt)
        return weights
