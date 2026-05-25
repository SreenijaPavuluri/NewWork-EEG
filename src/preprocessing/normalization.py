"""
EEG normalization strategies.

Trial-level normalization is critical for cross-subject robustness.
We use z-score per channel per trial as the default.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch


def normalize_trial(
    data: np.ndarray,
    method: str = "zscore",
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize a single EEG trial.

    Parameters
    ----------
    data : (C, T)
    method : "zscore" | "minmax" | "robust"
    eps : numerical stability

    Returns
    -------
    (C, T) normalized
    """
    if method == "zscore":
        mu = data.mean(axis=-1, keepdims=True)
        sigma = data.std(axis=-1, keepdims=True) + eps
        return ((data - mu) / sigma).astype(np.float32)

    elif method == "minmax":
        lo = data.min(axis=-1, keepdims=True)
        hi = data.max(axis=-1, keepdims=True)
        return ((data - lo) / (hi - lo + eps)).astype(np.float32)

    elif method == "robust":
        # Median + IQR-based normalization (robust to artifacts)
        med = np.median(data, axis=-1, keepdims=True)
        q75, q25 = np.percentile(data, [75, 25], axis=-1, keepdims=True)
        iqr = q75 - q25 + eps
        return ((data - med) / iqr).astype(np.float32)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


class GlobalScaler:
    """
    Fit global mean/std on training set, apply to val/test.

    Operates on (C, T) EEG trials.
    """

    def __init__(self):
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None

    def fit(self, trials: list[np.ndarray]) -> "GlobalScaler":
        """trials: list of (C, T) arrays."""
        stacked = np.concatenate([t.reshape(t.shape[0], -1) for t in trials], axis=-1)
        self._mu = stacked.mean(axis=-1, keepdims=True)
        self._sigma = stacked.std(axis=-1, keepdims=True) + 1e-8
        return self

    def transform(self, trial: np.ndarray) -> np.ndarray:
        assert self._mu is not None, "Call fit() first"
        return ((trial - self._mu) / self._sigma).astype(np.float32)

    def fit_transform(self, trials: list[np.ndarray]) -> list[np.ndarray]:
        self.fit(trials)
        return [self.transform(t) for t in trials]
