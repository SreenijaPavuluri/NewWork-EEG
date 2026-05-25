"""
Modular EEG preprocessing pipeline.

Design:
  - Configurable via PreprocessingConfig dataclass
  - Applied per trial (C, T) arrays
  - CPU-safe, reproducible
  - Logging of each step

Steps (in order):
  1. Resample (optional)
  2. Notch filter (optional)
  3. Bandpass filter
  4. CAR spatial filter (optional)
  5. Artifact suppression via amplitude clipping
  6. Trial-level normalization
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import logging

from .filters import bandpass_filter, notch_filter, resample_signal, apply_car
from .normalization import normalize_trial

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Full preprocessing configuration."""

    # Resampling
    target_sfreq: float = 160.0

    # Filtering
    l_freq: float = 1.0
    h_freq: float = 40.0
    filter_order: int = 4
    notch_freq: float | None = 50.0    # set to None to disable
    notch_quality: float = 30.0

    # Spatial
    apply_car: bool = True

    # Artifact rejection
    clip_amplitude: float | None = 100.0   # μV; None to disable

    # Normalization
    normalization: str = "zscore"   # "zscore" | "minmax" | "robust" | "none"

    # Segmentation
    segment_len_sec: float = 4.0
    tmin_sec: float = 0.0            # crop start within trial


class EEGPreprocessingPipeline:
    """
    Stateless preprocessing pipeline applied per (C, T) EEG trial.

    Parameters
    ----------
    cfg : PreprocessingConfig
    orig_sfreq : float
        Original sampling frequency of the raw data.
    """

    def __init__(self, cfg: PreprocessingConfig, orig_sfreq: float):
        self.cfg = cfg
        self.orig_sfreq = orig_sfreq
        self.output_sfreq = cfg.target_sfreq

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline to one EEG trial.

        Parameters
        ----------
        data : (C, T) raw EEG in μV

        Returns
        -------
        (C, T') preprocessed EEG
        """
        cfg = self.cfg

        # 1. Resample
        if abs(self.orig_sfreq - cfg.target_sfreq) > 1.0:
            data = resample_signal(data, self.orig_sfreq, cfg.target_sfreq)

        # 2. Notch filter
        if cfg.notch_freq is not None:
            data = notch_filter(data, cfg.target_sfreq, cfg.notch_freq, cfg.notch_quality)

        # 3. Bandpass filter
        data = bandpass_filter(
            data, cfg.target_sfreq,
            l_freq=cfg.l_freq, h_freq=cfg.h_freq, order=cfg.filter_order,
        )

        # 4. CAR
        if cfg.apply_car:
            data = apply_car(data)

        # 5. Amplitude clip (artifact suppression)
        if cfg.clip_amplitude is not None:
            data = np.clip(data, -cfg.clip_amplitude, cfg.clip_amplitude)

        # 6. Crop to segment length
        target_len = int(cfg.segment_len_sec * cfg.target_sfreq)
        start = int(cfg.tmin_sec * cfg.target_sfreq)
        end = start + target_len
        if data.shape[-1] >= end:
            data = data[..., start:end]
        elif data.shape[-1] > start:
            # Pad with zeros if slightly short
            seg = data[..., start:]
            pad = np.zeros((*data.shape[:-1], target_len - seg.shape[-1]), dtype=np.float32)
            data = np.concatenate([seg, pad], axis=-1)

        # 7. Normalization
        if cfg.normalization != "none":
            data = normalize_trial(data, method=cfg.normalization)

        return data.astype(np.float32)

    def process_batch(self, trials: list[np.ndarray]) -> list[np.ndarray]:
        """Process a list of (C, T) trials."""
        return [self(t) for t in trials]
