"""
EEG-safe augmentations for self-supervised pretraining.

Principles:
  - Preserve EEG physiological plausibility
  - Augmentations create positive pairs (same trial, different views)
  - Avoid label-corrupting transformations (e.g., time-reversal for MI)
  - All operate on (C, T) numpy arrays or (B, C, T) tensors

Design inspired by:
  - Mohsenvand et al., "Contrastive Representation Learning for EEG", 2021
  - Banville et al., "Uncovering the structure of clinical EEG signals", 2021
"""

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable


# ---------------------------------------------------------------------------
# Functional augmentations (numpy, operate on (C, T))
# ---------------------------------------------------------------------------

def temporal_jitter(
    data: np.ndarray,
    sfreq: float = 160.0,
    max_jitter_sec: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Randomly shift the signal by up to max_jitter_sec, with wrap-around."""
    if rng is None:
        rng = np.random.default_rng()
    max_shift = int(max_jitter_sec * sfreq)
    shift = rng.integers(-max_shift, max_shift + 1)
    return np.roll(data, shift, axis=-1)


def gaussian_noise(
    data: np.ndarray,
    noise_std: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add i.i.d. Gaussian noise scaled to signal amplitude."""
    if rng is None:
        rng = np.random.default_rng()
    signal_std = data.std() + 1e-8
    noise = rng.standard_normal(data.shape).astype(np.float32)
    return data + noise_std * signal_std * noise


def amplitude_scale(
    data: np.ndarray,
    scale_range: tuple[float, float] = (0.7, 1.3),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Multiply all channels by a random scalar in scale_range."""
    if rng is None:
        rng = np.random.default_rng()
    scale = rng.uniform(*scale_range)
    return (data * scale).astype(np.float32)


def frequency_mask(
    data: np.ndarray,
    sfreq: float = 160.0,
    n_masks: int = 1,
    mask_width_hz: float = 3.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Zero out a random frequency band in the spectrum.

    Operates per-channel via rfft → mask → irfft.
    Preserves signal energy outside masked band.
    """
    if rng is None:
        rng = np.random.default_rng()
    C, T = data.shape
    spec = np.fft.rfft(data, axis=-1)               # (C, n_freqs)
    n_freqs = spec.shape[-1]
    nyq = sfreq / 2.0
    freq_res = nyq / n_freqs                         # Hz per bin

    mask_width_bins = max(1, int(mask_width_hz / freq_res))

    for _ in range(n_masks):
        start = rng.integers(0, n_freqs - mask_width_bins)
        spec[:, start:start + mask_width_bins] = 0.0

    return np.fft.irfft(spec, n=T, axis=-1).astype(np.float32)


def temporal_mask(
    data: np.ndarray,
    mask_ratio: float = 0.1,
    n_masks: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Zero out random temporal segments (masked autoencoding style).

    Parameters
    ----------
    mask_ratio : fraction of total time to mask per segment
    n_masks : number of separate segments
    """
    if rng is None:
        rng = np.random.default_rng()
    T = data.shape[-1]
    mask_len = max(1, int(mask_ratio * T))
    out = data.copy()
    for _ in range(n_masks):
        start = rng.integers(0, T - mask_len)
        out[..., start:start + mask_len] = 0.0
    return out


def channel_dropout(
    data: np.ndarray,
    drop_prob: float = 0.15,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Zero out each channel independently with probability drop_prob.

    Simulates missing/noisy electrodes.
    """
    if rng is None:
        rng = np.random.default_rng()
    C = data.shape[0]
    mask = rng.random(C) > drop_prob                # True = keep
    out = data.copy()
    out[~mask] = 0.0
    return out


def spectral_perturbation(
    data: np.ndarray,
    noise_scale: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Add small random perturbation to the power spectrum.

    Multiplies each frequency bin by (1 + ε) where ε ~ N(0, noise_scale).
    """
    if rng is None:
        rng = np.random.default_rng()
    spec = np.fft.rfft(data, axis=-1)
    T = data.shape[-1]
    perturbation = 1.0 + noise_scale * rng.standard_normal(spec.shape).astype(np.float32)
    spec = spec * perturbation
    return np.fft.irfft(spec, n=T, axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# High-level augmentation pipeline
# ---------------------------------------------------------------------------

@dataclass
class AugmentationConfig:
    """Probability of applying each augmentation."""
    p_temporal_jitter: float = 0.5
    p_gaussian_noise: float = 0.5
    p_amplitude_scale: float = 0.5
    p_frequency_mask: float = 0.3
    p_temporal_mask: float = 0.3
    p_channel_dropout: float = 0.3
    p_spectral_perturb: float = 0.3

    noise_std: float = 0.1
    scale_range: tuple[float, float] = (0.7, 1.3)
    max_jitter_sec: float = 0.1
    mask_ratio: float = 0.1
    freq_mask_width_hz: float = 3.0
    channel_drop_prob: float = 0.15


class AugmentationPipeline:
    """
    Stochastic EEG augmentation pipeline for generating contrastive views.

    Applies each augmentation independently with its configured probability.
    Two calls with the same input produce statistically independent views.

    Parameters
    ----------
    cfg : AugmentationConfig
    sfreq : float
    seed : int | None
    """

    def __init__(
        self,
        cfg: AugmentationConfig | None = None,
        sfreq: float = 160.0,
        seed: int | None = None,
    ):
        self.cfg = cfg or AugmentationConfig()
        self.sfreq = sfreq
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply stochastic augmentations to one (C, T) trial.

        Returns
        -------
        augmented (C, T) array
        """
        cfg = self.cfg
        rng = self._rng

        if rng.random() < cfg.p_temporal_jitter:
            data = temporal_jitter(data, self.sfreq, cfg.max_jitter_sec, rng)

        if rng.random() < cfg.p_gaussian_noise:
            data = gaussian_noise(data, cfg.noise_std, rng)

        if rng.random() < cfg.p_amplitude_scale:
            data = amplitude_scale(data, cfg.scale_range, rng)

        if rng.random() < cfg.p_frequency_mask:
            data = frequency_mask(data, self.sfreq, mask_width_hz=cfg.freq_mask_width_hz, rng=rng)

        if rng.random() < cfg.p_temporal_mask:
            data = temporal_mask(data, cfg.mask_ratio, rng=rng)

        if rng.random() < cfg.p_channel_dropout:
            data = channel_dropout(data, cfg.channel_drop_prob, rng)

        if rng.random() < cfg.p_spectral_perturb:
            data = spectral_perturbation(data, rng=rng)

        return data.astype(np.float32)

    def get_two_views(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generate two independent augmented views of the same trial."""
        return self(data.copy()), self(data.copy())


# Alias for external use
EEGAugmentation = AugmentationPipeline
