"""
Dual-Stream EEG Tokenizer for STELLA.

Implements two parallel tokenization pathways:
  1. Temporal Patch Tokens  — depthwise patch convolution captures local waveform morphology
  2. Spectral Band Tokens   — learnable band-power encoder captures oscillatory content

These two token streams are designed to be fused by the GatedFusion module.

Design principles:
  - Channel-separable (depthwise) operations to keep params low
  - Spectral branch uses fixed Gaussian band filters + learnable weighting
    (interpretable + robust vs. purely learnable filters)
  - Output tokens are projected to a unified d_model dimension
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


# ---------------------------------------------------------------------------
# Temporal stream
# ---------------------------------------------------------------------------

class TemporalPatchEmbed(nn.Module):
    """
    Hierarchical temporal patch embedding via depthwise 1D convolution.

    Strategy:
      - Stage 1: channel-wise (depthwise) conv with kernel=patch_size, stride=patch_stride
        captures local waveform features per channel independently
      - Stage 2: pointwise conv to project to d_model
      - Output: (B, n_patches, d_model) — channels collapsed via mean pooling

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    patch_size : int
        Temporal window length (samples) per patch.
    patch_stride : int
        Stride between patches (default = patch_size // 2 for 50% overlap).
    d_model : int
        Output embedding dimension.
    max_patches : int
        Maximum number of patches (for positional embedding).
    """

    def __init__(
        self,
        n_channels: int,
        patch_size: int = 40,
        patch_stride: int = 20,
        d_model: int = 256,
        max_patches: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.d_model = d_model

        d_inter = max(32, d_model // 4)

        # Stage 1: depthwise temporal conv (one filter per channel)
        self.dw_conv = nn.Conv1d(
            n_channels, n_channels,
            kernel_size=patch_size,
            stride=patch_stride,
            groups=n_channels,
            bias=False,
        )
        # Stage 2: pointwise projection per channel
        self.pw_proj = nn.Conv1d(n_channels, n_channels * d_inter // n_channels * n_channels, kernel_size=1)

        # Simpler: project via linear after reshape
        self.channel_proj = nn.Linear(n_channels, d_inter)
        self.temporal_proj = nn.Linear(d_inter, d_model)
        self.norm = nn.LayerNorm(d_model)

        # Learnable positional embedding
        self.pos_embed = nn.Embedding(max_patches + 1, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, T)

        Returns
        -------
        tokens : (B, P, D)  where P = number of patches
        """
        B, C, T = x.shape

        # Depthwise conv → (B, C, P)
        x_patches = self.dw_conv(x)           # (B, C, P)
        x_patches = F.gelu(x_patches)

        # Rearrange: (B, C, P) → (B, P, C)
        x_patches = rearrange(x_patches, "b c p -> b p c")

        # Project channel dim: (B, P, C) → (B, P, d_inter) → (B, P, D)
        tokens = self.channel_proj(x_patches)
        tokens = F.gelu(tokens)
        tokens = self.temporal_proj(tokens)

        tokens = self.norm(tokens)

        # Add positional embeddings
        P = tokens.shape[1]
        pos_ids = torch.arange(P, device=x.device)
        tokens = tokens + self.pos_embed(pos_ids).unsqueeze(0)
        tokens = self.dropout(tokens)
        return tokens


# ---------------------------------------------------------------------------
# Spectral stream
# ---------------------------------------------------------------------------

def _gaussian_band_filters(
    n_freqs: int,
    sfreq: float,
    bands: list[tuple[float, float]],
    sigma_hz: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build Gaussian frequency-domain filters for each EEG band.

    Returns (n_bands, n_freqs) weight tensor (normalized, real-valued).
    """
    freqs = torch.linspace(0, sfreq / 2, n_freqs, device=device)
    filters = []
    for (flo, fhi) in bands:
        center = (flo + fhi) / 2.0
        # Gaussian centered at band center, width ∝ band width
        sigma = max(sigma_hz, (fhi - flo) / 4.0)
        w = torch.exp(-0.5 * ((freqs - center) / sigma) ** 2)
        w = w / (w.sum() + 1e-8)
        filters.append(w)
    return torch.stack(filters, dim=0)   # (n_bands, n_freqs)


class SpectralBandEncoder(nn.Module):
    """
    Learnable spectral-band token encoder.

    For each EEG channel, computes band-power features across standard
    oscillatory bands then projects to d_spectral.

    Architecture:
      1. rfft along time axis → complex spectrum
      2. Apply fixed Gaussian band filters (per canonical EEG band)
      3. Learnable mixing weights on top of fixed filters (interpretable)
      4. Compute log-power per band → (B, C, n_bands)
      5. Project: (B, C, n_bands) → (B, C, d_spectral)

    Parameters
    ----------
    n_channels : int
    sfreq : float
        Sampling frequency (Hz).
    d_spectral : int
        Output spectral token dimension.
    n_fft : int
        FFT length.
    """

    EEG_BANDS = [
        (0.5, 4.0),   # delta
        (4.0, 8.0),   # theta
        (8.0, 13.0),  # alpha
        (13.0, 30.0), # beta
        (30.0, 45.0), # low-gamma
    ]
    BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]

    def __init__(
        self,
        n_channels: int,
        sfreq: float = 160.0,
        d_spectral: int = 128,
        n_fft: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.d_spectral = d_spectral
        self.n_fft = n_fft
        self.n_bands = len(self.EEG_BANDS)
        n_freqs = n_fft // 2 + 1

        # Register fixed Gaussian band filters as buffer (non-trainable)
        band_filters = _gaussian_band_filters(n_freqs, sfreq, self.EEG_BANDS)
        self.register_buffer("band_filters", band_filters)  # (n_bands, n_freqs)

        # Learnable mixing: allow the model to reweight bands
        self.band_mix = nn.Parameter(torch.ones(self.n_bands, n_freqs))

        # Learnable spectral attention over channels
        self.channel_attn = nn.Linear(n_channels, n_channels)

        # Projection: band features → d_spectral
        # Input: (B, C, n_bands) → (B, n_bands, d_spectral)
        self.band_proj = nn.Sequential(
            nn.Linear(n_channels, d_spectral // 2),
            nn.GELU(),
            nn.Linear(d_spectral // 2, d_spectral),
        )
        self.norm = nn.LayerNorm(d_spectral)
        self.dropout = nn.Dropout(dropout)

        # Positional: one token per band
        self.band_pos = nn.Embedding(self.n_bands, d_spectral)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, T)

        Returns
        -------
        spectral_tokens : (B, n_bands, d_spectral)
        """
        B, C, T = x.shape

        # rfft: (B, C, n_freqs) complex → real power
        spec = torch.fft.rfft(x, n=self.n_fft, dim=-1)  # (B, C, n_freqs)
        power = spec.real ** 2 + spec.imag ** 2          # (B, C, n_freqs)

        # Apply band filters: (n_bands, n_freqs) × learnable mix
        eff_filters = self.band_filters * torch.sigmoid(self.band_mix)  # (n_bands, n_freqs)
        eff_filters = F.normalize(eff_filters, p=1, dim=-1)

        # Band power: (B, C, n_bands)
        band_power = torch.einsum("bcf,nf->bcn", power, eff_filters)
        band_power = torch.log1p(band_power)             # log-power (stable)

        # Rearrange: (B, n_bands, C)
        band_power = rearrange(band_power, "b c n -> b n c")

        # Project across channel dimension → (B, n_bands, d_spectral)
        tokens = self.band_proj(band_power)
        tokens = self.norm(tokens)

        # Add band positional embeddings
        band_ids = torch.arange(self.n_bands, device=x.device)
        tokens = tokens + self.band_pos(band_ids).unsqueeze(0)
        tokens = self.dropout(tokens)
        return tokens


# ---------------------------------------------------------------------------
# Combined dual-stream tokenizer
# ---------------------------------------------------------------------------

class DualStreamTokenizer(nn.Module):
    """
    Dual-stream EEG tokenizer producing temporal + spectral tokens.

    This is the primary novel tokenization component of STELLA.
    Both streams output tokens in the same d_model space to enable
    downstream cross-stream fusion.

    Parameters
    ----------
    n_channels : int
    sfreq : float
    patch_size : int
    patch_stride : int
    d_model : int
    n_fft : int
    max_patches : int
    dropout : float
    """

    def __init__(
        self,
        n_channels: int = 64,
        sfreq: float = 160.0,
        patch_size: int = 40,
        patch_stride: int = 20,
        d_model: int = 256,
        n_fft: int = 256,
        max_patches: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_spectral = d_model  # same dimension for fusion compatibility

        self.temporal = TemporalPatchEmbed(
            n_channels=n_channels,
            patch_size=patch_size,
            patch_stride=patch_stride,
            d_model=d_model,
            max_patches=max_patches,
            dropout=dropout,
        )
        self.spectral = SpectralBandEncoder(
            n_channels=n_channels,
            sfreq=sfreq,
            d_spectral=d_spectral,
            n_fft=n_fft,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, C, T)

        Returns
        -------
        temporal_tokens : (B, P, D)
        spectral_tokens : (B, n_bands, D)
        """
        temporal_tokens = self.temporal(x)
        spectral_tokens = self.spectral(x)
        return temporal_tokens, spectral_tokens
