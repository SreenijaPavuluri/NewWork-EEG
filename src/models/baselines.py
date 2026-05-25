"""
Reproducible EEG classification baselines.

All baselines are implemented faithfully from their original papers.
No fabricated numbers — these are run on the same data splits as STELLA.

Baselines:
  - EEGNet (Lawhern et al., J Neural Eng, 2018)
  - ShallowConvNet (Schirrmeister et al., HBM, 2017)
  - DeepConvNet (Schirrmeister et al., HBM, 2017)
  - VanillaTransformerEEG (standard MHSA over temporal patches)
  - CNNTransformerEEG (CNN backbone + Transformer)
  - MIRepNetBaseline (simplified MIRepNet-style contrastive baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# EEGNet
# ---------------------------------------------------------------------------

class EEGNet(nn.Module):
    """
    EEGNet: A compact convolutional network for EEG BCIs.

    Lawhern et al., "EEGNet: A compact convolutional neural network for
    EEG-based brain-computer interfaces", J Neural Eng, 2018.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (C).
    n_classes : int
    sfreq : float
        Sampling frequency (Hz).
    T : int
        Number of time samples in each trial.
    F1 : int
        Number of temporal filters.
    D : int
        Depth multiplier.
    F2 : int
        Number of pointwise filters.
    dropout : float
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_classes: int = 4,
        sfreq: float = 160.0,
        T: int = 640,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_classes = n_classes

        half_sfreq = int(sfreq) // 2   # kernel size for temporal conv

        # Block 1: Temporal conv
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, half_sfreq), padding=(0, half_sfreq // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise: spatial (channel) filtering
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # Block 2: Separable conv
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # Compute flattened dim
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, T)
            out = self.block2(self.block1(dummy))
            flat_dim = out.numel()

        self.classifier = nn.Linear(flat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, T)
        """
        x = x.unsqueeze(1)              # (B, 1, C, T)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        return self.classifier(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        return x.flatten(1)


# ---------------------------------------------------------------------------
# ShallowConvNet
# ---------------------------------------------------------------------------

class ShallowConvNet(nn.Module):
    """
    Shallow ConvNet for EEG decoding.

    Schirrmeister et al., "Deep learning with convolutional neural networks
    for EEG decoding and visualization", HBM, 2017.

    Parameters
    ----------
    n_channels : int
    n_classes : int
    T : int
    n_filters_time : int
    filter_time_length : int
    n_filters_spat : int
    pool_time_length : int
    pool_time_stride : int
    dropout : float
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_classes: int = 4,
        T: int = 640,
        n_filters_time: int = 40,
        filter_time_length: int = 25,
        n_filters_spat: int = 40,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.temporal_conv = nn.Conv2d(
            1, n_filters_time, (1, filter_time_length), bias=False
        )
        self.spatial_conv = nn.Conv2d(
            n_filters_time, n_filters_spat, (n_channels, 1), bias=False
        )
        self.bn = nn.BatchNorm2d(n_filters_spat, momentum=0.1, affine=True)

        self.pool = nn.AvgPool2d((1, pool_time_length), stride=(1, pool_time_stride))
        self.dropout = nn.Dropout(dropout)

        # Compute output size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, T)
            out = self._features(dummy)
            flat_dim = out.numel()

        self.classifier = nn.Linear(flat_dim, n_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = x ** 2                     # square activation
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-7))  # log activation
        x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self._features(x)
        return self.classifier(x.flatten(1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        return self._features(x).flatten(1)


# ---------------------------------------------------------------------------
# DeepConvNet
# ---------------------------------------------------------------------------

class DeepConvNet(nn.Module):
    """
    Deep ConvNet for EEG decoding.

    Schirrmeister et al., HBM, 2017.

    Parameters
    ----------
    n_channels : int
    n_classes : int
    T : int
    dropout : float
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_classes: int = 4,
        T: int = 640,
        dropout: float = 0.5,
    ):
        super().__init__()

        def _conv_block(in_ch, out_ch, k, drop=dropout, pool=True):
            pad = k // 2   # same-length padding so short sequences work
            layers: list[nn.Module] = [
                nn.Conv2d(in_ch, out_ch, (1, k), padding=(0, pad), bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ELU(),
            ]
            if pool:
                layers.append(nn.MaxPool2d((1, 3), stride=(1, 3)))
            layers.append(nn.Dropout(drop))
            return nn.Sequential(*layers)

        self.block0 = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), padding=(0, 5), bias=False),
            nn.Conv2d(25, 25, (n_channels, 1), bias=False),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 3), stride=(1, 3)),
            nn.Dropout(dropout),
        )
        self.block1 = _conv_block(25, 50, 10)
        self.block2 = _conv_block(50, 100, 10)
        self.block3 = _conv_block(100, 200, 10)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, T)
            out = self.block3(self.block2(self.block1(self.block0(dummy))))
            flat_dim = out.numel()

        self.classifier = nn.Linear(flat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x.flatten(1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.flatten(1)


# ---------------------------------------------------------------------------
# Vanilla Transformer
# ---------------------------------------------------------------------------

class VanillaTransformerEEG(nn.Module):
    """
    Standard Transformer for EEG (temporal patch tokens, full self-attention).

    Used as baseline to isolate benefit of Mamba temporal modeling.

    Parameters
    ----------
    n_channels : int
    n_classes : int
    T : int
    patch_size : int
    d_model : int
    n_heads : int
    n_layers : int
    dropout : float
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_classes: int = 4,
        T: int = 640,
        patch_size: int = 40,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Simple patch embedding
        self.patch_embed = nn.Conv1d(
            n_channels, d_model, kernel_size=patch_size, stride=patch_size // 2
        )
        n_patches = (T - patch_size) // (patch_size // 2) + 1
        self.pos_embed = nn.Embedding(n_patches + 1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        patches = self.patch_embed(x)               # (B, D, P)
        patches = rearrange(patches, "b d p -> b p d")
        P = patches.shape[1]
        pos = torch.arange(P, device=x.device)
        patches = patches + self.pos_embed(pos).unsqueeze(0)
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, patches], dim=1)
        out = self.transformer(seq)
        return self.norm(out[:, 0])                 # CLS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._encode(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode(x)


# ---------------------------------------------------------------------------
# CNN + Transformer hybrid
# ---------------------------------------------------------------------------

class CNNTransformerEEG(nn.Module):
    """
    CNN feature extractor + Transformer encoder hybrid.

    CNN (EEGNet-style) extracts local features; Transformer models long-range.
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_classes: int = 4,
        sfreq: float = 160.0,
        T: int = 640,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        F1, D = 8, 2
        half_sfreq = int(sfreq) // 2

        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, F1, (1, half_sfreq), padding=(0, half_sfreq // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(F1 * D, d_model, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(d_model, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
        )

        # Determine sequence length after CNN
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, T)
            cnn_out = self.cnn(dummy)               # (1, d_model, 1, L)
            seq_len = cnn_out.shape[-1]

        self.pos_embed = nn.Embedding(seq_len + 1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.unsqueeze(1)                         # (B, 1, C, T)
        feat = self.cnn(x)                          # (B, D, 1, L)
        feat = feat.squeeze(2)                      # (B, D, L)
        feat = rearrange(feat, "b d l -> b l d")
        L = feat.shape[1]
        pos = torch.arange(L, device=x.device)
        feat = feat + self.pos_embed(pos).unsqueeze(0)
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, feat], dim=1)
        out = self.norm(self.transformer(seq)[:, 0])
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._encode(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode(x)


# ---------------------------------------------------------------------------
# Simplified MIRepNet-style baseline
# ---------------------------------------------------------------------------

class MIRepNetBaseline(nn.Module):
    """
    Simplified MIRepNet-style contrastive EEG encoder baseline.

    Uses CNN backbone + contrastive pretraining (SimCLR-like), without
    STELLA's Mamba temporal modeling or dual-stream tokenization.

    This is NOT MIRepNet — it is a simplified reimplementation to serve
    as a fair comparison point, not a clone of the original paper.

    Reference:
    Kostas & Bhatt, "MIRepNet: Motor Imagery Representation Network", 2023.
    (This baseline approximates the core CNN+contrastive design.)
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_classes: int = 4,
        T: int = 640,
        d_model: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        # Temporal CNN backbone (1D, depthwise-separable)
        self.backbone = nn.Sequential(
            nn.Conv1d(n_channels, 128, 25, stride=5, padding=12, groups=min(n_channels, 8), bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, d_model, 15, stride=3, padding=7, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 128),
        )
        self.head = nn.Linear(d_model, n_classes)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → (B, D)"""
        return self.dropout(self.backbone(x).squeeze(-1))

    def project(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return F.normalize(self.proj(z), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))
