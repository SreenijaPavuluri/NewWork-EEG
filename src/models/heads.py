"""Task heads for STELLA — linear probe, MLP, and projection heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProbe(nn.Module):
    """Single linear layer for frozen-encoder transfer learning."""

    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


class MLPHead(nn.Module):
    """Two-layer MLP classification head."""

    def __init__(self, d_model: int, n_classes: int, hidden_dim: int | None = None, dropout: float = 0.3):
        super().__init__()
        h = hidden_dim or d_model // 2
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ProjectionHead(nn.Module):
    """
    Contrastive learning projection head (SimCLR-style).

    Maps encoder output to a lower-dimensional space for contrastive loss.
    Discarded after pretraining; encoder output used for downstream tasks.
    """

    def __init__(self, d_model: int, proj_dim: int = 128, hidden_dim: int | None = None):
        super().__init__()
        h = hidden_dim or d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1)


class SpectralTemporalConsistencyHead(nn.Module):
    """
    Projection head for spectral-temporal consistency loss.

    Both spectral and temporal representations are projected to the same
    space for VICReg-style alignment.
    """

    def __init__(self, d_model: int, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DomainAdversarialHead(nn.Module):
    """
    Gradient-reversal domain classifier for domain-adaptive alignment.

    The gradient reversal layer flips gradients during backprop,
    encouraging the encoder to learn domain-invariant features.
    """

    def __init__(self, d_model: int, n_domains: int, lambda_grl: float = 1.0):
        super().__init__()
        self.lambda_grl = lambda_grl
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_domains),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Apply gradient reversal manually
        z_rev = GradientReversal.apply(z, self.lambda_grl)
        return self.classifier(z_rev)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None
