"""
Spectral-Temporal Consistency Loss (Objective 3 in STELLA pretraining).

Encourages the spectral branch representation and the temporal branch
representation of the same trial to agree in latent space.

This is motivated by: spectral features (band power) and temporal features
(waveform morphology) are two views of the same underlying neural process.
Making them consistent improves representation robustness.

Implementation: VICReg-style (variance-invariance-covariance regularization)
applied to the projected spectral and temporal representations.

Reference:
  Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization
  for Self-Supervised Learning", ICLR 2022.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegLoss(nn.Module):
    """
    VICReg: Variance-Invariance-Covariance Regularization.

    Three complementary terms:
      - Invariance: MSE between the two projections (pull views together)
      - Variance:  variance hinge keeps each dim above γ (avoid collapse)
      - Covariance: off-diagonal covariance penalty (decorrelates features)

    Parameters
    ----------
    lambda_inv : float
        Weight for invariance term.
    mu_var : float
        Weight for variance term.
    nu_cov : float
        Weight for covariance term.
    gamma : float
        Target variance level.
    eps : float
        Numerical stability.
    """

    def __init__(
        self,
        lambda_inv: float = 25.0,
        mu_var: float = 25.0,
        nu_cov: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.lambda_inv = lambda_inv
        self.mu_var = mu_var
        self.nu_cov = nu_cov
        self.gamma = gamma
        self.eps = eps

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z1, z2 : (B, D) — projected representations from two branches

        Returns
        -------
        loss : scalar
        """
        B, D = z1.shape

        # Invariance: MSE(z1, z2) — pull two views together
        inv_loss = F.mse_loss(z1, z2)

        # Variance: hinge on per-dim std to prevent collapse
        def _var_loss(z):
            std = torch.sqrt(z.var(dim=0) + self.eps)
            return F.relu(self.gamma - std).mean()

        var_loss = _var_loss(z1) + _var_loss(z2)

        # Covariance: penalize off-diagonal correlations
        def _cov_loss(z):
            z = z - z.mean(dim=0)
            cov = (z.T @ z) / (B - 1)             # (D, D)
            # Off-diagonal squared values
            off_diag = cov ** 2
            off_diag.fill_diagonal_(0.0)
            return off_diag.sum() / D

        cov_loss = _cov_loss(z1) + _cov_loss(z2)

        total = (
            self.lambda_inv * inv_loss
            + self.mu_var * var_loss
            + self.nu_cov * cov_loss
        )
        return total


class SpectralTemporalConsistencyLoss(nn.Module):
    """
    Lightweight consistency loss between spectral and temporal representations.

    Uses cosine similarity loss (1 - cosine_similarity) as a softer alternative
    to MSE, which is invariant to scale differences between branches.

    Parameters
    ----------
    reduction : str
        "mean" | "sum"
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        z_temporal: torch.Tensor,
        z_spectral: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z_temporal, z_spectral : (B, D)

        Returns
        -------
        loss : scalar in [0, 2]
        """
        z_t = F.normalize(z_temporal, dim=-1)
        z_s = F.normalize(z_spectral, dim=-1)
        cosine_sim = (z_t * z_s).sum(dim=-1)       # (B,)
        loss = 1.0 - cosine_sim
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()
