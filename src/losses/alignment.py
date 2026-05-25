"""
Domain-adaptive alignment losses for inter-subject/inter-dataset invariance.

Two implementations:
  1. MMDLoss — Maximum Mean Discrepancy with RBF kernel
     (lightweight, no adversarial training needed)
  2. DomainAdversarialLoss — cross-entropy on domain classifier output
     (used with GradientReversal in DomainAdversarialHead)

MMD is preferred for CPU-only training as it avoids the instability of
min-max adversarial optimization.

References:
  Gretton et al., "A Kernel Two-Sample Test", JMLR 2012.
  Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) with RBF kernel.

    Used to minimize the distribution gap between representations from
    different subjects/domains, encouraging domain-invariant features.

    MMD^2(P, Q) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
    where k(a,b) = exp(-||a-b||^2 / (2σ^2))

    Parameters
    ----------
    kernel_bandwidths : list[float]
        List of RBF bandwidths σ (multi-kernel MMD for robustness).
    """

    def __init__(self, kernel_bandwidths: list[float] | None = None):
        super().__init__()
        self.bandwidths = kernel_bandwidths or [0.1, 0.5, 1.0, 5.0, 10.0]

    def _rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """RBF kernel matrix K(X, Y). Returns (n, m) tensor."""
        # Pairwise squared distances
        XX = (X * X).sum(dim=-1, keepdim=True)
        YY = (Y * Y).sum(dim=-1, keepdim=True)
        XY = torch.mm(X, Y.T)
        dist_sq = XX - 2 * XY + YY.T                 # (n, m)

        K = torch.zeros_like(dist_sq)
        for bw in self.bandwidths:
            K = K + torch.exp(-dist_sq / (2 * bw ** 2))
        return K / len(self.bandwidths)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MMD between source and target distributions.

        Parameters
        ----------
        source, target : (B, D) — representations from two domains/subjects

        Returns
        -------
        mmd : scalar ≥ 0
        """
        n, m = source.shape[0], target.shape[0]

        K_ss = self._rbf_kernel(source, source)
        K_tt = self._rbf_kernel(target, target)
        K_st = self._rbf_kernel(source, target)

        mmd = (
            K_ss.sum() / (n * n)
            - 2 * K_st.sum() / (n * m)
            + K_tt.sum() / (m * m)
        )
        return mmd.clamp(min=0.0)


class DomainAdversarialLoss(nn.Module):
    """
    Cross-entropy loss on domain classifier output.

    Used with DomainAdversarialHead (which applies gradient reversal).
    The encoder is trained to fool the domain classifier by maximizing
    this loss (via gradient reversal), while the classifier minimizes it.

    Parameters
    ----------
    label_smoothing : float
    """

    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(
        self,
        domain_logits: torch.Tensor,
        domain_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        domain_logits : (B, n_domains)
        domain_labels : (B,) — integer domain/subject IDs

        Returns
        -------
        loss : scalar
        """
        return F.cross_entropy(
            domain_logits,
            domain_labels,
            label_smoothing=self.label_smoothing,
        )
