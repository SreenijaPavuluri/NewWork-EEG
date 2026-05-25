"""
Subject-aware NT-Xent contrastive loss for EEG pretraining.

Standard NT-Xent (SimCLR) treats all other samples in a batch as negatives.
For EEG, same-subject samples from different trials are "easy" negatives that
provide less learning signal.  This variant optionally reweights negatives
based on subject identity to encourage harder cross-subject discrimination.

References:
  Chen et al., "SimCLR: A Simple Framework for Contrastive Learning", ICML 2020.
  Banville et al., "Uncovering the structure of clinical EEG signals", 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubjectAwareNTXentLoss(nn.Module):
    """
    NT-Xent loss with optional subject-aware negative weighting.

    For a batch of N samples, generates 2N representations (2 views per sample).
    Positives: same-sample different views.
    Negatives: all other samples, with downweighted same-subject pairs.

    Parameters
    ----------
    temperature : float
        Softmax temperature τ. Lower = sharper distribution.
    subject_weight : float
        Weight for same-subject negatives (0 = ignore, 1 = standard NT-Xent).
        Set < 1 to reduce contribution of easy same-subject negatives.
    """

    def __init__(self, temperature: float = 0.07, subject_weight: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.subject_weight = subject_weight

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z1, z2 : (B, D) — L2-normalized projection vectors
        subject_ids : (B,) — integer subject IDs (optional)

        Returns
        -------
        loss : scalar
        """
        B = z1.shape[0]
        device = z1.device

        # Concatenate: (2B, D)
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, dim=-1)

        # Similarity matrix: (2B, 2B)
        sim = torch.mm(z, z.T) / self.temperature

        # Mask out self-similarities
        mask_self = torch.eye(2 * B, device=device).bool()
        sim.masked_fill_(mask_self, -1e9)

        # Positive pair indices
        # For i in [0,B), positive = i+B; for i in [B,2B), positive = i-B
        pos_idx = torch.cat([
            torch.arange(B, 2 * B, device=device),
            torch.arange(0, B, device=device),
        ])

        # Optional subject-aware negative weighting
        if subject_ids is not None and self.subject_weight < 1.0:
            ids = torch.cat([subject_ids, subject_ids], dim=0)  # (2B,)
            same_subj = (ids.unsqueeze(0) == ids.unsqueeze(1))   # (2B, 2B)
            # Reduce weight for same-subject (non-positive) negatives
            weight_mat = torch.where(
                same_subj & ~mask_self,
                torch.full_like(sim, self.subject_weight),
                torch.ones_like(sim),
            )
            sim = sim * weight_mat

        # NT-Xent loss
        loss = F.cross_entropy(sim, pos_idx)
        return loss
