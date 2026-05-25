"""
Combined STELLA pretraining loss (all 4 objectives).

Weighted sum of:
  1. Subject-aware contrastive (NT-Xent)        — λ_contrastive
  2. Auxiliary supervised classification         — λ_auxiliary
  3. Spectral-temporal consistency (VICReg)      — λ_consistency
  4. Domain-adaptive alignment (MMD)             — λ_alignment

Loss weights are configured in the training YAML and can be ablated by
setting individual weights to 0.0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contrastive import SubjectAwareNTXentLoss
from .consistency import VICRegLoss, SpectralTemporalConsistencyLoss
from .alignment import MMDLoss


class STELLAPretrainLoss(nn.Module):
    """
    Multi-objective pretraining loss for STELLA.

    Parameters
    ----------
    lambda_contrastive : float
    lambda_auxiliary : float
    lambda_consistency : float
    lambda_alignment : float
    temperature : float
        NT-Xent temperature.
    subject_weight : float
        Subject-aware negative weighting.
    """

    def __init__(
        self,
        lambda_contrastive: float = 1.0,
        lambda_auxiliary: float = 0.5,
        lambda_consistency: float = 0.3,
        lambda_alignment: float = 0.1,
        temperature: float = 0.07,
        subject_weight: float = 0.5,
    ):
        super().__init__()
        self.lambda_contrastive = lambda_contrastive
        self.lambda_auxiliary = lambda_auxiliary
        self.lambda_consistency = lambda_consistency
        self.lambda_alignment = lambda_alignment

        self.contrastive_loss = SubjectAwareNTXentLoss(temperature, subject_weight)
        self.consistency_loss = SpectralTemporalConsistencyLoss()
        self.mmd_loss = MMDLoss()

    def forward(
        self,
        model_output: dict[str, torch.Tensor],
        labels: torch.Tensor | None = None,
        domain_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Parameters
        ----------
        model_output : dict from STELLA.pretrain_forward()
        labels : (B,) — class labels for auxiliary head
        domain_ids : (B,) — subject IDs for alignment

        Returns
        -------
        total_loss : scalar
        loss_components : dict of individual scalar losses (for logging)
        """
        components: dict[str, float] = {}
        total = torch.tensor(0.0, device=model_output["z1"].device)

        # 1. Contrastive
        if self.lambda_contrastive > 0:
            l_c = self.contrastive_loss(
                model_output["z1"],
                model_output["z2"],
                subject_ids=domain_ids,
            )
            total = total + self.lambda_contrastive * l_c
            components["contrastive"] = l_c.item()

        # 2. Auxiliary supervised
        if self.lambda_auxiliary > 0 and labels is not None:
            aux_logits = model_output.get("aux_logits")
            if aux_logits is not None:
                l_a = F.cross_entropy(aux_logits, labels)
                total = total + self.lambda_auxiliary * l_a
                components["auxiliary"] = l_a.item()

        # 3. Spectral-temporal consistency
        if self.lambda_consistency > 0:
            stc_t = model_output.get("stc_temporal")
            stc_s = model_output.get("stc_spectral")
            if stc_t is not None and stc_s is not None:
                l_stc = self.consistency_loss(stc_t, stc_s)
                total = total + self.lambda_consistency * l_stc
                components["consistency"] = l_stc.item()

        # 4. Domain alignment (MMD between random subject splits)
        if self.lambda_alignment > 0 and domain_ids is not None:
            repr1 = model_output.get("repr1")
            if repr1 is not None:
                # Split batch into two halves for MMD estimate
                B = repr1.shape[0]
                half = B // 2
                if half > 1:
                    l_mmd = self.mmd_loss(repr1[:half], repr1[half:half * 2])
                    total = total + self.lambda_alignment * l_mmd
                    components["alignment"] = l_mmd.item()

        components["total"] = total.item()
        return total, components
