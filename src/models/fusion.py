"""
Spectral-Temporal Gated Fusion for STELLA.

Core idea: spectral tokens carry frequency-band context (WHAT oscillations
are present), while temporal tokens carry waveform morphology (WHEN and HOW
they evolve).  A gated cross-attention mechanism lets spectral context
modulate which temporal tokens matter — analogous to how the brain's
frequency-specific networks gate attention to relevant time windows.

Fusion design (GatedSpectralFusion):
  1. Spectral tokens act as KEY/VALUE in cross-attention
  2. Temporal tokens act as QUERY
  3. A sigmoid gate computed from both streams controls information flow
  4. The fused output matches temporal token shape (B, P, D)

This is lightweight (one cross-attention + one gate MLP) and interpretable:
the gate activations reveal which spectral bands inform which time patches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GatedSpectralFusion(nn.Module):
    """
    Cross-attention based gated fusion of spectral and temporal streams.

    Temporal tokens query into spectral tokens.  A learnable gate
    combines the attended spectral context with the temporal tokens.

    Parameters
    ----------
    d_model : int
        Token dimension (must match both streams).
    n_heads : int
        Cross-attention heads.
    dropout : float
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        # Cross-attention projections
        # Q: from temporal tokens
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        # K, V: from spectral tokens
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        # Gate: sigmoid gate computed from concatenation of temporal + attended spectral
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid(),
        )

        self.norm_temporal = nn.LayerNorm(d_model)
        self.norm_spectral = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        temporal_tokens: torch.Tensor,
        spectral_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        temporal_tokens : (B, P, D)  — temporal waveform patches
        spectral_tokens  : (B, S, D)  — spectral band tokens (S = n_bands = 5)

        Returns
        -------
        fused : (B, P, D)  — spectrally-gated temporal tokens
        """
        B, P, D = temporal_tokens.shape
        H = self.n_heads

        t = self.norm_temporal(temporal_tokens)
        s = self.norm_spectral(spectral_tokens)

        # Cross-attention: Q from temporal, K/V from spectral
        Q = rearrange(self.q_proj(t), "b p (h d) -> b h p d", h=H)
        K = rearrange(self.k_proj(s), "b s (h d) -> b h s d", h=H)
        V = rearrange(self.v_proj(s), "b s (h d) -> b h s d", h=H)

        # Attention: (B, H, P, S)
        attn = torch.einsum("bhpd,bhsd->bhps", Q, K) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Attended spectral context: (B, H, P, d_head) → (B, P, D)
        ctx = torch.einsum("bhps,bhsd->bhpd", attn, V)
        ctx = rearrange(ctx, "b h p d -> b p (h d)")
        ctx = self.out_proj(ctx)                          # (B, P, D)

        # Gated fusion: gate from temporal+spectral context
        gate_input = torch.cat([temporal_tokens, ctx], dim=-1)  # (B, P, 2D)
        gate = self.gate_mlp(gate_input)                         # (B, P, D) ∈ (0,1)

        # Residual + gate: fused = temporal + gate * spectral_context
        fused = temporal_tokens + self.drop(gate * ctx)
        fused = self.norm_out(fused)
        return fused


class AdaptiveTokenRouter(nn.Module):
    """
    Optional alternative to GatedSpectralFusion: soft token routing.

    Routes each temporal token to either 'spectral-focused' or
    'temporal-focused' processing via a learned routing score.
    Included as an ablation option.

    Parameters
    ----------
    d_model : int
    """

    def __init__(self, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.router = nn.Linear(d_model, 2)   # 2-way soft routing
        self.spectral_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        temporal_tokens: torch.Tensor,
        spectral_summary: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        temporal_tokens  : (B, P, D)
        spectral_summary : (B, D)  — mean of spectral tokens

        Returns
        -------
        (B, P, D)
        """
        # Routing weights
        route = F.softmax(self.router(temporal_tokens), dim=-1)  # (B, P, 2)
        r_t, r_s = route[..., 0:1], route[..., 1:2]

        # Spectral branch: broadcast spectral summary over patches
        spec = self.spectral_proj(spectral_summary.unsqueeze(1))  # (B, 1, D)

        # Soft blend
        out = r_t * temporal_tokens + r_s * spec
        return self.norm(self.drop(out) + temporal_tokens)
