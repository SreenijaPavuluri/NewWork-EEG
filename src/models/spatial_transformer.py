"""
Shallow Channel-Spatial Transformer for EEG inter-channel modeling.

Design rationale:
  - Transformer depth is intentionally shallow (2 layers) to keep compute low
  - Operates on the channel dimension of each temporal patch
  - Uses topology-aware relative bias derived from EEG channel adjacency
  - Multi-head attention with compact dimension (d_model // n_heads per head)

This module handles inter-channel spatial dependencies that Mamba cannot
capture, since Mamba processes a temporal sequence, not a channel sequence.

The channel topology bias encodes prior knowledge about electrode placement
without a full graph neural network — a lightweight graph prior.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Topology-aware relative position bias
# ---------------------------------------------------------------------------

def build_adjacency_bias(
    n_channels: int,
    adj: torch.Tensor | None = None,
    n_heads: int = 4,
) -> torch.Tensor:
    """
    Build a learnable bias initialized from channel adjacency.

    If no adjacency matrix is provided, returns a learned scalar bias per
    head (equivalent to no topology prior).

    Parameters
    ----------
    n_channels : int
    adj : (C, C) binary/weighted adjacency matrix or None
    n_heads : int

    Returns
    -------
    bias : (n_heads, C, C)
    """
    if adj is None:
        return torch.zeros(n_heads, n_channels, n_channels)
    # Normalize adjacency and expand to n_heads
    adj = adj.float()
    adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)
    bias = adj.unsqueeze(0).expand(n_heads, -1, -1).clone()
    return bias


class ChannelSelfAttention(nn.Module):
    """
    Multi-head self-attention over the channel (spatial) dimension.

    Input: (B*P, C, D)  — batch_patches × channels × features
    Output: (B*P, C, D)

    Parameters
    ----------
    d_model : int
    n_heads : int
    dropout : float
    use_topology_bias : bool
        If True, adds a learnable topology-prior attention bias.
    n_channels : int
        Required when use_topology_bias=True.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_topology_bias: bool = True,
        n_channels: int = 64,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        # Topology bias: (n_heads, C, C)
        if use_topology_bias:
            self.topo_bias = nn.Parameter(
                torch.zeros(n_heads, n_channels, n_channels)
            )
        else:
            self.topo_bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, C, D) where N = B*P

        Returns
        -------
        (N, C, D)
        """
        N, C, D = x.shape
        H = self.n_heads

        qkv = self.qkv(x)                      # (N, C, 3D)
        q, k, v = qkv.chunk(3, dim=-1)          # each (N, C, D)

        # (N, H, C, d_head)
        q = rearrange(q, "n c (h d) -> n h c d", h=H)
        k = rearrange(k, "n c (h d) -> n h c d", h=H)
        v = rearrange(v, "n c (h d) -> n h c d", h=H)

        # Attention scores: (N, H, C, C)
        attn = torch.einsum("nhcd,nhkd->nhck", q, k) * self.scale

        # Add topology bias if available and shapes match
        if self.topo_bias is not None and C == self.topo_bias.shape[-1]:
            attn = attn + self.topo_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Aggregate: (N, H, C, d_head)
        out = torch.einsum("nhck,nhkd->nhcd", attn, v)
        out = rearrange(out, "n h c d -> n c (h d)")
        out = self.out_proj(out)
        return out


class SpatialTransformerBlock(nn.Module):
    """
    Single Pre-LN spatial Transformer block.

    Pre-LayerNorm for training stability on CPU (no gradient scaling needed).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        ffn_expand: int = 2,
        dropout: float = 0.1,
        n_channels: int = 64,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ChannelSelfAttention(d_model, n_heads, dropout, n_channels=n_channels)
        self.norm2 = nn.LayerNorm(d_model)

        # Gated linear unit FFN (smaller than standard FFN, gating helps)
        d_ff = ffn_expand * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_ff),
            nn.GLU(dim=-1),              # halves dimension back to d_ff
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN self-attention
        x = x + self.drop(self.attn(self.norm1(x)))
        # Pre-LN FFN
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class ChannelSpatialTransformer(nn.Module):
    """
    Shallow Transformer for spatial (inter-channel) EEG modeling.

    Applied per-patch on the channel dimension.  The shallow depth (2 layers
    by default) is deliberate: channels form a small set (typically 22–128),
    so deep stacks are wasteful.

    Parameters
    ----------
    d_model : int
    n_heads : int
    n_layers : int
    ffn_expand : int
    dropout : float
    n_channels : int
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_expand: int = 2,
        dropout: float = 0.1,
        n_channels: int = 64,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SpatialTransformerBlock(d_model, n_heads, ffn_expand, dropout, n_channels)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, C, D) where N = B*P (batch × patches)

        Returns
        -------
        (N, C, D)
        """
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)
