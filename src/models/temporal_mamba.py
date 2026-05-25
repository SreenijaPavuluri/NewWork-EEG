"""
CPU-friendly Selective State Space Module (S3M) for EEG temporal modeling.

Implements a Mamba-inspired selective scan without CUDA dependencies.
Key differences from mamba-ssm:
  - Uses sequential scan (O(L) time, O(N) memory) for CPU efficiency
  - Diagonal real A matrix (stable, interpretable)
  - Input-dependent B, C, delta (selectivity preserved)
  - No hardware-aware parallel scan required

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective
State Spaces", arXiv 2312.00752, 2023.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class S3MBlock(nn.Module):
    """
    Selective State Space Module — CPU-native Mamba approximation.

    For an input sequence x of shape (B, L, D), applies:
      1. Local depthwise conv for short-range context
      2. Selective scan: h_t = diag(A_t) h_{t-1} + B_t x_t
                         y_t = C_t h_t + D x_t
      3. Gated output projection

    Parameters
    ----------
    d_model : int
        Input/output feature dimension.
    d_state : int
        State dimension N (typically 8–32; smaller = faster on CPU).
    d_conv : int
        Kernel size for local depthwise conv.
    expand : int
        Inner expansion factor (inner_dim = expand * d_model).
    dt_rank : int or 'auto'
        Rank of the delta projection. 'auto' = ceil(d_model / 16).
    dropout : float
        Dropout on output.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # Input projection: x → [x_branch, z_gate]
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Local depthwise conv for short-range context (acts before SSM)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM parameters
        # x_proj maps x_branch → (delta, B, C)
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * d_state, bias=False
        )
        # dt_proj maps low-rank delta → d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A: fixed log-parameterized diagonal state matrix; shape (d_inner, d_state)
        # Initialized with HiPPO-like spectrum: A_n = -(n+1)
        A_log = torch.log(
            torch.arange(1, d_state + 1, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(self.d_inner, 1)
        )
        self.A_log = nn.Parameter(A_log)

        # D: skip (residual) connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # LayerNorm for pre-norm residual
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, D)

        Returns
        -------
        (B, L, D)
        """
        residual = x
        x = self.norm(x)

        B, L, D = x.shape

        # Expand: (B, L, 2*d_inner)
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Local conv on x_branch — causal: trim future context
        x_conv = rearrange(x_branch, "b l d -> b d l")
        x_conv = self.conv1d(x_conv)[..., :L]          # (B, d_inner, L)
        x_conv = rearrange(x_conv, "b d l -> b l d")
        x_conv = F.silu(x_conv)

        # SSM
        y = self._selective_scan(x_conv)                # (B, L, d_inner)

        # Gate with z
        y = y * F.silu(z)

        # Output project
        out = self.out_proj(y)
        out = self.dropout(out)
        return out + residual

    def _selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sequential selective scan — O(L·d_inner·d_state) per forward pass.

        Parameters
        ----------
        x : (B, L, d_inner)

        Returns
        -------
        y : (B, L, d_inner)
        """
        B, L, d_inner = x.shape
        d_state = self.d_state

        # Project to (delta, B_mat, C_mat)
        x_dbc = self.x_proj(x)                          # (B, L, dt_rank + 2*N)
        dt_raw, B_mat, C_mat = x_dbc.split(
            [self.dt_rank, d_state, d_state], dim=-1
        )
        # dt: low-rank → full, softplus to ensure positivity
        dt = F.softplus(self.dt_proj(dt_raw))           # (B, L, d_inner)

        # Discretize A: dA_t = exp(dt_t * (-exp(A_log)))
        A = -torch.exp(self.A_log)                       # (d_inner, N)
        # dA: (B, L, d_inner, N) — exponential ZOH discretization
        dA = torch.exp(
            dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )
        # dB: (B, L, d_inner, N) — input matrix after discretization
        dB = dt.unsqueeze(-1) * B_mat.unsqueeze(2)      # (B, L, d_inner, N)

        # Sequential scan over time
        h = torch.zeros(B, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys: list[torch.Tensor] = []

        for t in range(L):
            # h: (B, d_inner, N)
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            # y_t: (B, d_inner) — C_mat[:, t]: (B, N) → (B, 1, N) broadcast over d_inner
            y_t = (h * C_mat[:, t].unsqueeze(1)).sum(-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                       # (B, L, d_inner)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)    # skip connection
        return y


class TemporalMamba(nn.Module):
    """
    Stack of S3M blocks for long-range EEG temporal modeling.

    Parameters
    ----------
    d_model : int
    n_layers : int
        Number of S3M blocks.
    d_state : int
    d_conv : int
    expand : int
    dropout : float
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                S3MBlock(d_model, d_state, d_conv, expand, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, D)

        Returns
        -------
        (B, L, D)
        """
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)
