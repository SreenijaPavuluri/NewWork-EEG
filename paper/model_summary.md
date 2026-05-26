# STELLA — Detailed Model Summary
**Spectro-Temporal EEG Learning with Latent Alignment**

> Config: `n_channels=64, sfreq=160 Hz, segment_len=640 (4 s), d_model=256`

---

## Input

| Property | Value |
|---|---|
| Channels (C) | 64 |
| Samples (T) | 640 (4 s × 160 Hz) |
| Input tensor | `(B, 64, 640)` |

---

## 1. Dual-Stream Tokenizer

### 1-A. Temporal Patch Embedding
Captures waveform morphology via overlapping depthwise convolution.

| Layer | Operation | Input Shape | Output Shape | Params |
|---|---|---|---|---|
| DW Conv1d | `Conv1d(64, 64, k=40, stride=20, groups=64, bias=False)` | `(B, 64, 640)` | `(B, 64, 31)` | 2,560 |
| Rearrange | `(B,C,P)→(B,P,C)` | `(B, 64, 31)` | `(B, 31, 64)` | — |
| Channel Proj | `Linear(64, 64) + GELU` | `(B, 31, 64)` | `(B, 31, 64)` | 4,160 |
| Temporal Proj | `Linear(64, 256)` | `(B, 31, 64)` | `(B, 31, 256)` | 16,640 |
| LayerNorm | `LayerNorm(256)` | `(B, 31, 256)` | `(B, 31, 256)` | 512 |
| Pos Embed | `Embedding(65, 256)` | pos_ids (31,) | `(B, 31, 256)` | 16,640 |
| **Sub-total** | | | `(B, 31, 256)` | **40,512** |

> **P = (640 − 40) / 20 + 1 = 31 patches** (50% overlap, each 0.25 s = 40 samples)

### 1-B. Spectral Band Encoder
Captures oscillatory content via learnable Gaussian band filters.

| Layer | Operation | Input Shape | Output Shape | Params |
|---|---|---|---|---|
| rfft | `torch.fft.rfft(n=256)` | `(B, 64, 640)` | `(B, 64, 129)` complex | — |
| Power | `real² + imag²` | `(B, 64, 129)` | `(B, 64, 129)` | — |
| Band filters | Fixed Gaussian × 5 bands (δ,θ,α,β,γ) | `(B, 64, 129)` | `(B, 64, 5)` | — |
| Band mix | `Parameter(5, 129)` learnable re-weighting | — | — | 645 |
| log1p | Stabilised log-power | `(B, 64, 5)` | `(B, 64, 5)` | — |
| Rearrange | `(B,C,N)→(B,N,C)` | `(B, 64, 5)` | `(B, 5, 64)` | — |
| Band Proj-1 | `Linear(64, 128) + GELU` | `(B, 5, 64)` | `(B, 5, 128)` | 8,320 |
| Band Proj-2 | `Linear(128, 256)` | `(B, 5, 128)` | `(B, 5, 256)` | 33,024 |
| LayerNorm | `LayerNorm(256)` | `(B, 5, 256)` | `(B, 5, 256)` | 512 |
| Band Pos Embed | `Embedding(5, 256)` | band_ids (5,) | `(B, 5, 256)` | 1,280 |
| **Sub-total** | | | `(B, 5, 256)` | **43,781** |

> 5 canonical EEG bands: δ (0.5–4 Hz), θ (4–8 Hz), α (8–13 Hz), β (13–30 Hz), γ (30–45 Hz)

**Tokenizer total params: 354,693** *(includes pw_proj buffer; active params ≈ 84,293)*

---

## 2. Channel Spatial Transformer
Shallow Transformer over the P=31 patch sequence modeling patch-level dependencies with topology-aware attention bias.

### Per-Block (× 2 layers)

| Layer | Operation | Shape | Params |
|---|---|---|---|
| Pre-LayerNorm 1 | `LayerNorm(256)` | `(B, 31, 256)` | 512 |
| QKV projection | `Linear(256, 768, bias=False)` | → `(B, 31, 768)` | 196,608 |
| Multi-head attn | 4 heads, d_head=64, scale=1/8 | `(B, 4, 31, 31)` | — |
| Topology bias | `Parameter(4, 64, 64)` (learnable) | added to attn | 16,384 |
| Out projection | `Linear(256, 256, bias=False)` | `(B, 31, 256)` | 65,536 |
| Pre-LayerNorm 2 | `LayerNorm(256)` | `(B, 31, 256)` | 512 |
| GLU-FFN expand | `Linear(256, 512)` (→ GLU → 256) | `(B, 31, 256)` | 131,328 |
| GLU-FFN contract | `Linear(512→256)` wait, `Linear(256→256)` | `(B, 31, 256)` | 131,072→65,536* |
| Residual + Dropout | — | `(B, 31, 256)` | — |
| **Per-block total** | | | **≈ 476,480** |

*FFN: `Linear(256→1024)` then `GLU` halves to 512, then `Linear(512, 256)` = `263,168 + 131,328 = 394,496`*

| | Params |
|---|---|
| Block 0 | 657,472 |
| Block 1 | 657,472 |
| Final LayerNorm | 512 |
| **Spatial Transformer total** | **1,348,608** |

Output: `(B, 31, 256)`

---

## 3. Gated Spectral-Temporal Fusion *(primary novel component)*
Cross-attention fusion where spectral tokens (K, V) gate temporal patches (Q).

| Component | Operation | Shape | Params |
|---|---|---|---|
| LayerNorm temporal | `LayerNorm(256)` | `(B, 31, 256)` | 512 |
| LayerNorm spectral | `LayerNorm(256)` | `(B,  5, 256)` | 512 |
| Q projection | `Linear(256, 256, bias=False)` from temporal | `(B, 4, 31, 64)` | 65,536 |
| K projection | `Linear(256, 256, bias=False)` from spectral | `(B, 4,  5, 64)` | 65,536 |
| V projection | `Linear(256, 256, bias=False)` from spectral | `(B, 4,  5, 64)` | 65,536 |
| Cross-attention | `softmax(QKᵀ/√64)·V`, scale=1/8 | `(B, 4, 31, 5)→(B, 31, 256)` | — |
| Out projection | `Linear(256, 256, bias=False)` | `(B, 31, 256)` | 65,536 |
| Gate MLP | `Linear(512, 256)` + Sigmoid | `(B, 31, 256)` | 131,328 |
| LayerNorm out | `LayerNorm(256)` | `(B, 31, 256)` | 512 |
| **Fusion total** | | `(B, 31, 256)` | **395,008** |

Gate equation: `gate = σ(Linear([temporal ‖ ctx]))` → `fused = temporal + gate ⊙ ctx`

---

## 4. CLS Token Prepend

| Operation | Shape |
|---|---|
| `Parameter(1, 1, 256)` learnable | 256 params |
| `cat([cls.expand(B,-1,-1), fused], dim=1)` | `(B, 32, 256)` |

---

## 5. Temporal Mamba — S3M Blocks (× 3)
CPU-native selective state space module. Sequential scan, no CUDA required.

### Per S3M Block

| Layer | Operation | Shape | Params |
|---|---|---|---|
| Pre-LayerNorm | `LayerNorm(256)` | `(B, 32, 256)` | 512 |
| Input projection | `Linear(256, 1024, bias=False)` → split to `x`(512) + `z`(512) | `(B, 32, 1024)` | 262,144 |
| DW Conv1d | `Conv1d(512, 512, k=4, padding=3, groups=512)` | `(B, 32, 512)` + SiLU | 2,560 |
| x_proj | `Linear(512, 48, bias=False)` → `[Δ(16), B(16), C(16)]` | `(B, 32, 48)` | 24,576 |
| dt_proj | `Linear(16, 512, bias=True)` + softplus → Δ | `(B, 32, 512)` | 8,704 |
| A_log | `Parameter(512, 16)` HiPPO-init | state matrix | 8,192 |
| D skip | `Parameter(512)` | residual | 512 |
| SSM scan | `h_t = exp(Δ⊙A)·h_{t-1} + Δ⊙B·x_t` | state `(B, 512, 16)` | — |
| SSM output | `y_t = (h ⊙ C[t]).sum(-1)` | `(B, 32, 512)` | — |
| Gate | `y = y * SiLU(z)` | `(B, 32, 512)` | — |
| Out projection | `Linear(512, 256, bias=False)` | `(B, 32, 256)` | 131,072 |
| Residual | `out + residual` | `(B, 32, 256)` | — |
| **Per-block total** | | | **438,272** |

| | Params |
|---|---|
| S3M Block 0 | 438,272 |
| S3M Block 1 | 438,272 |
| S3M Block 2 | 438,272 |
| Final LayerNorm | 512 |
| **Temporal Mamba total** | **1,315,328** |

Output: `(B, 32, 256)`

SSM complexity: **O(L · d_inner · d_state) = O(32 × 512 × 16) = 262,144** ops/layer (vs. O(L²·d) = O(1024·256) = 262,144 for attention at L=32 — same scale here, but S3M scales linearly with L for longer recordings)

---

## 6. Readout

| Operation | Shape |
|---|---|
| Final LayerNorm | `(B, 32, 256)` |
| CLS token slice `[:, 0]` | `(B, 256)` |

---

## 7. Downstream Classification Head (MLPHead)

| Layer | Input | Output | Params |
|---|---|---|---|
| LayerNorm | `(B, 256)` | `(B, 256)` | 512 |
| Linear + GELU | `(B, 256)` | `(B, 128)` | 32,896 |
| Dropout(0.3) | — | — | — |
| Linear | `(B, 128)` | `(B, 4)` | 516 |
| **Head total** | | | **33,924** |

---

## 8. Pretraining Heads (discarded after pretraining)

| Head | Purpose | Loss | Params |
|---|---|---|---|
| ProjectionHead `Linear(256→256→128)` | NT-Xent contrastive | L₁ | 66,048 |
| MLPHead `Linear(256→128→4)` | Auxiliary supervised (CE) | L₂ | 33,924 |
| STC-Temporal `LayerNorm+Linear(256→256→64)` | Spectral-temporal consistency | L₃ | 82,880 |
| STC-Spectral `LayerNorm+Linear(256→256→64)` | Spectral-temporal consistency | L₃ | 82,880 |
| DomainHead `Linear(256→128→109)` | MMD domain alignment | L₄ | 79,853 |
| **Pretraining heads total** | | | **345,585** |

**Loss weights:** λ_c = 1.0, λ_aux = 0.5, λ_stc = 0.3, λ_mmd = 0.1

---

## 9. Complete Parameter Budget

| Module | Params | % of Encoder |
|---|---|---|
| Temporal Patch Embed | 40,512 | 1.2% |
| Spectral Band Encoder | 43,781 | 1.3% |
| (pw_proj — unused buffer) | (266,240) | — |
| Channel Spatial Transformer | 1,348,608 | 39.5% |
| Gated Spectral Fusion | 395,008 | 11.6% |
| S3M Temporal Mamba ×3 | 1,315,328 | 38.5% |
| CLS param + Final LN | 512 | < 0.1% |
| **Encoder Total** | **3,414,405** | **100%** |
| Pretraining heads | 345,585 | — |
| **Grand Total (pretrain)** | **3,759,990** | — |
| Downstream head only | 33,924 | — |
| **Inference Total** | **3,448,329** | — |

---

## 10. Tensor Flow Summary

```
Input                (B, 64, 640)
  │
  ├─ Temporal ──► DW-Conv → Channel/Temporal Proj → LayerNorm + PosEmbed → (B, 31, 256)
  │                                                                              │
  └─ Spectral ──► FFT → Band Filter → log1p → Band Proj → LN + BandEmbed ─► (B,  5, 256)
                                                                              │
                              ┌───────────────────────────────────────────────┘
                              │ temporal (B,31,256)
                              ▼
                Spatial Transformer (2L, 4H)  ──────────────────────► (B, 31, 256)
                              │                         spectral (B,5,256)
                              ▼                              │
                 Gated Spectral Fusion ◄─────────────────────┘
                         gate = σ([temporal ‖ ctx])
                         fused = temporal + gate ⊙ cross_attn(Q=t, KV=s)
                              │
                              ▼
                     cat([CLS], fused)  ──────────────────────────► (B, 32, 256)
                              │
                              ▼
                    S3M Block × 3 (Temporal Mamba)  ─────────────► (B, 32, 256)
                              │
                              ▼
                    LayerNorm + CLS readout  ──────────────────────► (B, 256)
                              │
                              ▼
                    MLPHead(256→128→4)  ─────────────────────────► (B, 4) logits
```

---

## 11. Computational Complexity

| Component | Complexity | Notes |
|---|---|---|
| DW Patch Conv | O(C·P·k) | Linear in sequence |
| Spectral Encoder | O(C·T·log T) | FFT dominant |
| Spatial Transformer | O(P² · d · H · layers) | P=31, cheap |
| Gated Fusion | O(P · S · d) | S=5 bands |
| S3M Temporal Mamba | **O(L · d · N · layers)** | Linear in L |
| **Attention comparison** | O(L² · d · H · layers) | Quadratic |

For T=4 s (L=32 after CLS): S3M complexity ≈ 32 × 512 × 16 × 3 = **786,432 ops/forward**  
Equivalent full attention: 32² × 256 × 4 × 3 = **3,145,728 ops/forward** — 4× cheaper

For T=60 s (L=480 patches): S3M saves ~17× vs attention.
