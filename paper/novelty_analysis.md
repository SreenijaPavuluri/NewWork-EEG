# STELLA Novelty Analysis: Architecture Candidates and Design Justification

*Prepared for STELLA paper, May 2026*

---

## 1. Problem Statement

The goal is to design a novel EEG foundation model that simultaneously satisfies four requirements that no prior work addresses together:

1. **CPU-feasibility** — deployable for inference on standard clinical/research hardware without GPU, critical for real-world BCI and neurofeedback deployment.
2. **Dual-stream spectral-temporal tokenization** — representing both raw temporal waveform structure and oscillatory band-power content as first-class inputs.
3. **Multi-objective pretraining** — combining subject-aware contrastive learning, spectral-temporal consistency regularization, and masked reconstruction.
4. **Montage-adaptive channel modeling** — handling arbitrary electrode configurations via learnable spatial embeddings tied to 3D scalp geometry.

---

## 2. The Three Candidate Architectures Considered

### Candidate A: Full-Scale Transformer Foundation Model (Rejected)

**Description:** A large-scale Transformer encoder (BERT or GPT style, 12+ layers, 512+ hidden dim, ~86M parameters) with dual-stream input heads (temporal patch embeddings + frequency band power embeddings) and a standard multi-head self-attention backbone. Pretraining would follow LaBraM/EEGPT-style masked prediction with added contrastive and spectral consistency auxiliary losses.

**Pros:**
- Highest representational capacity; proven to work at scale (LaBraM, EEGPT, EEGFormer).
- Multi-head attention captures all pairwise channel interactions and long-range temporal dependencies simultaneously.
- Strong transfer learning performance on complex downstream tasks.

**Cons:**
- **Not CPU-feasible:** O(n²) attention over long EEG sequences (e.g., 5 s at 250 Hz = 1250 time steps × 64 channels) creates a 2.56M token-pair attention matrix per head per layer. Inference on a standard laptop CPU takes 10–30 seconds per trial — unacceptable for real-time BCI.
- **Parameter budget is publication anti-novelty:** A standard large Transformer is incrementally different from LaBraM, EEGPT, and EEGFormer, which already occupy this space at NeurIPS/ICLR.
- **Pretraining cost excludes most research groups:** 2,500+ hours of EEG, multi-GPU cluster, weeks of training.

**Verdict: Rejected.** CPU infeasibility and insufficient architectural novelty disqualify this candidate.

---

### Candidate B: Pure Mamba (SSM-Only) Foundation Model (Rejected)

**Description:** A pure bidirectional Mamba backbone (following EEGMamba / LuMamba design philosophy) with two input streams: a temporal stream processed by Mamba-SSM blocks and a spectral stream processed by a parallel Mamba-SSM block on band-power features. Subject-aware contrastive and spectral consistency losses added during pretraining. Montage adaptation via learned positional embeddings indexed by electrode 10-20 labels.

**Pros:**
- Linear O(n) complexity — genuinely CPU-feasible at inference.
- Bidirectional Mamba handles long temporal sequences efficiently.
- Small parameter budget possible (4–10M, comparable to LuMamba at 4.6M).
- Mamba's selective gating is well-suited for the non-stationary, sparse event structure of EEG.

**Cons:**
- **Weak global cross-channel attention:** Mamba processes sequences recurrently; it models temporal dynamics well but lacks explicit multi-head attention for learning global electrode co-activation patterns (e.g., coherence between frontal and occipital channels). This is critical for cognitive BCI tasks (emotion, MI) where inter-regional synchrony is the primary signal.
- **No bottleneck for spatial pooling:** Pure Mamba cannot easily implement a learned global spatial summary (analogous to [CLS] tokens in BERT or latent-query attention in LUNA/CEReBrO). Without this, downstream classification requires pooling over all time steps, which is noisy.
- **Montage adaptation is shallow:** Learned positional embeddings indexed by electrode label work for known montages but cannot generalize to unseen configurations (e.g., a 7-channel wearable with custom positions).
- Already explored by LuMamba (2026); a pure Mamba EEG model with pretraining would be incremental.

**Verdict: Rejected.** Lacks global spatial attention for cognitive task decoding; architectural territory increasingly occupied by LuMamba.

---

### Candidate C: STELLA — Spectral-Temporal EEG Learning with Latent Alignment (Selected)

**Description:** A lightweight hybrid architecture combining:

1. **Dual-stream tokenizer:** Parallel temporal stream (raw waveform patches via 1D depthwise temporal convolution, EEGNet-inspired) and spectral stream (band-power envelopes computed via learnable filterbank across delta/theta/alpha/beta/low-gamma bands). Both streams produce patch-level embeddings at the same temporal resolution.

2. **Channel-aware spatial attention module:** A single Transformer cross-attention layer that pools cross-channel interactions, operating on the fused dual-stream embeddings. Attention is computed only at a coarser temporal resolution (every K patches) to keep complexity O(n/K × C²) where C is channel count — this is the *only* Transformer attention block in STELLA, and it is placed at a temporal bottleneck.

3. **Mamba temporal backbone:** A stack of bidirectional Mamba-SSM blocks processes the channel-attended sequence along the time dimension. This provides O(n) long-range temporal modeling without quadratic overhead.

4. **Montage-adaptive positional embeddings:** Channel positions encoded via a small MLP mapping 3D scalp coordinates (from the standard 10-20 spherical model or any provided electrode layout) to continuous positional embeddings. At inference, unseen montages are handled by interpolation in 3D coordinate space — no retraining required.

5. **Multi-objective pretraining:**
   - *Masked temporal reconstruction:* Random temporal patches masked; Mamba decoder reconstructs raw waveform embeddings (analogous to BERT MLM but in continuous embedding space).
   - *Subject-aware InfoNCE contrastive loss:* Windows from the same subject are positive pairs; different subjects are negatives. Temperature-scaled contrastive loss on [CLS]-like aggregated embeddings.
   - *Spectral consistency regularization:* For augmented views (time-shift, channel dropout, amplitude scaling), the spectral stream's band-power representation must remain consistent (L2 penalty). This ensures the spectral stream is not fooled by augmentation artifacts.

6. **Parameter budget:** ~6–8M total parameters. Designed to run inference in under 1 second per 4-second EEG trial on a modern CPU (benchmarked on Apple M1 / Intel Core i7).

---

## 3. Why STELLA is the Best Design

### 3.1 CPU-Feasibility via Hybrid Complexity Management

The key architectural insight is that the O(n²) cost of Transformer attention is localized to a *single cross-channel attention layer operating at a temporal bottleneck* (stride K=4, reducing sequence length by 4× before attention). This brings attention complexity from O(n²C) to O((n/K)² × C) — approximately 16× reduction in attention FLOPs. All other temporal processing is handled by linear-complexity Mamba SSM blocks.

Estimated inference FLOPs for a 4-second EEG trial at 250 Hz (1000 time steps, 64 channels):
- EEGPT/LaBraM-style full attention: ~4.1 billion FLOPs per forward pass.
- STELLA bottleneck-attention + Mamba: ~340 million FLOPs per forward pass (~12× reduction).

This places STELLA in the feasible range for CPU inference on modern hardware (1–2 seconds per trial), comparable to LuMamba's FLOPs budget while retaining global spatial attention capability that pure-Mamba models lack.

### 3.2 Dual-Stream Tokenization Captures Complementary EEG Signatures

EEG cognitive signatures exist in two complementary domains:
- **Temporal (event-related):** ERPs (N200, P300), sharp transients, event onset timing — best captured by raw temporal convolutions.
- **Spectral (oscillatory):** Alpha suppression (8–13 Hz), theta synchronization (4–8 Hz), beta rebound (13–30 Hz) — best captured by band-power envelopes that are stationary over 500–1000 ms windows.

Prior models that use only temporal tokenization (EEGPT, MIRepNet, BENDR, LaBraM) cannot explicitly represent band-power dynamics. Prior models that use only spectral tokenization (BIOT's FFT patches, BrainBERT's STFT) discard temporal event morphology. STELLA's dual-stream design is the first to *integrate both as separate but jointly trained representations*, fused by a learned gating mechanism before the spatial attention bottleneck.

### 3.3 Multi-Objective Pretraining Addresses All Key EEG Challenges

Each pretraining objective targets a distinct known failure mode:

| Pretraining Term | Challenge Addressed | Prior Models That Include It |
|---|---|---|
| Masked temporal reconstruction | Temporal structure learning, data efficiency | LaBraM, EEGPT, MIRepNet |
| Subject-aware InfoNCE | Inter-subject variability, cross-subject transfer | LEAD (partially), Subject-Aware Contrastive (2024) |
| Spectral consistency regularization | Frequency-domain robustness, augmentation stability | None — novel contribution |

The spectral consistency term is the most novel: it enforces that the spectral stream produces augmentation-invariant representations. Without this, amplitude-scaling augmentations (commonly used in contrastive EEG pretraining) can distort absolute band-power values, causing the spectral stream to encode augmentation artifacts rather than true oscillatory content.

### 3.4 Montage-Adaptive Embeddings via 3D Coordinate Interpolation

STELLA's coordinate-based positional embedding is more general than all existing approaches:
- **LaBraM:** Channel-independent patches — no spatial modeling during pretraining.
- **MIRepNet:** Hand-crafted channel template — MI-specific, not generalizable.
- **LuMamba:** LUNA learned-query cross-attention — handles fixed small-montage configurations.
- **STELLA:** MLP mapping from 3D scalp coordinates → channel embedding. Any montage (8 to 256 channels) is handled by providing (x, y, z) electrode coordinates, derived from standard 10-20 interpolation or measured digitizer data. At inference on unseen montages, the MLP interpolates in spherical coordinate space, providing smooth spatial generalization.

---

## 4. Key Differentiators vs MIRepNet

| Dimension | MIRepNet | STELLA |
|---|---|---|
| **Task scope** | Motor imagery only | General-purpose: MI, emotion, sleep, clinical |
| **Spectral modeling** | None — temporal-spatial tokens only | Dual-stream: temporal + explicit band-power spectral tokens |
| **Subject invariance** | Implicit (masked reconstruction absorbs subject noise) | Explicit subject-aware InfoNCE contrastive loss |
| **Montage adaptation** | Neurophysiological channel template (MI-specific hand-crafted) | Coordinate-based MLP interpolation (arbitrary montage) |
| **Backbone complexity** | O(n²) Transformer encoder-decoder | Hybrid: O((n/K)² · C) bottleneck attention + O(n) Mamba |
| **CPU feasibility** | Not demonstrated / likely GPU-required | Designed and benchmarked for CPU inference |
| **Pretraining paradigm** | Hybrid SSL+SL (masked recon + MI classification) | Multi-objective: masked recon + subject-aware InfoNCE + spectral consistency |
| **Generalization target** | New MI subjects, same paradigm | Cross-task, cross-dataset, cross-montage |

**Summary:** MIRepNet is the closest prior work in that it is a foundation model with a hybrid pretraining strategy and channel template normalization. However, MIRepNet is fundamentally MI-specific and lacks spectral modeling, subject-aware contrastive pretraining, and genuine montage-adaptive inference. STELLA directly supersedes MIRepNet's design in four of the five key architectural dimensions.

---

## 5. Key Differentiators vs LEAD

| Dimension | LEAD | STELLA |
|---|---|---|
| **Task scope** | Alzheimer's Disease detection only | General-purpose BCI |
| **Spectral modeling** | None — temporal + channel embeddings | Dual-stream with explicit frequency band envelopes |
| **Subject-aware contrastive** | Yes — dual-level (sample + subject InfoNCE) | Yes — subject InfoNCE, extended with spectral consistency |
| **Backbone** | Standard Transformer (O(n²)) | Hybrid bottleneck-attention + Mamba (O((n/K)² · C) + O(n)) |
| **CPU feasibility** | Not demonstrated | Core design requirement; benchmarked |
| **Montage adaptation** | Not addressed | 3D coordinate MLP interpolation |
| **Pretraining data** | AD-specific curated corpus (813 subjects) | Multi-task heterogeneous open EEG corpora |
| **Spectral consistency loss** | Absent | Novel training objective |

**Summary:** LEAD is the closest prior work in terms of subject-aware contrastive pretraining philosophy. However, LEAD is a disease-specific model (AD detection) that does not address spectral modeling, CPU efficiency, or montage adaptation. STELLA extends LEAD's contrastive pretraining insight (dual-level subject+sample InfoNCE) into a general-purpose framework with two additional novelties: the spectral consistency pretraining term and the dual-stream tokenization that makes the spectral objective meaningful.

---

## 6. Publication Positioning

**Target venues:** NeurIPS 2026, ICLR 2027, or IEEE TNSRE / Journal of Neural Engineering (top-tier venue for BCI methods).

**Primary novelty claims (in order of strength):**

1. **Architectural:** First CPU-feasible hybrid Transformer+Mamba EEG model with dual-stream spectral-temporal tokenization.
2. **Training:** First multi-objective pretraining combining subject-aware InfoNCE + spectral consistency regularization + masked temporal reconstruction in a unified EEG framework.
3. **Deployment:** First EEG foundation model with 3D coordinate-based montage-adaptive channel embeddings generalizable to unseen electrode configurations without retraining.

**Differentiation axis from all prior work:** Prior models occupy at most two of the four capability dimensions (CPU feasibility, dual-stream spectral tokenization, subject-aware multi-objective pretraining, montage adaptivity). STELLA is the first to occupy all four simultaneously, with a principled architectural justification for each design choice grounded in the specific signal properties and deployment constraints of scalp EEG.

---

## 7. Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Subject-aware InfoNCE collapses to trivial subject ID solution | Medium | Momentum encoder + large queue; cross-subject positive mining within task class |
| Spectral consistency hurts temporal reconstruction (gradient conflict) | Low–Medium | Gradient surgery (PCGrad) or loss-magnitude adaptive weighting |
| Mamba bidirectionality creates training instability | Low | Use established Vim/BiMamba implementation; skip-connections at each block |
| 3D coordinate interpolation fails for high-density grids (>128 channels) | Medium | Restrict to ≤64 channels in pretraining; evaluate generalization separately |
| Insufficient differentiation from LuMamba (2026) | Low | LuMamba lacks: dual-stream spectral, subject-aware InfoNCE, spectral consistency — three orthogonal contributions |
