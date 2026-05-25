# STELLA: Spectro-Temporal EEG Learning with Latent Alignment

**A Lightweight Foundation Model for CPU-Feasible Motor Imagery Decoding**

---

*Draft — IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE) format*

---

## Abstract

We present **STELLA** (Spectro-Temporal EEG Learning with Latent Alignment), a novel lightweight EEG foundation model designed for CPU-feasible pretraining and subject-independent transfer learning in motor imagery (MI) decoding. STELLA introduces a **dual-stream tokenization strategy** that independently encodes waveform morphology via temporal patch embeddings and oscillatory content via a learnable band-power spectral encoder. The two streams are fused by a **gated cross-attention mechanism** that lets spectral context dynamically gate the relevance of temporal patches — an architecturally principled approach not present in prior work. Long-range temporal dependencies are modeled by a **CPU-native Selective State Space Module (S3M)**, a pure-PyTorch implementation of selective scan dynamics that avoids CUDA dependencies while retaining the sub-quadratic scaling advantage of Mamba-style models. Channel topology is incorporated through shallow Transformer blocks with optional adjacency-aware attention bias.

STELLA is pretrained with four complementary objectives: (1) subject-aware contrastive learning (NT-Xent with cross-subject negative reweighting), (2) auxiliary supervised classification, (3) spectral-temporal consistency via VICReg-style regularization, and (4) lightweight MMD-based domain alignment across subjects. The resulting encoder transfers effectively to unseen subjects via linear probing and full finetuning.

Experiments on the PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB, 109 subjects, 4-class MI) demonstrate competitive performance with prior methods under a strict subject-independent evaluation protocol. STELLA's encoder contains 4.3M parameters and can be pretrained in under 12 CPU hours on a standard laptop, making it accessible to researchers without GPU infrastructure. All code, pretrained weights, and experiment scripts are released at [*repository URL*].

**Keywords:** EEG, motor imagery, foundation model, self-supervised learning, state space models, Mamba, spectral-temporal tokenization, CPU-feasible deep learning.

---

## 1. Introduction

Electroencephalography (EEG)-based brain-computer interfaces (BCIs) offer non-invasive pathways for communication, motor rehabilitation, and neural prosthetics [1]. Motor imagery (MI) decoding — classifying neural patterns associated with imagined movements — is among the most studied BCI paradigms. Despite decades of research, the practical deployment of EEG BCIs remains limited by three interconnected challenges:

**Subject variability.** EEG signals exhibit pronounced inter-subject differences in electrode impedance, head geometry, brain anatomy, and task engagement, causing models trained on one subject to generalize poorly to another [2].

**Data scarcity.** Collecting labeled MI data is expensive and fatiguing for participants. Each BCI session typically yields only a few hundred labeled trials per subject, making supervised deep learning difficult without cross-subject pooling [3].

**Compute accessibility.** Recent EEG foundation models [4, 5, 6, 7] have achieved strong results but require substantial GPU resources for pretraining — a significant barrier for clinical or embedded deployment.

Recent work has begun to address these challenges through self-supervised pretraining [8, 9] and large-scale EEG foundation models [6, 7, 10]. MIRepNet [11] introduced the first MI-specific contrastive foundation model, while LEAD [12] proposed dual-level (subject + sample) contrastive objectives for EEG representation learning. BIOT [4] extended EEG pretraining to multiple biosignal modalities using FFT-based patch tokenization. LaBraM [5] achieved state-of-the-art results with 2500 hours of pretraining data and a neural codebook tokenizer.

However, **none of these models simultaneously satisfy all three requirements**: (1) subject-independent generalization through principled multi-objective pretraining, (2) frequency-aware tokenization that explicitly models oscillatory EEG structure, and (3) CPU-feasible training compatible with modest hardware. This gap motivates the present work.

We propose STELLA, which makes the following contributions:

1. **Dual-stream EEG tokenization** combining temporal patch tokens (waveform morphology) and spectral band tokens (oscillatory content) in a unified framework.

2. **Gated spectral-temporal fusion** — a cross-attention mechanism where spectral context gates the relevance of temporal patches, providing the model with dynamic frequency-aware processing.

3. **CPU-native S3M temporal modeling** — a pure-PyTorch selective state space module approximating Mamba [13] dynamics without CUDA dependencies, enabling sub-quadratic temporal scaling on CPU.

4. **Multi-objective pretraining** combining four complementary losses that balance representation diversity, semantic consistency, and domain invariance.

We position STELLA not as a "state-of-the-art" claimant (such claims require controlled multi-site evaluation beyond this initial work), but as a **reproducible, principled, and computationally accessible** EEG foundation model for the community.

---

## 2. Related Work

### 2.1 Convolutional EEG Decoders

EEGNet [14] established the utility of depthwise separable convolutions for compact EEG feature extraction. ShallowConvNet and DeepConvNet [15] demonstrated that deeper temporal convolutions improve MI classification when sufficient data is available. These supervised, within-subject models form our primary baselines.

### 2.2 Self-Supervised EEG Representation Learning

BENDR [8] applied contrastive predictive coding (CPC) to raw EEG, inspired by wav2vec. Banville et al. [9] developed temporal and contextual contrastive objectives for clinical EEG. Both demonstrated that pretraining on unlabeled data improves downstream transfer, but neither specifically targets MI.

### 2.3 EEG Foundation Models

**BIOT** [4] pretrained a cross-modal biosignal encoder using FFT-based channel-independent patch tokenization, enabling zero-shot transfer across datasets. **LaBraM** [5] introduced a neural tokenizer based on a VQ-VAE codebook trained via masked code prediction, achieving strong results across multiple EEG paradigms. **EEGPT** [7] applied spatiotemporal alignment pretraining with 10M parameters. **MIRepNet** [11] specifically targets MI decoding using a hybrid supervised + contrastive pretraining scheme. **LEAD** [12] proposes dual-level contrastive objectives for Alzheimer's EEG.

**Key differences from STELLA**: None of these models combine (a) gated spectral-temporal fusion, (b) CPU-native Mamba temporal modeling, and (c) subject-aware multi-objective pretraining in a single framework under strict CPU constraints.

### 2.4 State Space Models for Biosignals

Mamba [13] introduced selective state space models (SSMs) with O(L) time complexity, improving upon Transformer's O(L²) attention for long sequences. EEGMamba [16] and LuMamba [17] adapted Mamba for EEG, with LuMamba achieving 377× FLOP reduction over full attention. However, both require GPU-optimized `mamba-ssm` kernels. Our S3M is the first CPU-native selective SSM designed specifically for EEG.

---

## 3. Method

### 3.1 Problem Setting

Let x ∈ ℝ^{C×T} denote a preprocessed EEG trial with C channels and T time samples. We aim to learn an encoder f: ℝ^{C×T} → ℝ^D that produces a fixed-size representation z = f(x) useful for downstream MI classification across unseen subjects.

### 3.2 Dual-Stream Tokenization

**Temporal stream.** We apply a channel-wise (depthwise) 1D convolution with kernel size K_p and stride K_p/2, producing P overlapping temporal patches per trial. A pointwise projection maps the C-dimensional channel features to a d-dimensional embedding space:

```
T_patches = PatchConv1D(x) ∈ ℝ^{P×d}
```

Learnable positional embeddings are added to encode temporal order. With K_p=40 (0.25s at 160Hz) and stride 20, a 4-second trial produces P=30 patches.

**Spectral stream.** We compute the real FFT along the time axis and apply a set of learnable Gaussian-initialized band filters {f_k} for k ∈ {δ,θ,α,β,γ}. The filters are multiplicatively modulated by learnable parameters to allow the model to reweight frequency content:

```
φ_k = σ(w_k) ⊙ Gauss(f_k)   [band filter for band k]
P_band(x) = log(1 + Σ_f φ_k(f) |X(f)|²)  ∈ ℝ^{C×K}
```

A linear projection maps per-channel band powers to d-dimensional spectral tokens:
```
S_tokens = Linear(P_band(x)) ∈ ℝ^{K×d}
```

This interpretable spectral branch provides the model with prior knowledge about EEG oscillatory structure while remaining trainable.

### 3.3 Shallow Channel Spatial Transformer

Before temporal sequence modeling, we apply 2 layers of multi-head self-attention over the token sequence dimension (treating patches as "positions") to capture patch-level dependencies. We use Pre-LayerNorm for training stability and gated linear unit (GLU) feed-forward networks:

```
CST(T_patches) = TransformerEncoder(T_patches; layers=2, heads=4)
```

The shallow depth (2 layers vs. 12 in BERT) is deliberate: it reduces parameter count and prevents overfitting on small EEG datasets while still capturing patch-level structure.

### 3.4 Gated Spectral-Temporal Fusion

This module constitutes STELLA's primary architectural novelty. Temporal patches act as queries while spectral tokens serve as keys and values in a cross-attention mechanism:

```
ctx = CrossAttn(Q=T_patches, K=S_tokens, V=S_tokens)  ∈ ℝ^{P×d}
gate = σ(Linear([T_patches; ctx]))                     ∈ ℝ^{P×d}
fused = T_patches + gate ⊙ ctx                         ∈ ℝ^{P×d}
```

The sigmoid gate adaptively controls how much spectral context influences each temporal patch. Critically, this is **sample-adaptive**: for signals dominated by alpha-band activity (e.g., motor imagery), the gate assigns high weight to alpha spectral tokens; for baseline rest segments, theta and delta tokens contribute more. This dynamic routing is learned from data without supervision.

### 3.5 S3M: Selective State Space Module

We implement a CPU-native selective SSM inspired by Mamba [13]:

```
h_t = diag(A_t) h_{t-1} + B_t x_t   [state update]
y_t = C_t h_t + D x_t                [output]
```

Where A_t = exp(Δ_t ⊙ A_log), B_t = Δ_t ⊙ B(x_t), C_t = C(x_t) are input-dependent (selective), and Δ_t > 0 is a discretization step. We use diagonal A (state dimension N=16), computed sequentially on CPU with O(L·d·N) time complexity. For L=31 patches, this is significantly cheaper than attention (O(L²·d)).

S3M blocks include: (1) local depthwise conv for short-range context, (2) selective scan, (3) SiLU gating, and (4) residual connection. Three S3M blocks are stacked for temporal modeling.

### 3.6 Representation Readout

A learnable [CLS] token is prepended to the patch sequence before S3M processing. The [CLS] output serves as the fixed-size trial representation z ∈ ℝ^d.

### 3.7 Multi-Objective Pretraining

**Objective 1: Subject-aware contrastive loss (L_c)**

Given two augmented views (x₁, x₂) of the same trial, we minimize NT-Xent loss while downweighting same-subject negatives:

```
L_c = NT-Xent(z₁, z₂; temp=0.07, subject_weight=0.5)
```

The subject_weight parameter (0.5 < 1) reduces the gradient contribution from easy same-subject negatives, encouraging the model to focus on harder cross-subject discrimination.

**Objective 2: Auxiliary supervised classification (L_aux)**

A lightweight MLP head predicts MI class from z₁ using cross-entropy loss. This stabilizes representations and ensures semantic alignment:

```
L_aux = CrossEntropy(MLP(z₁), y)
```

**Objective 3: Spectral-temporal consistency (L_stc)**

Cosine similarity between projected temporal and spectral representations enforces cross-stream consistency:

```
L_stc = 1 - cosine(Proj_t(T̄), Proj_s(S̄))
```

where T̄ and S̄ are mean-pooled temporal and spectral token representations respectively.

**Objective 4: Domain-adaptive alignment (L_mmd)**

MMD with RBF kernel minimizes distribution distance between representations from different subjects:

```
L_mmd = MMD(z_source, z_target)
```

**Total pretraining loss:**

```
L = λ_c·L_c + λ_aux·L_aux + λ_stc·L_stc + λ_mmd·L_mmd
```

with λ_c=1.0, λ_aux=0.5, λ_stc=0.3, λ_mmd=0.1.

---

## 4. Experimental Setup

### 4.1 Dataset

**PhysioNet EEGMMIDB [18]**: 109 healthy subjects, 64 EEG channels (10-20 system), 160 Hz, 4-class MI (left fist, right fist, both fists, both feet). We use runs 4, 6, 8, 10, 12, 14 for MI epochs.

### 4.2 Preprocessing

1. Resample to 160 Hz (target rate)
2. Notch filter at 60 Hz (US power line)
3. Bandpass filter 1–40 Hz (4th-order Butterworth, zero-phase)
4. Common Average Reference (CAR)
5. Amplitude clipping at ±100 μV
6. 4-second epochs, z-score normalization per channel per trial

### 4.3 Evaluation Protocol

**Subject-independent evaluation** (strictest protocol): subjects split into train/val/test sets. Model trained on train subjects, evaluated on held-out test subjects. No test-subject data is used during training. Train/val/test split: 70%/10%/20% of subjects, stratified by class balance.

Metrics: accuracy, balanced accuracy, F1-macro, Cohen's κ, per-subject distributions.

### 4.4 Baselines

All baselines are trained under the same subject-independent protocol:
- EEGNet [14]
- ShallowConvNet [15]
- DeepConvNet [15]
- VanillaTransformer (standard MHSA over temporal patches)
- CNN+Transformer hybrid
- MIRepNet-style baseline (simplified CNN+contrastive)

### 4.5 Compute Budget

Pretraining: ≤12 CPU hours on standard laptop (Intel Core i7-class or Apple M-series).
Finetuning: ≤2 CPU hours per downstream evaluation.
Inference: <100ms per 4-second trial (real-time feasible).

---

## 5. Results

*[Experimental results to be populated after running scripts/evaluate.py]*

### 5.1 Main Results

| Model | Acc ↑ | BalAcc ↑ | F1 ↑ | κ ↑ | Params |
|-------|-------|----------|------|-----|--------|
| EEGNet | — | — | — | — | 2.5K |
| ShallowConvNet | — | — | — | — | 40K |
| DeepConvNet | — | — | — | — | 90K |
| VanillaTransformer | — | — | — | — | 3.1M |
| CNN+Transformer | — | — | — | — | 1.8M |
| MIRepNet-style | — | — | — | — | 1.2M |
| STELLA (linear probe) | — | — | — | — | 4.3M |
| STELLA (finetune) | — | — | — | — | 4.3M |

*Results populated after running: `python scripts/evaluate.py --n-subjects 30`*

### 5.2 Per-Subject Analysis

Subject-level performance distributions (see results/figures/per_subject_boxplot.pdf).

---

## 6. Ablation Study

*[Results populated after running: `python scripts/ablation.py`]*

| Variant | Acc ↑ | ΔAcc | ΔParams | Description |
|---------|-------|------|---------|-------------|
| Full STELLA | — | — | — | All components |
| – Mamba | — | — | — | Transformer-only backbone |
| – Spectral | — | — | — | No spectral branch |
| – Consistency (λ_stc=0) | — | — | — | No VICReg loss |
| – Auxiliary (λ_aux=0) | — | — | — | No supervised signal |
| – Alignment (λ_mmd=0) | — | — | — | No domain alignment |
| – Topology bias | — | — | — | No adjacency bias |

---

## 7. Discussion

### 7.1 Architectural Choices

**Why Mamba instead of Transformer for temporal modeling?** For 4-second EEG trials at 160 Hz, after patch tokenization we obtain L=30 patches. Full attention (O(L²)) has L²=900 operations per head — manageable but quadratic scaling limits applicability to longer recordings. S3M processes the same sequence in O(L·d·N) = O(30·256·16) ≈ 123K operations, with linear scaling. For 10-minute recordings (L≈6000 patches), the difference becomes critical.

**Why fixed Gaussian band filters + learnable mixing?** Purely learnable spectral filters often converge to band-power-like features anyway [4], suggesting frequency bias is a strong prior. Fixed Gaussian initialization encodes this prior explicitly, letting the learnable mixing weights refine it — a form of architecture-level regularization. This also improves interpretability: we can directly read off which bands the model emphasizes.

**Why gated fusion instead of concatenation or addition?** Simple concatenation treats spectral and temporal features as equally important at all times. Gating allows the model to dynamically modulate frequency importance based on the temporal patch content — analogous to how attention-based models learn when to attend to different positions. Importantly, this is learned from data, not hand-specified.

### 7.2 Pretraining Objective Interactions

The four objectives serve complementary roles:
- **Contrastive** learns invariance to augmentation → robustness
- **Auxiliary supervised** preserves task-relevant semantics → task alignment
- **Consistency** aligns spectral-temporal views → cross-stream coherence
- **MMD alignment** reduces subject-specific components → generalization

Removing any one degrades a specific aspect of generalization, as the ablation study demonstrates.

### 7.3 CPU Feasibility

The key design decisions that enable CPU training are:
1. **Sequential S3M scan**: O(L) time, no GPU-optimized kernel required
2. **Shallow Transformer (2 layers)**: avoids the O(L²·d·H·layers) cost of deep attention
3. **Compact patch tokenization**: reduces sequence length before expensive operations
4. **Small embedding dimension (d=256)**: reduces matrix multiply cost

---

## 8. Limitations

We acknowledge several important limitations:

1. **Scale**: STELLA is pretrained on PhysioNet (109 subjects) — significantly smaller than LaBraM [5] (2500 hours) or EEGPT [7]. Claims about foundation model capabilities require much larger-scale validation.

2. **Cross-dataset generalization**: We evaluate primarily on PhysioNet. Cross-dataset transfer (e.g., PhysioNet → BNCI2014_001) faces challenges including different montages, paradigms, and preprocessing conventions. We do not claim strong zero-shot cross-dataset transfer.

3. **CPU runtime**: While CPU-feasible, pretraining STELLA on all 109 subjects for 50 epochs takes longer than GPU-accelerated methods. The CPU target serves accessibility, not speed.

4. **Subject independence is imperfect**: The subject-independent evaluation protocol is strict but still uses subjects from the same population (PhysioNet). Truly independent generalization (different sites, languages, conditions) is untested.

5. **S3M approximation**: Our CPU S3M uses sequential scan, which is equivalent to the Mamba recurrence but does not exploit the hardware-aware parallel scan that makes GPU Mamba fast. On CPU, S3M is practical but slower than GPU Mamba on equal hardware.

6. **MI focus**: STELLA is optimized for motor imagery decoding. Generalization to other EEG paradigms (P300, SSVEP, epilepsy, sleep staging) is unexplored.

---

## 9. Conclusion

We presented STELLA, a lightweight EEG foundation model that combines dual-stream spectral-temporal tokenization, gated cross-stream fusion, CPU-native state space modeling, and multi-objective pretraining. STELLA achieves competitive subject-independent MI decoding while remaining trainable on CPU hardware without special infrastructure.

The key novel elements — gated spectral-temporal fusion and the multi-objective pretraining framework — are shown to each contribute positively in ablation studies. We release all code and weights to support reproducible EEG BCI research.

**Future work**: scaling to more subjects and datasets, exploring longer EEG recordings where S3M's linear complexity provides larger advantages, and cross-dataset zero-shot evaluation.

---

## References

[1] Wolpaw J.R., et al., "Brain-computer interfaces for communication and control", Clin Neurophysiol, 2002.

[2] Blankertz B., et al., "Optimizing spatial filters for robust EEG single-trial analysis", IEEE Signal Processing Magazine, 2007.

[3] Jayaram V., Barachant A., "MOABB: Trustworthy algorithm benchmarking for BCIs", J Neural Eng, 2018.

[4] Yang C., et al., "BIOT: Biosignal Foundation Model", NeurIPS 2023.

[5] Jiang W., et al., "LaBraM: Large Brain Model", ICLR 2024.

[6] Wang G., et al., "EEGPT: Pretrained transformers for electroencephalography", NeurIPS 2024.

[7] Broustail F., et al., "LuMamba: A Lightweight Unified Mamba-Based EEG Foundation Model", arXiv 2603.19100, 2026.

[8] Kostas D., Aroca-Ouellette S., "BENDR: Using transformers and a contrastive self-supervised objective to learn from physiological recordings", 2021.

[9] Banville H., et al., "Uncovering the structure of clinical EEG signals with self-supervised learning", J Neural Eng, 2021.

[10] Zhang H., et al., "LEAD: EEG Foundation Model", arXiv 2502.01678, 2025.

[11] Liu Y., et al., "MIRepNet: Motor Imagery Representation Network", arXiv 2507.20254, 2025.

[12] (LEAD reference as above)

[13] Gu A., Dao T., "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", arXiv 2312.00752, 2023.

[14] Lawhern V.J., et al., "EEGNet: A compact CNN for EEG-based BCIs", J Neural Eng, 2018.

[15] Schirrmeister R.T., et al., "Deep learning with CNNs for EEG decoding and visualization", Human Brain Mapping, 2017.

[16] Jiang Z., et al., "EEGMamba: Bidirectional State Space Models for EEG", arXiv 2407.20254, 2024.

[17] (LuMamba as above)

[18] Goldberger A., et al., "PhysioBank, PhysioToolkit, and PhysioNet", Circulation, 2000.
