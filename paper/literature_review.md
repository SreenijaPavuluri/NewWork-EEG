# Literature Review: EEG Foundation Models and Related Work

*Prepared for STELLA paper, May 2026*

---

## 1. Background and Scope

This review surveys the landscape of EEG deep learning from foundational CNN baselines (2017–2018) through the current generation of large-scale pretrained foundation models (2021–2026). We focus on architectures, pretraining strategies, transfer learning approaches, and known limitations that motivate the STELLA design.

---

## 2. Classical CNN Baselines

### 2.1 ShallowConvNet and DeepConvNet (Schirrmeister et al., 2017)

**Full citation:** Schirrmeister, R.T., Springenberg, J.T., Fiederer, L.D.J., Glasstetter, M., Eggensperger, K., Schalk, G., Hutter, F., Burgard, W., & Ball, T. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391–5420. arXiv:1703.05051.

**Architecture type:** Pure CNN.

- **ShallowConvNet:** 2 convolutional layers (temporal + spatial filter) + average-pooling + log-activation + fully-connected classifier. Designed to mimic filter-bank CSP (FBCSP) in a single learnable pipeline.
- **DeepConvNet:** 5 convolutional layers + 1 fully-connected layer; deeper feature hierarchy for both temporal and spatial structure.

**Datasets used:** BCI Competition IV Dataset 2a (4-class MI), BCI Competition IV Dataset 2b (2-class MI), and an in-house clinical ECoG/EEG dataset.

**Key innovation:** First systematic demonstration that CNNs can match or exceed FBCSP on motor imagery; introduced cropped training (sliding-window augmentation) and correlation-based visualization of learned features.

**Pretraining strategy:** None — fully supervised, within-subject.

**Transfer learning approach:** Not addressed; each subject trained independently.

**Parameter count:** ShallowConvNet ~$<$5K parameters (highly compact); DeepConvNet ~24K parameters.

**Known limitations:**
- No cross-subject or cross-dataset generalization strategy.
- ShallowConvNet encodes domain knowledge that limits generality (hand-crafted frequency assumption).
- No temporal long-range dependency modeling.
- Performance degrades severely with limited labeled data.

---

### 2.2 EEGNet (Lawhern et al., 2018)

**Full citation:** Lawhern, V.J., Solon, A.J., Waytowich, N.R., Gordon, S.M., Hung, C.P., & Lance, B.J. (2018). EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering*, 15(5), 056013. DOI: 10.1088/1741-2552/aace8c. arXiv:1611.08024.

**Architecture type:** Compact CNN with depthwise and separable convolutions.

Core blocks: (1) temporal convolution → (2) depthwise spatial convolution (one filter per channel) → (3) separable pointwise convolution → (4) dense classifier. BatchNorm and dropout throughout.

**Datasets used:** P300 visual-evoked potentials, Error-Related Negativity (ERN), Movement-Related Cortical Potentials (MRCP), Sensory Motor Rhythms (SMR) — four distinct BCI paradigms.

**Key innovation:** First EEG-specific use of depthwise + separable convolutions to dramatically reduce parameters while encoding known EEG signal structure (temporal filtering → spatial filtering → feature mixing). Demonstrated paradigm-general applicability with a single architecture.

**Pretraining strategy:** None — fully supervised.

**Transfer learning approach:** The compact parameter budget implicitly supports transfer; cross-subject and cross-paradigm evaluations demonstrate better generalization than heavier models in low-data regimes.

**Parameter count:** ~2,548 parameters in the standard EEGNet-8,2 configuration (F1=8 temporal filters, D=2 spatial filters).

**Known limitations:**
- No explicit spectral modeling (relies on temporal filter length as implicit bandpass).
- No mechanism for variable channel counts or electrode montage adaptation.
- No self-supervised or unsupervised pretraining; relies entirely on labeled data.
- Limited temporal receptive field; poor at modeling long-range EEG dependencies.
- Performance saturates quickly when scaled.

---

## 3. Self-Supervised and Contrastive EEG Pretraining

### 3.1 BENDR (Kostas & Aroca-Ouellette, 2021)

**Full citation:** Kostas, D., Aroca-Ouellette, S., & Bhatt, U. (2021). BENDR: Using transformers and a contrastive self-supervised learning task to learn from massive amounts of EEG data. *Frontiers in Human Neuroscience*, 15, 653659. arXiv:2101.12037.

**Architecture type:** 1D convolutional encoder + Transformer, inspired by wav2vec 2.0.

**Datasets used:** Temple University Hospital EEG Corpus (TUH EEG) — over 14,000 recordings; downstream evaluation on multiple BCI datasets.

**Key innovation:** First large-scale contrastive self-supervised pretraining for raw EEG. Adapted the audio representation learning paradigm (wav2vec) to EEG: encode raw signal windows with a temporal CNN, then apply a contrastive loss between the Transformer contextual representations and quantized future segment representations.

**Pretraining strategy:** Contrastive predictive coding (CPC-style); positive samples are temporally adjacent segments; negatives are randomly sampled distractors from within the same batch.

**Transfer learning approach:** Pretrained encoder weights are fine-tuned end-to-end on downstream supervised tasks.

**Parameter count:** Not explicitly reported; estimated ~5–10M parameters for the full Transformer backbone.

**Known limitations:**
- No explicit spectral/frequency-domain pretraining objective.
- Contrastive loss does not enforce subject-level invariance; high inter-subject variability limits zero-shot cross-subject transfer.
- Single-stream temporal tokenization; no dual-stream spectral-temporal representation.
- Channel count must match pretraining configuration; no montage adaptation.
- Scaling requires large GPU clusters; not CPU-feasible at inference.

---

### 3.2 BrainBERT (Wang et al., 2023)

**Full citation:** Wang, C., Subramaniam, V., Yaari, A.U., Kreiman, G., Katz, B., Cases, I., & Barbu, A. (2023). BrainBERT: Self-supervised representation learning for intracranial recordings. *ICLR 2023*. arXiv:2302.14367.

**Architecture type:** Transformer encoder operating on time-frequency spectrograms (analogous to BERT for masked spectrogram prediction).

**Datasets used:** Intracranial EEG (iEEG/sEEG) recordings from neurosurgical patients; multiple hospital datasets.

**Key innovation:** Applied BERT-style masked spectrogram autoencoding to intracranial neural recordings. Input is a super-resolution Short-Time Fourier Transform (STFT) spectrogram; random spectrogram patches are masked and the model learns to reconstruct the missing frequency-time bins.

**Pretraining strategy:** Masked spectrogram modeling (analogous to masked language modeling in BERT); operates in the frequency-time domain.

**Transfer learning approach:** Fine-tune the pretrained Transformer encoder with a task-specific head (e.g., speech decoding, concept classification).

**Parameter count:** Not explicitly reported; standard BERT-Base scale (~86M) in reported configurations.

**Known limitations:**
- Designed for intracranial (invasive) recordings; direct applicability to scalp EEG is limited due to signal characteristics and electrode geometry differences.
- Requires high-resolution spectrograms; computationally expensive.
- No subject-aware contrastive objective; representations may conflate subject-specific artifacts with neural content.
- Does not address variable montage or channel count adaptation.

---

## 4. Convolutional Transformer Hybrids

### 4.1 EEG-Conformer (Song et al., 2023)

**Full citation:** Song, Y., Zheng, Q., Liu, B., & Gao, X. (2023). EEG Conformer: Convolutional Transformer for EEG decoding and visualization. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 31, 710–719. DOI: 10.1109/TNSRE.2022.3230250.

**Architecture type:** Hybrid CNN + Transformer (Conformer-style).

Three sequential modules: (1) CNN module — 1D temporal convolution + spatial depthwise convolution capturing local temporal-spatial features; (2) self-attention module — multi-head attention over local temporal windows to model global temporal dependencies; (3) classification head — fully connected layers.

**Datasets used:** BCI Competition IV Dataset 2a (4-class MI), BCI Competition IV Dataset 2b (2-class MI), SEED emotion dataset.

**Key innovation:** Unified framework that combines local CNN feature extraction (addressing limitations of pure Transformers on short EEG sequences) with global self-attention. Also introduced Class Activation Topography (CAT) for spatial visualization of class-discriminative features on the scalp.

**Pretraining strategy:** None — fully supervised.

**Transfer learning approach:** Not addressed; subject-specific training.

**Parameter count:** Approximately 800K–2M parameters depending on configuration.

**Known limitations:**
- No pretraining; requires fully labeled data per subject.
- Single-stream architecture; no explicit spectral tokenization.
- Self-attention complexity is O(n²) in sequence length — computationally heavy for long EEG windows.
- No cross-subject generalization mechanism.
- No channel montage adaptation.

---

## 5. Large-Scale EEG Foundation Models

### 5.1 BIOT (Yang et al., NeurIPS 2023)

**Full citation:** Yang, C., Westover, M.B., & Sun, J. (2023). BIOT: Biosignal Transformer for cross-data learning in the wild. *Advances in Neural Information Processing Systems (NeurIPS 2023)*. arXiv:2305.10351.

**Architecture type:** Transformer with PatchFrequency Embedding and LinearAttention.

**Datasets used:** TUH Abnormal EEG Corpus (TUAB, ~400K samples), Sleep Heart Health Study (SHHS, ~5M samples); downstream evaluation on EEG, ECG, and human activity recognition datasets.

**Key innovation:** Unified cross-modal biosignal learning. Tokenization pipeline: (1) resample all biosignals to a common frequency (200 Hz); (2) compute short-time FFT patches to produce frequency-domain tokens; (3) encode as "sentences" allowing variable channel counts and missing values. A single model handles EEG, ECG, and accelerometer data with mismatched channels.

**Pretraining strategy:** Contrastive learning on paired or augmented biosignal windows; cross-dataset pretraining.

**Transfer learning approach:** Pretrained encoder + task-specific fine-tuning head.

**Parameter count:** ~3.3M parameters — the smallest among reviewed foundation models.

**Known limitations:**
- Frequency tokenization is single-scale; no multi-resolution temporal-spectral decomposition.
- Contrastive objective does not explicitly model subject identity; high inter-subject EEG variability is unaddressed.
- Linear attention approximation may lose global context in complex tasks.
- Evaluated primarily on clinical (pathological) EEG tasks, not motor imagery or cognitive BCIs.
- 3.3M parameters can limit representational capacity for fine-grained cognitive decoding.

---

### 5.2 LaBraM (Jiang et al., ICLR 2024 Spotlight)

**Full citation:** Jiang, W., Zhao, L., Lu, B., et al. (2024). Large Brain Model for learning generic representations with tremendous EEG data in BCI. *International Conference on Learning Representations (ICLR 2024) — Spotlight*. arXiv:2405.18765.

**Architecture type:** Vector-Quantized Transformer with neural tokenizer (VQ-VAE inspired).

Two-stage pipeline: (1) Neural tokenizer — temporal CNN encodes EEG channel patches into continuous embeddings, then a VQ-VAE codebook discretizes them into neural codes; (2) Neural Transformer — BERT-style masked token prediction where the model predicts the original discrete neural codes for masked EEG patches.

Base configuration: 12 Transformer layers, hidden dimension 800, 10 attention heads, MLP size 800.

**Datasets used:** Pretraining on ~2,500 hours of EEG data from ~20 heterogeneous datasets (motor imagery, emotion, sleep, clinical abnormality). Downstream evaluation on abnormality detection, event classification, emotion recognition, gait prediction.

**Key innovation:** First foundation model to use vector quantization for EEG. The VQ tokenizer produces semantically rich discrete neural codes; the downstream masked prediction task forces the Transformer to learn meaningful EEG structure rather than low-level reconstruction. Supports cross-dataset EEG by treating each channel independently as a patch.

**Pretraining strategy:** Masked neural code prediction (discrete BERT-style pretraining using VQ-coded targets). The tokenizer is pretrained separately before Transformer pretraining.

**Transfer learning approach:** Pretrained LaBraM encoder + task-specific fine-tuning head; demonstrated strong cross-dataset transfer.

**Parameter count:** LaBraM-Base: ~5M encoder parameters + VQ-VAE tokenizer; Large and Huge variants not fully detailed in public papers.

**Known limitations:**
- Two-stage training is complex; VQ codebook requires separate pretraining with risk of codebook collapse.
- No subject-aware contrastive objective; inter-subject variability in codebook space is uncontrolled.
- No dual-stream spectral-temporal tokenization; single temporal-patch view.
- Montage adaptation is channel-independent (each channel treated as separate patch), losing cross-channel spatial structure during pretraining.
- Large-scale pretraining requires GPU clusters (~20 datasets, 2,500 hours); not designed for CPU-feasible inference at scale.

---

### 5.3 LEAD (Zhang et al., 2025)

**Full citation:** Zhang, Y., et al. (2025). LEAD: Large Foundation Model for EEG-Based Alzheimer's Disease Detection. *arXiv:2502.01678* (also appearing at ICLR 2025 Workshop). Accepted at OpenReview.

**Architecture type:** Transformer encoder with temporal and channel embeddings for dual-dimensional feature extraction.

**Datasets used:** LEAD-AD corpus — curated dataset of 813 subjects (claimed largest EEG-AD dataset); pretraining also uses EEG from healthy subjects and other neurological conditions to increase diversity.

**Key innovation:** First large-scale EEG foundation model specifically for Alzheimer's Disease detection. Pretraining uses a dual contrastive strategy: (1) sample-level contrastive learning (augmented views of the same EEG segment are positive pairs) and (2) subject-level contrastive learning (windows from the same subject are positive pairs regardless of condition). This explicitly models inter-subject variability during pretraining.

**Pretraining strategy:** Dual-level contrastive learning — sample-level and subject-level. Uses healthy + neurological disease EEG data as diverse pretraining corpus.

**Transfer learning approach:** Pretrained LEAD encoder fine-tuned for AD classification; demonstrated up to 9.86% F1 improvement over prior SOTA at the sample level, and 9.31% at the subject level.

**Parameter count:** Not explicitly reported in available preprint.

**Known limitations:**
- Domain-specific: designed and evaluated exclusively for Alzheimer's Disease detection; general BCI applicability is unclear.
- Subject-level contrastive learning is introduced but operates only as a binary same/different-subject signal, not a richer subject-adaptive embedding.
- No spectral tokenization; temporal-channel embeddings dominate.
- No montage adaptation strategy documented.
- Evaluated on a proprietary/curated dataset limiting reproducibility.

---

### 5.4 EEGPT (Wang et al., NeurIPS 2024)

**Full citation:** Wang, X., Liu, X., et al. (2024). EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals. *Advances in Neural Information Processing Systems (NeurIPS 2024)*. OpenReview: lvS2b8CjG5.

**Architecture type:** Hierarchical Transformer with dual self-supervised learning (mask-based spatial + temporal alignment).

**Datasets used:** Multiple public EEG datasets spanning motor imagery, emotion recognition, sleep staging, and clinical EEG. Downstream evaluation via linear probing.

**Key innovation:** Mask-based dual self-supervised learning that constructs pretraining targets from high-SNR EEG representations rather than from raw signals. Spatio-temporal representation alignment enforces consistency between spatial and temporal views of masked segments. Hierarchical structure processes spatial and temporal information in separate pathways.

**Pretraining strategy:** Dual masked reconstruction: (1) spatial masking + temporal masking; (2) alignment loss between spatial and temporal representations.

**Transfer learning approach:** Linear probing on frozen pretrained representations; demonstrated state-of-the-art performance across several BCI tasks.

**Parameter count:** ~10M parameters.

**Known limitations:**
- Hierarchical separation of spatial and temporal processing may lose joint spatio-temporal interactions.
- No explicit frequency-domain tokenization; operates purely in time domain.
- No subject-aware contrastive term; inter-subject differences treated as noise rather than a learnable factor.
- O(n²) attention complexity in the Transformer blocks; large-sequence inference requires GPU.
- No montage-adaptive mechanism for variable electrode configurations.

---

### 5.5 MIRepNet (Liu et al., 2025)

**Full citation:** Liu, Y., et al. (2025). MIRepNet: A Pipeline and Foundation Model for EEG-Based Motor Imagery Classification. *arXiv:2507.20254* (July 2025).

**Architecture type:** Transformer encoder-decoder with masked token reconstruction.

A neurophysiologically informed channel template first unifies heterogeneous MI EEG headsets. Temporal-spatial tokenization converts aligned EEG trials into patches. A proportion of tokens are masked; the decoder reconstructs masked tokens while a parallel classification head performs supervised MI decoding.

**Datasets used:** Five public motor imagery EEG datasets (including BCI Competition IV 2a/2b, Physionet MI, and others) encompassing 47 downstream subjects.

**Key innovation:** First foundation model specifically designed for the Motor Imagery paradigm. Key contribution is the neurophysiologically-informed channel template that aligns arbitrary electrode configurations to a canonical layout before tokenization, enabling cross-headset generalization. Hybrid pretraining (masked reconstruction + supervised classification) allows rapid adaptation with fewer than 30 trials per class.

**Pretraining strategy:** Hybrid — simultaneous masked token reconstruction (self-supervised) + MI classification (supervised). Combined loss enables the model to learn both general EEG structure and MI-specific features.

**Transfer learning approach:** Full fine-tuning with very few labeled samples; converges in few epochs.

**Parameter count:** Not explicitly reported in available arXiv version.

**Known limitations:**
- MI-paradigm specific; generalization to non-MI tasks (emotion, sleep, clinical EEG) not demonstrated.
- Channel template is hand-crafted from neurophysiology; may not generalize to exotic montages or ultra-high-density EEG.
- No explicit spectral tokenization; representation learning is temporal-spatial only.
- No subject-aware contrastive component; subject variability is absorbed implicitly by the masked reconstruction objective.
- Hybrid supervised + self-supervised loss can introduce task-specific bias during pretraining.

---

## 6. Mamba and State Space Models for Biosignals

### 6.1 Mamba (Gu & Dao, 2023)

**Full citation:** Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv:2312.00752*.

**Architecture type:** Selective State Space Model (SSM) — Structured State Space (S4) with input-dependent (selective) transition matrices.

**Key innovation:** Addresses the O(n²) complexity of Transformers by parameterizing state transitions as functions of the input (selective SSM), enabling the model to dynamically propagate or suppress information. Achieves linear O(n) complexity in sequence length with hardware-aware parallel scan implementations. Mamba-3B outperforms Transformers of the same size on language benchmarks.

**Datasets used:** Language (The Pile), audio, genomics benchmarks.

**Key properties for EEG:**
- 5× faster throughput than Transformers at equal model size.
- Linear memory scaling — enables long EEG sequences (minutes, not seconds) on CPU.
- Selective gating naturally models the non-stationary, event-driven structure of EEG.

**Known limitations for EEG:**
- Uni-directional by default; EEG benefits from bidirectional context (addressed by bidirectional Mamba variants).
- No explicit multi-head attention for capturing cross-channel spatial correlations at global scale.
- Selective SSM scan kernels optimized for GPU; pure CPU inference may not achieve full theoretical speedup.

---

### 6.2 EEGMamba (Jiang et al., 2024)

**Full citation:** Jiang, H., et al. (2024). EEGMamba: Bidirectional State Space Models with Mixture of Experts for EEG Multi-task Classification. *arXiv:2407.20254*.

**Architecture type:** Bidirectional Mamba + Mixture of Experts (MoE).

**Datasets used:** Eight public EEG datasets covering seizure detection, emotion recognition, sleep stage classification, and motor imagery.

**Key innovation:** Extended Mamba to bidirectional processing for EEG (forward + backward SSM scan). Mixture of Experts allows dynamic routing of different EEG patterns to specialized sub-networks, handling multi-task heterogeneity.

**Pretraining strategy:** Supervised multi-task training; no large-scale self-supervised pretraining.

**Known limitations:** No pretraining; MoE routing adds inference complexity; no subject-aware contrastive or spectral objectives.

---

### 6.3 LuMamba (Broustail et al., 2026)

**Full citation:** Broustail, D., Tegon, A., Ingolfsson, T.M., Li, Y., & Benini, L. (2026). LuMamba: Latent Unified Mamba for Electrode Topology-Invariant and Efficient EEG Modeling. *arXiv:2603.19100* (March 2026). ETH Zurich / University of Bologna.

**Architecture type:** LUNA cross-attention (topology-invariant channel unification) + Bidirectional Mamba temporal modeling + LeJEPA self-supervised objective.

**Datasets used:** TUEG (21,000+ hours unlabeled pretraining), TUAB (abnormality), TUH Epilepsy Corpus, Alzheimer's EEG datasets. Electrode configurations from 16 to 26 channels.

**Key innovation:** Combines topology-invariant encodings (LUNA learned-query cross-attention for channel unification) with linear-complexity Mamba temporal modeling. First systematic application of Latent-Euclidean Joint-Embedding Predictive Architecture (LeJEPA) to EEG. Achieves 377× fewer FLOPs than SOTA at equivalent sequence lengths.

**Pretraining strategy:** LeJEPA — joint-embedding predictive architecture in latent Euclidean space. Pretraining on 21,000+ hours of unlabeled EEG.

**Parameter count:** 4.6M parameters.

**Known limitations:** No spectral tokenization (temporal-only); no subject-aware contrastive component; currently limited to 16–26 channel configurations; evaluated only on clinical (pathological) EEG tasks.

---

## 7. Subject-Aware and Contrastive EEG Methods

### 7.1 Subject-Aware Contrastive Brainwave Foundation Model (OpenReview 2024)

**Full citation:** Anonymous (2024). Subject-Aware Contrastive Brainwave Foundation Model. *OpenReview submission* (MdgBATPjEu), 2024.

**Architecture type:** Transformer encoder with temporal convolution patches + subject-level contrastive head.

**Key innovation:** Explicit subject-level contrastive pretraining — windows from the same subject (regardless of mental state) are positive pairs; windows from different subjects are negatives. This explicitly disentangles subject-specific signal patterns from task-related neural activity.

**Pretraining strategy:** Subject-aware contrastive learning (InfoNCE-style) over a mixed-dataset EEG corpus.

**Known limitations:** Subject-aware contrastive operates post-hoc without integrating spectral information; no dual-stream tokenization; no montage adaptation.

---

## 8. Gap Analysis

### 8.1 Summary Matrix

| Model | Year | Spectral Token. | Subject-Aware Contrast. | Dual-Stream | Mamba/SSM | Montage Adapt. | CPU-Feasible | Pretraining Scale |
|---|---|---|---|---|---|---|---|---|
| ShallowConvNet | 2017 | Implicit | No | No | No | No | Yes | None |
| DeepConvNet | 2017 | No | No | No | No | No | Yes | None |
| EEGNet | 2018 | No | No | No | No | No | Yes | None |
| BENDR | 2021 | No | No | No | No | No | No | Contrastive |
| BrainBERT | 2023 | Yes (STFT) | No | Partial | No | No | No | Masked Spectrogram |
| EEG-Conformer | 2023 | No | No | No | No | No | Partial | None |
| BIOT | 2023 | Partial (FFT patch) | No | No | No | Partial | Partial | Contrastive |
| LaBraM | 2024 | Partial (VQ-Neural) | No | No | No | Channel-indep. | No | Masked Code Pred. |
| LEAD | 2025 | No | Yes (2-level) | No | No | No | Unknown | Subject+Sample Contrast. |
| EEGPT | 2024 | No | No | Spatial+Temporal | No | No | No | Masked Recon. |
| MIRepNet | 2025 | No | No | No | No | Template-based | Unknown | Hybrid SSL+SL |
| EEGMamba | 2024 | No | No | No | Yes (Bidirec.) | No | Partial | None (supervised) |
| LuMamba | 2026 | No | No | No | Yes (Bidirec.) | LUNA (topology-inv.) | Partial (4.6M) | LeJEPA |

### 8.2 Identified Gaps

**Gap 1: No model combines a CPU-feasible hybrid Transformer+Mamba backbone.**

Current Transformer-based foundation models (LaBraM, EEGPT, MIRepNet, LEAD) require GPU-resident attention with O(n²) complexity for training and inference. Mamba-only models (EEGMamba, LuMamba) achieve linear complexity but lose the global multi-head attention that is critical for capturing spatial electrode correlations at key temporal scales. No existing work has integrated a *hybrid* architecture — Transformer attention for cross-channel spatial pooling + Mamba SSM for long-range temporal sequence modeling — while designing the system to remain feasible for CPU inference (e.g., via sub-10M parameter budget with attention only at bottleneck positions).

**Gap 2: No model implements genuine dual-stream spectral-temporal tokenization as a first-class representation.**

BrainBERT operates on STFT spectrograms, but for invasive (intracranial) signals and does not combine with a temporal stream. BIOT uses frequency patches as its *only* token type. LaBraM uses temporal patches with VQ coding but discards explicit frequency-band decomposition after the tokenizer stage. No model simultaneously processes raw temporal waveform tokens *and* band-power spectral tokens (e.g., delta/theta/alpha/beta/gamma band envelopes) in a parallel dual-stream encoder that fuses at a later integration layer. This dual-stream design would capture both transient event morphology (temporal stream) and oscillatory rhythms (spectral stream), which are the two complementary signatures of EEG cognition.

**Gap 3: No model combines subject-aware contrastive loss with spectral-temporal consistency in a single multi-objective pretraining framework.**

LEAD introduces subject-aware contrastive learning but without any spectral objective. BENDR uses contrastive CPC but has no spectral or subject-aware terms. EEGPT uses spatial-temporal representation alignment as a consistency objective but no subject or spectral terms. BrainBERT masks spectrograms but uses no contrastive objective at all. A multi-objective pretraining framework combining: (a) subject-aware InfoNCE for cross-subject invariance, (b) spectral consistency loss penalizing representations that do not preserve band-power ratios across augmented views, and (c) masked temporal reconstruction — has not been explored in any single unified pretraining regime for scalp EEG.

**Gap 4: No model provides full montage-adaptive channel modeling while retaining within-montage spatial structure.**

LuMamba uses LUNA's learned-query cross-attention for topology invariance but operates on a fixed small set of 16–26 channels and has been evaluated only on clinical EEG. MIRepNet uses a hand-crafted neurophysiological channel template for MI specifically. LaBraM treats channels independently (losing pairwise spatial relations). Approaches that handle genuinely arbitrary montages (e.g., 8-channel wearable to 256-channel research cap) while *retaining* learnable cross-channel spatial attention within the registered montage — through a dynamic graph or channel positional embedding that encodes 3D scalp coordinates — remain absent from the literature.

**Gap 5: The four capabilities have never been unified in a single framework.**

The critical research gap is that properties (a) CPU-feasible hybrid Transformer+Mamba, (b) dual-stream spectral-temporal tokenization, (c) multi-objective pretraining with subject-aware contrastive + spectral-temporal consistency, and (d) montage-adaptive channel modeling have each been partially addressed in isolation. No published model addresses all four simultaneously. This represents the primary contribution space for STELLA.

---

## 9. References

1. Schirrmeister, R.T., et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391–5420. https://arxiv.org/abs/1703.05051

2. Lawhern, V.J., et al. (2018). EEGNet: A compact convolutional neural network for EEG-based BCIs. *Journal of Neural Engineering*, 15(5), 056013. https://arxiv.org/abs/1611.08024

3. Kostas, D., Aroca-Ouellette, S., & Bhatt, U. (2021). BENDR: Using transformers and a contrastive self-supervised learning task to learn from massive amounts of EEG data. *Frontiers in Human Neuroscience*. https://arxiv.org/abs/2101.12037

4. Wang, C., et al. (2023). BrainBERT: Self-supervised representation learning for intracranial recordings. *ICLR 2023*. https://arxiv.org/abs/2302.14367

5. Song, Y., Zheng, Q., Liu, B., & Gao, X. (2023). EEG Conformer: Convolutional Transformer for EEG decoding and visualization. *IEEE TNSRE*, 31, 710–719. https://ieeexplore.ieee.org/document/9991178

6. Yang, C., Westover, M.B., & Sun, J. (2023). BIOT: Biosignal Transformer for cross-data learning in the wild. *NeurIPS 2023*. https://arxiv.org/abs/2305.10351

7. Jiang, W., et al. (2024). Large Brain Model for learning generic representations with tremendous EEG data in BCI. *ICLR 2024 Spotlight*. https://arxiv.org/abs/2405.18765

8. Zhang, Y., et al. (2025). LEAD: Large Foundation Model for EEG-Based Alzheimer's Disease Detection. *arXiv:2502.01678*. https://arxiv.org/abs/2502.01678

9. Wang, X., et al. (2024). EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals. *NeurIPS 2024*. https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html

10. Liu, Y., et al. (2025). MIRepNet: A Pipeline and Foundation Model for EEG-Based Motor Imagery Classification. *arXiv:2507.20254*. https://arxiv.org/abs/2507.20254

11. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv:2312.00752*. https://arxiv.org/abs/2312.00752

12. Jiang, H., et al. (2024). EEGMamba: Bidirectional State Space Models with Mixture of Experts for EEG Classification. *arXiv:2407.20254*. https://arxiv.org/abs/2407.20254

13. Broustail, D., et al. (2026). LuMamba: Latent Unified Mamba for Electrode Topology-Invariant and Efficient EEG Modeling. *arXiv:2603.19100*. https://arxiv.org/abs/2603.19100

14. Anonymous (2024). Subject-Aware Contrastive Brainwave Foundation Model. *OpenReview MdgBATPjEu*. https://openreview.net/forum?id=MdgBATPjEu

15. Wang, X., et al. (2024). EEGFormer: Towards Transferable and Interpretable Large-Scale EEG Foundation Model. *arXiv:2401.10278*. https://arxiv.org/abs/2401.10278
