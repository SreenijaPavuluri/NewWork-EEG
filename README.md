# NewWork-EEG: STELLA Foundation Model

**STELLA** — Spectro-Temporal EEG Learning with Latent Alignment

A novel lightweight EEG foundation model for CPU-feasible pretraining and subject-independent Motor Imagery decoding.

---

## What is STELLA?

STELLA is a novel EEG foundation model that combines:

1. **Dual-stream tokenization** — temporal patch tokens (waveform morphology) + spectral band tokens (oscillatory content)
2. **Gated spectral-temporal fusion** — cross-attention where spectral context gates temporal patch relevance
3. **CPU-native S3M** — pure-PyTorch Selective State Space Module for long-range temporal modeling (no CUDA required)
4. **Multi-objective pretraining** — subject-aware contrastive + auxiliary supervised + spectral-temporal consistency + domain alignment

**Key properties:**
- ~4.3M encoder parameters
- Pretraining < 12 CPU hours
- Subject-independent evaluation protocol
- 37/37 unit tests passing
- Full reproducibility: seeds, configs, datasets all documented

---

## Repository Structure

```
NewWork-EEG/
├── src/
│   ├── models/         # STELLA + baselines
│   ├── datasets/       # PhysioNet, MOABB loaders
│   ├── preprocessing/  # Bandpass, notch, CAR, normalize
│   ├── augmentations/  # EEG-safe augmentations
│   ├── losses/         # NT-Xent, VICReg, MMD, combined
│   ├── training/       # Pretrain/finetune loops
│   ├── evaluation/     # Metrics, per-subject analysis
│   ├── visualization/  # Publication figures
│   └── utils/          # Seed, logging, profiling
├── configs/            # YAML configs (model, dataset, training)
├── scripts/            # Runnable experiment scripts
├── notebooks/          # Jupyter experiment notebooks
├── tests/              # pytest unit tests (37 tests)
├── paper/              # Manuscript draft + literature review
└── results/            # Outputs (figures, tables, logs)
```

---

## Quick Start

### 1. Install dependencies

```bash
# Option A: pip
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate newwork-eeg
```

### 2. Verify installation

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest tests/ -v
# Expected: 37 passed
```

### 3. Download datasets

```bash
python scripts/download_data.py --dataset physionet --n-subjects 5
# Generates: results/dataset_report.md
```

### 4. Validate preprocessing

```bash
python scripts/validate_datasets.py --n-subjects 5
```

### 5. Sanity check (verify training loop)

```bash
python scripts/pretrain.py --sanity-check --n-subjects 5 --epochs 1
```

### 6. Full pretraining

```bash
# Quick (10 subjects, 10 epochs, ~30min CPU)
python scripts/pretrain.py --n-subjects 10 --epochs 10

# Full (20 subjects, 50 epochs, ~4h CPU)
python scripts/pretrain.py --n-subjects 20 --epochs 50
```

### 7. Evaluate all models

```bash
python scripts/evaluate.py --n-subjects 30
# Outputs: results/tables/evaluation_results.csv
```

### 8. Ablation study

```bash
python scripts/ablation.py --n-subjects 15 --epochs 15
# Outputs: results/tables/ablation_results.csv
```

### 9. Run everything

```bash
bash run_all.sh
```

---

## Model Architecture

```
Input EEG (B, C, T)
       |
  ┌────┴────────────────────┐
  │                         │
TemporalPatch          SpectralBand
Embed (depthwise conv)  Encoder (FFT + Gaussian band filters)
(B, P, D)               (B, 5, D)
  │                         │
  └─────┬───────────────────┘
        │ GatedSpectralFusion
        │ (cross-attention: spectral→ gates temporal)
   (B, P, D)
        │
  ChannelSpatialTransformer (2 layers)
        │
   [CLS] ++ fused patches
        │
  TemporalMamba / S3M (3 × S3MBlock)
        │
  CLS token → representation z (B, D)
```

### Key design decisions

| Decision | Why |
|----------|-----|
| Shallow Transformer (2L) instead of deep | Channels form small set; 2L captures inter-channel structure without quadratic cost |
| Mamba/S3M instead of full attention for temporal | O(L) vs O(L²) scaling; linear in sequence length |
| Gaussian-initialized spectral filters | Encodes EEG frequency prior explicitly; interpretable; learnable mixing refines |
| Gated fusion (cross-attention) | Allows dynamic frequency conditioning vs. simple concatenation |
| Subject-aware NT-Xent | Reduces easy same-subject negatives; harder cross-subject discrimination |

---

## Pretraining Objectives

| # | Objective | Loss | Weight |
|---|-----------|------|--------|
| 1 | Subject-aware contrastive | NT-Xent (τ=0.07) | λ=1.0 |
| 2 | Auxiliary supervised classification | Cross-entropy | λ=0.5 |
| 3 | Spectral-temporal consistency | Cosine (+ VICReg) | λ=0.3 |
| 4 | Domain-adaptive alignment | MMD (RBF kernel) | λ=0.1 |

---

## Baselines

| Model | Type | Reference |
|-------|------|-----------|
| EEGNet | Supervised CNN | Lawhern 2018 |
| ShallowConvNet | Supervised CNN | Schirrmeister 2017 |
| DeepConvNet | Supervised CNN | Schirrmeister 2017 |
| VanillaTransformer | Self-attn patches | — |
| CNN+Transformer | Hybrid | — |
| MIRepNet-style | CNN + contrastive | Liu 2025 (simplified) |

---

## Compute Requirements

| Task | Time | Memory |
|------|------|--------|
| Unit tests | <30s | <500MB |
| Sanity check | <2min | <1GB |
| Quick pretraining (10 subj, 10ep) | ~30min CPU | ~2GB |
| Full pretraining (20 subj, 50ep) | ~4-8h CPU | ~3GB |
| Evaluation (30 subjects) | ~1h CPU | ~2GB |
| Ablation study | ~1h CPU | ~2GB |

No GPU required. Tested on Apple Silicon (M-series) and Intel Core i7.

---

## Results

Run `python scripts/evaluate.py` to generate results.
Tables: `results/tables/evaluation_results.csv`
Figures: `results/figures/`
Logs: `results/logs/`

---

## Reproducibility

- All seeds set via `src/utils/seed.py` (default: 42)
- Fixed train/val/test subject splits stored in `SubjectSplit` 
- All configs version-controlled in `configs/`
- Requirements pinned in `requirements.txt`

---

## Citation

If you use this code, please cite:

```
@article{stella2026,
  title={STELLA: Spectro-Temporal EEG Learning with Latent Alignment},
  author={[Authors]},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

MIT License. Dataset licenses:
- PhysioNet EEGMMIDB: Public domain
- MOABB/BNCI2014_001: CC-BY 4.0
