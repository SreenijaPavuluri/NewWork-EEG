#!/bin/bash
# NewWork-EEG / STELLA — Full experiment pipeline
# Usage: bash run_all.sh [--quick]
# --quick: uses fewer subjects/epochs for fast validation

set -e   # exit on any error

PYTHON=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3

echo "============================================"
echo "NewWork-EEG: STELLA Full Experiment Pipeline"
echo "============================================"
echo ""

# Parse arguments
QUICK=false
if [[ "$1" == "--quick" ]]; then
    QUICK=true
    N_SUBJECTS=10
    PRETRAIN_EPOCHS=5
    FINETUNE_SUBJECTS=15
    echo "[QUICK MODE] Reduced subjects and epochs for fast validation"
else
    N_SUBJECTS=20
    PRETRAIN_EPOCHS=50
    FINETUNE_SUBJECTS=30
fi

echo ""
echo "Config:"
echo "  Pretraining subjects: $N_SUBJECTS"
echo "  Pretraining epochs:   $PRETRAIN_EPOCHS"
echo "  Evaluation subjects:  $FINETUNE_SUBJECTS"
echo ""

# ---- Step 1: Download datasets ----
echo "=== Step 1: Download & Verify Datasets ==="
$PYTHON scripts/download_data.py --dataset physionet --n-subjects 5
echo "Dataset download OK"
echo ""

# ---- Step 2: Validate preprocessing ----
echo "=== Step 2: Validate Preprocessing ==="
$PYTHON scripts/validate_datasets.py --n-subjects 5
echo "Preprocessing validation OK"
echo ""

# ---- Step 3: Run unit tests ----
echo "=== Step 3: Unit Tests ==="
$PYTHON -m pytest tests/ -v --tb=short
echo "All tests passed"
echo ""

# ---- Step 4: Sanity check ----
echo "=== Step 4: Sanity Check (overfit 1 batch) ==="
$PYTHON scripts/pretrain.py --sanity-check --n-subjects 5 --epochs 1
echo "Sanity check OK"
echo ""

# ---- Step 5: Pretrain STELLA ----
echo "=== Step 5: STELLA Pretraining ($N_SUBJECTS subjects, $PRETRAIN_EPOCHS epochs) ==="
$PYTHON scripts/pretrain.py \
    --n-subjects $N_SUBJECTS \
    --epochs $PRETRAIN_EPOCHS \
    --batch-size 32
echo "Pretraining complete"
echo ""

# ---- Step 6: Evaluate all models ----
echo "=== Step 6: Evaluate (subject-independent) ==="
CHECKPOINT=""
if [ -f "results/checkpoints/pretrain/stella_pretrain_best.pt" ]; then
    CHECKPOINT="results/checkpoints/pretrain/stella_pretrain_best.pt"
fi
$PYTHON scripts/evaluate.py \
    --n-subjects $FINETUNE_SUBJECTS \
    ${CHECKPOINT:+--checkpoint $CHECKPOINT}
echo "Evaluation complete"
echo ""

# ---- Step 7: Ablation study ----
echo "=== Step 7: Ablation Study ==="
$PYTHON scripts/ablation.py \
    --n-subjects $((N_SUBJECTS < 15 ? N_SUBJECTS : 15)) \
    --epochs 15
echo "Ablation complete"
echo ""

# ---- Summary ----
echo "============================================"
echo "COMPLETE — Results:"
echo "  Tables:  results/tables/"
echo "  Figures: results/figures/"
echo "  Logs:    results/logs/"
echo "  Report:  results/dataset_report.md"
echo ""
echo "Key result files:"
ls results/tables/*.csv 2>/dev/null || echo "  (no CSV files yet)"
echo "============================================"
