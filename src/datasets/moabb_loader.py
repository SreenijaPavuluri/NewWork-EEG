"""
MOABB-based EEG dataset loader.

MOABB (Mother of all BCI Benchmarks) provides standardized access to
public EEG BCI datasets.  We use it as a secondary dataset source for:
  - BNCI2014_001 (BCI Competition IV-2a): 9 subjects, 4-class MI
  - Zhou2016: lightweight 3-subject dataset for quick validation

Reference:
  Chevallier et al., "MOABB: Trustworthy algorithm benchmarking for BCIs",
  arXiv 2303.03842, 2023.
"""

from __future__ import annotations
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import moabb
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery
    MOABB_AVAILABLE = True
except ImportError:
    MOABB_AVAILABLE = False
    logger.warning("MOABB not available — install with: pip install moabb")

from .base import EEGDataset
from ..preprocessing.pipeline import EEGPreprocessingPipeline, PreprocessingConfig


SUPPORTED_DATASETS = {
    "BNCI2014_001": {
        "n_subjects": 9,
        "n_classes": 4,
        "sfreq": 250.0,
        "description": "BCI Competition IV-2a, 9 subjects, 4-class MI",
    },
}


def load_moabb_dataset(
    dataset_name: str = "BNCI2014_001",
    subjects: list[int] | None = None,
    data_dir: str = "data/moabb",
    preprocess_cfg: PreprocessingConfig | None = None,
) -> tuple[EEGDataset, dict]:
    """
    Load a MOABB dataset and return an EEGDataset.

    Parameters
    ----------
    dataset_name : str
        Name of the MOABB dataset.
    subjects : list[int] or None
        Subject IDs. None = all subjects.
    data_dir : str
        Cache directory.
    preprocess_cfg : PreprocessingConfig or None

    Returns
    -------
    dataset : EEGDataset
    meta : dict
    """
    if not MOABB_AVAILABLE:
        raise RuntimeError(
            "MOABB is required. Install with: pip install moabb\n"
            "Then re-run: python scripts/download_data.py"
        )

    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. "
            f"Choose from: {list(SUPPORTED_DATASETS.keys())}"
        )

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # Set MOABB data path
    moabb.set_download_dir(data_dir)

    dataset_info = SUPPORTED_DATASETS[dataset_name]

    if subjects is None:
        subjects = list(range(1, dataset_info["n_subjects"] + 1))

    # Initialize dataset and paradigm
    if dataset_name == "BNCI2014_001":
        dataset = BNCI2014_001()
        paradigm = MotorImagery(n_classes=4, fmin=1.0, fmax=40.0, tmin=0.0, tmax=4.0)

    if preprocess_cfg is None:
        preprocess_cfg = PreprocessingConfig(
            target_sfreq=160.0,
            l_freq=1.0,
            h_freq=40.0,
            notch_freq=50.0,
            apply_car=True,
            normalization="zscore",
            segment_len_sec=4.0,
        )

    all_trials: list[np.ndarray] = []
    all_labels: list[int] = []
    all_subject_ids: list[int] = []

    try:
        X, y, metadata = paradigm.get_data(
            dataset=dataset,
            subjects=subjects,
            return_epochs=False,
        )
        # X: (n_trials, n_channels, n_times)
        # y: class labels as strings or ints

        # Build label map
        unique_labels = sorted(set(y))
        label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}

        orig_sfreq = dataset_info["sfreq"]
        pipeline = EEGPreprocessingPipeline(preprocess_cfg, orig_sfreq=orig_sfreq)

        for i, (trial, label) in enumerate(zip(X, y)):
            try:
                processed = pipeline(trial.astype(np.float32))
                all_trials.append(processed)
                all_labels.append(label_map[label])
                subj = int(metadata.iloc[i]["subject"]) if hasattr(metadata, 'iloc') else i
                all_subject_ids.append(subj)
            except Exception as e:
                logger.debug(f"Skipped trial {i}: {e}")

    except Exception as e:
        raise RuntimeError(
            f"Failed to load {dataset_name}: {e}\n"
            "Ensure internet access for first download."
        ) from e

    from collections import Counter
    meta = {
        "dataset": dataset_name,
        "n_subjects": len(set(all_subject_ids)),
        "n_trials": len(all_trials),
        "n_classes": len(set(all_labels)),
        "class_distribution": dict(Counter(all_labels)),
        "n_channels": all_trials[0].shape[0] if all_trials else 0,
        "trial_length": all_trials[0].shape[1] if all_trials else 0,
        "sfreq": preprocess_cfg.target_sfreq,
    }

    logger.info(
        f"Loaded {dataset_name}: {meta['n_trials']} trials, "
        f"{meta['n_subjects']} subjects"
    )

    eeg_dataset = EEGDataset(all_trials, all_labels, all_subject_ids)
    return eeg_dataset, meta


class MOABBDataset(EEGDataset):
    """Thin wrapper with MOABB-specific metadata."""

    def __init__(self, *args, meta: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta = meta or {}
