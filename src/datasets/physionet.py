"""
PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB) loader.

Dataset: Goldberger et al., PhysioBank/PhysioToolkit (2000)
URL: https://physionet.org/content/eegmmidb/1.0.0/
Subjects: 109 healthy subjects
Tasks: Rest, Left/Right Fist MI, Both Fists/Feet MI
Channels: 64 (10-20 system)
Sampling rate: 160 Hz

We focus on the 4-class Motor Imagery task:
  Class 0: Left fist imagery
  Class 1: Right fist imagery
  Class 2: Both fists imagery
  Class 3: Both feet imagery

Runs used:
  Runs 4, 8, 12: Left vs Right fist imagery
  Runs 6, 10, 14: Both fists vs Both feet imagery
"""

from __future__ import annotations
import os
import logging
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Try MNE for PhysioNet loading
try:
    import mne
    from mne.datasets import eegbci
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    logger.warning("MNE not available — PhysioNet loading disabled")

from .base import EEGDataset
from ..preprocessing.pipeline import EEGPreprocessingPipeline, PreprocessingConfig


# Task → (run numbers, event descriptions, label mapping)
TASK_CONFIG = {
    "4class": {
        "runs": [4, 6, 8, 10, 12, 14],
        # Events: T1=left/both-fists, T2=right/both-feet
        # Combined: T1 in runs 4,8,12 = left; T1 in runs 6,10,14 = both-fists
        "label_map": {
            "left": 0,
            "right": 1,
            "both_fists": 2,
            "both_feet": 3,
        },
    },
    "2class_lr": {
        "runs": [4, 8, 12],
        "label_map": {"left": 0, "right": 1},
    },
}

# Run-specific event mapping
_RUN_EVENTS = {
    4: {"T1": "left", "T2": "right"},
    6: {"T1": "both_fists", "T2": "both_feet"},
    8: {"T1": "left", "T2": "right"},
    10: {"T1": "both_fists", "T2": "both_feet"},
    12: {"T1": "left", "T2": "right"},
    14: {"T1": "both_fists", "T2": "both_feet"},
}


def _load_subject_raw(
    subject_id: int,
    runs: list[int],
    data_dir: str | None = None,
) -> Optional["mne.io.Raw"]:
    """Load and concatenate raw EEG for one subject and given runs."""
    if not MNE_AVAILABLE:
        raise RuntimeError("MNE is required to load PhysioNet data")

    try:
        raw_list = []
        for run in runs:
            fnames = eegbci.load_data(
                subject_id,
                runs=[run],
                path=data_dir,
                verbose=False,
            )
            raw = mne.io.read_raw_edf(fnames[0], preload=True, verbose=False)
            eegbci.standardize(raw)   # standardize channel names to 10-20
            raw_list.append(raw)
        return mne.concatenate_raws(raw_list)
    except Exception as e:
        logger.error(f"Failed to load subject {subject_id}: {e}")
        return None


def _extract_epochs(
    raw: "mne.io.Raw",
    run: int,
    label_map: dict[str, int],
    tmin: float = 0.0,
    tmax: float = 4.0,
) -> tuple[list[np.ndarray], list[int]]:
    """Extract epochs for a single run from a raw object."""
    import mne as _mne

    events, event_id = _mne.events_from_annotations(raw, verbose=False)
    if not events.size:
        return [], []

    run_event_map = _RUN_EVENTS.get(run, {})

    trials, labels = [], []
    sfreq = raw.info["sfreq"]
    tmin_samp = int(tmin * sfreq)
    tmax_samp = int(tmax * sfreq)

    for evt_name, evt_class in run_event_map.items():
        if evt_class not in label_map:
            continue
        class_id = label_map[evt_class]

        # Find matching event code
        matching_codes = [v for k, v in event_id.items() if evt_name in k]
        if not matching_codes:
            continue

        for evt_code in matching_codes:
            evt_onsets = events[events[:, 2] == evt_code, 0]
            for onset in evt_onsets:
                start = onset + tmin_samp
                end = onset + tmax_samp
                if end <= raw.n_times:
                    data = raw.get_data(start=start, stop=end, verbose=False)
                    trials.append(data.astype(np.float32) * 1e6)  # V → μV
                    labels.append(class_id)

    return trials, labels


def load_physionet(
    data_dir: str = "data/physionet",
    subjects: list[int] | None = None,
    task: str = "4class",
    preprocess_cfg: PreprocessingConfig | None = None,
    n_subjects: int = 109,
) -> tuple[EEGDataset, dict]:
    """
    Load and preprocess PhysioNet MI dataset.

    Parameters
    ----------
    data_dir : str
        Directory to cache downloaded data.
    subjects : list[int] or None
        Subject IDs (1-indexed). None = all 109 subjects.
    task : str
        "4class" | "2class_lr"
    preprocess_cfg : PreprocessingConfig or None
    n_subjects : int
        Maximum number of subjects to load (for quick testing).

    Returns
    -------
    dataset : EEGDataset
    meta : dict with dataset statistics
    """
    if not MNE_AVAILABLE:
        raise RuntimeError(
            "MNE is required. Install with: pip install mne\n"
            "Then re-run: python scripts/download_data.py"
        )

    os.makedirs(data_dir, exist_ok=True)
    task_cfg = TASK_CONFIG[task]

    if subjects is None:
        subjects = list(range(1, min(n_subjects + 1, 110)))

    if preprocess_cfg is None:
        preprocess_cfg = PreprocessingConfig(
            target_sfreq=160.0,
            l_freq=1.0,
            h_freq=40.0,
            notch_freq=60.0,   # US power line
            apply_car=True,
            normalization="zscore",
            segment_len_sec=4.0,
        )

    all_trials: list[np.ndarray] = []
    all_labels: list[int] = []
    all_subject_ids: list[int] = []

    failed_subjects = []

    for subj_idx, subject_id in enumerate(subjects):
        logger.info(f"Loading subject {subject_id} ({subj_idx+1}/{len(subjects)})")

        run_trials: list[np.ndarray] = []
        run_labels: list[int] = []

        for run in task_cfg["runs"]:
            raw = _load_subject_raw(subject_id, [run], data_dir)
            if raw is None:
                failed_subjects.append(subject_id)
                continue

            # Preprocessing
            sfreq = raw.info["sfreq"]
            pipeline = EEGPreprocessingPipeline(preprocess_cfg, orig_sfreq=sfreq)

            trials, labels = _extract_epochs(
                raw, run, task_cfg["label_map"],
                tmin=0.0, tmax=4.0
            )

            for trial, lbl in zip(trials, labels):
                try:
                    processed = pipeline(trial)
                    run_trials.append(processed)
                    run_labels.append(lbl)
                except Exception as e:
                    logger.debug(f"Skipped trial: {e}")

        all_trials.extend(run_trials)
        all_labels.extend(run_labels)
        all_subject_ids.extend([subject_id] * len(run_trials))

    if not all_trials:
        raise RuntimeError(
            "No trials loaded. Check dataset download and preprocessing config.\n"
            "Run: python scripts/download_data.py"
        )

    # Build metadata
    from collections import Counter
    label_counts = Counter(all_labels)
    meta = {
        "n_subjects": len(set(all_subject_ids)),
        "n_trials": len(all_trials),
        "n_classes": len(set(all_labels)),
        "class_distribution": dict(label_counts),
        "n_channels": all_trials[0].shape[0] if all_trials else 0,
        "trial_length": all_trials[0].shape[1] if all_trials else 0,
        "sfreq": preprocess_cfg.target_sfreq,
        "failed_subjects": failed_subjects,
    }

    logger.info(
        f"Loaded {meta['n_trials']} trials from {meta['n_subjects']} subjects. "
        f"Class distribution: {meta['class_distribution']}"
    )

    dataset = EEGDataset(all_trials, all_labels, all_subject_ids)
    return dataset, meta


class PhysioNetMIDataset(EEGDataset):
    """Thin wrapper around EEGDataset with PhysioNet-specific metadata."""

    def __init__(self, *args, meta: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta = meta or {}
