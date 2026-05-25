"""Tests for EEG preprocessing pipeline."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.filters import bandpass_filter, notch_filter, resample_signal, apply_car
from src.preprocessing.normalization import normalize_trial, GlobalScaler
from src.preprocessing.pipeline import EEGPreprocessingPipeline, PreprocessingConfig


@pytest.fixture
def sample_eeg():
    """Random EEG trial (C=32, T=1000) at 500Hz."""
    np.random.seed(42)
    return np.random.randn(32, 1000).astype(np.float32) * 50.0   # ~50μV scale


class TestFilters:
    def test_bandpass_output_shape(self, sample_eeg):
        filtered = bandpass_filter(sample_eeg, sfreq=500.0, l_freq=1.0, h_freq=40.0)
        assert filtered.shape == sample_eeg.shape

    def test_bandpass_no_nan(self, sample_eeg):
        filtered = bandpass_filter(sample_eeg, sfreq=500.0)
        assert not np.isnan(filtered).any()

    def test_notch_shape(self, sample_eeg):
        filtered = notch_filter(sample_eeg, sfreq=500.0, notch_freq=50.0)
        assert filtered.shape == sample_eeg.shape

    def test_resample_reduces_length(self, sample_eeg):
        resampled = resample_signal(sample_eeg, orig_sfreq=500.0, target_sfreq=160.0)
        expected_len = int(1000 * 160 / 500)
        assert abs(resampled.shape[-1] - expected_len) <= 2

    def test_car_zero_mean(self, sample_eeg):
        car = apply_car(sample_eeg)
        assert np.allclose(car.mean(axis=0), 0.0, atol=1e-5)


class TestNormalization:
    def test_zscore_stats(self, sample_eeg):
        norm = normalize_trial(sample_eeg, method="zscore")
        # Per-channel mean ~0, std ~1
        assert np.allclose(norm.mean(axis=-1), 0.0, atol=0.1)
        assert np.allclose(norm.std(axis=-1), 1.0, atol=0.1)

    def test_minmax_range(self, sample_eeg):
        norm = normalize_trial(sample_eeg, method="minmax")
        assert norm.min() >= -0.01
        assert norm.max() <= 1.01

    def test_global_scaler(self, sample_eeg):
        trials = [sample_eeg + i for i in range(5)]
        scaler = GlobalScaler()
        scaler.fit(trials)
        for t in trials:
            scaled = scaler.transform(t)
            assert not np.isnan(scaled).any()


class TestPipeline:
    def test_full_pipeline(self, sample_eeg):
        cfg = PreprocessingConfig(
            target_sfreq=160.0, l_freq=1.0, h_freq=40.0,
            notch_freq=50.0, apply_car=True,
            normalization="zscore", segment_len_sec=4.0,
        )
        pipeline = EEGPreprocessingPipeline(cfg, orig_sfreq=500.0)
        out = pipeline(sample_eeg)
        expected_len = int(4.0 * 160.0)
        assert out.shape == (32, expected_len)
        assert not np.isnan(out).any()
        assert out.dtype == np.float32

    def test_pipeline_reproducible(self, sample_eeg):
        cfg = PreprocessingConfig(target_sfreq=160.0, normalization="zscore",
                                   segment_len_sec=4.0)
        pipeline = EEGPreprocessingPipeline(cfg, orig_sfreq=500.0)
        out1 = pipeline(sample_eeg.copy())
        out2 = pipeline(sample_eeg.copy())
        assert np.allclose(out1, out2)
