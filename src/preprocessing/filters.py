"""
EEG signal filtering utilities.

All filters use scipy.signal for CPU-based filtering.
Uses zero-phase (filtfilt) by default for offline preprocessing.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly
from math import gcd


def bandpass_filter(
    data: np.ndarray,
    sfreq: float,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    data : (..., T) last dim = time
    sfreq : sampling frequency
    l_freq : lower cutoff
    h_freq : upper cutoff
    order : filter order

    Returns
    -------
    filtered : same shape as data
    """
    nyq = sfreq / 2.0
    low = l_freq / nyq
    high = min(h_freq / nyq, 0.99)
    b, a = butter(order, [low, high], btype="band")
    # Apply along last axis
    return filtfilt(b, a, data, axis=-1).astype(np.float32)


def notch_filter(
    data: np.ndarray,
    sfreq: float,
    notch_freq: float = 50.0,
    quality: float = 30.0,
) -> np.ndarray:
    """
    IIR notch filter for power-line interference removal.

    Parameters
    ----------
    data : (..., T)
    sfreq : sampling frequency
    notch_freq : notch center frequency (50 or 60 Hz)
    quality : quality factor Q
    """
    b, a = iirnotch(notch_freq / (sfreq / 2.0), quality)
    return filtfilt(b, a, data, axis=-1).astype(np.float32)


def resample_signal(
    data: np.ndarray,
    orig_sfreq: float,
    target_sfreq: float,
) -> np.ndarray:
    """
    Rational resample using polyphase filter.

    Parameters
    ----------
    data : (..., T)
    orig_sfreq : original sampling frequency
    target_sfreq : target sampling frequency

    Returns
    -------
    resampled : (..., T')
    """
    if abs(orig_sfreq - target_sfreq) < 1e-6:
        return data
    g = gcd(int(orig_sfreq), int(target_sfreq))
    up = int(target_sfreq) // g
    down = int(orig_sfreq) // g
    return resample_poly(data, up, down, axis=-1).astype(np.float32)


def apply_car(data: np.ndarray) -> np.ndarray:
    """
    Common Average Reference (CAR) spatial filter.

    Subtracts the mean across channels from each channel.

    Parameters
    ----------
    data : (C, T)

    Returns
    -------
    (C, T)
    """
    return (data - data.mean(axis=0, keepdims=True)).astype(np.float32)
