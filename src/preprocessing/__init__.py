from .pipeline import EEGPreprocessingPipeline, PreprocessingConfig
from .filters import bandpass_filter, notch_filter, resample_signal
from .normalization import normalize_trial, GlobalScaler
