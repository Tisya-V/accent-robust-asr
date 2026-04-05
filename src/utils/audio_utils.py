from __future__ import annotations

import io
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def bytes_to_array(bytes_data: bytes, target_sr: int = 16000) -> np.ndarray:
    """Decode WAV bytes → float32 mono array at target_sr via soundfile."""
    waveform, sr = sf.read(io.BytesIO(bytes_data), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        waveform = resample_poly(waveform, target_sr // g, sr // g).astype(np.float32)
    return waveform