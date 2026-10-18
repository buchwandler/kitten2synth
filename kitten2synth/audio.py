from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def validate_audio(audio: np.ndarray) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("generated audio must be a non-empty mono waveform")
    if not np.isfinite(values).all():
        raise ValueError("generated audio contains non-finite samples")
    return values


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    values = np.clip(validate_audio(audio), -1.0, 1.0)
    pcm = np.round(values * 32767.0).astype("<i2")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())
    return output
