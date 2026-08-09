"""Audio input/output contracts used by the speech front-end."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class AudioData:
    """A validated mono audio signal and its sample rate."""

    samples: FloatArray
    sample_rate: int


def validate_mono_audio(samples: FloatArray, sample_rate: int) -> None:
    """Validate the internal audio representation.

    The project uses one-dimensional, finite floating-point arrays. Amplitude
    is not normalized here because clean/noisy pairs must retain their common
    scale for objective evaluation.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if samples.ndim != 1:
        raise ValueError("audio must be a one-dimensional mono array")
    if not np.issubdtype(samples.dtype, np.floating):
        raise TypeError("audio samples must use a floating-point dtype")
    if not np.all(np.isfinite(samples)):
        raise ValueError("audio contains NaN or infinite values")


def read_audio(path: str | Path) -> AudioData:
    """Read a mono audio file without normalizing its amplitude."""

    samples, sample_rate = sf.read(
        Path(path),
        dtype="float32",
        always_2d=False,
    )
    samples = np.asarray(samples)
    validate_mono_audio(samples, sample_rate)
    return AudioData(samples=samples, sample_rate=sample_rate)


def write_audio(path: str | Path, audio: AudioData) -> None:
    """Write validated floating-point audio as a WAV file."""

    validate_mono_audio(audio.samples, audio.sample_rate)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        output_path,
        audio.samples,
        audio.sample_rate,
        subtype="FLOAT",
    )


def validate_aligned_pair(clean: AudioData, noisy: AudioData) -> None:
    """Validate a clean/noisy pair before intrusive evaluation."""

    validate_mono_audio(clean.samples, clean.sample_rate)
    validate_mono_audio(noisy.samples, noisy.sample_rate)
    if clean.sample_rate != noisy.sample_rate:
        raise ValueError("clean and noisy sample rates do not match")
    if clean.samples.shape != noisy.samples.shape:
        raise ValueError("clean and noisy lengths do not match")
