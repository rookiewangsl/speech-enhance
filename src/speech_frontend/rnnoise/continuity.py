"""Output-only speech-continuity control for RNNoise experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from speech_frontend.audio import validate_mono_audio

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class ContinuityConfig:
    """Parameters for bounded voiced-frame envelope stabilization."""

    sample_rate: int = 16_000
    frame_samples: int = 160
    vad_threshold: float = 0.60
    allowed_drop_db: float = 4.0
    envelope_release_db_per_frame: float = 1.5
    max_boost_db: float = 3.0
    minimum_rms: float = 1e-5

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.frame_samples <= 0:
            raise ValueError("frame_samples must be positive")
        if not 0.0 <= self.vad_threshold <= 1.0:
            raise ValueError("vad_threshold must be within [0, 1]")
        if self.allowed_drop_db < 0.0:
            raise ValueError("allowed_drop_db must be nonnegative")
        if self.envelope_release_db_per_frame < 0.0:
            raise ValueError(
                "envelope_release_db_per_frame must be nonnegative"
            )
        if self.max_boost_db < 0.0:
            raise ValueError("max_boost_db must be nonnegative")
        if self.minimum_rms <= 0.0:
            raise ValueError("minimum_rms must be positive")


@dataclass(frozen=True)
class ContinuityResult:
    """Enhanced samples plus diagnostics for each processed frame."""

    samples: NDArray[np.float32]
    frame_gain_db: NDArray[np.float32]
    frame_rms_db: NDArray[np.float32]
    vad_probability: NDArray[np.float32]
    clipping_samples_before_limit: int


def stabilize_output_continuity(
    enhanced: FloatArray,
    vad_probabilities: FloatArray,
    *,
    config: ContinuityConfig | None = None,
) -> ContinuityResult:
    """Bound abrupt voiced-frame level drops without adding noisy residual.

    The controller only amplifies samples already present in the RNNoise
    output. It never mixes the raw noisy waveform back, so it cannot restore a
    component that RNNoise removed completely. This is intentionally a small,
    deployable experiment rather than a second enhancement model.
    """

    settings = config or ContinuityConfig()
    validate_mono_audio(enhanced, settings.sample_rate)
    samples = np.asarray(enhanced, dtype=np.float64)
    vad = np.asarray(vad_probabilities, dtype=np.float64)
    if vad.ndim != 1 or not np.all(np.isfinite(vad)):
        raise ValueError("VAD probabilities must be a finite vector")
    if np.any((vad < 0.0) | (vad > 1.0)):
        raise ValueError("VAD probabilities must be within [0, 1]")
    if samples.size and vad.size == 0:
        raise ValueError("non-empty audio requires VAD probabilities")

    output = np.empty_like(samples)
    frame_gain_db: list[float] = []
    frame_rms_db: list[float] = []
    used_vad: list[float] = []
    envelope_db: float | None = None
    previous_gain = 1.0
    epsilon = settings.minimum_rms**2

    for frame_index, offset in enumerate(
        range(0, samples.size, settings.frame_samples)
    ):
        stop = min(offset + settings.frame_samples, samples.size)
        frame = samples[offset:stop]
        probability = float(vad[min(frame_index, vad.size - 1)])
        rms = float(np.sqrt(np.mean(frame**2) + epsilon))
        current_db = 20.0 * np.log10(rms)

        boost_db = 0.0
        if probability >= settings.vad_threshold:
            if envelope_db is None:
                envelope_db = current_db
            else:
                envelope_db = max(
                    current_db,
                    envelope_db
                    - settings.envelope_release_db_per_frame,
                )
            target_floor_db = envelope_db - settings.allowed_drop_db
            boost_db = min(
                settings.max_boost_db,
                max(0.0, target_floor_db - current_db),
            )
        else:
            envelope_db = None

        current_gain = 10.0 ** (boost_db / 20.0)
        interpolation = np.linspace(
            previous_gain,
            current_gain,
            frame.size,
            endpoint=True,
            dtype=np.float64,
        )
        output[offset:stop] = frame * interpolation
        previous_gain = current_gain
        frame_gain_db.append(boost_db)
        frame_rms_db.append(current_db)
        used_vad.append(probability)

    clipping_samples = int(np.count_nonzero(np.abs(output) > 1.0))
    output = np.clip(output, -1.0, 1.0)
    return ContinuityResult(
        samples=np.asarray(output, dtype=np.float32),
        frame_gain_db=np.asarray(frame_gain_db, dtype=np.float32),
        frame_rms_db=np.asarray(frame_rms_db, dtype=np.float32),
        vad_probability=np.asarray(used_vad, dtype=np.float32),
        clipping_samples_before_limit=clipping_samples,
    )
