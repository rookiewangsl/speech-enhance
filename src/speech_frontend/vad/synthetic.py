"""Controlled speech/noise mixtures with sample-exact VAD labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class VADMixtureConfig:
    """Synthetic VAD mixture timing and SNR settings."""

    sample_rate: int = 16_000
    duration_seconds: float = 10.0
    minimum_segments: int = 1
    maximum_segments: int = 4
    minimum_segment_seconds: float = 0.5
    maximum_segment_seconds: float = 2.0
    minimum_gap_seconds: float = 0.3
    maximum_gap_seconds: float = 2.0
    maximum_peak: float = 0.98

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.duration_seconds <= 0.0:
            raise ValueError("sample rate and duration must be positive")
        if not 1 <= self.minimum_segments <= self.maximum_segments:
            raise ValueError("invalid segment-count range")
        if not 0.0 < self.minimum_segment_seconds <= self.maximum_segment_seconds:
            raise ValueError("invalid segment duration range")
        if not 0.0 <= self.minimum_gap_seconds <= self.maximum_gap_seconds:
            raise ValueError("invalid gap duration range")
        if self.maximum_peak <= 0.0:
            raise ValueError("maximum_peak must be positive")


@dataclass(frozen=True)
class VADMixture:
    """A clean/noisy mixture plus ground-truth active-speech intervals."""

    clean: NDArray[np.float64]
    noisy: NDArray[np.float64]
    speech_intervals: tuple[tuple[int, int], ...]
    snr_db: float


def create_vad_mixture(
    speech_pool: list[NDArray[np.floating]],
    noise_pool: list[NDArray[np.floating]],
    *,
    snr_db: float,
    config: VADMixtureConfig | None = None,
    rng: np.random.Generator | None = None,
) -> VADMixture:
    """Create one labeled mixture without independently normalizing signals."""

    settings = config or VADMixtureConfig()
    random = rng or np.random.default_rng()
    speech = _validated_pool(speech_pool, "speech_pool")
    noise = _validated_pool(noise_pool, "noise_pool")
    total_samples = round(settings.duration_seconds * settings.sample_rate)
    clean = np.zeros(total_samples, dtype=np.float64)
    intervals = _place_speech(clean, speech, settings, random)
    noise_track = _build_noise_track(noise, total_samples, random)

    active = clean != 0.0
    speech_power = float(np.mean(clean[active] ** 2))
    noise_power = float(np.mean(noise_track[active] ** 2))
    if speech_power <= 1e-12 or noise_power <= 1e-12:
        raise ValueError("speech and noise pools must have non-zero power")
    noise_scale = np.sqrt(
        speech_power / (noise_power * 10.0 ** (snr_db / 10.0))
    )
    noisy = clean + noise_scale * noise_track

    peak = max(float(np.max(np.abs(clean))), float(np.max(np.abs(noisy))))
    if peak > settings.maximum_peak:
        common_gain = settings.maximum_peak / peak
        clean *= common_gain
        noisy *= common_gain
    return VADMixture(
        clean=clean,
        noisy=noisy,
        speech_intervals=tuple(intervals),
        snr_db=snr_db,
    )


def _validated_pool(
    pool: list[NDArray[np.floating]],
    name: str,
) -> list[NDArray[np.float64]]:
    if not pool:
        raise ValueError(f"{name} cannot be empty")
    validated = []
    for item in pool:
        samples = np.asarray(item, dtype=np.float64)
        if samples.ndim != 1 or samples.size == 0:
            raise ValueError(f"{name} items must be non-empty mono arrays")
        if not np.all(np.isfinite(samples)):
            raise ValueError(f"{name} contains NaN or infinite values")
        validated.append(samples)
    return validated


def _place_speech(
    target: NDArray[np.float64],
    pool: list[NDArray[np.float64]],
    settings: VADMixtureConfig,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    total = target.size
    min_duration = round(settings.minimum_segment_seconds * settings.sample_rate)
    max_duration = round(settings.maximum_segment_seconds * settings.sample_rate)
    min_gap = round(settings.minimum_gap_seconds * settings.sample_rate)
    max_gap = round(settings.maximum_gap_seconds * settings.sample_rate)
    count = int(rng.integers(settings.minimum_segments, settings.maximum_segments + 1))

    while count > 1 and count * min_duration + (count - 1) * min_gap > total:
        count -= 1
    if count * min_duration > total:
        raise ValueError("mixture duration is too short for one speech segment")

    intervals: list[tuple[int, int]] = []
    cursor = int(rng.integers(0, min(max_gap, total - count * min_duration) + 1))
    for index in range(count):
        remaining_segments = count - index - 1
        required_after = remaining_segments * (min_duration + min_gap)
        maximum_here = min(max_duration, total - cursor - required_after)
        duration = int(rng.integers(min_duration, maximum_here + 1))
        source = pool[int(rng.integers(0, len(pool)))]
        if source.size < duration:
            repeats = int(np.ceil(duration / source.size))
            source = np.tile(source, repeats)
        source_start = int(rng.integers(0, source.size - duration + 1))
        stop = cursor + duration
        target[cursor:stop] += source[source_start : source_start + duration]
        intervals.append((cursor, stop))
        if remaining_segments:
            minimum_after_gap = (
                remaining_segments * min_duration
                + (remaining_segments - 1) * min_gap
            )
            gap_max = min(max_gap, total - stop - minimum_after_gap)
            gap = int(rng.integers(min_gap, gap_max + 1))
            cursor = stop + gap
    return intervals


def _build_noise_track(
    pool: list[NDArray[np.float64]],
    length: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    parts: list[NDArray[np.float64]] = []
    collected = 0
    while collected < length:
        source = pool[int(rng.integers(0, len(pool)))]
        if source.size > 1:
            offset = int(rng.integers(0, source.size))
            source = np.concatenate((source[offset:], source[:offset]))
        parts.append(source)
        collected += source.size
    noise = np.concatenate(parts)[:length]
    return noise - np.mean(noise)
