"""Deployable residual-mixing controllers around aligned RNNoise output."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from speech_frontend.audio import validate_mono_audio

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class ControllerResult:
    samples: NDArray[np.float32]
    frame_strength: NDArray[np.float32]
    correction_ratio_db: NDArray[np.float32]


@dataclass(frozen=True)
class CorrectionAwareConfig:
    frame_samples: int = 160
    energy_smoothing: float = 0.80
    correction_threshold_db: float = -9.0
    correction_slope_db: float = 1.5
    speech_protection: float = 0.0
    strength_attack: float = 0.30
    strength_release: float = 0.85
    initial_strength: float = 0.5

    def __post_init__(self) -> None:
        if self.frame_samples <= 0:
            raise ValueError("frame_samples must be positive")
        for name in (
            "energy_smoothing",
            "speech_protection",
            "strength_attack",
            "strength_release",
            "initial_strength",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.correction_slope_db <= 0:
            raise ValueError("correction_slope_db must be positive")


def _validate_pair(
    noisy: FloatArray,
    enhanced: FloatArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    validate_mono_audio(noisy, 16_000)
    validate_mono_audio(enhanced, 16_000)
    if noisy.shape != enhanced.shape:
        raise ValueError("noisy and enhanced lengths do not match")
    return (
        np.asarray(noisy, dtype=np.float64),
        np.asarray(enhanced, dtype=np.float64),
    )


def fixed_residual_mix(
    noisy: FloatArray,
    enhanced: FloatArray,
    strength: float,
) -> NDArray[np.float32]:
    """Mix RNNoise output with its aligned input using a fixed strength."""

    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be within [0, 1]")
    input_signal, enhanced_signal = _validate_pair(noisy, enhanced)
    output = strength * enhanced_signal + (1.0 - strength) * input_signal
    return np.asarray(output, dtype=np.float32)


def vad_aware_mix(
    noisy: FloatArray,
    enhanced: FloatArray,
    vad_probabilities: FloatArray,
    *,
    speech_protection: float,
    frame_samples: int = 160,
) -> NDArray[np.float32]:
    """Reduce enhancement strength during high speech probability."""

    if not 0.0 <= speech_protection <= 1.0:
        raise ValueError("speech_protection must be within [0, 1]")
    if frame_samples <= 0:
        raise ValueError("frame_samples must be positive")
    input_signal, enhanced_signal = _validate_pair(noisy, enhanced)
    vad = np.asarray(vad_probabilities, dtype=np.float64)
    if vad.ndim != 1 or not np.all(np.isfinite(vad)):
        raise ValueError("VAD probabilities must be a finite vector")
    if np.any((vad < 0.0) | (vad > 1.0)):
        raise ValueError("VAD probabilities must be within [0, 1]")
    if vad.size == 0:
        return np.asarray(enhanced_signal, dtype=np.float32)
    sample_probability = np.repeat(vad, frame_samples)
    sample_probability = np.pad(
        sample_probability,
        (0, max(0, input_signal.size - sample_probability.size)),
        mode="edge",
    )[: input_signal.size]
    strength = 1.0 - speech_protection * sample_probability
    output = strength * enhanced_signal + (1.0 - strength) * input_signal
    return np.asarray(output, dtype=np.float32)


def correction_aware_mix(
    noisy: FloatArray,
    enhanced: FloatArray,
    *,
    vad_probabilities: FloatArray | None = None,
    config: CorrectionAwareConfig | None = None,
) -> ControllerResult:
    """Adapt enhancement strength using only deployable signals.

    The smoothed energy ratio between the RNNoise correction ``noisy-enhanced``
    and the input acts as a noise-severity proxy. Small corrections imply that
    the input is already relatively clean, so the controller moves toward
    bypass. Large corrections retain more RNNoise output.
    """

    settings = config or CorrectionAwareConfig()
    input_signal, enhanced_signal = _validate_pair(noisy, enhanced)
    if vad_probabilities is None:
        vad = None
    else:
        vad = np.asarray(vad_probabilities, dtype=np.float64)
        if vad.ndim != 1 or not np.all(np.isfinite(vad)):
            raise ValueError("VAD probabilities must be a finite vector")
        if np.any((vad < 0.0) | (vad > 1.0)):
            raise ValueError("VAD probabilities must be within [0, 1]")

    output = np.empty_like(input_signal)
    strengths: list[float] = []
    correction_ratios: list[float] = []
    smoothed_input_energy = 0.0
    smoothed_correction_energy = 0.0
    previous_strength = settings.initial_strength
    epsilon = 1e-12

    for frame_index, offset in enumerate(
        range(0, input_signal.size, settings.frame_samples)
    ):
        stop = min(offset + settings.frame_samples, input_signal.size)
        input_frame = input_signal[offset:stop]
        enhanced_frame = enhanced_signal[offset:stop]
        correction = input_frame - enhanced_frame
        input_energy = float(np.mean(input_frame**2))
        correction_energy = float(np.mean(correction**2))
        if frame_index == 0:
            smoothed_input_energy = input_energy
            smoothed_correction_energy = correction_energy
        else:
            alpha = settings.energy_smoothing
            smoothed_input_energy = (
                alpha * smoothed_input_energy
                + (1.0 - alpha) * input_energy
            )
            smoothed_correction_energy = (
                alpha * smoothed_correction_energy
                + (1.0 - alpha) * correction_energy
            )
        correction_ratio_db = 10.0 * np.log10(
            (smoothed_correction_energy + epsilon)
            / (smoothed_input_energy + epsilon)
        )
        normalized = (
            correction_ratio_db - settings.correction_threshold_db
        ) / settings.correction_slope_db
        correction_strength = 1.0 / (
            1.0 + np.exp(-np.clip(normalized, -30.0, 30.0))
        )
        if vad is not None and vad.size:
            probability = vad[min(frame_index, vad.size - 1)]
            correction_strength *= (
                1.0 - settings.speech_protection * probability
            )
        smoothing = (
            settings.strength_attack
            if correction_strength > previous_strength
            else settings.strength_release
        )
        current_strength = (
            smoothing * previous_strength
            + (1.0 - smoothing) * correction_strength
        )
        interpolation = np.linspace(
            previous_strength,
            current_strength,
            stop - offset,
            endpoint=True,
        )
        output[offset:stop] = (
            interpolation * enhanced_frame
            + (1.0 - interpolation) * input_frame
        )
        strengths.append(current_strength)
        correction_ratios.append(correction_ratio_db)
        previous_strength = current_strength

    return ControllerResult(
        samples=np.asarray(output, dtype=np.float32),
        frame_strength=np.asarray(strengths, dtype=np.float32),
        correction_ratio_db=np.asarray(
            correction_ratios,
            dtype=np.float32,
        ),
    )
