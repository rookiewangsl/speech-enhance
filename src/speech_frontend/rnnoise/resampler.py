"""Causal stateful 16↔48 kHz integer-ratio FIR resampling."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import firwin, lfilter

from speech_frontend.audio import validate_mono_audio

INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 48_000
UPSAMPLE_FACTOR = 3
RESAMPLER_TAPS = 127
RESAMPLER_GROUP_DELAY_48K = (RESAMPLER_TAPS - 1) // 2

FloatArray = NDArray[np.floating]


def _lowpass_coefficients() -> NDArray[np.float64]:
    return np.asarray(
        firwin(
            RESAMPLER_TAPS,
            cutoff=7_800.0,
            fs=OUTPUT_SAMPLE_RATE,
            window=("kaiser", 8.6),
        ),
        dtype=np.float64,
    )


class StreamingUpsampler3:
    """Causal 16→48 kHz zero-insertion and anti-imaging FIR."""

    def __init__(self) -> None:
        self.coefficients = _lowpass_coefficients() * UPSAMPLE_FACTOR
        self._state = np.zeros(RESAMPLER_TAPS - 1, dtype=np.float64)
        self.total_input_samples = 0
        self.total_output_samples = 0

    def process_chunk(self, samples: FloatArray) -> NDArray[np.float32]:
        validate_mono_audio(samples, INPUT_SAMPLE_RATE)
        normalized = np.asarray(samples, dtype=np.float64)
        if normalized.size == 0:
            return np.empty(0, dtype=np.float32)
        upsampled = np.zeros(
            normalized.size * UPSAMPLE_FACTOR,
            dtype=np.float64,
        )
        upsampled[::UPSAMPLE_FACTOR] = normalized
        filtered, self._state = lfilter(
            self.coefficients,
            [1.0],
            upsampled,
            zi=self._state,
        )
        self.total_input_samples += normalized.size
        self.total_output_samples += filtered.size
        return np.asarray(filtered, dtype=np.float32)

    def reset(self) -> None:
        self._state.fill(0.0)
        self.total_input_samples = 0
        self.total_output_samples = 0


class StreamingDownsampler3:
    """Causal 48→16 kHz anti-aliasing FIR with persistent phase."""

    def __init__(self) -> None:
        self.coefficients = _lowpass_coefficients()
        self._state = np.zeros(RESAMPLER_TAPS - 1, dtype=np.float64)
        self.total_input_samples = 0
        self.total_output_samples = 0

    def process_chunk(self, samples: FloatArray) -> NDArray[np.float32]:
        validate_mono_audio(samples, OUTPUT_SAMPLE_RATE)
        normalized = np.asarray(samples, dtype=np.float64)
        if normalized.size == 0:
            return np.empty(0, dtype=np.float32)
        filtered, self._state = lfilter(
            self.coefficients,
            [1.0],
            normalized,
            zi=self._state,
        )
        first_index = (-self.total_input_samples) % UPSAMPLE_FACTOR
        output = filtered[first_index::UPSAMPLE_FACTOR]
        self.total_input_samples += normalized.size
        self.total_output_samples += output.size
        return np.asarray(output, dtype=np.float32)

    def reset(self) -> None:
        self._state.fill(0.0)
        self.total_input_samples = 0
        self.total_output_samples = 0
