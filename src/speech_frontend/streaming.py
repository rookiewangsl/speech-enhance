"""Stateful streaming weighted overlap-add processing."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from speech_frontend.framing import sqrt_periodic_hann
from speech_frontend.stft import STFTConfig

FloatArray = NDArray[np.floating]
ComplexArray = NDArray[np.complexfloating]
SpectrumProcessor = Callable[[ComplexArray], ComplexArray]


def _identity(spectrum: ComplexArray) -> ComplexArray:
    return spectrum


class StreamingWOLA:
    """Process arbitrary input chunks through a frame-wise spectral function.

    The class prepends the same implicit padding used by :class:`STFT` and
    emits samples only after their overlap-add values can no longer change.
    A processor receives one ``rfft`` spectrum and must return the same shape.
    """

    def __init__(
        self,
        config: STFTConfig | None = None,
        processor: SpectrumProcessor | None = None,
    ) -> None:
        self.config = config or STFTConfig()
        self.processor = processor or _identity
        self.window = sqrt_periodic_hann(self.config.frame_length)
        self.reset()

    @property
    def latency_samples(self) -> int:
        """Return the steady-state look-ahead in samples."""

        return self.config.frame_length - self.config.hop_length

    def reset(self) -> None:
        """Clear all signal and lifecycle state for a new stream."""

        self._left_padding = self.latency_samples
        self._analysis_buffer = np.zeros(
            self._left_padding,
            dtype=np.float64,
        )
        self._ola_signal = np.zeros(
            self.config.frame_length,
            dtype=np.float64,
        )
        self._ola_normalization = np.zeros(
            self.config.frame_length,
            dtype=np.float64,
        )
        self._input_samples = 0
        self._output_samples = 0
        self._padded_samples_emitted = 0
        self._frames_processed = 0
        self._flushed = False

    def process_chunk(self, samples: FloatArray) -> NDArray[np.float64]:
        """Consume a finite one-dimensional chunk and return ready samples."""

        if self._flushed:
            raise RuntimeError("cannot process samples after flush")
        chunk = np.asarray(samples)
        if chunk.ndim != 1:
            raise ValueError("samples must be one-dimensional")
        if not np.all(np.isfinite(chunk)):
            raise ValueError("samples contain NaN or infinite values")
        if chunk.size == 0:
            return np.empty(0, dtype=np.float64)

        self._analysis_buffer = np.concatenate(
            (self._analysis_buffer, chunk.astype(np.float64, copy=False))
        )
        self._input_samples += chunk.size
        return self._process_available_frames()

    def flush(self) -> NDArray[np.float64]:
        """Finish the stream, returning its exact remaining output once."""

        if self._flushed:
            return np.empty(0, dtype=np.float64)
        self._flushed = True

        frame_count = self._required_frame_count(self._input_samples)
        frames_remaining = frame_count - self._frames_processed
        if frames_remaining < 0:
            raise RuntimeError("stream processed more frames than expected")

        required_buffer_size = (
            (frames_remaining - 1) * self.config.hop_length
            + self.config.frame_length
            if frames_remaining
            else 0
        )
        padding = required_buffer_size - self._analysis_buffer.size
        if padding < 0:
            raise RuntimeError("stream has inconsistent framing state")
        if padding:
            self._analysis_buffer = np.pad(
                self._analysis_buffer,
                (0, padding),
            )

        output = self._process_available_frames(
            maximum_frames=frames_remaining
        )

        if self._output_samples != self._input_samples:
            raise RuntimeError("flush did not reconstruct the exact input length")
        return output

    def _required_frame_count(self, input_length: int) -> int:
        minimum_length = (
            self._left_padding + input_length + self._left_padding
        )
        if minimum_length <= self.config.frame_length:
            return 1
        return (
            int(
                np.ceil(
                    (minimum_length - self.config.frame_length)
                    / self.config.hop_length
                )
            )
            + 1
        )

    def _process_available_frames(
        self,
        *,
        maximum_frames: int | None = None,
    ) -> NDArray[np.float64]:
        chunks: list[NDArray[np.float64]] = []
        processed_here = 0

        while (
            self._analysis_buffer.size >= self.config.frame_length
            and (
                maximum_frames is None
                or processed_here < maximum_frames
            )
        ):
            frame = self._analysis_buffer[: self.config.frame_length]
            spectrum = np.fft.rfft(
                frame * self.window,
                n=self.config.fft_length,
            )
            processed = np.asarray(self.processor(spectrum))
            if processed.shape != spectrum.shape:
                raise ValueError(
                    "spectrum processor must preserve the spectrum shape"
                )
            if not np.all(np.isfinite(processed)):
                raise ValueError(
                    "spectrum processor returned NaN or infinite values"
                )

            synthesis_frame = np.fft.irfft(
                processed,
                n=self.config.fft_length,
            )[: self.config.frame_length]
            self._ola_signal += synthesis_frame * self.window
            self._ola_normalization += self.window**2
            chunks.append(self._emit_hop())

            hop = self.config.hop_length
            self._analysis_buffer = self._analysis_buffer[hop:]
            self._frames_processed += 1
            processed_here += 1

        if not chunks:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(chunks)

    def _emit_hop(self) -> NDArray[np.float64]:
        hop = self.config.hop_length
        signal = self._ola_signal[:hop].copy()
        normalization = self._ola_normalization[:hop]
        valid = normalization > 1e-12
        signal[valid] /= normalization[valid]
        signal[~valid] = 0.0

        segment_start = self._padded_samples_emitted
        crop_offset = max(self._left_padding - segment_start, 0)
        ready = signal[crop_offset:]
        samples_remaining = self._input_samples - self._output_samples
        ready = ready[: max(samples_remaining, 0)]
        self._padded_samples_emitted += hop
        self._output_samples += ready.size

        self._ola_signal[:-hop] = self._ola_signal[hop:]
        self._ola_signal[-hop:] = 0.0
        self._ola_normalization[:-hop] = self._ola_normalization[hop:]
        self._ola_normalization[-hop:] = 0.0
        return ready
