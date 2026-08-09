"""Arbitrary-chunk streaming around RNNoise's fixed 480-sample API."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from speech_frontend.audio import validate_mono_audio
from speech_frontend.rnnoise.backend import (
    RNNOISE_FRAME_SAMPLES,
    RNNOISE_SAMPLE_RATE,
    RNNoiseLibrary,
    RNNoiseState,
)
from speech_frontend.rnnoise.resampler import (
    RESAMPLER_GROUP_DELAY_48K,
    StreamingDownsampler3,
    StreamingUpsampler3,
)

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class RNNoiseChunkResult:
    """Audio emitted now and VAD values used for those delayed frames."""

    samples: NDArray[np.float32]
    vad_probabilities: NDArray[np.float32]


class StreamingRNNoise48k:
    """Preserve RNNoise state across arbitrary normalized 48 kHz chunks.

    RNNoise emits the previous frame when processing the current frame. The
    first C output is therefore discarded, matching the official demo. At end
    of stream, samples for which no future frame exists are zero-filled so the
    complete file output has exactly the input length.
    """

    def __init__(
        self,
        library: RNNoiseLibrary | None = None,
        *,
        pcm_compatible: bool = False,
    ) -> None:
        self.library = library or RNNoiseLibrary()
        self.pcm_compatible = pcm_compatible
        self._state: RNNoiseState | None = None
        self._input_buffer = np.empty(0, dtype=np.float32)
        self._primed = False
        self._flushed = False
        self._total_input_samples = 0
        self._total_output_samples = 0
        self._create_state()

    @property
    def total_input_samples(self) -> int:
        return self._total_input_samples

    @property
    def total_output_samples(self) -> int:
        return self._total_output_samples

    @property
    def algorithmic_delay_samples(self) -> int:
        return RNNOISE_FRAME_SAMPLES

    @property
    def alignment_delay_samples(self) -> int:
        """Fixed waveform lag measured against the original input timeline."""

        return RNNOISE_FRAME_SAMPLES

    def _create_state(self) -> None:
        self._state = self.library.create_state()

    def _require_active(self) -> RNNoiseState:
        if self._flushed:
            raise RuntimeError("cannot process audio after flush")
        if self._state is None:
            raise RuntimeError("RNNoise stream is closed")
        return self._state

    @staticmethod
    def _validate_chunk(chunk: FloatArray) -> NDArray[np.float32]:
        validate_mono_audio(chunk, RNNOISE_SAMPLE_RATE)
        if np.max(np.abs(chunk), initial=0.0) > 1.0:
            raise ValueError("RNNoise normalized input must be within [-1, 1]")
        return np.asarray(chunk, dtype=np.float32)

    def _process_available_frames(self) -> RNNoiseChunkResult:
        state = self._require_active()
        outputs: list[NDArray[np.float32]] = []
        vad_probabilities: list[float] = []

        while self._input_buffer.size >= RNNOISE_FRAME_SAMPLES:
            frame = self._input_buffer[:RNNOISE_FRAME_SAMPLES]
            self._input_buffer = self._input_buffer[RNNOISE_FRAME_SAMPLES:]
            output, probability = state.process_frame(
                frame,
                pcm_compatible=self.pcm_compatible,
            )
            if self._primed:
                outputs.append(output)
                vad_probabilities.append(probability)
            else:
                self._primed = True

        if outputs:
            samples = np.concatenate(outputs)
        else:
            samples = np.empty(0, dtype=np.float32)
        self._total_output_samples += samples.size
        return RNNoiseChunkResult(
            samples=samples,
            vad_probabilities=np.asarray(
                vad_probabilities,
                dtype=np.float32,
            ),
        )

    def process_chunk(self, chunk: FloatArray) -> RNNoiseChunkResult:
        """Consume an arbitrary chunk and emit all newly available frames."""

        self._require_active()
        normalized = self._validate_chunk(chunk)
        self._total_input_samples += normalized.size
        if normalized.size:
            self._input_buffer = np.concatenate(
                (self._input_buffer, normalized)
            )
        return self._process_available_frames()

    def flush(self) -> RNNoiseChunkResult:
        """Finish the stream, zero-filling the unavailable delayed tail."""

        if self._flushed:
            return RNNoiseChunkResult(
                samples=np.empty(0, dtype=np.float32),
                vad_probabilities=np.empty(0, dtype=np.float32),
            )
        self._require_active()
        outputs: list[NDArray[np.float32]] = []
        vad: list[NDArray[np.float32]] = []
        if self._input_buffer.size:
            padded = np.pad(
                self._input_buffer,
                (0, RNNOISE_FRAME_SAMPLES - self._input_buffer.size),
            )
            self._input_buffer = padded.astype(np.float32, copy=False)
            available = self._process_available_frames()
            outputs.append(available.samples)
            vad.append(available.vad_probabilities)

        missing = self._total_input_samples - self._total_output_samples
        if missing < 0:
            raise RuntimeError("RNNoise stream emitted more samples than input")
        if missing:
            outputs.append(np.zeros(missing, dtype=np.float32))
            self._total_output_samples += missing

        self._flushed = True
        if self._state is not None:
            self._state.close()
            self._state = None
        samples = (
            np.concatenate(outputs)
            if outputs
            else np.empty(0, dtype=np.float32)
        )
        probabilities = (
            np.concatenate(vad)
            if vad
            else np.empty(0, dtype=np.float32)
        )
        return RNNoiseChunkResult(samples, probabilities)

    def reset(self) -> None:
        """Discard buffered audio and recurrent state for a new stream."""

        if self._state is not None:
            self._state.close()
        self._input_buffer = np.empty(0, dtype=np.float32)
        self._primed = False
        self._flushed = False
        self._total_input_samples = 0
        self._total_output_samples = 0
        self._create_state()

    def close(self) -> None:
        """Release C state without synthesizing or padding the tail."""

        if self._state is not None:
            self._state.close()
            self._state = None

    def __enter__(self) -> StreamingRNNoise48k:
        self._require_active()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class StreamingRNNoise16k:
    """Stateful 16 kHz adapter around the native 48 kHz RNNoise stream."""

    def __init__(
        self,
        library: RNNoiseLibrary | None = None,
        *,
        pcm_compatible: bool = False,
    ) -> None:
        self.upsampler = StreamingUpsampler3()
        self.rnnoise = StreamingRNNoise48k(
            library,
            pcm_compatible=pcm_compatible,
        )
        self.downsampler = StreamingDownsampler3()
        self._flushed = False
        self._total_input_samples = 0
        self._total_output_samples = 0
        self.resampler_clipping_samples = 0

    @property
    def total_input_samples(self) -> int:
        return self._total_input_samples

    @property
    def total_output_samples(self) -> int:
        return self._total_output_samples

    @property
    def algorithmic_delay_samples(self) -> int:
        core_delay_16k = RNNOISE_FRAME_SAMPLES // 3
        resampler_delay_16k = (
            2 * RESAMPLER_GROUP_DELAY_48K
        ) // 3
        return core_delay_16k + resampler_delay_16k

    @property
    def alignment_delay_samples(self) -> int:
        """Fixed waveform lag measured against the original input timeline."""

        return self.algorithmic_delay_samples

    def process_chunk(self, chunk: FloatArray) -> RNNoiseChunkResult:
        if self._flushed:
            raise RuntimeError("cannot process audio after flush")
        validate_mono_audio(chunk, 16_000)
        if np.max(np.abs(chunk), initial=0.0) > 1.0:
            raise ValueError("RNNoise normalized input must be within [-1, 1]")
        self._total_input_samples += chunk.size
        upsampled = self.upsampler.process_chunk(chunk)
        clipped = np.count_nonzero(np.abs(upsampled) > 1.0)
        self.resampler_clipping_samples += int(clipped)
        if clipped:
            upsampled = np.clip(upsampled, -1.0, 1.0)
        enhanced = self.rnnoise.process_chunk(upsampled)
        output = self.downsampler.process_chunk(enhanced.samples)
        self._total_output_samples += output.size
        return RNNoiseChunkResult(output, enhanced.vad_probabilities)

    def flush(self) -> RNNoiseChunkResult:
        if self._flushed:
            return RNNoiseChunkResult(
                samples=np.empty(0, dtype=np.float32),
                vad_probabilities=np.empty(0, dtype=np.float32),
            )
        enhanced = self.rnnoise.flush()
        output = self.downsampler.process_chunk(enhanced.samples)
        self._total_output_samples += output.size
        if self._total_output_samples != self._total_input_samples:
            raise RuntimeError(
                "16 kHz RNNoise length mismatch: "
                f"{self._total_output_samples} output samples for "
                f"{self._total_input_samples} input samples"
            )
        self._flushed = True
        return RNNoiseChunkResult(output, enhanced.vad_probabilities)

    def reset(self) -> None:
        self.upsampler.reset()
        self.rnnoise.reset()
        self.downsampler.reset()
        self._flushed = False
        self._total_input_samples = 0
        self._total_output_samples = 0
        self.resampler_clipping_samples = 0

    def close(self) -> None:
        self.rnnoise.close()

    def __enter__(self) -> StreamingRNNoise16k:
        if self._flushed:
            raise RuntimeError("RNNoise stream has been flushed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
