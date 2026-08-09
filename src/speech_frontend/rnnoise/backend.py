"""Stateful Python bindings for the official RNNoise C API."""

from __future__ import annotations

import ctypes
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from speech_frontend.audio import validate_mono_audio

RNNOISE_SAMPLE_RATE = 48_000
RNNOISE_FRAME_SAMPLES = 480
PCM_SCALE = 32_767.0

FloatArray = NDArray[np.floating]


def default_library_path() -> Path:
    """Return the expected output of ``scripts/setup_rnnoise.sh``."""

    project_root = Path(__file__).resolve().parents[3]
    system = platform.system()
    if system == "Darwin":
        library_name = "librnnoise.dylib"
    elif system == "Linux":
        library_name = "librnnoise.so"
    else:
        raise RuntimeError(f"unsupported RNNoise platform: {system}")
    return project_root / "build" / "rnnoise" / "lib" / library_name


@dataclass(frozen=True)
class RNNoiseResult:
    """Enhanced samples and one VAD probability per processed frame."""

    samples: NDArray[np.float32]
    vad_probabilities: NDArray[np.float32]
    padding_samples: int


class RNNoiseLibrary:
    """Loaded official RNNoise shared library with declared C signatures."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_library_path()
        if not self.path.is_file():
            raise FileNotFoundError(
                f"RNNoise library not found: {self.path}. "
                "Run scripts/setup_rnnoise.sh first."
            )

        self._library = ctypes.CDLL(str(self.path))
        self._library.rnnoise_get_frame_size.argtypes = []
        self._library.rnnoise_get_frame_size.restype = ctypes.c_int
        self._library.rnnoise_create.argtypes = [ctypes.c_void_p]
        self._library.rnnoise_create.restype = ctypes.c_void_p
        self._library.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        self._library.rnnoise_destroy.restype = None
        float_pointer = ctypes.POINTER(ctypes.c_float)
        self._library.rnnoise_process_frame.argtypes = [
            ctypes.c_void_p,
            float_pointer,
            float_pointer,
        ]
        self._library.rnnoise_process_frame.restype = ctypes.c_float

        frame_size = self.frame_size
        if frame_size != RNNOISE_FRAME_SAMPLES:
            raise RuntimeError(
                "unexpected RNNoise frame size: "
                f"{frame_size} != {RNNOISE_FRAME_SAMPLES}"
            )

    @property
    def frame_size(self) -> int:
        """Number of 48 kHz samples processed by one C API call."""

        return int(self._library.rnnoise_get_frame_size())

    def create_state(self) -> RNNoiseState:
        """Allocate a state that must persist across consecutive frames."""

        return RNNoiseState(self)


class RNNoiseState:
    """Owned RNNoise ``DenoiseState`` and frame processing contract."""

    def __init__(self, library: RNNoiseLibrary) -> None:
        self.library = library
        self._state: int | None = None
        self._create()

    @property
    def is_closed(self) -> bool:
        """Whether the underlying C state has already been destroyed."""

        return self._state is None

    def _create(self) -> None:
        state = self.library._library.rnnoise_create(None)
        if not state:
            raise MemoryError("rnnoise_create returned a null pointer")
        self._state = int(state)

    def _require_open(self) -> int:
        if self._state is None:
            raise RuntimeError("RNNoise state is closed")
        return self._state

    @staticmethod
    def _validate_normalized_frame(frame: FloatArray) -> None:
        validate_mono_audio(frame, RNNOISE_SAMPLE_RATE)
        if frame.shape != (RNNOISE_FRAME_SAMPLES,):
            raise ValueError(
                "RNNoise frame must contain exactly "
                f"{RNNOISE_FRAME_SAMPLES} samples"
            )
        if np.max(np.abs(frame), initial=0.0) > 1.0:
            raise ValueError("RNNoise normalized input must be within [-1, 1]")

    def process_frame(
        self,
        frame: FloatArray,
        *,
        pcm_compatible: bool = False,
    ) -> tuple[NDArray[np.float32], float]:
        """Enhance one normalized 10 ms frame.

        The public RNNoise API uses floating-point buffers whose magnitude is
        that of 16-bit PCM. The project boundary is normalized ``[-1, 1]``
        audio, so the conversion is explicit. ``pcm_compatible`` additionally
        quantizes the output like the official raw-PCM demo for equivalence
        testing.
        """

        self._validate_normalized_frame(frame)
        state = self._require_open()
        input_pcm = (
            np.rint(np.asarray(frame, dtype=np.float64) * PCM_SCALE)
            .astype(np.int16)
            .astype(np.float32)
        )
        output_pcm = np.empty(RNNOISE_FRAME_SAMPLES, dtype=np.float32)
        float_pointer = ctypes.POINTER(ctypes.c_float)
        vad_probability = self.library._library.rnnoise_process_frame(
            ctypes.c_void_p(state),
            output_pcm.ctypes.data_as(float_pointer),
            input_pcm.ctypes.data_as(float_pointer),
        )

        if pcm_compatible:
            output_pcm = (
                np.clip(output_pcm, -32_768.0, 32_767.0)
                .astype(np.int16)
                .astype(np.float32)
            )
        output = output_pcm / np.float32(PCM_SCALE)
        if not np.all(np.isfinite(output)):
            raise RuntimeError("RNNoise produced non-finite output")
        if not 0.0 <= vad_probability <= 1.0:
            raise RuntimeError(
                f"RNNoise returned invalid VAD probability: {vad_probability}"
            )
        return output, float(vad_probability)

    def process_audio(
        self,
        samples: FloatArray,
        *,
        pcm_compatible: bool = False,
    ) -> RNNoiseResult:
        """Enhance a complete 48 kHz signal through consecutive C frames."""

        validate_mono_audio(samples, RNNOISE_SAMPLE_RATE)
        if np.max(np.abs(samples), initial=0.0) > 1.0:
            raise ValueError("RNNoise normalized input must be within [-1, 1]")
        original_length = samples.size
        padding_samples = (-original_length) % RNNOISE_FRAME_SAMPLES
        padded = np.pad(np.asarray(samples), (0, padding_samples))
        output = np.empty(padded.size, dtype=np.float32)
        vad_probabilities = np.empty(
            padded.size // RNNOISE_FRAME_SAMPLES,
            dtype=np.float32,
        )

        for index, offset in enumerate(
            range(0, padded.size, RNNOISE_FRAME_SAMPLES)
        ):
            output_frame, vad_probability = self.process_frame(
                padded[offset : offset + RNNOISE_FRAME_SAMPLES],
                pcm_compatible=pcm_compatible,
            )
            output[offset : offset + RNNOISE_FRAME_SAMPLES] = output_frame
            vad_probabilities[index] = vad_probability

        return RNNoiseResult(
            samples=output[:original_length],
            vad_probabilities=vad_probabilities,
            padding_samples=padding_samples,
        )

    def reset(self) -> None:
        """Destroy all recurrent history and create a fresh state."""

        self.close()
        self._create()

    def close(self) -> None:
        """Release the owned C state; repeated calls are safe."""

        if self._state is not None:
            self.library._library.rnnoise_destroy(
                ctypes.c_void_p(self._state)
            )
            self._state = None

    def __enter__(self) -> RNNoiseState:
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, TypeError):
            pass
