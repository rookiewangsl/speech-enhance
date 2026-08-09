"""Bindings and streaming utilities for the external RNNoise library."""

from speech_frontend.rnnoise.backend import (
    RNNOISE_FRAME_SAMPLES,
    RNNOISE_SAMPLE_RATE,
    RNNoiseLibrary,
    RNNoiseResult,
    RNNoiseState,
)
from speech_frontend.rnnoise.continuity import (
    ContinuityConfig,
    ContinuityResult,
    stabilize_output_continuity,
)
from speech_frontend.rnnoise.resampler import (
    RESAMPLER_GROUP_DELAY_48K,
    StreamingDownsampler3,
    StreamingUpsampler3,
)
from speech_frontend.rnnoise.streaming import (
    RNNoiseChunkResult,
    StreamingRNNoise16k,
    StreamingRNNoise48k,
)

__all__ = [
    "RNNOISE_FRAME_SAMPLES",
    "RNNOISE_SAMPLE_RATE",
    "RNNoiseLibrary",
    "RNNoiseResult",
    "RNNoiseState",
    "ContinuityConfig",
    "ContinuityResult",
    "RNNoiseChunkResult",
    "RESAMPLER_GROUP_DELAY_48K",
    "StreamingDownsampler3",
    "StreamingRNNoise16k",
    "StreamingRNNoise48k",
    "StreamingUpsampler3",
    "stabilize_output_continuity",
]
