"""Voice activity detection implementations."""

from speech_frontend.vad.energy import (
    EnergyVAD,
    EnergyVADConfig,
    VADResult,
)
from speech_frontend.vad.external import WebRTCVAD, WebRTCVADConfig
from speech_frontend.vad.metrics import BinaryVADMetrics, binary_metrics, labels_from_intervals
from speech_frontend.vad.synthetic import VADMixture, VADMixtureConfig, create_vad_mixture
from speech_frontend.vad.statistical import StatisticalVAD, StatisticalVADConfig

__all__ = [
    "BinaryVADMetrics",
    "EnergyVAD",
    "EnergyVADConfig",
    "StatisticalVAD",
    "StatisticalVADConfig",
    "VADMixture",
    "VADMixtureConfig",
    "VADResult",
    "WebRTCVAD",
    "WebRTCVADConfig",
    "binary_metrics",
    "create_vad_mixture",
    "labels_from_intervals",
]
