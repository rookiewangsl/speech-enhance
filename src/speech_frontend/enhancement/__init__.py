"""Classical single-channel spectral enhancement algorithms."""

from speech_frontend.enhancement.spectral_subtraction import (
    oracle_wiener_spectrum,
    spectral_subtraction,
)
from speech_frontend.enhancement.om_lsa import OMLSA, OMLSAConfig
from speech_frontend.enhancement.wiener import (
    DecisionDirectedWiener,
    DualUncertaintyWiener,
    WienerConfig,
    instantaneous_wiener_gain,
)

__all__ = [
    "DecisionDirectedWiener",
    "DualUncertaintyWiener",
    "WienerConfig",
    "instantaneous_wiener_gain",
    "OMLSA",
    "OMLSAConfig",
    "oracle_wiener_spectrum",
    "spectral_subtraction",
]
