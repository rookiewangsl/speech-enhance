"""Frozen Raw/S-WPE/M-WPE frontend conditions."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .wpe import WPEBackend, WPEConfig, offline_wpe_waveform

FrontendName = Literal["raw", "s_wpe_10", "s_wpe_40", "m_wpe_10"]


def apply_frontend(
    multichannel: NDArray[np.floating],
    condition: FrontendName,
    *,
    backend: WPEBackend = "nara_wpe",
    reference_channel: int = 0,
) -> NDArray[np.float32]:
    """Return the mono signal presented to ASR for one frozen condition."""

    signals = np.asarray(multichannel, dtype=np.float64)
    if signals.ndim != 2 or signals.shape[0] == 0 or signals.shape[1] == 0:
        raise ValueError("multichannel must have shape (channels, samples)")
    if not 0 <= reference_channel < signals.shape[0]:
        raise ValueError("reference_channel is out of range")
    if not np.all(np.isfinite(signals)):
        raise ValueError("multichannel contains NaN or infinite values")
    if condition == "raw":
        output = signals[reference_channel]
    elif condition in {"s_wpe_10", "s_wpe_40"}:
        taps = 10 if condition == "s_wpe_10" else 40
        enhanced = offline_wpe_waveform(
            signals[reference_channel : reference_channel + 1],
            WPEConfig(taps=taps),
            backend=backend,
        )
        output = enhanced[0]
    elif condition == "m_wpe_10":
        enhanced = offline_wpe_waveform(
            signals,
            WPEConfig(taps=10),
            backend=backend,
        )
        output = enhanced[reference_channel]
    else:
        raise ValueError(f"unknown frontend condition: {condition}")
    if not np.all(np.isfinite(output)):
        raise FloatingPointError(f"frontend {condition} produced non-finite audio")
    return np.asarray(output, dtype=np.float32)
