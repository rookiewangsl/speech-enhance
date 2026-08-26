"""RIR validation, multichannel convolution, common gain, and DRR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve

FloatArray = NDArray[np.floating]


def rt60_within_tolerance(measured: float, target: float) -> bool:
    """Apply the frozen max(50 ms, 10%) RT60 acceptance rule."""

    if not np.isfinite(measured) or not np.isfinite(target):
        raise ValueError("RT60 values must be finite")
    if measured <= 0 or target <= 0:
        raise ValueError("RT60 values must be positive")
    tolerance = max(0.05, 0.1 * target)
    return abs(measured - target) <= tolerance


def direct_to_reverberant_ratio(
    full_rirs: FloatArray,
    direct_rirs: FloatArray,
    *,
    epsilon: float = 1e-12,
) -> NDArray[np.float64]:
    """Compute per-channel oracle DRR from direct-only and full RIRs."""

    full = np.asarray(full_rirs, dtype=np.float64)
    direct = np.asarray(direct_rirs, dtype=np.float64)
    if full.ndim != 2 or direct.ndim != 2:
        raise ValueError("RIR arrays must have shape (channels, samples)")
    if full.shape[0] != direct.shape[0]:
        raise ValueError("full and direct RIRs must have the same channel count")
    if not np.all(np.isfinite(full)) or not np.all(np.isfinite(direct)):
        raise ValueError("RIR arrays contain NaN or infinite values")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    length = max(full.shape[1], direct.shape[1])
    full_padded = np.pad(full, ((0, 0), (0, length - full.shape[1])))
    direct_padded = np.pad(
        direct, ((0, 0), (0, length - direct.shape[1]))
    )
    reverberant = full_padded - direct_padded
    direct_energy = np.sum(direct_padded**2, axis=1)
    reverberant_energy = np.sum(reverberant**2, axis=1)
    return 10.0 * np.log10(
        np.maximum(direct_energy, epsilon)
        / np.maximum(reverberant_energy, epsilon)
    )


@dataclass(frozen=True)
class ConvolutionResult:
    """Four synchronized microphone signals and their common scaling."""

    signals: NDArray[np.float64]
    common_gain: float
    peak_guard_gain: float
    reference_rms_dbfs: float


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


def convolve_multichannel(
    clean: FloatArray,
    rirs: FloatArray,
    *,
    reference_channel: int = 0,
    target_rms_dbfs: float = -25.0,
    peak_headroom_db: float = 1.0,
) -> ConvolutionResult:
    """Convolve mono speech with all RIRs and apply one shared gain."""

    source = np.asarray(clean, dtype=np.float64)
    filters = np.asarray(rirs, dtype=np.float64)
    if source.ndim != 1 or source.size == 0:
        raise ValueError("clean speech must be a non-empty mono signal")
    if filters.ndim != 2 or filters.shape[0] == 0 or filters.shape[1] == 0:
        raise ValueError("rirs must have shape (channels, samples)")
    if not 0 <= reference_channel < filters.shape[0]:
        raise ValueError("reference_channel is out of range")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(filters)):
        raise ValueError("speech and RIRs must contain only finite values")
    if not np.isfinite(target_rms_dbfs) or target_rms_dbfs >= 0:
        raise ValueError("target_rms_dbfs must be finite and below 0 dBFS")
    if not np.isfinite(peak_headroom_db) or peak_headroom_db < 0:
        raise ValueError("peak_headroom_db must be finite and non-negative")

    signals = np.stack(
        [
            fftconvolve(source, impulse_response, mode="full")
            for impulse_response in filters
        ]
    )
    reference_rms = _rms(signals[reference_channel])
    if reference_rms <= np.finfo(np.float64).tiny:
        raise ValueError("reference channel is silent after convolution")
    target_rms = 10.0 ** (target_rms_dbfs / 20.0)
    common_gain = target_rms / reference_rms
    signals *= common_gain

    peak_limit = 10.0 ** (-peak_headroom_db / 20.0)
    peak = float(np.max(np.abs(signals)))
    peak_guard_gain = 1.0 if peak <= peak_limit else peak_limit / peak
    signals *= peak_guard_gain
    final_rms = _rms(signals[reference_channel])
    final_rms_dbfs = 20.0 * np.log10(max(final_rms, np.finfo(float).tiny))
    return ConvolutionResult(
        signals=signals,
        common_gain=float(common_gain),
        peak_guard_gain=float(peak_guard_gain),
        reference_rms_dbfs=float(final_rms_dbfs),
    )
