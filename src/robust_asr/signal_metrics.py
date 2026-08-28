"""Reference-based speech metrics for controlled dereverberation analysis."""

from __future__ import annotations

import math

import numpy as np


def match_signal_lengths(
    reference: np.ndarray, estimate: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad two mono signals to a shared length without time shifting."""

    target = np.asarray(reference, dtype=np.float64)
    output = np.asarray(estimate, dtype=np.float64)
    for name, values in (("reference", target), ("estimate", output)):
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"{name} must be non-empty mono audio")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or infinity")
    length = max(target.size, output.size)
    if target.size < length:
        target = np.pad(target, (0, length - target.size))
    if output.size < length:
        output = np.pad(output, (0, length - output.size))
    return target, output


def scale_invariant_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Return zero-mean SI-SDR in dB using the direct-path target."""

    target, output = match_signal_lengths(reference, estimate)
    target = target - np.mean(target)
    output = output - np.mean(output)
    target_energy = float(np.dot(target, target))
    if target_energy <= np.finfo(np.float64).tiny:
        raise ValueError("reference energy is zero")
    projection = float(np.dot(output, target)) / target_energy * target
    residual = output - projection
    projected_energy = float(np.dot(projection, projection))
    residual_energy = float(np.dot(residual, residual))
    floor = np.finfo(np.float64).eps * target_energy
    value = 10.0 * math.log10(
        max(projected_energy, floor) / max(residual_energy, floor)
    )
    if not math.isfinite(value):
        raise FloatingPointError("SI-SDR is non-finite")
    return value


def stoi_score(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    sample_rate: int = 16_000,
) -> float:
    """Return classical STOI after deterministic length matching."""

    if sample_rate != 16_000:
        raise ValueError("the frozen signal-metric protocol requires 16 kHz")
    try:
        from pystoi import stoi
    except ImportError as exc:  # pragma: no cover - optional evaluation stack
        raise RuntimeError("STOI evaluation requires pystoi") from exc
    target, output = match_signal_lengths(reference, estimate)
    value = float(stoi(target, output, sample_rate, extended=False))
    if not math.isfinite(value):
        raise FloatingPointError("STOI is non-finite")
    return value
