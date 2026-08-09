"""Small, dependency-light intrusive metrics for paired experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def si_sdr(reference: NDArray[np.floating], estimate: NDArray[np.floating]) -> float:
    """Return scale-invariant SDR in dB for aligned one-dimensional signals."""

    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.ndim != 1 or estimate.ndim != 1:
        raise ValueError("reference and estimate must be one-dimensional")
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate lengths do not match")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(estimate)):
        raise ValueError("reference and estimate must be finite")
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    reference_energy = np.dot(reference, reference)
    if reference_energy <= 1e-12:
        raise ValueError("reference energy is too small for SI-SDR")
    target = np.dot(estimate, reference) / reference_energy * reference
    residual = estimate - target
    return float(
        10.0
        * np.log10(
            (np.dot(target, target) + 1e-12)
            / (np.dot(residual, residual) + 1e-12)
        )
    )


def stoi(
    reference: NDArray[np.floating],
    estimate: NDArray[np.floating],
    *,
    sample_rate: int,
) -> float:
    """Return short-time objective intelligibility for aligned speech signals."""

    try:
        from pystoi import stoi as pystoi_score
    except ImportError as error:
        raise RuntimeError(
            "STOI requires the optional evaluation dependencies; install '.[evaluation]'"
        ) from error
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.ndim != 1 or estimate.ndim != 1:
        raise ValueError("reference and estimate must be one-dimensional")
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate lengths do not match")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(estimate)):
        raise ValueError("reference and estimate must be finite")
    return float(pystoi_score(reference, estimate, sample_rate, extended=False))
