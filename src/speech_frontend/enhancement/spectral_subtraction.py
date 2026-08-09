"""Spectral-subtraction and oracle-Wiener diagnostic baselines."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complexfloating]
FloatArray = NDArray[np.floating]


def _validate_spectrum_and_psd(
    spectrum: ComplexArray,
    noise_psd: FloatArray,
) -> tuple[ComplexArray, FloatArray]:
    spectrum = np.asarray(spectrum)
    noise_psd = np.asarray(noise_psd, dtype=np.float64)
    if spectrum.shape != noise_psd.shape:
        raise ValueError("spectrum and noise_psd shapes must match")
    if spectrum.ndim not in (1, 2):
        raise ValueError("spectrum must be frame- or batch-major")
    if not np.all(np.isfinite(spectrum)):
        raise ValueError("spectrum contains NaN or infinite values")
    if not np.all(np.isfinite(noise_psd)) or np.any(noise_psd < 0.0):
        raise ValueError("noise_psd must be finite and non-negative")
    return spectrum, noise_psd


def spectral_subtraction(
    noisy_spectrum: ComplexArray,
    noise_psd: FloatArray,
    *,
    over_subtraction: float = 1.0,
    gain_floor: float = 0.05,
    epsilon: float = 1e-12,
) -> ComplexArray:
    """Subtract estimated noise power while preserving noisy phase."""

    spectrum, noise = _validate_spectrum_and_psd(
        noisy_spectrum,
        noise_psd,
    )
    if over_subtraction < 0.0:
        raise ValueError("over_subtraction cannot be negative")
    if not 0.0 <= gain_floor <= 1.0:
        raise ValueError("gain_floor must be in [0, 1]")

    noisy_power = np.abs(spectrum) ** 2
    estimated_power = np.maximum(
        noisy_power - over_subtraction * noise,
        (gain_floor**2) * noisy_power,
    )
    gain = np.sqrt(estimated_power / (noisy_power + epsilon))
    gain = np.clip(gain, gain_floor, 1.0)
    return gain * spectrum


def oracle_wiener_spectrum(
    noisy_spectrum: ComplexArray,
    clean_spectrum: ComplexArray,
    *,
    epsilon: float = 1e-12,
) -> ComplexArray:
    """Apply an ideal ratio mask derived from aligned clean/noisy spectra."""

    noisy = np.asarray(noisy_spectrum)
    clean = np.asarray(clean_spectrum)
    if noisy.shape != clean.shape:
        raise ValueError("clean and noisy spectrum shapes must match")
    if noisy.ndim not in (1, 2):
        raise ValueError("spectra must be frame- or batch-major")
    if not np.all(np.isfinite(noisy)) or not np.all(np.isfinite(clean)):
        raise ValueError("spectra contain NaN or infinite values")

    speech_psd = np.abs(clean) ** 2
    noise_psd = np.abs(noisy - clean) ** 2
    gain = speech_psd / (speech_psd + noise_psd + epsilon)
    return gain * noisy
