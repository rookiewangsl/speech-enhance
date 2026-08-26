"""Offline WPE protocol and a NumPy reference backend.

The formal experiment is configured to use NARA-WPE. The local NumPy backend
exists for dependency-free mathematical smoke tests and is never labelled as a
formal NARA-WPE result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.signal import istft, stft

ComplexArray = NDArray[np.complexfloating]
FloatArray = NDArray[np.floating]
WPEBackend = Literal["numpy_reference", "nara_wpe"]


@dataclass(frozen=True)
class WPEConfig:
    """Parameters shared by the reference and formal NARA-WPE backends."""

    sample_rate: int = 16_000
    n_fft: int = 512
    win_length: int = 512
    hop_length: int = 128
    delay: int = 3
    taps: int = 10
    iterations: int = 3
    psd_context: int = 0
    statistics_mode: Literal["full", "valid"] = "full"
    diagonal_loading: float = 1e-6
    power_floor: float = 1e-10

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.win_length <= 0 or self.n_fft < self.win_length:
            raise ValueError("n_fft must be at least the positive win_length")
        if not 0 < self.hop_length < self.win_length:
            raise ValueError("hop_length must be in [1, win_length)")
        if self.delay <= 0:
            raise ValueError("delay must be positive")
        if self.taps <= 0:
            raise ValueError("taps must be positive")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.psd_context != 0:
            raise ValueError("v0.1 reference backend supports psd_context=0")
        if self.statistics_mode not in {"full", "valid"}:
            raise ValueError("statistics_mode must be 'full' or 'valid'")
        if self.diagonal_loading < 0 or not np.isfinite(self.diagonal_loading):
            raise ValueError("diagonal_loading must be finite and non-negative")
        if self.power_floor <= 0 or not np.isfinite(self.power_floor):
            raise ValueError("power_floor must be finite and positive")

def build_delayed_history(
    spectrum: ComplexArray,
    *,
    taps: int,
    delay: int,
) -> NDArray[np.complex128]:
    """Stack delayed `(frequency, channel, time)` observations."""

    observation = np.asarray(spectrum, dtype=np.complex128)
    if observation.ndim != 3:
        raise ValueError("spectrum must have shape (frequency, channel, time)")
    if taps <= 0 or delay <= 0:
        raise ValueError("taps and delay must be positive")
    frequencies, channels, frames = observation.shape
    history = np.zeros(
        (frequencies, channels * taps, frames), dtype=np.complex128
    )
    for tap in range(taps):
        lag = delay + tap
        if lag >= frames:
            continue
        start = tap * channels
        history[:, start : start + channels, lag:] = observation[:, :, :-lag]
    return history


def _stable_solve(
    correlation: np.ndarray,
    covariance: np.ndarray,
    *,
    diagonal_loading: float,
) -> np.ndarray:
    dimension = correlation.shape[0]
    scale = float(np.trace(correlation).real / max(dimension, 1))
    loading = diagonal_loading * max(scale, np.finfo(np.float64).eps)
    regularized = correlation + loading * np.eye(
        dimension, dtype=correlation.dtype
    )
    try:
        return np.linalg.solve(regularized, covariance)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(regularized, covariance, rcond=None)[0]


def _numpy_reference_wpe(
    spectrum: ComplexArray,
    config: WPEConfig,
) -> NDArray[np.complex128]:
    observation = np.asarray(spectrum, dtype=np.complex128)
    if observation.ndim != 3:
        raise ValueError("spectrum must have shape (frequency, channel, time)")
    if not np.all(np.isfinite(observation)):
        raise ValueError("spectrum contains NaN or infinite values")
    _, _, frames = observation.shape
    minimum_frames = config.delay + config.taps
    if frames <= minimum_frames:
        raise ValueError(
            f"WPE requires more than {minimum_frames} STFT frames; got {frames}"
        )

    history = build_delayed_history(
        observation, taps=config.taps, delay=config.delay
    )
    estimate = observation.copy()
    valid_start = config.delay + config.taps - 1
    selection = slice(None) if config.statistics_mode == "full" else slice(
        valid_start, None
    )

    for _ in range(config.iterations):
        for frequency in range(observation.shape[0]):
            current = estimate[frequency]
            power = np.mean(np.abs(current) ** 2, axis=0)
            inverse_power = 1.0 / np.maximum(power, config.power_floor)
            delayed = history[frequency]
            delayed_selected = delayed[:, selection]
            inverse_selected = inverse_power[selection]
            observation_selected = observation[frequency, :, selection]
            weighted_delayed = delayed_selected * inverse_selected[np.newaxis, :]
            correlation = weighted_delayed @ delayed_selected.conj().T
            covariance = weighted_delayed @ observation_selected.conj().T
            filters = _stable_solve(
                correlation,
                covariance,
                diagonal_loading=config.diagonal_loading,
            )
            prediction = filters.conj().T @ delayed
            estimate[frequency] = observation[frequency] - prediction
    if not np.all(np.isfinite(estimate)):
        raise FloatingPointError("WPE produced NaN or infinite values")
    return estimate


def analyze_multichannel(
    signals: FloatArray,
    config: WPEConfig,
) -> NDArray[np.complex128]:
    """Use the formal periodic Hann analysis for synchronized channels."""

    waveforms = np.asarray(signals, dtype=np.float64)
    if waveforms.ndim != 2 or waveforms.shape[0] == 0:
        raise ValueError("signals must have shape (channels, samples)")
    _, _, spectrum = stft(
        waveforms,
        fs=config.sample_rate,
        window="hann",
        nperseg=config.win_length,
        noverlap=config.win_length - config.hop_length,
        nfft=config.n_fft,
        detrend=False,
        return_onesided=True,
        boundary="zeros",
        padded=True,
        axis=-1,
    )
    # SciPy returns (channel, frequency, time).
    return np.asarray(spectrum.transpose(1, 0, 2), dtype=np.complex128)


def synthesize_multichannel(
    spectrum: ComplexArray,
    config: WPEConfig,
    *,
    output_length: int,
) -> NDArray[np.float64]:
    """Invert formal Hann spectra and restore the exact input length."""

    values = np.asarray(spectrum, dtype=np.complex128)
    if values.ndim != 3:
        raise ValueError("spectrum must have shape (frequency, channel, time)")
    if output_length < 0:
        raise ValueError("output_length cannot be negative")
    _, reconstructed = istft(
        values.transpose(1, 0, 2),
        fs=config.sample_rate,
        window="hann",
        nperseg=config.win_length,
        noverlap=config.win_length - config.hop_length,
        nfft=config.n_fft,
        input_onesided=True,
        boundary=True,
        time_axis=-1,
        freq_axis=-2,
    )
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    if reconstructed.shape[-1] < output_length:
        reconstructed = np.pad(
            reconstructed,
            ((0, 0), (0, output_length - reconstructed.shape[-1])),
        )
    return reconstructed[:, :output_length]


def _nara_wpe(
    spectrum: ComplexArray,
    config: WPEConfig,
) -> NDArray[np.complex128]:
    try:
        from nara_wpe.wpe import wpe
    except ImportError as exc:  # pragma: no cover - installed with data phase
        raise RuntimeError(
            "formal WPE requires the optional nara_wpe dependency"
        ) from exc
    output = wpe(
        np.asarray(spectrum, dtype=np.complex128),
        taps=config.taps,
        delay=config.delay,
        iterations=config.iterations,
        psd_context=config.psd_context,
        statistics_mode=config.statistics_mode,
    )
    output = np.asarray(output, dtype=np.complex128)
    if output.shape != np.asarray(spectrum).shape:
        raise RuntimeError("NARA-WPE returned an unexpected shape")
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("NARA-WPE produced NaN or infinite values")
    return output


def offline_wpe_spectrum(
    spectrum: ComplexArray,
    config: WPEConfig | None = None,
    *,
    backend: WPEBackend = "nara_wpe",
) -> NDArray[np.complex128]:
    """Dereverberate an STFT tensor through an explicit backend."""

    settings = config or WPEConfig()
    if backend == "numpy_reference":
        return _numpy_reference_wpe(spectrum, settings)
    if backend == "nara_wpe":
        return _nara_wpe(spectrum, settings)
    raise ValueError(f"unknown WPE backend: {backend}")


def offline_wpe_waveform(
    signals: FloatArray,
    config: WPEConfig | None = None,
    *,
    backend: WPEBackend = "nara_wpe",
) -> NDArray[np.float64]:
    """Apply offline WPE to synchronized `(channels, samples)` signals."""

    settings = config or WPEConfig()
    waveforms = np.asarray(signals, dtype=np.float64)
    if waveforms.ndim != 2 or waveforms.shape[0] == 0:
        raise ValueError("signals must have shape (channels, samples)")
    if not np.all(np.isfinite(waveforms)):
        raise ValueError("signals contain NaN or infinite values")
    spectrum = analyze_multichannel(waveforms, settings)
    enhanced = offline_wpe_spectrum(spectrum, settings, backend=backend)
    waveforms_out = synthesize_multichannel(
        enhanced,
        settings,
        output_length=waveforms.shape[1],
    )
    if waveforms_out.shape != waveforms.shape:
        raise RuntimeError("WPE waveform output length changed")
    return waveforms_out
