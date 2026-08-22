"""Frozen local DNSMOS P.835 scoring primitives."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly

from speech_frontend.audio import AudioData, validate_mono_audio


FloatArray = NDArray[np.floating]


class InferenceSession(Protocol):
    """Small protocol implemented by onnxruntime.InferenceSession."""

    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, NDArray[np.float32]],
    ) -> list[NDArray[np.float32]]: ...


@dataclass(frozen=True)
class DNSMOSProtocol:
    """Validated inference and calibration settings."""

    protocol_version: str
    sample_rate: int
    input_samples: int
    hop_samples: int
    input_name: str
    output_name: str
    model_sha256: str
    inference_batch_size: int
    calibration: dict[str, tuple[float, ...]]


def canonical_json_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(config: dict[str, Any]) -> DNSMOSProtocol:
    """Parse the frozen JSON protocol and reject inconsistent dimensions."""

    audio = config["audio"]
    model = config["model"]
    sample_rate = int(audio["sample_rate_hz"])
    input_samples = int(audio["input_samples"])
    expected_samples = int(round(float(audio["input_length_seconds"]) * sample_rate))
    if input_samples != expected_samples:
        raise ValueError("DNSMOS input_samples does not match input duration")
    hop_samples = int(round(float(audio["hop_seconds"]) * sample_rate))
    if sample_rate <= 0 or input_samples <= 0 or hop_samples <= 0:
        raise ValueError("DNSMOS audio dimensions must be positive")
    if audio.get("short_clip_policy") != "repeat_by_doubling_until_input_length":
        raise ValueError("unsupported DNSMOS short clip policy")
    if audio.get("amplitude_normalization") is not False:
        raise ValueError("DNSMOS protocol must not normalize audio amplitude")
    calibration = {
        metric: tuple(float(item) for item in config["calibration"][metric])
        for metric in ("sig", "bak", "ovrl")
    }
    if any(len(coefficients) != 3 for coefficients in calibration.values()):
        raise ValueError("DNSMOS regular calibration must use quadratic polynomials")
    inference_batch_size = int(config["runtime"]["inference_batch_size"])
    if inference_batch_size <= 0:
        raise ValueError("DNSMOS inference batch size must be positive")
    return DNSMOSProtocol(
        protocol_version=str(config["protocol_version"]),
        sample_rate=sample_rate,
        input_samples=input_samples,
        hop_samples=hop_samples,
        input_name=str(model["input_name"]),
        output_name=str(model["output_name"]),
        model_sha256=str(model["sha256"]),
        inference_batch_size=inference_batch_size,
        calibration=calibration,
    )


def resample_for_dnsmos(audio: AudioData, target_sample_rate: int) -> FloatArray:
    """Resample without changing level or applying peak normalization."""

    validate_mono_audio(audio.samples, audio.sample_rate)
    if audio.samples.size == 0:
        raise ValueError("DNSMOS audio must not be empty")
    if audio.sample_rate == target_sample_rate:
        return np.asarray(audio.samples, dtype=np.float32)
    common = gcd(audio.sample_rate, target_sample_rate)
    output = resample_poly(
        audio.samples,
        target_sample_rate // common,
        audio.sample_rate // common,
    )
    return np.asarray(output, dtype=np.float32)


def official_windows(
    samples: FloatArray,
    *,
    input_samples: int,
    hop_samples: int,
) -> list[NDArray[np.float32]]:
    """Reproduce the official local script's short-clip repeat and hopping."""

    values = np.asarray(samples, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("DNSMOS samples must be a non-empty mono array")
    if not np.all(np.isfinite(values)):
        raise ValueError("DNSMOS samples contain NaN or infinite values")
    if input_samples <= 0 or hop_samples <= 0:
        raise ValueError("DNSMOS window dimensions must be positive")
    while values.size < input_samples:
        values = np.concatenate((values, values))
    num_hops = (values.size - input_samples) // hop_samples + 1
    windows = [
        values[index * hop_samples : index * hop_samples + input_samples]
        for index in range(num_hops)
    ]
    complete = [window for window in windows if window.size == input_samples]
    if not complete:
        raise RuntimeError("DNSMOS window preparation produced no complete window")
    return complete


def calibrate(raw: float, coefficients: tuple[float, ...]) -> float:
    """Apply the official regular-DNSMOS polynomial calibration."""

    return float(np.polyval(np.asarray(coefficients, dtype=np.float64), raw))


def score_audio(
    audio: AudioData,
    session: InferenceSession,
    protocol: DNSMOSProtocol,
) -> dict[str, float | int]:
    """Return raw and calibrated clip-level DNSMOS P.835 predictions."""

    samples = resample_for_dnsmos(audio, protocol.sample_rate)
    windows = official_windows(
        samples,
        input_samples=protocol.input_samples,
        hop_samples=protocol.hop_samples,
    )
    raw_scores: list[NDArray[np.float64]] = []
    for offset in range(0, len(windows), protocol.inference_batch_size):
        batch = np.stack(windows[offset : offset + protocol.inference_batch_size])
        output = session.run(
            [protocol.output_name],
            {protocol.input_name: batch.astype(np.float32, copy=False)},
        )[0]
        values = np.asarray(output, dtype=np.float64)
        if values.shape != (batch.shape[0], 3) or not np.all(np.isfinite(values)):
            raise ValueError(f"unexpected DNSMOS output: shape={values.shape}")
        raw_scores.extend(values)
    stacked = np.stack(raw_scores)
    raw_means = np.mean(stacked, axis=0)
    result: dict[str, float | int] = {
        "source_sample_rate": audio.sample_rate,
        "sample_rate": protocol.sample_rate,
        "source_num_samples": int(audio.samples.size),
        "resampled_num_samples": int(samples.size),
        "duration_seconds": float(audio.samples.size / audio.sample_rate),
        "num_hops": len(windows),
    }
    for index, metric in enumerate(("sig", "bak", "ovrl")):
        segment_raw = stacked[:, index]
        result[f"{metric}_raw"] = float(raw_means[index])
        result[metric] = float(
            np.mean(
                [
                    calibrate(float(raw), protocol.calibration[metric])
                    for raw in segment_raw
                ]
            )
        )
    return result
