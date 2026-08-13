"""Reference-free diagnostics for autoregressive ASR decoding failures."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable


TokenCounter = Callable[[str], int]


def repeated_ngram_fraction(words: list[str], ngram_size: int) -> float:
    """Return the fraction of n-gram positions occupied by repeated n-grams."""

    if ngram_size <= 0:
        raise ValueError("ngram_size must be positive")
    if len(words) < ngram_size:
        return 0.0
    ngrams = [tuple(words[index : index + ngram_size]) for index in range(len(words) - ngram_size + 1)]
    counts = Counter(ngrams)
    repeated_positions = sum(count for count in counts.values() if count > 1)
    return repeated_positions / len(ngrams)


def validate_detector_config(config: dict[str, Any]) -> None:
    positive = (
        "compression_ratio_threshold",
        "repetition_ngram_size",
        "repetition_min_words",
        "max_words_per_second",
        "token_limit",
        "no_speech_conflict_min_words",
    )
    for field in positive:
        value = config.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"detector.{field} must be finite and positive")
    bounded = (
        "repeated_ngram_fraction_threshold",
        "token_limit_fraction",
        "no_speech_conflict_threshold",
    )
    for field in bounded:
        value = config.get(field)
        if not isinstance(value, (int, float)) or not 0.0 < float(value) <= 1.0:
            raise ValueError(f"detector.{field} must be in (0, 1]")
    logprob = config.get("logprob_threshold")
    if not isinstance(logprob, (int, float)) or not math.isfinite(float(logprob)):
        raise ValueError("detector.logprob_threshold must be finite")
    tolerance = config.get("timestamp_tolerance_seconds")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(float(tolerance)) or tolerance < 0:
        raise ValueError("detector.timestamp_tolerance_seconds must be finite and non-negative")
    if not isinstance(config.get("require_segment_metadata"), bool):
        raise ValueError("detector.require_segment_metadata must be boolean")
    if not isinstance(config.get("timestamp_check_enabled"), bool):
        raise ValueError("detector.timestamp_check_enabled must be boolean")


def _finite_segment_values(segments: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for segment in segments:
        value = segment.get(field)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def diagnose_hypothesis(
    row: dict[str, Any],
    config: dict[str, Any],
    *,
    token_counter: TokenCounter | None = None,
) -> dict[str, Any]:
    """Diagnose one hypothesis without inspecting reference text or WER."""

    validate_detector_config(config)
    text = row.get("hypothesis_normalized", row.get("hypothesis_raw", ""))
    if not isinstance(text, str):
        raise ValueError("hypothesis text must be a string")
    words = text.split()
    duration = row.get("duration_seconds")
    if not isinstance(duration, (int, float)) or not math.isfinite(float(duration)) or duration <= 0:
        raise ValueError("duration_seconds must be finite and positive")
    duration = float(duration)
    word_rate = len(words) / duration
    repeated_fraction = repeated_ngram_fraction(words, int(config["repetition_ngram_size"]))

    raw_segments = row.get("segments")
    segments = raw_segments if isinstance(raw_segments, list) else []
    valid_segments = [segment for segment in segments if isinstance(segment, dict)]
    compression_ratios = _finite_segment_values(valid_segments, "compression_ratio")
    avg_logprobs = _finite_segment_values(valid_segments, "avg_logprob")
    no_speech_probs = _finite_segment_values(valid_segments, "no_speech_prob")
    max_compression = max(compression_ratios, default=None)
    min_logprob = min(avg_logprobs, default=None)
    max_no_speech = max(no_speech_probs, default=None)

    generated_token_count = None
    if token_counter is not None:
        generated_token_count = int(token_counter(row.get("hypothesis_raw", text)))
        if generated_token_count < 0:
            raise ValueError("token counter returned a negative count")

    reasons: list[str] = []
    if bool(config["require_segment_metadata"]) and not valid_segments:
        reasons.append("missing_segment_metadata")
    if max_compression is not None and max_compression > float(config["compression_ratio_threshold"]):
        reasons.append("compression_ratio")
    if min_logprob is not None and min_logprob < float(config["logprob_threshold"]):
        reasons.append("low_average_logprob")
    if len(words) >= int(config["repetition_min_words"]) and repeated_fraction > float(
        config["repeated_ngram_fraction_threshold"]
    ):
        reasons.append("repeated_ngram")
    if word_rate > float(config["max_words_per_second"]):
        reasons.append("word_rate")
    if generated_token_count is not None and generated_token_count >= math.ceil(
        int(config["token_limit"]) * float(config["token_limit_fraction"])
    ):
        reasons.append("token_limit")
    if (
        max_no_speech is not None
        and max_no_speech > float(config["no_speech_conflict_threshold"])
        and len(words) >= int(config["no_speech_conflict_min_words"])
    ):
        reasons.append("no_speech_text_conflict")

    if bool(config["timestamp_check_enabled"]):
        tolerance = float(config["timestamp_tolerance_seconds"])
        for segment in valid_segments:
            start = segment.get("start")
            end = segment.get("end")
            if not all(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in (start, end)
            ):
                reasons.append("invalid_timestamp")
                break
            if float(start) < -tolerance or float(end) + tolerance < float(start) or float(end) > duration + tolerance:
                reasons.append("invalid_timestamp")
                break

    return {
        "anomalous": bool(reasons),
        "reasons": reasons,
        "word_count": len(words),
        "words_per_second": word_rate,
        "repeated_ngram_fraction": repeated_fraction,
        "generated_token_count": generated_token_count,
        "token_limit": int(config["token_limit"]),
        "max_segment_compression_ratio": max_compression,
        "min_segment_avg_logprob": min_logprob,
        "max_segment_no_speech_prob": max_no_speech,
        "segment_metadata_count": len(valid_segments),
    }
