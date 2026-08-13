from __future__ import annotations

import pytest

from speech_frontend.asr.anomaly_detection import diagnose_hypothesis, repeated_ngram_fraction


CONFIG = {
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    "repetition_ngram_size": 4,
    "repeated_ngram_fraction_threshold": 0.5,
    "repetition_min_words": 12,
    "max_words_per_second": 8.0,
    "token_limit": 224,
    "token_limit_fraction": 0.95,
    "no_speech_conflict_threshold": 0.8,
    "no_speech_conflict_min_words": 5,
    "timestamp_check_enabled": True,
    "timestamp_tolerance_seconds": 0.05,
    "require_segment_metadata": True,
}


def _row(text: str = "please call stella") -> dict[str, object]:
    return {
        "hypothesis_raw": text,
        "hypothesis_normalized": text,
        "duration_seconds": 2.0,
        "segments": [
            {
                "start": 0.0,
                "end": 1.8,
                "compression_ratio": 0.9,
                "avg_logprob": -0.2,
                "no_speech_prob": 0.1,
            }
        ],
    }


def test_repeated_ngram_fraction_counts_repeated_positions() -> None:
    words = "a b c d a b c d a b c d".split()
    assert repeated_ngram_fraction(words, 4) == pytest.approx(1.0)
    assert repeated_ngram_fraction(["a", "b"], 4) == 0.0


def test_normal_hypothesis_passes_without_reference_fields() -> None:
    diagnostic = diagnose_hypothesis(_row(), CONFIG, token_counter=lambda text: 3)
    assert diagnostic["anomalous"] is False
    assert diagnostic["reasons"] == []


def test_repetition_failure_triggers_independent_signals() -> None:
    text = "this is not a club " * 30
    row = _row(text.strip())
    row["duration_seconds"] = 3.5
    row["segments"][0]["end"] = 3.49  # type: ignore[index]
    row["segments"][0]["compression_ratio"] = 17.88  # type: ignore[index]
    diagnostic = diagnose_hypothesis(row, CONFIG, token_counter=lambda text: 223)
    assert diagnostic["anomalous"] is True
    assert set(diagnostic["reasons"]) == {
        "compression_ratio",
        "repeated_ngram",
        "word_rate",
        "token_limit",
    }


def test_invalid_timestamp_and_missing_metadata_are_detected() -> None:
    row = _row()
    row["segments"][0]["end"] = 3.0  # type: ignore[index]
    assert "invalid_timestamp" in diagnose_hypothesis(row, CONFIG)["reasons"]
    row["segments"] = []
    assert diagnose_hypothesis(row, CONFIG)["reasons"] == ["missing_segment_metadata"]
