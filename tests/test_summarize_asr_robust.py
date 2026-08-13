from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_asr_robust.py"
SPEC = importlib.util.spec_from_file_location("summarize_asr_robust", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _attempt(text: str, *, anomalous: bool = False) -> dict[str, object]:
    return {
        "hypothesis_raw": text,
        "hypothesis_normalized": text,
        "audio_sha256": "a" * 64,
        "diagnostic": {"anomalous": anomalous},
    }


def _rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    v1: list[dict[str, object]] = []
    v2: list[dict[str, object]] = []
    for utterance_id in ("u1", "u2"):
        for condition in MODULE.CONDITIONS:
            text = "one wrong" if condition == "rnnoise_r3" and utterance_id == "u1" else "one two"
            first = {
                "id": utterance_id,
                "condition": condition,
                "reference_raw_sha256": "r" * 64,
                "reference_normalized": "one two",
                "hypothesis_raw": text,
                "hypothesis_normalized": text,
                "audio_sha256": "a" * 64,
                "asr_seconds": 1.0,
            }
            v1.append(first)
            recovered = condition == "rnnoise_r3" and utterance_id == "u1"
            final_attempt = _attempt("one two") if recovered else _attempt(text)
            v2.append(
                {
                    **first,
                    "protocol_version": "asr_whisper_small_en_robust_v2",
                    "first_pass_attempt": _attempt(text, anomalous=recovered),
                    "first_pass_anomalous": recovered,
                    "trigger_reasons": ["compression_ratio"] if recovered else [],
                    "retry_attempts": [_attempt("one two")] if recovered else [],
                    "final_status": "accepted",
                    "final_source": "temperature_retry" if recovered else "first_pass",
                    "final_attempt": final_attempt,
                    "hypothesis_raw": final_attempt["hypothesis_raw"],
                    "hypothesis_normalized": final_attempt["hypothesis_normalized"],
                    "first_pass_asr_seconds": 1.0,
                    "recovery_asr_seconds": 0.5 if recovered else 0.0,
                    "total_service_asr_seconds": 1.5 if recovered else 1.0,
                    "duration_seconds": 2.0,
                }
            )
    return v1, v2


def test_summary_reports_recovery_and_latency() -> None:
    v1, v2 = _rows()
    scored = MODULE.score_comparison(v1, v2)
    summary = MODULE.summarize(scored, bootstrap_draws=100, bootstrap_seed=3)
    rnnoise = summary["conditions"]["rnnoise_r3"]

    assert rnnoise["v1_first_pass"]["wer"] == 0.25
    assert rnnoise["v2_full_policy"]["wer"] == 0.0
    assert rnnoise["paired_v1_noisy_baseline"]["wer"] == 0.0
    assert rnnoise["oracle_v1_condition_or_noisy"]["wer"] == 0.0
    assert rnnoise["oracle_v1_condition_or_noisy"]["deployable"] is False
    assert rnnoise["v2_full_policy"]["absolute_wer_change_vs_v1_noisy"] == 0.0
    assert rnnoise["triggered_utterances"] == 1
    assert rnnoise["retry_accepted_utterances"] == 1
    assert rnnoise["final_source_counts"] == {"first_pass": 1, "temperature_retry": 1}
    assert rnnoise["latency"]["v2_extra_asr_seconds"] == 0.5
    assert rnnoise["detector_offline_evaluation"]["true_positive"] == 1


def test_abstention_reduces_coverage_without_silent_deletion() -> None:
    v1, v2 = _rows()
    target = next(row for row in v2 if row["condition"] == "rnnoise_r3" and row["id"] == "u1")
    target["final_status"] = "abstained"
    target["final_source"] = "abstained"
    target["final_attempt"] = None
    target["hypothesis_raw"] = ""
    target["hypothesis_normalized"] = ""
    scored = MODULE.score_comparison(v1, v2)
    summary = MODULE.summarize(scored, bootstrap_draws=10)
    final = summary["conditions"]["rnnoise_r3"]["v2_full_policy"]
    assert final["coverage"] == 0.5
    assert final["full_coverage_wer"] is None
    assert summary["conditions"]["rnnoise_r3"]["paired_bootstrap_final_minus_v1_ci95"] is None
    assert summary["conditions"]["rnnoise_r3"]["paired_bootstrap_final_minus_v1_noisy_ci95"] is None


def test_rejects_cross_condition_normalized_reference_mismatch() -> None:
    v1, v2 = _rows()
    target_v1 = next(row for row in v1 if row["id"] == "u1" and row["condition"] == "rnnoise_r3")
    target_v2 = next(row for row in v2 if row["id"] == "u1" and row["condition"] == "rnnoise_r3")
    target_v1["reference_normalized"] = "different reference"
    target_v2["reference_normalized"] = "different reference"

    try:
        MODULE.score_comparison(v1, v2)
    except ValueError as error:
        assert "paired normalized reference mismatch" in str(error)
    else:
        raise AssertionError("expected normalized reference mismatch to be rejected")
