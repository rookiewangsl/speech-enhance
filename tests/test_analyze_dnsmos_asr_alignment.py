from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_dnsmos_asr_alignment.py"
SPEC = importlib.util.spec_from_file_location("analyze_dnsmos_asr_alignment", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def dns(utterance_id: str, condition: str, score: float):  # type: ignore[no-untyped-def]
    return {
        "id": utterance_id,
        "condition": condition,
        "speaker_id": "p1",
        "noise": "babble",
        "snr_db": 5.0,
        "sig": score,
        "bak": score,
        "ovrl": score,
        "protocol_version": "p1",
        "model_sha256": "dm1",
        "config_digest": "dc1",
    }


def asr(utterance_id: str, condition: str, errors: int):  # type: ignore[no-untyped-def]
    return {
        "id": utterance_id,
        "condition": condition,
        "speaker_id": "p1",
        "noise": "babble",
        "snr_db": 5.0,
        "errors": errors,
        "reference_words": 10,
        "status": "completed",
        "model_sha256": "am1",
        "asr_config_digest": "ac1",
    }


def test_reports_perceptual_asr_conflicts() -> None:
    dnsmos_rows = [
        dns("u1", "noisy", 2.0),
        dns("u1", "rnnoise_r3", 3.0),
        dns("u2", "noisy", 2.0),
        dns("u2", "rnnoise_r3", 1.0),
    ]
    asr_rows = [
        asr("u1", "noisy", 1),
        asr("u1", "rnnoise_r3", 4),
        asr("u2", "noisy", 3),
        asr("u2", "rnnoise_r3", 1),
    ]

    report = MODULE.analyze(dnsmos_rows, asr_rows, conditions=["rnnoise_r3"])
    overall = report["overall"]["rnnoise_r3"]

    assert overall["asr"]["corpus_wer_delta_percentage_points"] == 5.0
    assert overall["asr"]["utterances_harmed"] == 1
    assert overall["asr"]["utterances_improved"] == 1
    assert overall["metrics"]["ovrl"]["score_improved_but_asr_harmed"] == 1
    assert overall["metrics"]["ovrl"]["asr_harmed_fraction_among_score_improved"] == 1.0
    assert overall["metrics"]["ovrl"]["spearman_with_asr_error_reduction"] == pytest.approx(-1.0)


def test_rejects_cross_source_metadata_mismatch() -> None:
    dnsmos_rows = [dns("u1", "noisy", 2.0), dns("u1", "rnnoise_r3", 3.0)]
    asr_rows = [asr("u1", "noisy", 1), asr("u1", "rnnoise_r3", 2)]
    asr_rows[-1]["snr_db"] = 10.0

    with pytest.raises(ValueError, match="paired metadata mismatch"):
        MODULE.analyze(dnsmos_rows, asr_rows, conditions=["rnnoise_r3"])


def test_rejects_incomplete_pairing() -> None:
    dnsmos_rows = [dns("u1", "noisy", 2.0), dns("u1", "rnnoise_r3", 3.0)]
    asr_rows = [asr("u1", "noisy", 1)]

    with pytest.raises(ValueError, match="missing required condition"):
        MODULE.analyze(dnsmos_rows, asr_rows, conditions=["rnnoise_r3"])
