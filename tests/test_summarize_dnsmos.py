from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_dnsmos.py"
SPEC = importlib.util.spec_from_file_location("summarize_dnsmos", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(utterance_id: str, condition: str, sig: float, bak: float, ovrl: float):  # type: ignore[no-untyped-def]
    return {
        "id": utterance_id,
        "condition": condition,
        "speaker_id": "p1",
        "split": "validation",
        "noise": "babble" if utterance_id == "u1" else "car",
        "snr_db": 0.0,
        "sig": sig,
        "bak": bak,
        "ovrl": ovrl,
        "num_hops": 2,
        "rtf_processing_only": 0.01,
        "protocol_version": "p1",
        "model_sha256": "m1",
        "config_digest": "c1",
    }


def test_summary_reports_paired_deltas_and_is_order_stable() -> None:
    rows = [
        row("u1", "noisy", 3.0, 2.0, 2.5),
        row("u1", "rnnoise_r3", 2.5, 4.0, 3.0),
        row("u2", "noisy", 4.0, 3.0, 3.5),
        row("u2", "rnnoise_r3", 3.5, 4.0, 3.7),
    ]

    report, paired = MODULE.summarize(rows, draws=100, seed=7)
    repeated, _ = MODULE.summarize(list(reversed(rows)), draws=100, seed=7)

    rnnoise = report["overall"]["rnnoise_r3"]
    assert rnnoise["sig"]["mean"] == 3.0
    assert rnnoise["sig"]["mean_delta_vs_baseline"] == -0.5
    assert rnnoise["bak"]["mean_delta_vs_baseline"] == 1.5
    assert rnnoise["ovrl"]["mean_delta_vs_baseline"] == pytest.approx(0.35)
    assert report["overall"]["noisy"]["sig"]["mean_delta_vs_baseline_ci95"] == [0.0, 0.0]
    assert len(paired) == 4
    assert report == repeated


def test_summary_rejects_incomplete_pairing() -> None:
    rows = [
        row("u1", "noisy", 3.0, 2.0, 2.5),
        row("u1", "rnnoise_r3", 2.5, 4.0, 3.0),
        row("u2", "noisy", 4.0, 3.0, 3.5),
    ]

    with pytest.raises(ValueError, match="incomplete paired conditions"):
        MODULE.summarize(rows, draws=10)


def test_summary_rejects_mixed_model_identity() -> None:
    rows = [
        row("u1", "noisy", 3.0, 2.0, 2.5),
        row("u1", "rnnoise_r3", 2.5, 4.0, 3.0),
    ]
    rows[1]["model_sha256"] = "m2"

    with pytest.raises(ValueError, match="mix protocol, model, or config"):
        MODULE.summarize(rows, draws=10)
