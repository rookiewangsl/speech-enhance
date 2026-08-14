from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_asr_stability.py"
SPEC = importlib.util.spec_from_file_location("analyze_asr_stability", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _score(errors: int) -> dict[str, int]:
    return {"errors": errors}


def test_stability_reports_speaker_outlier_and_coverage() -> None:
    rows = []
    for condition in MODULE.ENHANCED_CONDITIONS:
        for speaker in ("p1", "p2"):
            for index, noise in enumerate(("babble", "car")):
                rows.append(
                    {
                        "id": f"{speaker}_{index}",
                        "condition": condition,
                        "speaker_id": speaker,
                        "noise": noise,
                        "snr_db": 0.0,
                        "reference_words": 10,
                        "first_pass": _score(3),
                        "paired_v1_noisy": _score(1),
                        "final": None if speaker == "p2" and index == 1 else _score(2),
                    }
                )
    summary = {
        "overall": {
            condition: {"paired_bootstrap_absolute_wer_change_vs_noisy_ci95": {"lower": 0.1, "upper": 0.3}}
            for condition in MODULE.ENHANCED_CONDITIONS
        }
    }
    report = MODULE.analyze(rows, summary)
    rnnoise = report["conditions"]["rnnoise_r3"]

    assert rnnoise["v1_first_pass"]["absolute_wer_change_vs_paired_noisy"] == 0.2
    assert rnnoise["v1_first_pass"]["speaker_direction_counts"] == {"better": 0, "equal": 0, "worse": 2}
    assert rnnoise["v1_first_pass"]["all_leave_one_speaker_out_same_direction"] is True
    assert rnnoise["v1_first_pass"]["babble_and_non_babble_same_direction"] is True
    assert rnnoise["v2_final"]["coverage"] == 0.75
