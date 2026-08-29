from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from robust_asr.scoring import (
    CharacterErrorCounts,
    paired_bootstrap_cer_interaction,
)


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "robust_asr"
    / "summarize_lora_wpe_interaction.py"
)


def _module():
    specification = importlib.util.spec_from_file_location(
        "summarize_lora_wpe_interaction", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _counts(errors: int) -> CharacterErrorCounts:
    return CharacterErrorCounts(errors, 0, 0, 100)


def test_paired_bootstrap_interaction_detects_redundancy() -> None:
    baseline_raw = {"u1": _counts(30), "u2": _counts(20)}
    baseline_enhanced = {"u1": _counts(10), "u2": _counts(10)}
    candidate_raw = {"u1": _counts(20), "u2": _counts(10)}
    candidate_enhanced = {"u1": _counts(15), "u2": _counts(10)}

    interval = paired_bootstrap_cer_interaction(
        baseline_raw,
        baseline_enhanced,
        candidate_raw,
        candidate_enhanced,
        draws=100,
        seed=7,
    )

    assert interval.median > 0


def test_paired_bootstrap_interaction_rejects_unpaired_rows() -> None:
    rows = {"u1": _counts(10)}
    with pytest.raises(ValueError, match="identical utterance ids"):
        paired_bootstrap_cer_interaction(
            rows,
            rows,
            rows,
            {"u2": _counts(10)},
        )


def _rows(*, raw_errors: int, enhanced_errors: int, revision: str) -> list[dict]:
    rows = []
    for utterance in ("u1", "u2"):
        rows.append(
            {
                "utterance_id": utterance,
                "frontend": "clean",
                "target_rt60_seconds": None,
                "model_revision": revision,
                "substitutions": 5,
                "deletions": 0,
                "insertions": 0,
                "reference_characters": 100,
            }
        )
        for rt60 in (0.2, 0.4, 0.6, 0.8, 1.0):
            for frontend, errors in (
                ("raw", raw_errors),
                ("m_wpe_10", enhanced_errors),
            ):
                rows.append(
                    {
                        "utterance_id": utterance,
                        "frontend": frontend,
                        "target_rt60_seconds": rt60,
                        "model_revision": revision,
                        "substitutions": errors,
                        "deletions": 0,
                        "insertions": 0,
                        "reference_characters": 100,
                    }
                )
    return rows


def test_interaction_summary_builds_three_model_paired_matrix() -> None:
    module = _module()
    summary = module.build_summary(
        {
            "w0_pretrained": _rows(
                raw_errors=30, enhanced_errors=15, revision="w0"
            ),
            "w1_clean_lora": _rows(
                raw_errors=20, enhanced_errors=12, revision="clean"
            ),
            "w2_mct_lora": _rows(
                raw_errors=18, enhanced_errors=11, revision="mct"
            ),
        },
        draws=100,
        seed=7,
    )

    assert summary["utterances"] == 2
    assert summary["models"]["w0_pretrained"]["robust"]["raw_cer"] == 0.30
    assert summary["models"]["w2_mct_lora"]["robust"]["m_wpe_10_cer"] == 0.11
    assert summary["wpe_model_interactions"]["w2_mct_lora"]["value"] > 0
    assert summary["test_split_accessed"] is False
