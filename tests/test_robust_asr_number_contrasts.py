from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from robust_asr.text import ChineseTextNormalizer


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "robust_asr"
    / "compare_number_normalization.py"
)


def _module():
    specification = importlib.util.spec_from_file_location(
        "compare_number_normalization", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _row(utterance: str, frontend: str, rt60: float | None, hypothesis: str) -> dict:
    return {
        "utterance_id": utterance,
        "frontend": frontend,
        "target_rt60_seconds": rt60,
        "reference_raw": "收入为百分之七",
        "hypothesis_raw": hypothesis,
    }


def test_condition_scores_cluster_selected_rt60_rows_by_utterance() -> None:
    module = _module()
    rows = [
        _row("u1", "raw", 0.4, "收入为7%"),
        _row("u1", "raw", 0.6, "收入为7%"),
        _row("u1", "raw", 0.8, "收入为百分之八"),
        _row("u1", "raw", 1.0, "收入为百分之八"),
    ]
    scores = module.condition_scores(
        rows,
        frontend="raw",
        rt60_values=(0.4, 0.6, 0.8, 1.0),
        policy="contextual_cardinal",
        normalizer=ChineseTextNormalizer(traditional_to_simplified=False),
    )

    assert set(scores) == {"u1"}
    assert scores["u1"].reference_characters == 28
    assert scores["u1"].errors == 2


def test_condition_scores_reject_incomplete_rt60_cluster() -> None:
    module = _module()
    with pytest.raises(ValueError, match="do not have 4 selected rows"):
        module.condition_scores(
            [_row("u1", "raw", 0.4, "收入为7%")],
            frontend="raw",
            rt60_values=(0.4, 0.6, 0.8, 1.0),
            policy="formal",
            normalizer=ChineseTextNormalizer(traditional_to_simplified=False),
        )
