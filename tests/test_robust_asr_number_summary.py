from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "robust_asr"
    / "summarize_number_normalization.py"
)


def _module():
    specification = importlib.util.spec_from_file_location(
        "summarize_number_normalization", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _metric(errors: int, references: int = 100) -> dict[str, int | float]:
    return {
        "substitutions": errors,
        "deletions": 0,
        "insertions": 0,
        "reference_characters": references,
        "errors": errors,
        "cer": errors / references,
    }


def _condition(frontend: str, rt60: float | None, errors: int) -> dict:
    return {
        "frontend": frontend,
        "target_rt60_seconds": rt60,
        "utterances": 10,
        "hypotheses_with_ascii_digits": 2,
        "formal": _metric(errors),
        "deterministic_contextual": _metric(errors - 1),
        "deterministic_digit_by_digit": _metric(errors - 2),
        "number_equivalent_diagnostic": _metric(errors - 3),
    }


def test_summary_aggregates_predefined_robust_scope() -> None:
    module = _module()
    conditions = [_condition("clean", None, 10)]
    for rt60 in (0.2, 0.4, 0.6, 0.8, 1.0):
        conditions.append(_condition("raw", rt60, 20))
        conditions.append(_condition("m_wpe_10", rt60, 12))

    summary = module.summarize_audits(
        {
            "model": {
                "schema_version": 2,
                "result_rows": 110,
                "conditions": conditions,
            }
        }
    )

    robust = summary["models"]["model"]["scopes"]["robust_raw"]
    assert robust["utterances"] == 40
    assert robust["hypotheses_with_ascii_digits"] == 8
    assert robust["cer_percent"]["formal"] == pytest.approx(20.0)
    assert robust["cer_percent"]["deterministic_contextual"] == pytest.approx(
        19.0
    )
    assert robust["shift_from_formal_pp"]["deterministic_contextual"] == (
        pytest.approx(-1.0)
    )


def test_summary_rejects_legacy_oracle_only_audit() -> None:
    module = _module()
    with pytest.raises(ValueError, match="schema version 2"):
        module.summarize_audits(
            {"legacy": {"schema_version": 1, "conditions": []}}
        )
