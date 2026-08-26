from __future__ import annotations

import pytest

from robust_asr.scoring import (
    CharacterErrorCounts,
    aggregate_character_errors,
    paired_bootstrap_cer_delta,
    score_characters,
)
from robust_asr.text import ChineseTextNormalizer


def test_chinese_normalizer_is_transparent_without_opencc() -> None:
    normalizer = ChineseTextNormalizer(traditional_to_simplified=False)

    assert normalizer.normalize("ＡＢＣ，中文 １２3！") == "abc中文123"


def test_chinese_normalizer_applies_injected_t2s_symmetrically() -> None:
    replacements = str.maketrans({"臺": "台", "灣": "湾"})
    normalizer = ChineseTextNormalizer(
        converter=lambda value: value.translate(replacements)
    )

    reference, hypothesis = normalizer.normalize_pair("臺灣。", "台湾")

    assert reference == hypothesis == "台湾"


def test_normalizer_rejects_converter_when_conversion_disabled() -> None:
    with pytest.raises(ValueError, match="converter"):
        ChineseTextNormalizer(
            traditional_to_simplified=False,
            converter=lambda value: value,
        )


def test_character_scoring_reports_substitution_deletion_insertion() -> None:
    substitution = score_characters("你好", "你号")
    deletion = score_characters("你好", "你")
    insertion = score_characters("你好", "你们好")

    assert substitution == CharacterErrorCounts(1, 0, 0, 2)
    assert deletion == CharacterErrorCounts(0, 1, 0, 2)
    assert insertion == CharacterErrorCounts(0, 0, 1, 2)


def test_corpus_cer_aggregates_counts_before_division() -> None:
    aggregate = aggregate_character_errors(
        [score_characters("你好", "你"), score_characters("天气很好", "天气很好")]
    )

    assert aggregate.reference_characters == 6
    assert aggregate.cer == pytest.approx(1 / 6)


def test_paired_bootstrap_is_deterministic() -> None:
    baseline = {
        "u1": CharacterErrorCounts(1, 0, 0, 5),
        "u2": CharacterErrorCounts(0, 1, 0, 3),
    }
    candidate = {
        "u1": CharacterErrorCounts(0, 0, 0, 5),
        "u2": CharacterErrorCounts(0, 0, 0, 3),
    }

    first = paired_bootstrap_cer_delta(
        baseline, candidate, draws=101, seed=7
    )
    second = paired_bootstrap_cer_delta(
        dict(reversed(list(baseline.items()))),
        candidate,
        draws=101,
        seed=7,
    )

    assert first == second
    assert first.upper < 0


def test_paired_bootstrap_rejects_unpaired_ids() -> None:
    score = CharacterErrorCounts(0, 0, 0, 1)

    with pytest.raises(ValueError, match="identical utterance ids"):
        paired_bootstrap_cer_delta({"u1": score}, {"u2": score}, draws=2)

