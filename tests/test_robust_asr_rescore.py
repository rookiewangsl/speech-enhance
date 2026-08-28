from robust_asr.rescore import (
    best_number_equivalent_score,
    number_reading_variants,
    rescore_result_rows,
)
from robust_asr.text import ChineseTextNormalizer


def _normalizer() -> ChineseTextNormalizer:
    return ChineseTextNormalizer(traditional_to_simplified=False)


def test_number_reading_variants_cover_decimal_cardinal_and_digit_readings() -> None:
    assert number_reading_variants("约4.8万元") == ("约四点八万元",)
    assert set(number_reading_variants("花10万元")) == {
        "花一零万元",
        "花十万元",
    }
    assert set(number_reading_variants("达到2米40")) == {
        "达到二米四零",
        "达到二米四十",
    }


def test_number_equivalent_score_removes_representation_only_errors() -> None:
    score = best_number_equivalent_score(
        "均价约为四点八万元",
        "均价约为4.8万元",
        normalizer=_normalizer(),
    )
    assert score.errors == 0


def test_rescore_reports_but_does_not_replace_formal_cer() -> None:
    summary = rescore_result_rows(
        [
            {
                "frontend": "clean",
                "target_rt60_seconds": None,
                "reference_raw": "花十万元",
                "hypothesis_raw": "花10万元",
                "reference": "花十万元",
                "hypothesis": "花10万元",
            },
            {
                "frontend": "clean",
                "target_rt60_seconds": None,
                "reference_raw": "测试语音",
                "hypothesis_raw": "测式语音",
                "reference": "测试语音",
                "hypothesis": "测式语音",
            },
        ],
        normalizer=_normalizer(),
    )
    condition = summary["conditions"][0]

    assert summary["purpose"] == "diagnostic_only_primary_cer_is_unchanged"
    assert condition["hypotheses_with_ascii_digits"] == 1
    assert condition["utterances_improved_by_number_equivalence"] == 1
    assert condition["formal"]["errors"] == 3
    assert condition["number_equivalent_diagnostic"]["errors"] == 1
