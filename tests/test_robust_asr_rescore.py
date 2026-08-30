from robust_asr.rescore import (
    best_number_equivalent_score,
    deterministic_number_score,
    normalize_arabic_numbers,
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


def test_number_equivalent_bound_includes_fixed_percent_policy() -> None:
    fixed = deterministic_number_score(
        "同比下降百分之四十六",
        "同比下降46%",
        normalizer=_normalizer(),
    )
    bound = best_number_equivalent_score(
        "同比下降百分之四十六",
        "同比下降46%",
        normalizer=_normalizer(),
    )
    assert fixed.errors == 0
    assert bound.errors <= fixed.errors


def test_contextual_number_normalization_is_fixed_and_symmetric() -> None:
    assert normalize_arabic_numbers("同比下降46%") == "同比下降百分之四十六"
    assert normalize_arabic_numbers("2014年7月10日") == "二零一四年七月十日"
    assert normalize_arabic_numbers("15.107元") == "十五点一零七元"
    assert normalize_arabic_numbers("30-40万元") == "三十至四十万元"
    assert normalize_arabic_numbers("1,500元") == "一千五百元"


def test_digit_by_digit_policy_is_a_non_oracle_sensitivity_control() -> None:
    assert normalize_arabic_numbers("10万元", policy="digit_by_digit") == (
        "一零万元"
    )
    assert normalize_arabic_numbers("10万元") == "十万元"
    assert normalize_arabic_numbers("１０％", policy="digit_by_digit") == (
        "百分之一零"
    )


def test_deterministic_number_score_does_not_choose_a_reference_specific_reading() -> None:
    score = deterministic_number_score(
        "十月十八日",
        "10月18日",
        normalizer=_normalizer(),
    )
    assert score.errors == 0
    ambiguous = deterministic_number_score(
        "一零月十八日",
        "10月18日",
        normalizer=_normalizer(),
    )
    assert ambiguous.errors > 0


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

    assert summary["purpose"] == (
        "secondary_number_normalization_primary_cer_is_unchanged"
    )
    assert condition["hypotheses_with_ascii_digits"] == 1
    assert condition["utterances_improved_by_number_equivalence"] == 1
    assert condition["utterances_worsened_by_deterministic_contextual"] == 0
    assert condition["utterances_worsened_by_number_equivalence"] == 0
    assert condition["formal"]["errors"] == 3
    assert condition["deterministic_contextual"]["errors"] == 1
    assert condition["deterministic_digit_by_digit"]["errors"] == 3
    assert condition["number_equivalent_diagnostic"]["errors"] == 1
    assert condition["number_equivalent_diagnostic"]["errors"] <= condition[
        "deterministic_contextual"
    ]["errors"]
    assert condition["number_equivalent_diagnostic"]["errors"] <= condition[
        "deterministic_digit_by_digit"
    ]["errors"]
    assert condition["subsets"]["hypothesis_with_ascii_digits"]["utterances"] == 1
    assert condition["subsets"]["hypothesis_without_ascii_digits"]["utterances"] == 1
