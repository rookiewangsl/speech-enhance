"""Read-only diagnostic rescoring for Mandarin ASR result JSONL files."""

from __future__ import annotations

import itertools
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from robust_asr.scoring import (
    CharacterErrorCounts,
    aggregate_character_errors,
    score_characters,
)
from robust_asr.text import ChineseTextNormalizer

_NUMBER_PATTERN = r"[0-9]+(?:\.[0-9]+)?"
_ASCII_NUMBER = re.compile(_NUMBER_PATTERN)
_ASCII_PERCENT = re.compile(rf"({_NUMBER_PATTERN})\s*[%％]")
_ASCII_RANGE = re.compile(
    rf"({_NUMBER_PATTERN})\s*(?:-|~|—|–|至)\s*({_NUMBER_PATTERN})"
)
_GROUPED_DIGIT_SEPARATOR = re.compile(
    r"(?<=[0-9])[,，](?=[0-9]{3}(?:[,，][0-9]{3})*(?![0-9]))"
)
_ASCII_DIGITS = str.maketrans("0123456789", "零一二三四五六七八九")
NumberNormalizationPolicy = Literal["contextual_cardinal", "digit_by_digit"]


def _spoken_digits(value: str) -> str:
    return value.translate(_ASCII_DIGITS)


def _spoken_integer(value: str) -> str:
    """Return one conventional Mandarin cardinal reading up to 99,999,999."""

    if not value or not value.isascii() or not value.isdigit():
        raise ValueError("value must contain ASCII digits")
    if len(value) > 1 and value.startswith("0"):
        return _spoken_digits(value)
    number = int(value)
    if number == 0:
        return "零"
    if number >= 100_000_000:
        return _spoken_digits(value)

    digits = "零一二三四五六七八九"

    def below_ten_thousand(part: int) -> str:
        units = ((1000, "千"), (100, "百"), (10, "十"), (1, ""))
        output: list[str] = []
        pending_zero = False
        for divisor, unit in units:
            digit, part = divmod(part, divisor)
            if digit:
                if pending_zero and output:
                    output.append("零")
                if not (divisor == 10 and digit == 1 and not output):
                    output.append(digits[digit])
                output.append(unit)
                pending_zero = False
            elif output and part:
                pending_zero = True
        return "".join(output)

    high, low = divmod(number, 10_000)
    if not high:
        return below_ten_thousand(low)
    output = below_ten_thousand(high) + "万"
    if low:
        if low < 1000:
            output += "零"
        output += below_ten_thousand(low)
    return output


def _fixed_number_reading(
    value: str,
    *,
    policy: NumberNormalizationPolicy,
    year: bool = False,
) -> str:
    if policy not in ("contextual_cardinal", "digit_by_digit"):
        raise ValueError(f"unsupported number normalization policy: {policy}")
    if "." in value:
        integer, fraction = value.split(".", maxsplit=1)
        integer_reading = (
            _spoken_integer(integer)
            if policy == "contextual_cardinal"
            else _spoken_digits(integer)
        )
        return integer_reading + "点" + _spoken_digits(fraction)
    if policy == "digit_by_digit" or year:
        return _spoken_digits(value)
    return _spoken_integer(value)


def normalize_arabic_numbers(
    text: str,
    *,
    policy: NumberNormalizationPolicy = "contextual_cardinal",
) -> str:
    """Expand Arabic numbers using one fixed Mandarin reading policy.

    Unlike :func:`number_reading_variants`, this function never consults the
    paired transcript and never selects a per-utterance best reading.  The
    contextual policy handles percentages explicitly, reads four-digit years
    digit by digit, and otherwise uses conventional cardinal readings.  The
    digit policy is a deliberately simple sensitivity control.
    """

    if policy not in ("contextual_cardinal", "digit_by_digit"):
        raise ValueError(f"unsupported number normalization policy: {policy}")
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _GROUPED_DIGIT_SEPARATOR.sub("", normalized)

    def replace_percent(match: re.Match[str]) -> str:
        return "百分之" + _fixed_number_reading(match.group(1), policy=policy)

    normalized = _ASCII_PERCENT.sub(replace_percent, normalized)

    def replace_range(match: re.Match[str]) -> str:
        return (
            _fixed_number_reading(match.group(1), policy=policy)
            + "至"
            + _fixed_number_reading(match.group(2), policy=policy)
        )

    normalized = _ASCII_RANGE.sub(replace_range, normalized)

    def replace_number(match: re.Match[str]) -> str:
        value = match.group()
        following = normalized[match.end() : match.end() + 1]
        is_year = (
            policy == "contextual_cardinal"
            and following == "年"
            and "." not in value
            and len(value) == 4
        )
        return _fixed_number_reading(value, policy=policy, year=is_year)

    return _ASCII_NUMBER.sub(replace_number, normalized)


def deterministic_number_score(
    reference_raw: str,
    hypothesis_raw: str,
    *,
    normalizer: ChineseTextNormalizer,
    policy: NumberNormalizationPolicy = "contextual_cardinal",
) -> CharacterErrorCounts:
    """Score a pair after symmetric, deterministic Arabic-number expansion."""

    reference = normalizer.normalize(
        normalize_arabic_numbers(reference_raw, policy=policy)
    )
    hypothesis = normalizer.normalize(
        normalize_arabic_numbers(hypothesis_raw, policy=policy)
    )
    if not reference:
        raise ValueError("normalized reference cannot be empty")
    return score_characters(reference, hypothesis)


def number_reading_variants(text: str, *, limit: int = 64) -> tuple[str, ...]:
    """Expand Arabic numbers into plausible Mandarin written readings.

    This intentionally returns alternatives rather than claiming one universal
    inverse text-normalization rule. For example, ``40`` can be read as either
    ``四十`` or digit-by-digit ``四零`` depending on context.
    """

    if limit <= 0:
        raise ValueError("limit must be positive")
    normalized = unicodedata.normalize("NFKC", text)
    matches = list(_ASCII_NUMBER.finditer(normalized))
    if not matches:
        return (normalized,)

    pieces: list[str] = []
    choices: list[tuple[str, ...]] = []
    cursor = 0
    for match in matches:
        pieces.append(normalized[cursor : match.start()])
        value = match.group()
        if "." in value:
            integer, fraction = value.split(".", maxsplit=1)
            candidates = (
                _spoken_digits(integer) + "点" + _spoken_digits(fraction),
            )
        else:
            candidates = tuple(
                dict.fromkeys((_spoken_digits(value), _spoken_integer(value)))
            )
        choices.append(candidates)
        cursor = match.end()
    pieces.append(normalized[cursor:])

    variants: list[str] = []
    for readings in itertools.product(*choices):
        output: list[str] = []
        for index, reading in enumerate(readings):
            output.extend((pieces[index], reading))
        output.append(pieces[-1])
        variants.append("".join(output))
        if len(variants) >= limit:
            break
    return tuple(dict.fromkeys(variants))


def best_number_equivalent_score(
    reference_raw: str,
    hypothesis_raw: str,
    *,
    normalizer: ChineseTextNormalizer,
) -> CharacterErrorCounts:
    """Score the best plausible Arabic-number reading as a diagnostic bound."""

    references = {
        normalizer.normalize(value) for value in number_reading_variants(reference_raw)
    }
    hypotheses = {
        normalizer.normalize(value)
        for value in number_reading_variants(hypothesis_raw)
    }
    if "" in references:
        raise ValueError("normalized reference cannot be empty")
    candidates = (
        score_characters(reference, hypothesis)
        for reference in references
        for hypothesis in hypotheses
    )
    return min(
        candidates,
        key=lambda value: (
            value.errors,
            value.substitutions,
            value.deletions,
            value.insertions,
        ),
    )


def _condition_key(row: Mapping[str, Any]) -> tuple[str, float | None]:
    rt60 = row.get("target_rt60_seconds")
    if rt60 is None:
        rt60 = row.get("reverb_rt60_seconds", row.get("direct_rir_rt60_seconds"))
    return str(row.get("frontend", row.get("condition", "unknown"))), (
        None if rt60 is None else float(rt60)
    )


def rescore_result_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    normalizer: ChineseTextNormalizer | None = None,
) -> dict[str, Any]:
    """Compare formal CER with fixed and oracle-like number-aware CERs."""

    if not rows:
        raise ValueError("cannot rescore an empty result set")
    text_normalizer = normalizer or ChineseTextNormalizer()
    grouped: dict[
        tuple[str, float | None],
        list[
            tuple[
                CharacterErrorCounts,
                CharacterErrorCounts,
                CharacterErrorCounts,
                CharacterErrorCounts,
                bool,
                bool,
            ]
        ],
    ] = {}
    for row in rows:
        reference_source = row.get("reference_raw")
        if reference_source is None:
            reference_source = row["reference"]
        hypothesis_source = row.get("hypothesis_raw")
        if hypothesis_source is None:
            hypothesis_source = row["hypothesis"]
        reference_raw = str(reference_source)
        hypothesis_raw = str(hypothesis_source)
        reference = text_normalizer.normalize(reference_raw)
        hypothesis = text_normalizer.normalize(hypothesis_raw)
        formal = score_characters(reference, hypothesis)
        contextual = deterministic_number_score(
            reference_raw,
            hypothesis_raw,
            normalizer=text_normalizer,
            policy="contextual_cardinal",
        )
        digit_by_digit = deterministic_number_score(
            reference_raw,
            hypothesis_raw,
            normalizer=text_normalizer,
            policy="digit_by_digit",
        )
        diagnostic = best_number_equivalent_score(
            reference_raw,
            hypothesis_raw,
            normalizer=text_normalizer,
        )
        normalized_raw = unicodedata.normalize("NFKC", hypothesis_raw)
        grouped.setdefault(_condition_key(row), []).append(
            (
                formal,
                contextual,
                digit_by_digit,
                diagnostic,
                bool(re.search(r"[0-9]", normalized_raw)),
                bool(re.search(r"[A-Za-z]", normalized_raw)),
            )
        )

    conditions: list[dict[str, Any]] = []
    for (frontend, rt60), values in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1] or -1.0)
    ):
        formal = aggregate_character_errors(value[0] for value in values)
        contextual = aggregate_character_errors(value[1] for value in values)
        digit_by_digit = aggregate_character_errors(value[2] for value in values)
        diagnostic = aggregate_character_errors(value[3] for value in values)

        def subset_summary(
            selected: list[
                tuple[
                    CharacterErrorCounts,
                    CharacterErrorCounts,
                    CharacterErrorCounts,
                    CharacterErrorCounts,
                    bool,
                    bool,
                ]
            ],
        ) -> dict[str, Any]:
            if not selected:
                return {
                    "utterances": 0,
                    "formal": None,
                    "deterministic_contextual": None,
                    "deterministic_digit_by_digit": None,
                    "number_equivalent_diagnostic": None,
                }
            return {
                "utterances": len(selected),
                "formal": aggregate_character_errors(
                    value[0] for value in selected
                ).as_dict(),
                "deterministic_contextual": aggregate_character_errors(
                    value[1] for value in selected
                ).as_dict(),
                "deterministic_digit_by_digit": aggregate_character_errors(
                    value[2] for value in selected
                ).as_dict(),
                "number_equivalent_diagnostic": aggregate_character_errors(
                    value[3] for value in selected
                ).as_dict(),
            }

        with_digits = [value for value in values if value[4]]
        without_digits = [value for value in values if not value[4]]
        conditions.append(
            {
                "frontend": frontend,
                "target_rt60_seconds": rt60,
                "utterances": len(values),
                "hypotheses_with_ascii_digits": len(with_digits),
                "hypotheses_with_latin_letters": sum(value[5] for value in values),
                "utterances_improved_by_deterministic_contextual": sum(
                    value[1].errors < value[0].errors for value in values
                ),
                "utterances_improved_by_deterministic_digit_by_digit": sum(
                    value[2].errors < value[0].errors for value in values
                ),
                "utterances_improved_by_number_equivalence": sum(
                    value[3].errors < value[0].errors for value in values
                ),
                "formal": formal.as_dict(),
                "deterministic_contextual": contextual.as_dict(),
                "deterministic_digit_by_digit": digit_by_digit.as_dict(),
                "number_equivalent_diagnostic": diagnostic.as_dict(),
                "contextual_minus_formal_cer": contextual.cer - formal.cer,
                "digit_by_digit_minus_formal_cer": (
                    digit_by_digit.cer - formal.cer
                ),
                "diagnostic_minus_formal_cer": diagnostic.cer - formal.cer,
                "subsets": {
                    "hypothesis_with_ascii_digits": subset_summary(with_digits),
                    "hypothesis_without_ascii_digits": subset_summary(
                        without_digits
                    ),
                },
            }
        )
    return {
        "schema_version": 2,
        "purpose": "secondary_number_normalization_primary_cer_is_unchanged",
        "policies": {
            "formal": "pre_registered_no_number_expansion",
            "deterministic_contextual": (
                "percent_explicit_year_digitwise_otherwise_cardinal"
            ),
            "deterministic_digit_by_digit": "all_arabic_numbers_digitwise",
            "number_equivalent_diagnostic": (
                "per_utterance_best_plausible_reading_lower_bound"
            ),
        },
        "result_rows": len(rows),
        "conditions": conditions,
    }
