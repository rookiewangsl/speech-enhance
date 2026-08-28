"""Read-only diagnostic rescoring for Mandarin ASR result JSONL files."""

from __future__ import annotations

import itertools
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from robust_asr.scoring import (
    CharacterErrorCounts,
    aggregate_character_errors,
    score_characters,
)
from robust_asr.text import ChineseTextNormalizer

_ASCII_NUMBER = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_ASCII_DIGITS = str.maketrans("0123456789", "零一二三四五六七八九")


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
    """Compare formal CER with a number-equivalence diagnostic CER."""

    if not rows:
        raise ValueError("cannot rescore an empty result set")
    text_normalizer = normalizer or ChineseTextNormalizer()
    grouped: dict[
        tuple[str, float | None],
        list[tuple[CharacterErrorCounts, CharacterErrorCounts, bool, bool]],
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
        diagnostic = best_number_equivalent_score(
            reference_raw,
            hypothesis_raw,
            normalizer=text_normalizer,
        )
        normalized_raw = unicodedata.normalize("NFKC", hypothesis_raw)
        grouped.setdefault(_condition_key(row), []).append(
            (
                formal,
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
        diagnostic = aggregate_character_errors(value[1] for value in values)
        improved = sum(value[1].errors < value[0].errors for value in values)
        conditions.append(
            {
                "frontend": frontend,
                "target_rt60_seconds": rt60,
                "utterances": len(values),
                "hypotheses_with_ascii_digits": sum(value[2] for value in values),
                "hypotheses_with_latin_letters": sum(value[3] for value in values),
                "utterances_improved_by_number_equivalence": improved,
                "formal": formal.as_dict(),
                "number_equivalent_diagnostic": diagnostic.as_dict(),
                "diagnostic_minus_formal_cer": diagnostic.cer - formal.cer,
            }
        )
    return {
        "schema_version": 1,
        "purpose": "diagnostic_only_primary_cer_is_unchanged",
        "result_rows": len(rows),
        "conditions": conditions,
    }
