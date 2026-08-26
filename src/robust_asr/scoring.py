"""Dependency-free character error rate and paired bootstrap scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CharacterErrorCounts:
    """Character-level edit counts for one utterance or corpus."""

    substitutions: int
    deletions: int
    insertions: int
    reference_characters: int

    def __post_init__(self) -> None:
        values = (
            self.substitutions,
            self.deletions,
            self.insertions,
            self.reference_characters,
        )
        if any(value < 0 for value in values):
            raise ValueError("error counts cannot be negative")

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def cer(self) -> float:
        if self.reference_characters == 0:
            raise ValueError("CER is undefined for an empty reference")
        return self.errors / self.reference_characters

    def as_dict(self) -> dict[str, int | float]:
        return {
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "reference_characters": self.reference_characters,
            "errors": self.errors,
            "cer": self.cer,
            "substitution_rate": self.substitutions
            / self.reference_characters,
            "deletion_rate": self.deletions / self.reference_characters,
            "insertion_rate": self.insertions / self.reference_characters,
        }


def score_characters(reference: str, hypothesis: str) -> CharacterErrorCounts:
    """Score two already-normalized strings with deterministic tie-breaking."""

    if not reference:
        raise ValueError("cannot score an empty normalized reference")
    return _score_sequence(list(reference), list(hypothesis))


def _score_sequence(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> CharacterErrorCounts:
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    table: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(columns)] for _ in range(rows)
    ]
    for row in range(1, rows):
        table[row][0] = (row, 0, row, 0)
    for column in range(1, columns):
        table[0][column] = (column, 0, 0, column)

    for row in range(1, rows):
        for column in range(1, columns):
            if reference[row - 1] == hypothesis[column - 1]:
                table[row][column] = table[row - 1][column - 1]
                continue
            diagonal = table[row - 1][column - 1]
            substitution = (
                diagonal[0] + 1,
                diagonal[1] + 1,
                diagonal[2],
                diagonal[3],
            )
            above = table[row - 1][column]
            deletion = (above[0] + 1, above[1], above[2] + 1, above[3])
            left = table[row][column - 1]
            insertion = (left[0] + 1, left[1], left[2], left[3] + 1)
            candidates = (
                (substitution[0], 0, substitution),
                (deletion[0], 1, deletion),
                (insertion[0], 2, insertion),
            )
            table[row][column] = min(candidates, key=lambda item: item[:2])[2]

    _, substitutions, deletions, insertions = table[-1][-1]
    return CharacterErrorCounts(
        substitutions,
        deletions,
        insertions,
        len(reference),
    )


def aggregate_character_errors(
    scores: Iterable[CharacterErrorCounts],
) -> CharacterErrorCounts:
    """Aggregate edit counts before computing corpus CER."""

    substitutions = deletions = insertions = reference_characters = 0
    for score in scores:
        substitutions += score.substitutions
        deletions += score.deletions
        insertions += score.insertions
        reference_characters += score.reference_characters
    if reference_characters == 0:
        raise ValueError("cannot aggregate a corpus with no reference characters")
    return CharacterErrorCounts(
        substitutions,
        deletions,
        insertions,
        reference_characters,
    )


@dataclass(frozen=True)
class BootstrapInterval:
    """Percentile interval for candidate CER minus baseline CER."""

    lower: float
    median: float
    upper: float
    draws: int
    seed: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "lower": self.lower,
            "median": self.median,
            "upper": self.upper,
            "draws": self.draws,
            "seed": self.seed,
        }


def paired_bootstrap_cer_delta(
    baseline: Mapping[str, CharacterErrorCounts],
    candidate: Mapping[str, CharacterErrorCounts],
    *,
    draws: int = 10_000,
    seed: int = 2026,
) -> BootstrapInterval:
    """Bootstrap utterance pairs and return candidate-minus-baseline CER."""

    if draws <= 0:
        raise ValueError("draws must be positive")
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate must use identical utterance ids")
    identifiers = sorted(baseline)
    if not identifiers:
        raise ValueError("paired bootstrap requires at least one utterance")

    base = np.asarray(
        [
            (
                baseline[key].errors,
                baseline[key].reference_characters,
            )
            for key in identifiers
        ],
        dtype=np.int64,
    )
    cand = np.asarray(
        [
            (
                candidate[key].errors,
                candidate[key].reference_characters,
            )
            for key in identifiers
        ],
        dtype=np.int64,
    )
    if not np.array_equal(base[:, 1], cand[:, 1]):
        raise ValueError(
            "paired conditions disagree in reference character counts"
        )

    rng = np.random.default_rng(seed)
    deltas = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sample = rng.integers(0, len(identifiers), size=len(identifiers))
        reference_count = int(base[sample, 1].sum())
        if reference_count == 0:
            raise ValueError("bootstrap sample has an empty reference")
        deltas[index] = (
            cand[sample, 0].sum() - base[sample, 0].sum()
        ) / reference_count

    lower, median, upper = np.quantile(deltas, [0.025, 0.5, 0.975])
    return BootstrapInterval(
        lower=float(lower),
        median=float(median),
        upper=float(upper),
        draws=draws,
        seed=seed,
    )

