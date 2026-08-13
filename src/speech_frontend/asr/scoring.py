"""Dependency-free word-error-rate scoring.

The dynamic-programming alignment is deterministic.  When multiple minimum-cost
alignments exist it prefers substitution, then deletion, then insertion.  This
tie-break only affects the S/D/I decomposition, never the total edit distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ErrorCounts:
    """Word-level edit counts for one utterance or a corpus."""

    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        if self.reference_words == 0:
            raise ValueError("WER is undefined for an empty reference")
        return self.errors / self.reference_words

    def as_dict(self) -> dict[str, int | float]:
        return {
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "reference_words": self.reference_words,
            "errors": self.errors,
            "wer": self.wer,
            "substitution_rate": self.substitutions / self.reference_words,
            "deletion_rate": self.deletions / self.reference_words,
            "insertion_rate": self.insertions / self.reference_words,
        }


def score_text(reference: str, hypothesis: str) -> ErrorCounts:
    """Score two already-normalized, whitespace-tokenized strings."""

    return score_words(reference.split(), hypothesis.split())


def score_words(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> ErrorCounts:
    """Return deterministic minimum-edit S/D/I counts.

    Normalization is deliberately outside this function so that callers cannot
    accidentally apply different policies to references and hypotheses.
    """

    if not reference:
        raise ValueError("cannot score an empty reference")

    # Each cell is (edit cost, substitutions, deletions, insertions).  The
    # operation priority is used only after edit cost, making ties reproducible.
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
            # min() sees the explicit operation priority before count fields.
            candidates = (
                (substitution[0], 0, substitution),
                (deletion[0], 1, deletion),
                (insertion[0], 2, insertion),
            )
            table[row][column] = min(candidates, key=lambda item: item[:2])[2]

    _, substitutions, deletions, insertions = table[-1][-1]
    return ErrorCounts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_words=len(reference),
    )


def aggregate_error_counts(scores: Iterable[ErrorCounts]) -> ErrorCounts:
    """Sum edit counts before computing WER (corpus WER, not mean WER)."""

    substitutions = deletions = insertions = reference_words = 0
    for score in scores:
        substitutions += score.substitutions
        deletions += score.deletions
        insertions += score.insertions
        reference_words += score.reference_words
    if reference_words == 0:
        raise ValueError("cannot aggregate a corpus with no reference words")
    return ErrorCounts(substitutions, deletions, insertions, reference_words)
