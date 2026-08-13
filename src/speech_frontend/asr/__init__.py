"""Utilities for evaluating fixed automatic speech-recognition backends."""

from .scoring import ErrorCounts, aggregate_error_counts, score_text, score_words

__all__ = [
    "ErrorCounts",
    "aggregate_error_counts",
    "score_text",
    "score_words",
]
