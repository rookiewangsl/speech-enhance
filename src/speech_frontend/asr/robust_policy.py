"""Pure decision helpers for the robust ASR recovery policy."""

from __future__ import annotations

from typing import Any


def normalized_word_distance(first: str, second: str) -> float:
    """Return Levenshtein distance divided by the longer word sequence."""

    left, right = first.split(), second.split()
    if not left and not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for row_index, left_word in enumerate(left, start=1):
        current = [row_index]
        for column_index, right_word in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + (left_word != right_word),
                )
            )
        previous = current
    return previous[-1] / max(len(left), len(right))


def select_first_passing_retry(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the first retry that passes reference-free diagnostics."""

    for attempt in attempts:
        diagnostic = attempt.get("diagnostic")
        if isinstance(diagnostic, dict) and diagnostic.get("anomalous") is False:
            return attempt
    return None


def decide_final(
    *,
    condition: str,
    first_pass: dict[str, Any],
    retry_attempts: list[dict[str, Any]],
    fallback_conditions: set[str],
    noisy_result: dict[str, Any] | None,
    abstention_enabled: bool,
    retry_noisy_consistency_threshold: float | None = None,
) -> dict[str, Any]:
    """Return a reference-free final decision and its source."""

    diagnostic = first_pass.get("diagnostic")
    if not isinstance(diagnostic, dict):
        raise ValueError("first pass is missing its diagnostic")
    if diagnostic.get("anomalous") is False:
        return {"status": "accepted", "source": "first_pass", "attempt": first_pass}

    retry = select_first_passing_retry(retry_attempts)
    if retry is not None:
        if (
            condition in fallback_conditions
            and retry_noisy_consistency_threshold is not None
            and noisy_result is not None
            and noisy_result.get("final_status") == "accepted"
        ):
            noisy_attempt = noisy_result["final_attempt"]
            distance = normalized_word_distance(
                str(retry["hypothesis_normalized"]),
                str(noisy_attempt["hypothesis_normalized"]),
            )
            if distance > retry_noisy_consistency_threshold:
                return {
                    "status": "accepted",
                    "source": "noisy_fallback",
                    "attempt": noisy_attempt,
                    "decision_details": {
                        "retry_rejected_by_noisy_consistency": True,
                        "retry_noisy_word_distance": distance,
                        "max_retry_noisy_word_distance": retry_noisy_consistency_threshold,
                    },
                }
            return {
                "status": "accepted",
                "source": "temperature_retry",
                "attempt": retry,
                "decision_details": {
                    "retry_rejected_by_noisy_consistency": False,
                    "retry_noisy_word_distance": distance,
                    "max_retry_noisy_word_distance": retry_noisy_consistency_threshold,
                },
            }
        return {"status": "accepted", "source": "temperature_retry", "attempt": retry}

    if condition in fallback_conditions and noisy_result is not None:
        if noisy_result.get("final_status") == "accepted":
            return {
                "status": "accepted",
                "source": "noisy_fallback",
                "attempt": noisy_result["final_attempt"],
            }

    if abstention_enabled:
        return {"status": "abstained", "source": "abstained", "attempt": None}
    return {"status": "accepted", "source": "unsafe_first_pass", "attempt": first_pass}
