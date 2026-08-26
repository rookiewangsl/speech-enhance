"""Dataset-independent AISHELL manifest validation and duration sampling."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CleanUtterance:
    utterance_id: str
    speaker_id: str
    audio_path: str
    transcript: str
    duration_seconds: float
    domain: str | None = None

    def __post_init__(self) -> None:
        if not self.utterance_id or not self.speaker_id:
            raise ValueError("utterance_id and speaker_id cannot be empty")
        if not self.audio_path or not self.transcript:
            raise ValueError("audio_path and transcript cannot be empty")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")


def validate_unique_utterances(rows: Iterable[CleanUtterance]) -> None:
    seen: set[str] = set()
    for row in rows:
        if row.utterance_id in seen:
            raise ValueError(f"duplicate utterance id: {row.utterance_id}")
        seen.add(row.utterance_id)


def _rank(seed: int, utterance_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{utterance_id}".encode()).digest()


def select_speaker_balanced_duration_subset(
    rows: Iterable[CleanUtterance],
    *,
    target_hours: float,
    seed: int = 2026,
) -> tuple[CleanUtterance, ...]:
    """Round-robin speakers until a deterministic duration target is reached.

    This function operates on a future audited AISHELL manifest; it does not
    inspect or download audio. Overshoot is bounded by the final utterance.
    """

    if target_hours <= 0:
        raise ValueError("target_hours must be positive")
    candidates = tuple(rows)
    validate_unique_utterances(candidates)
    if not candidates:
        raise ValueError("rows cannot be empty")
    grouped: dict[str, list[CleanUtterance]] = {}
    for row in candidates:
        grouped.setdefault(row.speaker_id, []).append(row)
    for speaker_rows in grouped.values():
        speaker_rows.sort(key=lambda row: (_rank(seed, row.utterance_id), row.utterance_id))

    target_seconds = target_hours * 3600.0
    selected: list[CleanUtterance] = []
    total_seconds = 0.0
    indices = {speaker: 0 for speaker in grouped}
    speaker_order = sorted(grouped)
    while total_seconds < target_seconds:
        made_progress = False
        for speaker in speaker_order:
            index = indices[speaker]
            if index >= len(grouped[speaker]):
                continue
            row = grouped[speaker][index]
            indices[speaker] += 1
            selected.append(row)
            total_seconds += row.duration_seconds
            made_progress = True
            if total_seconds >= target_seconds:
                break
        if not made_progress:
            raise ValueError(
                f"available duration {total_seconds / 3600.0:.3f} h is below "
                f"target {target_hours:.3f} h"
            )
    return tuple(selected)


def select_speaker_balanced_count_subset(
    rows: Iterable[CleanUtterance],
    *,
    count: int,
    seed: int = 2026,
) -> tuple[CleanUtterance, ...]:
    """Round-robin speakers to a fixed utterance count deterministically."""

    if count <= 0:
        raise ValueError("count must be positive")
    candidates = tuple(rows)
    validate_unique_utterances(candidates)
    if len(candidates) < count:
        raise ValueError(
            f"available utterances {len(candidates)} are below requested count {count}"
        )
    grouped: dict[str, list[CleanUtterance]] = {}
    for row in candidates:
        grouped.setdefault(row.speaker_id, []).append(row)
    for speaker_rows in grouped.values():
        speaker_rows.sort(key=lambda row: (_rank(seed, row.utterance_id), row.utterance_id))
    speaker_order = sorted(
        grouped,
        key=lambda speaker: (_rank(seed, speaker), speaker),
    )
    selected: list[CleanUtterance] = []
    index = 0
    while len(selected) < count:
        made_progress = False
        for speaker in speaker_order:
            if index >= len(grouped[speaker]):
                continue
            selected.append(grouped[speaker][index])
            made_progress = True
            if len(selected) == count:
                break
        if not made_progress:
            raise RuntimeError("speaker-balanced count selection stalled")
        index += 1
    return tuple(selected)
