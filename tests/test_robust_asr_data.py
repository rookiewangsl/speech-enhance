from __future__ import annotations

from collections import Counter

import pytest

from robust_asr.data import (
    CleanUtterance,
    select_speaker_balanced_count_subset,
    select_speaker_balanced_duration_subset,
)


def rows() -> list[CleanUtterance]:
    return [
        CleanUtterance(
            utterance_id=f"{speaker}_{index}",
            speaker_id=speaker,
            audio_path=f"{speaker}_{index}.wav",
            transcript="测试语音",
            duration_seconds=10.0,
        )
        for speaker in ("s1", "s2", "s3")
        for index in range(5)
    ]


def test_duration_subset_is_deterministic_and_speaker_balanced() -> None:
    first = select_speaker_balanced_duration_subset(
        rows(), target_hours=60.0 / 3600.0, seed=7
    )
    second = select_speaker_balanced_duration_subset(
        reversed(rows()), target_hours=60.0 / 3600.0, seed=7
    )
    counts = Counter(row.speaker_id for row in first)

    assert first == second
    assert counts == {"s1": 2, "s2": 2, "s3": 2}
    assert sum(row.duration_seconds for row in first) == 60.0


def test_duration_subset_rejects_insufficient_audio() -> None:
    with pytest.raises(ValueError, match="below target"):
        select_speaker_balanced_duration_subset(rows(), target_hours=1.0)


def test_duration_subset_rejects_duplicate_ids() -> None:
    duplicate = rows()[:2]
    duplicate[1] = CleanUtterance(
        utterance_id=duplicate[0].utterance_id,
        speaker_id="s2",
        audio_path="other.wav",
        transcript="其他",
        duration_seconds=1.0,
    )

    with pytest.raises(ValueError, match="duplicate"):
        select_speaker_balanced_duration_subset(
            duplicate, target_hours=1.0 / 3600.0
        )


def test_count_subset_is_deterministic_and_speaker_balanced() -> None:
    first = select_speaker_balanced_count_subset(rows(), count=3, seed=5)
    second = select_speaker_balanced_count_subset(reversed(rows()), count=3, seed=5)

    assert first == second
    assert len(first) == 3
    assert len({row.speaker_id for row in first}) == 3
