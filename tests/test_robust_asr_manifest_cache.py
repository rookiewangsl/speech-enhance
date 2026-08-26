from __future__ import annotations

from pathlib import Path

import pytest

from robust_asr.cache import CacheIdentity
from robust_asr.config import canonical_sha256
from robust_asr.manifest import (
    choose_mct_condition,
    choose_rir_id,
    read_jsonl,
    validate_disjoint_groups,
    write_jsonl_atomic,
)


def test_manifest_round_trip_preserves_unicode(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    rows = [{"id": "u1", "text": "普通话"}, {"id": "u2", "value": 2}]

    write_jsonl_atomic(path, rows)

    assert read_jsonl(path) == rows


def test_speaker_and_room_group_leakage_is_rejected() -> None:
    splits = {
        "train": [{"speaker_id": "s1"}, {"speaker_id": "s2"}],
        "test": [{"speaker_id": "s2"}],
    }

    with pytest.raises(ValueError, match="leakage"):
        validate_disjoint_groups(splits, group_field="speaker_id")


def test_mct_condition_and_rir_choice_are_reproducible() -> None:
    first = choose_mct_condition("u1", epoch=2, seed=9)
    second = choose_mct_condition("u1", epoch=2, seed=9)
    rir_first = choose_rir_id("u1", ["r2", "r1", "r3"], epoch=2, seed=9)
    rir_second = choose_rir_id("u1", ["r3", "r2", "r1"], epoch=2, seed=9)

    assert first == second
    assert rir_first == rir_second


def test_mct_hash_sampling_is_close_to_requested_probability() -> None:
    conditions = [
        choose_mct_condition(f"u{index}", epoch=0, reverb_probability=0.5)
        for index in range(2_000)
    ]
    reverb_fraction = conditions.count("raw_reverb") / len(conditions)

    assert 0.46 < reverb_fraction < 0.54


def test_cache_digest_changes_with_decoder_protocol() -> None:
    digest = "a" * 64
    identity = CacheIdentity(digest, digest, "revision", None, digest, digest)
    changed = CacheIdentity(
        digest,
        digest,
        "revision",
        None,
        "b" * 64,
        digest,
    )

    assert identity.digest != changed.digest
    assert canonical_sha256({"b": 1, "a": 2}) == canonical_sha256(
        {"a": 2, "b": 1}
    )

