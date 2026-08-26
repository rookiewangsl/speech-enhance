"""Manifest utilities that enforce deterministic augmentation and split safety."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON objects from a UTF-8 JSONL file."""

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} is not an object")
        rows.append(value)
    return rows


def write_jsonl_atomic(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write JSONL through a temporary sibling and atomic replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            for row in rows:
                output.write(
                    json.dumps(
                        dict(row),
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def validate_disjoint_groups(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    group_field: str,
) -> None:
    """Reject speaker-, room-, or geometry-group leakage across splits."""

    groups: dict[str, set[str]] = {}
    for split, rows in splits.items():
        values: set[str] = set()
        for row in rows:
            value = row.get(group_field)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"split {split!r} has an invalid {group_field!r}"
                )
            values.add(value)
        groups[split] = values

    names = sorted(groups)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = sorted(groups[left_name] & groups[right_name])
            if overlap:
                raise ValueError(
                    f"{group_field} leakage between {left_name} and "
                    f"{right_name}: {overlap[:5]}"
                )


def stable_unit_interval(*parts: object) -> float:
    """Map protocol identity parts deterministically to [0, 1)."""

    payload = "\0".join(map(str, parts)).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / 2**64


def choose_mct_condition(
    utterance_id: str,
    *,
    epoch: int,
    seed: int = 2026,
    reverb_probability: float = 0.5,
) -> str:
    """Choose clean/raw-reverb reproducibly for one epoch and utterance."""

    if not utterance_id:
        raise ValueError("utterance_id cannot be empty")
    if epoch < 0:
        raise ValueError("epoch cannot be negative")
    if not 0.0 <= reverb_probability <= 1.0:
        raise ValueError("reverb_probability must be in [0, 1]")
    draw = stable_unit_interval("condition", seed, epoch, utterance_id)
    return "raw_reverb" if draw < reverb_probability else "clean"


def choose_rir_id(
    utterance_id: str,
    rir_ids: Sequence[str],
    *,
    epoch: int,
    seed: int = 2026,
) -> str:
    """Choose one train RIR without depending on process hash randomization."""

    if not rir_ids:
        raise ValueError("rir_ids cannot be empty")
    if len(set(rir_ids)) != len(rir_ids):
        raise ValueError("rir_ids must be unique")
    if epoch < 0:
        raise ValueError("epoch cannot be negative")
    ordered = sorted(rir_ids)
    draw = stable_unit_interval("rir", seed, epoch, utterance_id)
    index = min(int(draw * len(ordered)), len(ordered) - 1)
    return ordered[index]

