"""Deterministic train/dev/test Pyroomacoustics RIR-bank generation."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..config import canonical_sha256
from ..download import sha256_file
from ..manifest import write_jsonl_atomic
from .geometry import (
    GeometryProtocol,
    sample_room_dimensions,
    sample_scene_in_room,
)
from .pyroom import simulate_calibrated_rir

RIRSplit = Literal["train", "dev", "test", "smoke"]


def _targets_for_room(
    split: RIRSplit,
    *,
    room_index: int,
    positions_per_target: int,
    fixed_rt60_seconds: Sequence[float],
    train_positions: int,
    seed: int,
) -> list[float]:
    if split == "train":
        rng = np.random.default_rng(seed + room_index)
        return list(map(float, rng.uniform(0.2, 1.0, size=train_positions)))
    return [
        float(target)
        for target in fixed_rt60_seconds
        for _ in range(positions_per_target)
    ]


def _scene_target_groups(
    split: RIRSplit,
    *,
    room_index: int,
    positions_per_target: int,
    fixed_rt60_seconds: Sequence[float],
    train_positions_per_room: int,
    seed: int,
) -> list[tuple[int, tuple[float, ...]]]:
    """Return scene indices and the RT60 values sharing each geometry.

    Only the formal test split is paired across RT60.  Existing train/dev/smoke
    semantics remain one independently sampled scene per RIR, so regenerating a
    legacy dev bank does not silently change its protocol.
    """

    if split == "test":
        targets = tuple(map(float, fixed_rt60_seconds))
        return [
            (family_index, targets)
            for family_index in range(positions_per_target)
        ]
    targets = _targets_for_room(
        split,
        room_index=room_index,
        positions_per_target=positions_per_target,
        fixed_rt60_seconds=fixed_rt60_seconds,
        train_positions=train_positions_per_room,
        seed=seed,
    )
    return [(index, (target,)) for index, target in enumerate(targets)]


def generate_rir_bank(
    destination: str | Path,
    *,
    split: RIRSplit,
    rooms: int,
    positions_per_target: int = 1,
    train_positions_per_room: int = 10,
    fixed_rt60_seconds: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
    seed: int = 2026,
    protocol: GeometryProtocol | None = None,
) -> dict[str, Any]:
    """Generate one bank, its manifest, and a content-identity audit."""

    if split not in {"train", "dev", "test", "smoke"}:
        raise ValueError(f"unknown RIR split: {split}")
    if rooms <= 0 or positions_per_target <= 0 or train_positions_per_room <= 0:
        raise ValueError("room and position counts must be positive")
    if not fixed_rt60_seconds:
        raise ValueError("fixed_rt60_seconds cannot be empty")
    if any(value <= 0 for value in fixed_rt60_seconds):
        raise ValueError("RT60 targets must be positive")
    if len(set(map(float, fixed_rt60_seconds))) != len(fixed_rt60_seconds):
        raise ValueError("RT60 targets must be unique")
    root = Path(destination)
    array_root = root / split
    array_root.mkdir(parents=True, exist_ok=True)
    settings = protocol or GeometryProtocol()
    split_offset = {"train": 0, "dev": 1_000_000, "test": 2_000_000, "smoke": 3_000_000}[split]
    rows: list[dict[str, Any]] = []
    for room_index in range(rooms):
        room_seed = seed + split_offset + room_index * 10_000
        room_dimensions = sample_room_dimensions(room_seed, settings)
        groups = _scene_target_groups(
            split,
            room_index=room_index,
            positions_per_target=positions_per_target,
            fixed_rt60_seconds=fixed_rt60_seconds,
            train_positions_per_room=train_positions_per_room,
            seed=seed + split_offset,
        )
        for scene_index, targets in groups:
            scene_seed = room_seed + scene_index + 1
            scene = sample_scene_in_room(
                room_dimensions,
                seed=scene_seed,
                protocol=settings,
            )
            paired_family = split == "test"
            family_id = f"{split}_r{room_index:03d}_f{scene_index:03d}"
            geometry_id = f"{family_id}_geometry"
            for target_index, target_rt60 in enumerate(targets):
                simulated = simulate_calibrated_rir(
                    scene,
                    target_rt60_seconds=target_rt60,
                    seed=scene_seed,
                )
                if paired_family:
                    rir_id = f"{family_id}_t{target_index:03d}"
                    rir_family_id = family_id
                else:
                    rir_id = f"{split}_r{room_index:03d}_p{scene_index:03d}"
                    rir_family_id = rir_id
                relative_path = Path(split) / f"{rir_id}.npz"
                target = root / relative_path
                temporary = target.with_suffix(".npz.tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(
                        stream,
                        full=simulated.full,
                        direct=simulated.direct,
                    )
                os.replace(temporary, target)
                rows.append(
                    {
                        "rir_id": rir_id,
                        "rir_family_id": rir_family_id,
                        "split": split,
                        "room_id": f"{split}_room_{room_index:03d}",
                        "geometry_id": (
                            geometry_id
                            if paired_family
                            else f"{rir_id}_geometry"
                        ),
                        "path": relative_path.as_posix(),
                        "file_sha256": sha256_file(target),
                        "full_shape": list(simulated.full.shape),
                        "direct_shape": list(simulated.direct.shape),
                        "scene": scene.as_dict(),
                        **simulated.metadata(),
                    }
                )
    manifest_path = root / f"{split}.jsonl"
    write_jsonl_atomic(manifest_path, rows)
    audit = {
        "schema_version": 1,
        "split": split,
        "seed": seed,
        "rooms": rooms,
        "rirs": len(rows),
        "geometry_families": len({row["rir_family_id"] for row in rows}),
        "paired_rt60_geometry": split == "test",
        "rt60_targets": sorted({row["target_rt60_seconds"] for row in rows}),
        "manifest": manifest_path.name,
        "manifest_sha256": canonical_sha256(rows),
    }
    audit_path = root / f"{split}.audit.json"
    temporary_audit = audit_path.with_suffix(".json.tmp")
    temporary_audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_audit, audit_path)
    return audit
