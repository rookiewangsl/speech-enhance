"""Formal integrity and split-isolation checks for generated RIR banks."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from robust_asr.acoustics.rir import rt60_within_tolerance
from robust_asr.config import canonical_sha256
from robust_asr.download import sha256_file
from robust_asr.manifest import read_jsonl, validate_disjoint_groups


def _safe_path(root: Path, value: object) -> Path:
    relative = Path(str(value))
    if relative.is_absolute():
        raise ValueError(f"RIR path must be relative: {relative}")
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"RIR path escapes bank root: {relative}") from exc
    return target


def _validate_array_file(root: Path, row: Mapping[str, Any]) -> None:
    path = _safe_path(root, row.get("path"))
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_sha = row.get("file_sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError(f"invalid file SHA-256 for {row.get('rir_id')}")
    if sha256_file(path) != expected_sha:
        raise ValueError(f"RIR SHA-256 mismatch: {path}")
    with np.load(path) as archive:
        if set(archive.files) != {"direct", "full"}:
            raise ValueError(f"unexpected NPZ members: {path} {archive.files}")
        full = np.asarray(archive["full"])
        direct = np.asarray(archive["direct"])
    for key, values in (("full", full), ("direct", direct)):
        expected_shape = tuple(row.get(f"{key}_shape", ()))
        if values.ndim != 2 or values.shape[0] != 4 or values.shape[1] <= 0:
            raise ValueError(f"invalid {key} shape: {path} {values.shape}")
        if expected_shape != values.shape:
            raise ValueError(
                f"{key} shape disagrees with manifest: {path} "
                f"{values.shape} != {expected_shape}"
            )
        if not np.issubdtype(values.dtype, np.floating):
            raise ValueError(f"{key} is not floating point: {path}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{key} contains NaN or infinity: {path}")
    if direct.shape[1] > full.shape[1]:
        raise ValueError(f"direct RIR is longer than full RIR: {path}")


def _validate_rows(
    root: Path,
    *,
    split: str,
    expected_rirs: int,
    expected_rooms: int,
    expected_rt60: Sequence[float] | None,
    verify_files: bool,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = root / f"{split}.jsonl"
    audit_path = root / f"{split}.audit.json"
    rows = read_jsonl(manifest_path)
    if len(rows) != expected_rirs:
        raise ValueError(
            f"{split} RIR count mismatch: {len(rows)} != {expected_rirs}"
        )
    identifiers = [str(row.get("rir_id", "")) for row in rows]
    if any(not value for value in identifiers) or len(set(identifiers)) != len(rows):
        raise ValueError(f"{split} has empty or duplicate RIR ids")
    if {row.get("split") for row in rows} != {split}:
        raise ValueError(f"{split} manifest contains another split")
    room_ids = {str(row.get("room_id", "")) for row in rows}
    if "" in room_ids or len(room_ids) != expected_rooms:
        raise ValueError(
            f"{split} room count mismatch: {len(room_ids)} != {expected_rooms}"
        )
    geometry_ids = [str(row.get("geometry_id", "")) for row in rows]
    if any(not value for value in geometry_ids):
        raise ValueError(f"{split} has an empty geometry id")

    targets = {float(row["target_rt60_seconds"]) for row in rows}
    if split == "train":
        if any(value < 0.2 or value > 1.0 for value in targets):
            raise ValueError("train RT60 is outside [0.2, 1.0]")
    elif expected_rt60 is not None and targets != set(map(float, expected_rt60)):
        raise ValueError(f"{split} RT60 targets disagree with the protocol")

    for row in rows:
        target = float(row["target_rt60_seconds"])
        measured = tuple(map(float, row.get("measured_rt60_seconds", ())))
        if len(measured) != 4 or not all(
            np.isfinite(value) and value > 0 for value in measured
        ):
            raise ValueError(f"invalid measured RT60: {row['rir_id']}")
        if not rt60_within_tolerance(float(np.median(measured)), target):
            raise ValueError(f"RT60 outside tolerance: {row['rir_id']}")
        drr = tuple(map(float, row.get("drr_db", ())))
        if len(drr) != 4 or not all(np.isfinite(value) for value in drr):
            raise ValueError(f"invalid DRR: {row['rir_id']}")
        scene = row.get("scene")
        if not isinstance(scene, Mapping):
            raise ValueError(f"missing scene geometry: {row['rir_id']}")
    if verify_files:
        validate_file = partial(_validate_array_file, root)
        if workers == 1:
            for row in rows:
                validate_file(row)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Consume map in manifest order so the first reported failure is
                # deterministic even though hashing/decompression is parallel.
                tuple(executor.map(validate_file, rows))

    declared_paths = {_safe_path(root, row["path"]) for row in rows}
    actual_paths = set((root / split).glob("*.npz"))
    if declared_paths != actual_paths:
        raise ValueError(
            f"{split} NPZ inventory disagrees with manifest: "
            f"declared={len(declared_paths)} actual={len(actual_paths)}"
        )

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("manifest_sha256") != canonical_sha256(rows):
        raise ValueError(f"{split} audit manifest SHA-256 mismatch")
    if int(audit.get("rirs", -1)) != len(rows):
        raise ValueError(f"{split} audit RIR count mismatch")
    return rows, audit


def _validate_test_families(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_rt60: Sequence[float],
    expected_families: int,
) -> None:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        family = str(row.get("rir_family_id", ""))
        if not family:
            raise ValueError("test has an empty RIR family id")
        grouped.setdefault(family, []).append(row)
    if len(grouped) != expected_families:
        raise ValueError(
            f"test family count mismatch: {len(grouped)} != {expected_families}"
        )
    target_set = set(map(float, expected_rt60))
    for family, values in grouped.items():
        if {float(row["target_rt60_seconds"]) for row in values} != target_set:
            raise ValueError(f"test family lacks the full RT60 grid: {family}")
        if len({str(row["geometry_id"]) for row in values}) != 1:
            raise ValueError(f"test family changes geometry id: {family}")
        scenes = {json.dumps(row["scene"], sort_keys=True) for row in values}
        if len(scenes) != 1:
            raise ValueError(f"test family changes scene geometry: {family}")


def validate_formal_rir_banks(
    bank_root: str | Path,
    *,
    train_rooms: int,
    train_positions_per_room: int,
    dev_rooms: int,
    dev_positions_per_rt60: int,
    test_rooms: int,
    test_positions_per_rt60: int,
    fixed_rt60_seconds: Sequence[float],
    verify_files: bool = True,
    workers: int = 1,
) -> dict[str, Any]:
    """Validate formal train/dev/test banks and return a compact audit."""

    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    root = Path(bank_root).resolve()
    expected = {
        "train": (
            train_rooms * train_positions_per_room,
            train_rooms,
            None,
        ),
        "dev": (
            dev_rooms * dev_positions_per_rt60 * len(fixed_rt60_seconds),
            dev_rooms,
            fixed_rt60_seconds,
        ),
        "test": (
            test_rooms * test_positions_per_rt60 * len(fixed_rt60_seconds),
            test_rooms,
            fixed_rt60_seconds,
        ),
    }
    splits: dict[str, list[dict[str, Any]]] = {}
    audits: dict[str, dict[str, Any]] = {}
    for split, (rirs, rooms, rt60) in expected.items():
        rows, audit = _validate_rows(
            root,
            split=split,
            expected_rirs=rirs,
            expected_rooms=rooms,
            expected_rt60=rt60,
            verify_files=verify_files,
            workers=workers,
        )
        splits[split] = rows
        audits[split] = audit

    validate_disjoint_groups(splits, group_field="room_id")
    validate_disjoint_groups(splits, group_field="geometry_id")
    _validate_test_families(
        splits["test"],
        expected_rt60=fixed_rt60_seconds,
        expected_families=test_rooms * test_positions_per_rt60,
    )
    return {
        "schema_version": 1,
        "status": "PASS",
        "verify_files": verify_files,
        "workers": workers,
        "bank_root": str(root),
        "splits": {
            split: {
                "rirs": len(rows),
                "rooms": len({row["room_id"] for row in rows}),
                "geometry_families": int(
                    audits[split].get(
                        "geometry_families",
                        len(
                            {
                                row.get("rir_family_id", row["geometry_id"])
                                for row in rows
                            }
                        ),
                    )
                ),
                "paired_rt60_geometry": bool(
                    audits[split].get(
                        "paired_rt60_geometry",
                        split == "test",
                    )
                ),
                "manifest_sha256": audits[split]["manifest_sha256"],
            }
            for split, rows in splits.items()
        },
    }


def write_validation_atomic(path: str | Path, summary: Mapping[str, Any]) -> None:
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
