#!/usr/bin/env python3
"""Derive nested 5/10-hour manifests without rescanning AISHELL audio."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from robust_asr.aishell import derive_duration_subset_rows
from robust_asr.config import canonical_sha256
from robust_asr.manifest import read_jsonl, write_jsonl_atomic
from robust_asr.paths import require_data_root


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--source-name",
        default="aishell1_train_20h.jsonl",
        help="Audited source manifest under manifests/aishell1.",
    )
    parser.add_argument("--hours", type=float, nargs="+", default=(5.0, 10.0))
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if not args.hours or any(value <= 0 for value in args.hours):
        raise ValueError("--hours values must be positive")
    if len(set(args.hours)) != len(args.hours):
        raise ValueError("--hours values must be unique")
    root = require_data_root(args.data_root)
    manifest_root = root / "manifests" / "aishell1"
    source_path = manifest_root / args.source_name
    source_rows = read_jsonl(source_path)
    source_hours = sum(float(row["duration_seconds"]) for row in source_rows) / 3600
    if max(args.hours) > source_hours:
        raise ValueError(
            f"requested {max(args.hours):g} h exceeds source {source_hours:.3f} h"
        )

    derived: dict[float, list[dict]] = {}
    for hours in sorted(args.hours):
        rows = derive_duration_subset_rows(
            source_rows,
            target_hours=hours,
            seed=args.seed,
        )
        name = f"aishell1_train_{hours:g}h.jsonl"
        write_jsonl_atomic(manifest_root / name, rows)
        derived[hours] = rows

    ordered_hours = sorted(derived)
    for smaller, larger in zip(ordered_hours, ordered_hours[1:], strict=False):
        small_ids = {row["utterance_id"] for row in derived[smaller]}
        large_ids = {row["utterance_id"] for row in derived[larger]}
        if not small_ids < large_ids:
            raise ValueError("derived duration manifests are not strictly nested")
    if not {row["utterance_id"] for row in derived[ordered_hours[-1]]} <= {
        row["utterance_id"] for row in source_rows
    }:
        raise ValueError("derived duration manifest escapes the source manifest")

    audit = {
        "schema_version": 1,
        "seed": args.seed,
        "source": source_path.name,
        "source_utterances": len(source_rows),
        "source_hours": source_hours,
        "source_manifest_sha256": canonical_sha256(source_rows),
        "strictly_nested": True,
        "subsets": {
            f"{hours:g}h": {
                "path": f"aishell1_train_{hours:g}h.jsonl",
                "utterances": len(rows),
                "hours": sum(float(row["duration_seconds"]) for row in rows)
                / 3600,
                "speakers": len({row["speaker_id"] for row in rows}),
                "manifest_sha256": canonical_sha256(rows),
            }
            for hours, rows in sorted(derived.items())
        },
    }
    audit_path = manifest_root / "train_subsets.audit.json"
    temporary = audit_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, audit_path)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
