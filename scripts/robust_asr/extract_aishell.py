#!/usr/bin/env python3
"""Safely extract AISHELL-1 outer and per-speaker archives."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from robust_asr.aishell import extract_nested_wav_archives, extract_tar_safely
from robust_asr.paths import require_data_root


def archive_identity(path: Path) -> dict[str, object]:
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    identity: dict[str, object] = {
        "archive": str(path),
        "bytes": path.stat().st_size,
    }
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        identity["sha256"] = receipt["sha256"]
    return identity


def marker_matches(marker: Path, identity: dict[str, object]) -> bool:
    if not marker.is_file():
        return False
    value = json.loads(marker.read_text(encoding="utf-8"))
    return all(value.get(key) == expected for key, expected in identity.items())


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--outer-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    root = require_data_root(args.data_root)
    archive = root / "downloads" / "data_aishell.tgz"
    resource_archive = root / "downloads" / "resource_aishell.tgz"
    corpus_root = root / "corpora" / "aishell1"
    if not archive.is_file():
        raise FileNotFoundError(
            f"AISHELL archive is missing: {archive}; run download_aishell.py"
        )
    corpus_root.mkdir(parents=True, exist_ok=True)
    outer_marker = corpus_root / ".outer_extracted.json"
    outer_identity = archive_identity(archive)
    if marker_matches(outer_marker, outer_identity) and not args.force:
        outer_files = 0
    else:
        outer_files = extract_tar_safely(archive, corpus_root)
        temporary = outer_marker.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({**outer_identity, "files": outer_files}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, outer_marker)
    resource_files = 0
    resource_marker = corpus_root / ".resource_extracted.json"
    resource_identity = (
        archive_identity(resource_archive) if resource_archive.is_file() else None
    )
    if resource_identity is not None and (
        args.force or not marker_matches(resource_marker, resource_identity)
    ):
        resource_files = extract_tar_safely(resource_archive, corpus_root)
        temporary = resource_marker.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {**resource_identity, "files": resource_files},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, resource_marker)
    nested_archives = 0
    if not args.outer_only:
        nested_marker = corpus_root / ".nested_extracted.json"
        nested_identity = {**outer_identity, "nested_layout_version": 2}
        if args.force or not marker_matches(nested_marker, nested_identity):
            nested_archives = extract_nested_wav_archives(corpus_root)
            temporary = nested_marker.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(
                    {**nested_identity, "archives": nested_archives}, indent=2
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, nested_marker)
    print(
        json.dumps(
            {
                "archive": str(archive),
                "corpus_root": str(corpus_root),
                "outer_files_extracted": outer_files,
                "resource_files_extracted": resource_files,
                "nested_archives_extracted": nested_archives,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
