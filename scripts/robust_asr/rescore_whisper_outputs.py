#!/usr/bin/env python3
"""Rescore existing Mandarin ASR outputs under number-aware policies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from robust_asr.manifest import read_jsonl
from robust_asr.paths import require_data_root
from robust_asr.rescore import rescore_result_rows


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--input-name", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _basename(value: str, option: str) -> str:
    if Path(value).name != value:
        raise ValueError(f"{option} must be a basename")
    return value


def main() -> None:
    args = arguments()
    root = require_data_root(args.data_root)
    input_name = _basename(args.input_name, "--input-name")
    output_name = _basename(args.output_name, "--output-name")
    rows = read_jsonl(root / "outputs" / input_name)
    summary = rescore_result_rows(rows)
    destination = root / "outputs" / output_name
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    if args.quiet:
        print(
            f"rescored {summary['result_rows']} rows: "
            f"{input_name} -> {output_name}"
        )
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
