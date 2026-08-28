#!/usr/bin/env python3
"""Compare a beam audit with matching rows from the frozen greedy baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from robust_asr.decode_compare import compare_greedy_with_beam
from robust_asr.manifest import read_jsonl
from robust_asr.paths import require_data_root


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--greedy-name", required=True)
    parser.add_argument("--beam-name", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def _basename(value: str, option: str) -> str:
    if Path(value).name != value:
        raise ValueError(f"{option} must be a basename")
    return value


def main() -> None:
    args = arguments()
    root = require_data_root(args.data_root)
    greedy_name = _basename(args.greedy_name, "--greedy-name")
    beam_name = _basename(args.beam_name, "--beam-name")
    output_name = _basename(args.output_name, "--output-name")
    summary = compare_greedy_with_beam(
        read_jsonl(root / "outputs" / greedy_name),
        read_jsonl(root / "outputs" / beam_name),
        draws=args.bootstrap_draws,
        seed=args.seed,
    )
    destination = root / "outputs" / output_name
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
