#!/usr/bin/env python3
"""Print the formal 3×4×5 experiment matrix as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robust_asr.experiments import build_formal_reverb_matrix
from robust_asr.manifest import write_jsonl_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--utterances", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        cell.as_dict()
        for cell in build_formal_reverb_matrix(
            utterance_count=args.utterances
        )
    ]
    if args.output is not None:
        write_jsonl_atomic(args.output, rows)
    summary = {
        "cells": len(rows),
        "utterances_per_cell": args.utterances,
        "total_asr_inputs": sum(row["utterance_count"] for row in rows),
        "output": str(args.output) if args.output is not None else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

