#!/usr/bin/env python3
"""Validate frozen robust-ASR configs without downloading data or models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robust_asr.protocol import load_and_validate_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/robust_asr"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_and_validate_protocol(args.config_dir)
    print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

