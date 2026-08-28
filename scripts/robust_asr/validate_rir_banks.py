#!/usr/bin/env python3
"""Validate formal train/dev/test RIR banks against the frozen config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robust_asr.acoustics.validation import (
    validate_formal_rir_banks,
    write_validation_atomic,
)
from robust_asr.config import load_json_object
from robust_asr.paths import require_data_root


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/robust_asr/rir.json"),
    )
    parser.add_argument("--skip-file-verification", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    root = require_data_root(args.data_root)
    config = load_json_object(args.config)
    summary = validate_formal_rir_banks(
        root / "rir" / "pyroom_v1",
        train_rooms=int(config["train_rooms"]),
        train_positions_per_room=int(config["train_positions_per_room"]),
        dev_rooms=int(config["dev_rooms"]),
        dev_positions_per_rt60=int(config["dev_positions_per_rt60"]),
        test_rooms=int(config["test_rooms"]),
        test_positions_per_rt60=int(config["test_positions_per_rt60"]),
        fixed_rt60_seconds=tuple(map(float, config["test_rt60_seconds"])),
        verify_files=not args.skip_file_verification,
    )
    write_validation_atomic(
        root / "rir" / "pyroom_v1" / "validation.json",
        summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
