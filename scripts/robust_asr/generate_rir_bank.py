#!/usr/bin/env python3
"""Generate a formal or smoke Pyroomacoustics RIR bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robust_asr.acoustics.bank import generate_rir_bank
from robust_asr.paths import require_data_root


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--split", choices=("train", "dev", "test", "smoke"), default="smoke"
    )
    parser.add_argument("--rooms", type=int)
    parser.add_argument("--positions-per-target", type=int)
    parser.add_argument("--train-positions-per-room", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rt60", type=float, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    root = require_data_root(args.data_root)
    default_rooms = {"train": 100, "dev": 20, "test": 20, "smoke": 1}
    default_positions = {"train": 1, "dev": 2, "test": 3, "smoke": 1}
    audit = generate_rir_bank(
        root / "rir" / "pyroom_v1",
        split=args.split,
        rooms=args.rooms or default_rooms[args.split],
        positions_per_target=(
            args.positions_per_target or default_positions[args.split]
        ),
        train_positions_per_room=args.train_positions_per_room,
        fixed_rt60_seconds=args.rt60 or (0.2, 0.4, 0.6, 0.8, 1.0),
        seed=args.seed,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
