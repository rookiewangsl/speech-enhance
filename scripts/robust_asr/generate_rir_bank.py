#!/usr/bin/env python3
"""Generate a formal or smoke Pyroomacoustics RIR bank."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from robust_asr.acoustics.bank import RIRBankProgress, generate_rir_bank
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
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _duration(seconds: float) -> str:
    rounded = max(0, round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, final_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{final_seconds:02d}"
    return f"{minutes:02d}:{final_seconds:02d}"


def progress_printer(*, every: int, quiet: bool):
    if every <= 0:
        raise ValueError("--progress-every must be positive")
    started = time.perf_counter()

    def report(value: RIRBankProgress) -> None:
        if quiet:
            return
        if value.completed == 0:
            print(
                f"RIR bank {value.split}: 0/{value.total}",
                file=sys.stderr,
                flush=True,
            )
            return
        if value.completed != value.total and value.completed % every != 0:
            return
        elapsed = time.perf_counter() - started
        speed = value.completed / max(elapsed, 1e-9)
        eta = (value.total - value.completed) / max(speed, 1e-9)
        print(
            f"RIR {value.completed}/{value.total} "
            f"[{100.0 * value.completed / value.total:3.0f}%] "
            f"| RT60={value.target_rt60_seconds:.3f}s "
            f"| {speed:.2f} RIR/s | ETA {_duration(eta)}",
            file=sys.stderr,
            flush=True,
        )

    return report


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
        progress_callback=progress_printer(
            every=args.progress_every,
            quiet=args.quiet,
        ),
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
