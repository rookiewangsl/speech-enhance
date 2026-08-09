"""Create a deterministic speaker-balanced subset of a JSONL manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from speech_frontend.dataset import (
    sample_manifest_rows_by_speaker,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--items-per-speaker", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260724)
    arguments = parser.parse_args()

    rows = [
        json.loads(line)
        for line in arguments.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sampled = sample_manifest_rows_by_speaker(
        rows,
        items_per_speaker=arguments.items_per_speaker,
        seed=arguments.seed,
    )
    write_jsonl(sampled, arguments.output)
    speakers = sorted({str(row["speaker_id"]) for row in sampled})
    print(
        f"wrote {len(sampled)} rows from {len(speakers)} speakers to "
        f"{arguments.output}"
    )


if __name__ == "__main__":
    main()
