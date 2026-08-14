"""Create a deterministic nested speaker/noise/SNR validation subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from speech_frontend.dataset import sample_manifest_rows_by_strata, write_jsonl


FIELDS = ("speaker_id", "noise", "snr_db")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--items-per-stratum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sampled = sample_manifest_rows_by_strata(
        rows,
        fields=FIELDS,
        items_per_stratum=args.items_per_stratum,
        seed=args.seed,
    )
    cell_counts = Counter(tuple(row[field] for field in FIELDS) for row in sampled)
    speakers = sorted({str(row["speaker_id"]) for row in sampled})
    noises = sorted({str(row["noise"]) for row in sampled})
    snrs = sorted({float(row["snr_db"]) for row in sampled})
    expected_cells = len(speakers) * len(noises) * len(snrs)
    if len(cell_counts) != expected_cells or set(cell_counts.values()) != {args.items_per_stratum}:
        raise ValueError("sample does not provide complete balanced speaker/noise/SNR coverage")
    write_jsonl(sampled, args.output)
    audit = {
        "schema_version": 1,
        "strategy": "stable_sha256_rank_within_speaker_noise_snr",
        "seed": args.seed,
        "items_per_stratum": args.items_per_stratum,
        "stratification_fields": list(FIELDS),
        "source": str(args.input),
        "source_sha256": _sha256(args.input),
        "source_rows": len(rows),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "output_rows": len(sampled),
        "speakers": speakers,
        "noises": noises,
        "snr_db": snrs,
        "strata": len(cell_counts),
        "complete_cartesian_coverage": True,
    }
    _atomic_json(args.audit_output, audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
