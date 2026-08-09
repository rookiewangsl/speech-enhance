"""Summarize enhancement metrics by method, noise condition, and input SNR."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def aggregate(rows: list[dict[str, str]]) -> dict[str, float | int]:
    si_sdri = np.asarray([float(row["si_sdri_db"]) for row in rows])
    stoi = np.asarray([float(row["stoi_improvement"]) for row in rows])
    return {
        "files": len(rows),
        "mean_si_sdri_db": float(np.mean(si_sdri)),
        "median_si_sdri_db": float(np.median(si_sdri)),
        "positive_si_sdri_fraction": float(np.mean(si_sdri > 0.0)),
        "mean_stoi_improvement": float(np.mean(stoi)),
        "nonnegative_stoi_fraction": float(np.mean(stoi >= 0.0)),
    }


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("metrics file has no rows")
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        method = row.get("method", "unknown")
        grouped[("method", method, "all")].append(row)
        grouped[("noise", method, row.get("noise", "unknown"))].append(row)
        grouped[("snr_db", method, row.get("snr_db", "unknown"))].append(row)

    result: dict[str, Any] = {"overall": {}, "by_noise": {}, "by_snr_db": {}}
    section_for = {
        "method": "overall",
        "noise": "by_noise",
        "snr_db": "by_snr_db",
    }
    for (group_type, method, value), group_rows in sorted(grouped.items()):
        section = result[section_for[group_type]]
        if group_type == "method":
            section[method] = aggregate(group_rows)
        else:
            section.setdefault(method, {})[value] = aggregate(group_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    with arguments.input.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = summarize(rows)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
