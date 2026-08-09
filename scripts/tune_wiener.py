"""Small, interpretable dev-set sweep for DD-Wiener parameters."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from speech_frontend.audio import read_audio, validate_aligned_pair
from speech_frontend.enhancement.wiener import WienerConfig
from speech_frontend.metrics import si_sdr
from speech_frontend.pipeline import ClassicalEnhancer


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--alpha-dd", nargs="+", type=float, default=[0.92, 0.96, 0.98])
    parser.add_argument("--gain-floor", nargs="+", type=float, default=[0.03, 0.05, 0.10])
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/wiener_tuning.json"))
    arguments = parser.parse_args()
    project_root = arguments.project_root.resolve()
    rows = [
        json.loads(line)
        for line in resolve(project_root, str(arguments.manifest)).read_text().splitlines()
    ]
    if arguments.limit is not None:
        rows = rows[: arguments.limit]
    if not rows:
        raise ValueError("manifest has no rows")

    pairs = []
    for row in rows:
        clean = read_audio(resolve(project_root, row["clean"]))
        noisy = read_audio(resolve(project_root, row["noisy"]))
        validate_aligned_pair(clean, noisy)
        pairs.append((row["id"], clean.samples, noisy.samples))

    results: list[dict[str, float | int]] = []
    for alpha_dd, gain_floor in itertools.product(
        arguments.alpha_dd,
        arguments.gain_floor,
    ):
        config = WienerConfig(alpha_dd=alpha_dd, gain_floor=gain_floor)
        improvements = []
        for _, clean, noisy in pairs:
            enhanced = ClassicalEnhancer(wiener_config=config).enhance(
                noisy,
                method="mcra_dd_wiener",
            ).samples
            improvements.append(si_sdr(clean, enhanced) - si_sdr(clean, noisy))
        results.append(
            {
                "alpha_dd": alpha_dd,
                "gain_floor": gain_floor,
                "mean_si_sdri_db": float(np.mean(improvements)),
                "median_si_sdri_db": float(np.median(improvements)),
                "negative_files": int(np.count_nonzero(np.asarray(improvements) < 0.0)),
            }
        )
    results.sort(
        key=lambda result: (
            result["mean_si_sdri_db"],
            result["median_si_sdri_db"],
            -result["negative_files"],
        ),
        reverse=True,
    )
    output = arguments.output
    output = output if output.is_absolute() else project_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
