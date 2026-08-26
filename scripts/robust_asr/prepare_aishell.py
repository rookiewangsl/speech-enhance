#!/usr/bin/env python3
"""Create audited AISHELL-1 split manifests and a balanced train subset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from robust_asr.aishell import prepare_manifests
from robust_asr.paths import require_data_root


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--train-hours", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--minimum-seconds", type=float, default=0.5)
    parser.add_argument("--maximum-seconds", type=float, default=25.0)
    parser.add_argument("--maximum-clipped-fraction", type=float, default=0.01)
    parser.add_argument("--dev-model-utterances", type=int, default=1_000)
    parser.add_argument("--dev-frontend-utterances", type=int, default=500)
    parser.add_argument("--test-reverb-utterances", type=int, default=1_000)
    parser.add_argument("--measured-rir-test-utterances", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    root = require_data_root(args.data_root)
    manifest_root = root / "manifests" / "aishell1"
    audit = prepare_manifests(
        root / "corpora" / "aishell1",
        manifest_root,
        train_subset_hours=args.train_hours,
        seed=args.seed,
        minimum_duration_seconds=args.minimum_seconds,
        maximum_duration_seconds=args.maximum_seconds,
        maximum_clipped_fraction=args.maximum_clipped_fraction,
        dev_model_utterances=args.dev_model_utterances,
        dev_frontend_utterances=args.dev_frontend_utterances,
        test_reverb_utterances=args.test_reverb_utterances,
        measured_rir_test_utterances=args.measured_rir_test_utterances,
    )
    audit_path = manifest_root / "audit.json"
    temporary = audit_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, audit_path)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
