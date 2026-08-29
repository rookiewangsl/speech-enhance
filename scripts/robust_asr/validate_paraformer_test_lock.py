#!/usr/bin/env python3
"""Validate the frozen Paraformer held-out cross-model test lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robust_asr.config import canonical_sha256, load_json_object
from robust_asr.download import sha256_file
from robust_asr.paths import require_data_root


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/robust_asr/paraformer_test.json"),
    )
    return parser.parse_args()


def validate_lock(*, root: Path, config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != 1:
        raise ValueError("unsupported Paraformer test schema")
    if config.get("selection_basis") != "pre_registered_frozen_cross_model_extension":
        raise ValueError("Paraformer test is not a pre-registered extension")
    if config.get("whisper_test_used_for_parameter_selection") is not False:
        raise ValueError("Whisper test results cannot select Paraformer parameters")
    if config.get("test_time_tuning_allowed") is not False:
        raise ValueError("Paraformer test-time tuning must be disabled")
    if config.get("frontends") != ["raw", "m_wpe_10"]:
        raise ValueError("Paraformer frontends must be Raw and M-WPE-10")
    if config.get("rt60_seconds") != [0.2, 0.4, 0.6, 0.8, 1.0]:
        raise ValueError("Paraformer RT60 grid is not frozen")

    manifest = config["manifest"]
    manifest_path = root / "manifests" / "aishell1" / manifest["name"]
    observed_manifest_sha = sha256_file(manifest_path)
    if observed_manifest_sha != manifest["sha256"]:
        raise ValueError("Paraformer test manifest SHA-256 mismatch")
    with manifest_path.open(encoding="utf-8") as stream:
        rows = sum(1 for line in stream if line.strip())
    if rows != int(manifest["rows"]):
        raise ValueError("Paraformer test manifest row count mismatch")
    utterances = int(manifest["utterance_limit"])
    if utterances <= 0 or utterances > rows:
        raise ValueError("invalid Paraformer test utterance limit")

    rir_audit = load_json_object(root / "rir" / "pyroom_v1" / "test.audit.json")
    if rir_audit.get("manifest_sha256") != config["rir"]["manifest_sha256"]:
        raise ValueError("Paraformer test RIR manifest SHA-256 mismatch")
    validation = load_json_object(root / "rir" / "pyroom_v1" / "validation.json")
    if validation.get("status") != "PASS" or not validation.get("verify_files"):
        raise ValueError("formal RIR files have not passed full verification")

    model = config["model"]
    if any(
        model.get(field) is not None
        for field in ("vad_model", "punctuation_model", "language_model", "hotword")
    ):
        raise ValueError("Paraformer cross-check cannot use auxiliary models")
    model_path = root / model["path"] / "model.pt"
    observed_model_sha = sha256_file(model_path)
    if observed_model_sha != model["model_pt_sha256"]:
        raise ValueError("Paraformer model.pt SHA-256 mismatch")

    output_path = root / "outputs" / config["output"]
    output_rows = 0
    if output_path.is_file():
        with output_path.open(encoding="utf-8") as stream:
            output_rows = sum(1 for line in stream if line.strip())
    expected_rows = utterances * (
        1 + len(config["frontends"]) * len(config["rt60_seconds"])
    )
    if output_rows > expected_rows:
        raise ValueError("Paraformer test output has too many rows")
    return {
        "schema_version": 1,
        "status": "PASS",
        "lock_sha256": canonical_sha256(config),
        "manifest_sha256": observed_manifest_sha,
        "rir_manifest_sha256": rir_audit["manifest_sha256"],
        "model_pt_sha256": observed_model_sha,
        "utterances": utterances,
        "expected_rows": expected_rows,
        "output_rows": output_rows,
        "test_inference_started": output_rows > 0,
    }


def main() -> None:
    args = arguments()
    root = require_data_root(args.data_root)
    config = load_json_object(args.config)
    print(json.dumps(validate_lock(root=root, config=config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
