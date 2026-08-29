#!/usr/bin/env python3
"""Validate the immutable identities required by the final test protocol."""

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
        default=Path("configs/robust_asr/final_test.json"),
    )
    return parser.parse_args()


def validate_lock(*, root: Path, config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != 1:
        raise ValueError("unsupported final-test schema")
    if config.get("selection_basis") != "development_splits_only":
        raise ValueError("final test must be selected only from development results")
    if config.get("test_time_tuning_allowed") is not False:
        raise ValueError("test-time tuning must be disabled")
    if config.get("frontends") != ["raw", "m_wpe_10"]:
        raise ValueError("final frontends must be frozen to Raw and M-WPE-10")
    if config.get("rt60_seconds") != [0.2, 0.4, 0.6, 0.8, 1.0]:
        raise ValueError("final RT60 grid is not frozen")

    manifest = config["manifest"]
    manifest_path = root / "manifests" / "aishell1" / manifest["name"]
    observed_manifest_sha = sha256_file(manifest_path)
    if observed_manifest_sha != manifest["sha256"]:
        raise ValueError("final test manifest SHA-256 mismatch")
    with manifest_path.open(encoding="utf-8") as stream:
        utterances = sum(1 for line in stream if line.strip())
    if utterances != int(manifest["utterances"]):
        raise ValueError("final test manifest utterance count mismatch")

    rir = config["rir"]
    rir_audit = load_json_object(root / "rir" / "pyroom_v1" / "test.audit.json")
    for key in ("rooms", "geometry_families", "rirs", "paired_rt60_geometry"):
        if rir_audit.get(key) != rir[key]:
            raise ValueError(f"final test RIR {key} mismatch")
    if rir_audit.get("manifest_sha256") != rir["manifest_sha256"]:
        raise ValueError("final test RIR manifest SHA-256 mismatch")
    rir_validation = load_json_object(
        root / "rir" / "pyroom_v1" / "validation.json"
    )
    if rir_validation.get("status") != "PASS" or not rir_validation.get(
        "verify_files"
    ):
        raise ValueError("formal RIR files have not passed full verification")

    output_rows: dict[str, int] = {}
    for model_name, model in config["models"].items():
        adapter = model.get("adapter")
        expected_adapter_sha = model.get("adapter_model_sha256")
        if adapter is None:
            if expected_adapter_sha is not None:
                raise ValueError(f"{model_name} has a SHA without an adapter")
        else:
            adapter_path = root / adapter / "adapter_model.safetensors"
            if sha256_file(adapter_path) != expected_adapter_sha:
                raise ValueError(f"{model_name} adapter SHA-256 mismatch")
        output_path = root / "outputs" / model["output"]
        if output_path.is_file():
            with output_path.open(encoding="utf-8") as stream:
                output_rows[model_name] = sum(1 for line in stream if line.strip())
        else:
            output_rows[model_name] = 0

    return {
        "schema_version": 1,
        "status": "PASS",
        "lock_sha256": canonical_sha256(config),
        "manifest_sha256": observed_manifest_sha,
        "utterances": utterances,
        "rir_manifest_sha256": rir_audit["manifest_sha256"],
        "output_rows": output_rows,
        "test_inference_started": any(output_rows.values()),
    }


def main() -> None:
    args = arguments()
    root = require_data_root(args.data_root)
    config = load_json_object(args.config)
    print(
        json.dumps(
            validate_lock(root=root, config=config),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
