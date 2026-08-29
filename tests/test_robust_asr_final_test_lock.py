from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from robust_asr.download import sha256_file


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "robust_asr"
    / "validate_final_test_lock.py"
)


def _module():
    specification = importlib.util.spec_from_file_location(
        "validate_final_test_lock", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_final_test_lock_validates_frozen_identities(tmp_path: Path) -> None:
    module = _module()
    manifest = tmp_path / "manifests" / "aishell1" / "test.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    rir_root = tmp_path / "rir" / "pyroom_v1"
    rir_root.mkdir(parents=True)
    (rir_root / "test.audit.json").write_text(
        json.dumps(
            {
                "rooms": 20,
                "geometry_families": 60,
                "rirs": 300,
                "paired_rt60_geometry": True,
                "manifest_sha256": "r" * 64,
            }
        ),
        encoding="utf-8",
    )
    (rir_root / "validation.json").write_text(
        json.dumps({"status": "PASS", "verify_files": True}),
        encoding="utf-8",
    )
    adapter = tmp_path / "runs" / "adapter" / "adapter_model.safetensors"
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b"adapter")
    config = {
        "schema_version": 1,
        "selection_basis": "development_splits_only",
        "test_time_tuning_allowed": False,
        "frontends": ["raw", "m_wpe_10"],
        "rt60_seconds": [0.2, 0.4, 0.6, 0.8, 1.0],
        "manifest": {
            "name": "test.jsonl",
            "utterances": 1,
            "sha256": sha256_file(manifest),
        },
        "rir": {
            "rooms": 20,
            "geometry_families": 60,
            "rirs": 300,
            "paired_rt60_geometry": True,
            "manifest_sha256": "r" * 64,
        },
        "models": {
            "w0_pretrained": {
                "adapter": None,
                "adapter_model_sha256": None,
                "output": "w0.jsonl",
            },
            "w1_clean_lora": {
                "adapter": "runs/adapter",
                "adapter_model_sha256": sha256_file(adapter),
                "output": "clean.jsonl",
            },
        },
    }

    result = module.validate_lock(root=tmp_path, config=config)

    assert result["status"] == "PASS"
    assert result["test_inference_started"] is False
