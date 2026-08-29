from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from robust_asr.download import sha256_file


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "robust_asr"
    / "validate_paraformer_test_lock.py"
)


def _module():
    specification = importlib.util.spec_from_file_location(
        "validate_paraformer_test_lock", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_paraformer_test_lock_validates_frozen_inputs(tmp_path: Path) -> None:
    module = _module()
    manifest = tmp_path / "manifests" / "aishell1" / "test.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n{}\n", encoding="utf-8")
    rir = tmp_path / "rir" / "pyroom_v1"
    rir.mkdir(parents=True)
    (rir / "test.audit.json").write_text(
        json.dumps({"manifest_sha256": "r" * 64}), encoding="utf-8"
    )
    (rir / "validation.json").write_text(
        json.dumps({"status": "PASS", "verify_files": True}), encoding="utf-8"
    )
    model = tmp_path / "cache" / "model" / "model.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    config = {
        "schema_version": 1,
        "selection_basis": "pre_registered_frozen_cross_model_extension",
        "whisper_test_used_for_parameter_selection": False,
        "test_time_tuning_allowed": False,
        "frontends": ["raw", "m_wpe_10"],
        "rt60_seconds": [0.2, 0.4, 0.6, 0.8, 1.0],
        "manifest": {
            "name": "test.jsonl",
            "rows": 2,
            "utterance_limit": 1,
            "sha256": sha256_file(manifest),
        },
        "rir": {"manifest_sha256": "r" * 64},
        "model": {
            "path": "cache/model",
            "model_pt_sha256": sha256_file(model),
            "vad_model": None,
            "punctuation_model": None,
            "language_model": None,
            "hotword": None,
        },
        "output": "paraformer.jsonl",
    }

    result = module.validate_lock(root=tmp_path, config=config)

    assert result["status"] == "PASS"
    assert result["expected_rows"] == 11
    assert result["test_inference_started"] is False
