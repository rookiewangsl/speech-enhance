from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from robust_asr.manifest import read_jsonl, write_jsonl_atomic
from robust_asr.models.whisper_lora import lora_parameter_sha256


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "robust_asr"
    / "run_lora_whisper_baseline.py"
)


def _module():
    specification = importlib.util.spec_from_file_location(
        "run_lora_whisper_baseline", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_lora_parameter_fingerprint_includes_frozen_adapter() -> None:
    torch = pytest.importorskip("torch")

    class Adapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = torch.nn.Parameter(torch.tensor([1.0]))
            self.lora_A = torch.nn.Parameter(torch.tensor([2.0]))

    model = Adapter()
    model.requires_grad_(False)
    first = lora_parameter_sha256(model)
    with torch.no_grad():
        model.base.add_(1.0)
    assert lora_parameter_sha256(model) == first
    with torch.no_grad():
        model.lora_A.add_(1.0)
    assert lora_parameter_sha256(model) != first


def test_seed_predictions_requires_matching_adapter_identity(tmp_path: Path) -> None:
    module = _module()
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "destination.jsonl"
    revision = "base+lora:" + "a" * 64
    rows = [
        {
            "utterance_id": "u1",
            "frontend": "raw",
            "model_revision": revision,
        }
    ]
    write_jsonl_atomic(source, rows)

    assert module.seed_predictions(
        source,
        destination,
        expected_model_revision=revision,
    ) == 1
    assert read_jsonl(destination) == rows

    incompatible = tmp_path / "incompatible.jsonl"
    write_jsonl_atomic(
        incompatible,
        [{**rows[0], "model_revision": "base+lora:" + "b" * 64}],
    )
    with pytest.raises(ValueError, match="disagree with the loaded"):
        module.seed_predictions(
            incompatible,
            tmp_path / "other.jsonl",
            expected_model_revision=revision,
        )
