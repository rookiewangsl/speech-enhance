from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from robust_asr.protocol import CONFIG_FILES, load_and_validate_protocol


CONFIG_ROOT = Path(__file__).parents[1] / "configs" / "robust_asr"


def test_frozen_protocol_validates_without_data_or_models() -> None:
    summary = load_and_validate_protocol(CONFIG_ROOT)

    assert summary.sample_rate == 16_000
    assert summary.microphone_count == 4
    assert summary.model_count == 3
    assert summary.frontend_count == 4
    assert summary.rt60_count == 5
    assert summary.formal_reverb_inputs == 60_000
    assert len(summary.protocol_sha256) == 64


def test_protocol_rejects_cross_file_rt60_mismatch(tmp_path: Path) -> None:
    for filename in CONFIG_FILES:
        shutil.copy2(CONFIG_ROOT / filename, tmp_path / filename)
    evaluation_path = tmp_path / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["rt60_seconds"] = [0.2, 0.5]
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")

    with pytest.raises(ValueError, match="RT60 grids disagree"):
        load_and_validate_protocol(tmp_path)


def test_protocol_rejects_heavy_rt60_outside_evaluation_grid(tmp_path: Path) -> None:
    for filename in CONFIG_FILES:
        shutil.copy2(CONFIG_ROOT / filename, tmp_path / filename)
    lora_path = tmp_path / "lora.json"
    lora = json.loads(lora_path.read_text(encoding="utf-8"))
    lora["logging"]["heavy_rt60_seconds"] = [1.2]
    lora_path.write_text(json.dumps(lora), encoding="utf-8")

    with pytest.raises(ValueError, match="heavy RT60"):
        load_and_validate_protocol(tmp_path)
