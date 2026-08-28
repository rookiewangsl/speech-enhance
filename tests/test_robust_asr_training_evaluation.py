from __future__ import annotations

from pathlib import Path

import pytest

from robust_asr.manifest import write_jsonl_atomic
from robust_asr.training import evaluation
from robust_asr.training.evaluation import (
    WhisperDevEvaluator,
    trainable_parameter_sha256,
)


def test_trainable_parameter_fingerprint_changes_with_adapter() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1, bias=False)

    first = trainable_parameter_sha256(model)
    with torch.no_grad():
        model.weight.add_(1.0)
    second = trainable_parameter_sha256(model)

    assert len(first) == 64
    assert first != second


def test_whisper_dev_evaluator_builds_frozen_selection_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(1, 1, bias=False)
    rt60_grid = (0.2, 0.4, 0.6, 0.8, 1.0)

    def fake_baseline(**kwargs):
        conditions = [
            {
                "frontend": "clean",
                "target_rt60_seconds": None,
                "cer": 0.10,
            }
        ]
        predictions = []
        for index, rt60 in enumerate(rt60_grid, start=1):
            conditions.append(
                {
                    "frontend": "raw",
                    "target_rt60_seconds": rt60,
                    "cer": 0.10 + index * 0.01,
                }
            )
            predictions.append(
                {
                    "utterance_id": f"u{index}",
                    "frontend": "raw",
                    "target_rt60_seconds": rt60,
                    "substitutions": index,
                    "deletions": 1,
                    "insertions": 2,
                }
            )
        predictions.append(
            {
                "utterance_id": "clean",
                "frontend": "clean",
                "target_rt60_seconds": None,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
            }
        )
        write_jsonl_atomic(kwargs["output_path"], predictions)
        return {"conditions": conditions}

    monkeypatch.setattr(evaluation, "run_frozen_baseline", fake_baseline)
    evaluator = WhisperDevEvaluator(
        processor=object(),
        manifest_path=tmp_path / "dev.jsonl",
        corpus_root=tmp_path / "corpus",
        rir_manifest_path=tmp_path / "rir.jsonl",
        rir_root=tmp_path / "rir",
        output_dir=tmp_path / "run",
        model_id="openai/whisper-small",
        base_revision="a" * 40,
        limit=5,
        rt60_seconds=rt60_grid,
    )

    result = evaluator(model, 2)

    assert result.metrics.epoch == 2
    assert result.metrics.clean_cer == pytest.approx(0.10)
    assert result.metrics.reverb_cer == pytest.approx(0.135)
    assert result.metrics.heavy_cer == pytest.approx(0.145)
    assert result.substitutions == 14
    assert result.deletions == 4
    assert result.insertions == 8
    assert len(result.predictions) == 6


def test_whisper_dev_evaluator_rejects_selection_rt60_outside_grid(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="robust RT60"):
        WhisperDevEvaluator(
            processor=object(),
            manifest_path=tmp_path / "dev.jsonl",
            corpus_root=tmp_path,
            rir_manifest_path=tmp_path / "rir.jsonl",
            rir_root=tmp_path,
            output_dir=tmp_path,
            model_id="model",
            base_revision="revision",
            rt60_seconds=(0.2, 0.4),
            robust_rt60_seconds=(0.6,),
            heavy_rt60_seconds=(0.4,),
        )
