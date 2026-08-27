from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from robust_asr.training.reporting import (
    ConsoleTrainingReporter,
    EvaluationSummary,
    RunOverview,
    StructuredTrainingLogger,
    TrainingCompletion,
    TrainingProgress,
    TrainingReporter,
)


def overview(output_dir: Path) -> RunOverview:
    return RunOverview(
        experiment_id="mct_encoder_qv_seed2026",
        model_name="whisper-small",
        lora_target="encoder Q/V",
        lora_rank=8,
        trainable_ratio=0.00122,
        train_hours=20.0,
        train_utterances=15_943,
        dev_utterances=1_000,
        clean_probability=0.5,
        reverb_probability=0.5,
        precision="fp16",
        per_device_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        epochs=3,
        device_name="RTX 4070",
        device_memory_gib=12.0,
        output_dir=output_dir,
    )


def progress(step: int) -> TrainingProgress:
    return TrainingProgress(
        epoch=1,
        total_epochs=3,
        step=step,
        steps_per_epoch=100,
        loss=1.432,
        ema_loss=1.508,
        learning_rate=5.5e-5,
        grad_norm=0.84,
        steps_per_second=1.42,
        gpu_memory_gib=7.9,
        eta_seconds=182.0,
    )


def evaluation(tmp_path: Path) -> EvaluationSummary:
    return EvaluationSummary(
        epoch=1,
        total_epochs=3,
        clean_cer=0.0521,
        reverb_cer=0.1284,
        heavy_cer=0.1937,
        best_reverb_cer=0.1284,
        improved=True,
        checkpoint_path=tmp_path / "checkpoints/best",
        per_rt60_cer={0.2: 0.08, 0.8: 0.17, 1.0: 0.21},
        substitutions=100,
        deletions=20,
        insertions=5,
    )


def test_console_is_compact_and_filters_intermediate_steps(tmp_path: Path) -> None:
    stream = io.StringIO()
    console = ConsoleTrainingReporter(stream=stream, every_steps=20, live=False)

    console.start_run(overview(tmp_path))
    console.progress(progress(1))
    console.progress(progress(2))
    console.progress(progress(20))
    console.evaluation(evaluation(tmp_path))

    output = stream.getvalue()
    assert "Run: mct_encoder_qv_seed2026" in output
    assert "1/100" in output
    assert "2/100" not in output
    assert "20/100" in output
    assert "clean CER 5.21%" in output
    assert "reverb CER 12.84%" in output
    assert "per_rt60_cer" not in output
    assert "RT60=" not in output
    assert "protocol_sha256" not in output
    assert "substitutions" not in output


def test_structured_logger_keeps_full_metrics_off_console(tmp_path: Path) -> None:
    stream = io.StringIO()
    reporter = TrainingReporter(
        console=ConsoleTrainingReporter(
            stream=stream, every_steps=20, live=False
        ),
        structured=StructuredTrainingLogger(tmp_path),
        structured_every_steps=10,
    )
    reporter.start(
        overview(tmp_path),
        run_config={"protocol_sha256": "abc", "learning_rate": 1e-4},
        environment={"torch": "test", "cuda": "test"},
        data_audit={"speaker_leakage": "PASS", "room_leakage": "PASS"},
    )
    for step in (1, 2, 10, 20):
        reporter.progress(progress(step))
    reporter.evaluation(evaluation(tmp_path))
    reporter.predictions(
        epoch=1,
        rows=({"utterance_id": "u1", "reference": "你好", "hypothesis": "你好"},),
    )
    reporter.warning(
        code="LOW_GPU_UTILIZATION",
        message="GPU utilization below threshold",
        context={"utilization": 0.35},
    )
    reporter.complete(
        TrainingCompletion(
            best_epoch=1,
            best_reverb_cer=0.1284,
            elapsed_seconds=12_438,
            peak_gpu_memory_gib=7.9,
            checkpoint_path=tmp_path / "checkpoints/best",
        )
    )

    train_rows = [
        json.loads(line)
        for line in (tmp_path / "train_metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    eval_row = json.loads(
        (tmp_path / "eval_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    rt60 = json.loads((tmp_path / "eval_by_rt60.json").read_text(encoding="utf-8"))

    assert [row["step"] for row in train_rows] == [1, 10, 20]
    assert eval_row["substitutions"] == 100
    assert eval_row["per_rt60_cer"]["0.8"] == pytest.approx(0.17)
    assert rt60["epochs"]["1"]["1.0"] == pytest.approx(0.21)
    assert (tmp_path / "predictions.jsonl").is_file()
    assert (tmp_path / "warnings.jsonl").is_file()
    assert (tmp_path / "training_summary.json").is_file()
    assert "protocol_sha256" not in stream.getvalue()
    assert "LOW_GPU_UTILIZATION" in stream.getvalue()


def test_resume_rejects_changed_identity_metadata(tmp_path: Path) -> None:
    logger = StructuredTrainingLogger(tmp_path)
    logger.start(
        run_config={"learning_rate": 1e-4},
        environment={"torch": "test"},
        data_audit={"manifest_sha256": "one"},
    )

    with pytest.raises(ValueError, match="incompatible run metadata"):
        logger.start(
            run_config={"learning_rate": 2e-4},
            environment={"torch": "test"},
            data_audit={"manifest_sha256": "one"},
        )


def test_non_finite_training_metric_is_rejected() -> None:
    with pytest.raises(ValueError, match="loss"):
        TrainingProgress(
            epoch=1,
            total_epochs=1,
            step=1,
            steps_per_epoch=1,
            loss=float("nan"),
            ema_loss=1.0,
            learning_rate=1e-4,
            grad_norm=1.0,
            steps_per_second=1.0,
            gpu_memory_gib=1.0,
            eta_seconds=0.0,
        )
