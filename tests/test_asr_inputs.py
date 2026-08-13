from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from speech_frontend.audio import AudioData, read_audio, write_audio


SCRIPT = Path(__file__).parents[1] / "scripts" / "export_asr_inputs.py"
SPEC = importlib.util.spec_from_file_location("export_asr_inputs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _prepare_project(root: Path) -> tuple[Path, Path, Path]:
    source = root / "data" / "processed" / "voicebank"
    clean_path = source / "clean" / "p001_001.wav"
    noisy_path = source / "noisy" / "p001_001.wav"
    samples = np.linspace(-0.4, 0.4, 1_024, dtype=np.float32)
    write_audio(clean_path, AudioData(samples, 16_000))
    write_audio(noisy_path, AudioData(samples * 0.8, 16_000))
    manifest = root / "data" / "manifests" / "development.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "id": "p001_001",
                "speaker_id": "p001",
                "clean": clean_path.relative_to(root).as_posix(),
                "noisy": noisy_path.relative_to(root).as_posix(),
                "sample_rate": 16_000,
                "num_samples": samples.size,
                "split": "development",
                "source_partition": "official_train28",
                "noise": "babble",
                "snr_db": 5.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    configs = root / "configs"
    configs.mkdir(exist_ok=True)
    protocol = configs / "full_protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "classical_enhancement": {
                    "method": "mcra_dd_wiener",
                    "alpha_dd": 0.92,
                    "gain_floor": 0.2,
                },
                "neural_enhancement": {"chunk_size_16k": 137},
            }
        ),
        encoding="utf-8",
    )
    rnnoise = configs / "rnnoise.json"
    rnnoise.write_text(
        json.dumps(
            {
                "source": {"commit": MODULE.FROZEN_RNNOISE_COMMIT},
                "model": {
                    "archive_sha256": MODULE.FROZEN_RNNOISE_ARCHIVE_SHA256
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest, protocol, rnnoise


def _fake_rnnoise(
    audio: AudioData,
    _library: Path | None,
    chunk_size: int,
) -> tuple[AudioData, dict[str, Any]]:
    assert chunk_size == 137
    delayed = np.pad(audio.samples[:-2], (2, 0))
    return AudioData(delayed.astype(np.float32), audio.sample_rate), {
        "processing_seconds": 0.125,
        "algorithmic_delay_samples": 2,
        "alignment_delay_samples": 2,
        "resampler_input_clipping_samples": 0,
    }


def _export(root: Path, processor: Any = _fake_rnnoise) -> list[dict[str, Any]]:
    manifest, protocol, rnnoise = _prepare_project(root)
    return MODULE.export_asr_inputs(
        manifest_path=manifest,
        project_root=root,
        output_root=Path("exports"),
        output_manifest=Path("exports/manifests/asr_inputs.jsonl"),
        protocol_config_path=protocol,
        rnnoise_config_path=rnnoise,
        rnnoise_processor=processor,
    )


def test_config_digest_is_independent_of_key_order() -> None:
    assert MODULE.config_digest({"a": 1, "b": 2}) == MODULE.config_digest(
        {"b": 2, "a": 1}
    )


def test_export_writes_four_conditions_and_compensates_rnnoise_delay(
    tmp_path: Path,
) -> None:
    rows = _export(tmp_path)

    assert [row["condition"] for row in rows] == list(MODULE.CONDITIONS)
    required = {
        "id",
        "speaker_id",
        "split",
        "noise",
        "snr_db",
        "condition",
        "audio",
        "audio_sha256",
        "duration_seconds",
        "enhancement_config_digest",
        "processing_seconds",
    }
    assert all(required <= row.keys() for row in rows)
    assert all(row["sample_rate"] == 16_000 for row in rows)
    assert all(row["num_samples"] == 1_024 for row in rows)
    assert all(row["cache_status"] == "generated" for row in rows)

    rnnoise_row = next(row for row in rows if row["condition"] == "rnnoise_r3")
    noisy = read_audio(tmp_path / "data/processed/voicebank/noisy/p001_001.wav")
    enhanced = read_audio(tmp_path / rnnoise_row["audio"])
    np.testing.assert_allclose(enhanced.samples[:-2], noisy.samples[:-2])
    np.testing.assert_array_equal(enhanced.samples[-2:], 0.0)
    assert rnnoise_row["delay_compensated"] is True
    assert rnnoise_row["alignment_delay_samples"] == 2

    manifest_rows = [
        json.loads(line)
        for line in (tmp_path / "exports/manifests/asr_inputs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert manifest_rows == rows
    assert not list((tmp_path / "exports").rglob("*.tmp"))


def test_valid_sidecar_and_audio_cache_skips_all_processing(tmp_path: Path) -> None:
    first = _export(tmp_path)

    def fail_if_called(*_args: Any) -> Any:
        raise AssertionError("valid RNNoise cache should have been reused")

    second = _export(tmp_path, fail_if_called)

    assert all(row["cache_status"] == "generated" for row in first)
    assert all(row["cache_status"] == "reused" for row in second)
    assert [row["audio_sha256"] for row in second] == [
        row["audio_sha256"] for row in first
    ]


def test_corrupt_output_invalidates_only_its_cache(tmp_path: Path) -> None:
    rows = _export(tmp_path)
    rnnoise_row = next(row for row in rows if row["condition"] == "rnnoise_r3")
    output = tmp_path / rnnoise_row["audio"]
    write_audio(output, AudioData(np.zeros(1_024, dtype=np.float32), 16_000))
    calls = 0

    def count_calls(*args: Any) -> Any:
        nonlocal calls
        calls += 1
        return _fake_rnnoise(*args)

    rerun = _export(tmp_path, count_calls)

    assert calls == 1
    status = {row["condition"]: row["cache_status"] for row in rerun}
    assert status["rnnoise_r3"] == "generated"
    assert status["clean"] == "reused"
    assert status["noisy"] == "reused"
    assert status["mcra_dd_wiener"] == "reused"


def test_manifest_metadata_change_invalidates_stale_sidecars(tmp_path: Path) -> None:
    _export(tmp_path)
    manifest = tmp_path / "data/manifests/development.jsonl"
    row = json.loads(manifest.read_text(encoding="utf-8"))
    row["noise"] = "corrected-noise-label"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    rerun = MODULE.export_asr_inputs(
        manifest_path=manifest,
        project_root=tmp_path,
        output_root=Path("exports"),
        output_manifest=Path("exports/manifests/asr_inputs.jsonl"),
        protocol_config_path=tmp_path / "configs/full_protocol.json",
        rnnoise_config_path=tmp_path / "configs/rnnoise.json",
        rnnoise_processor=_fake_rnnoise,
    )

    assert all(row["cache_status"] == "generated" for row in rerun)
    assert all(row["noise"] == "corrected-noise-label" for row in rerun)


def test_manifest_replacement_waits_until_all_conditions_succeed(
    tmp_path: Path,
) -> None:
    manifest, protocol, rnnoise = _prepare_project(tmp_path)
    output_manifest = tmp_path / "exports/manifests/asr_inputs.jsonl"
    output_manifest.parent.mkdir(parents=True)
    output_manifest.write_text("previous manifest\n", encoding="utf-8")

    def fail_rnnoise(*_args: Any) -> Any:
        raise RuntimeError("simulated RNNoise failure")

    with pytest.raises(RuntimeError, match="simulated"):
        MODULE.export_asr_inputs(
            manifest_path=manifest,
            project_root=tmp_path,
            output_root=Path("exports"),
            output_manifest=output_manifest,
            protocol_config_path=protocol,
            rnnoise_config_path=rnnoise,
            conditions=("clean", "rnnoise_r3"),
            rnnoise_processor=fail_rnnoise,
        )

    assert output_manifest.read_text(encoding="utf-8") == "previous manifest\n"


def test_duplicate_manifest_ids_are_rejected(tmp_path: Path) -> None:
    manifest, protocol, rnnoise = _prepare_project(tmp_path)
    row = manifest.read_text(encoding="utf-8")
    manifest.write_text(row + row, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate utterance id"):
        MODULE.export_asr_inputs(
            manifest_path=manifest,
            project_root=tmp_path,
            output_root=Path("exports"),
            output_manifest=Path("exports/manifests/asr_inputs.jsonl"),
            protocol_config_path=protocol,
            rnnoise_config_path=rnnoise,
            conditions=("clean",),
        )


def test_manifest_audio_identity_and_declared_length_are_enforced(
    tmp_path: Path,
) -> None:
    manifest, protocol, rnnoise = _prepare_project(tmp_path)
    row = json.loads(manifest.read_text(encoding="utf-8"))
    row["num_samples"] += 1
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="num_samples mismatch"):
        MODULE.export_asr_inputs(
            manifest_path=manifest,
            project_root=tmp_path,
            output_root=Path("exports"),
            output_manifest=Path("exports/manifests/asr_inputs.jsonl"),
            protocol_config_path=protocol,
            rnnoise_config_path=rnnoise,
            conditions=("clean",),
        )
