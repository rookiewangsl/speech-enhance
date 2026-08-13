from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "assemble_range_download.py"
SPEC = importlib.util.spec_from_file_location("assemble_range_download", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_assemble_segments_checks_size_and_zip_crc(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("sample.wav", b"audio" * 100)
    payload = source.read_bytes()
    midpoint = len(payload) // 2
    segments = [tmp_path / "part.00", tmp_path / "part.01"]
    segments[0].write_bytes(payload[:midpoint])
    segments[1].write_bytes(payload[midpoint:])

    output = MODULE.assemble_segments(
        segments, tmp_path / "assembled.zip", expected_bytes=len(payload)
    )

    assert output.read_bytes() == payload


def test_assemble_segments_rejects_wrong_total(tmp_path: Path) -> None:
    segment = tmp_path / "part.00"
    segment.write_bytes(b"short")
    with pytest.raises(ValueError, match="bytes mismatch"):
        MODULE.assemble_segments(
            [segment], tmp_path / "assembled.zip", expected_bytes=100
        )


def test_assemble_slices_ignores_extra_tail_bytes(tmp_path: Path) -> None:
    original = tmp_path / "original.zip"
    with zipfile.ZipFile(original, "w") as archive:
        archive.writestr("sample.txt", "slice-aware assembly")
    payload = original.read_bytes()
    split = len(payload) // 2
    first = tmp_path / "first.part"
    second = tmp_path / "second.part"
    first.write_bytes(payload[:split] + b"ignored-tail")
    second.write_bytes(payload[split:] + b"also-ignored")

    output = MODULE.assemble_slices(
        [(first, split), (second, len(payload) - split)],
        tmp_path / "assembled.zip",
        expected_bytes=len(payload),
    )

    assert output.read_bytes() == payload
