"""Assemble and verify ordered HTTP Range segments for a large ZIP archive."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assemble_segments(
    segments: list[Path], output: Path, *, expected_bytes: int
) -> Path:
    """Concatenate ordered segments atomically and run ZIP CRC validation."""

    if not segments:
        raise ValueError("no segments were supplied")
    missing = [path for path in segments if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing segments: {missing}")
    actual_bytes = sum(path.stat().st_size for path in segments)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"segment bytes mismatch: expected {expected_bytes}, got {actual_bytes}"
        )

    temporary = output.with_suffix(output.suffix + ".assembling")
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as destination:
        for path in segments:
            with path.open("rb") as source:
                shutil.copyfileobj(source, destination, length=4 * 1024**2)
    if temporary.stat().st_size != expected_bytes:
        raise OSError("assembled archive size changed unexpectedly")
    with zipfile.ZipFile(temporary) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"ZIP CRC check failed for {bad_member}")
    temporary.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--segment-count", type=int, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    segments = [
        arguments.segments_dir / f"{arguments.prefix}.{index:02d}"
        for index in range(arguments.segment_count)
    ]
    path = assemble_segments(
        segments, arguments.output, expected_bytes=arguments.expected_bytes
    )
    print(
        f"verified {path}: {path.stat().st_size} bytes, "
        f"sha256={sha256(path)}"
    )


if __name__ == "__main__":
    main()
