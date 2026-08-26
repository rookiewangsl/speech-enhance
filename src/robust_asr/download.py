"""Auditable large-file download metadata and validation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO


@dataclass(frozen=True)
class DownloadReceipt:
    url: str
    path: str
    bytes: int
    sha256: str
    completed_at_utc: str


def sha256_stream(stream: BinaryIO, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    while chunk := stream.read(chunk_bytes):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return sha256_stream(stream)


def validate_download(
    path: str | Path,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> tuple[int, str]:
    """Validate length/checksum and return the observed identity."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    observed_bytes = source.stat().st_size
    if expected_bytes is not None and observed_bytes != expected_bytes:
        raise ValueError(
            f"unexpected byte length for {source}: {observed_bytes}, "
            f"expected {expected_bytes}"
        )
    observed_sha = sha256_file(source)
    if expected_sha256 is not None and observed_sha.lower() != expected_sha256.lower():
        raise ValueError(
            f"SHA-256 mismatch for {source}: {observed_sha}, "
            f"expected {expected_sha256}"
        )
    return observed_bytes, observed_sha


def finalize_download(
    partial_path: str | Path,
    destination: str | Path,
    *,
    url: str,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> DownloadReceipt:
    """Validate a `.part`, atomically publish it, and write a receipt."""

    partial = Path(partial_path)
    target = Path(destination)
    observed_bytes, observed_sha = validate_download(
        partial,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, target)
    receipt = DownloadReceipt(
        url=url,
        path=str(target),
        bytes=observed_bytes,
        sha256=observed_sha,
        completed_at_utc=datetime.now(UTC).isoformat(),
    )
    receipt_path = target.with_suffix(target.suffix + ".receipt.json")
    temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(receipt), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, receipt_path)
    return receipt
