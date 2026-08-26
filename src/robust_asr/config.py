"""Canonical JSON configuration helpers used by robust-ASR experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value with a stable byte representation."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 digest of canonical JSON data."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_json_object(path: str | Path) -> dict[str, Any]:
    """Load one JSON object and reject other JSON top-level types."""

    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must contain a JSON object: {source}")
    return value


def require_keys(
    value: Mapping[str, Any],
    required: set[str],
    *,
    context: str,
) -> None:
    """Reject a mapping that is missing required protocol fields."""

    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"{context} is missing required keys: {missing}")

