"""External-storage layout for the robust-ASR project."""

from __future__ import annotations

import os
from pathlib import Path


DATA_ROOT_ENV = "ROBUST_ASR_DATA_ROOT"
DATA_SUBDIRECTORIES = (
    "downloads",
    "corpora",
    "manifests",
    "rir",
    "cache/huggingface",
    "outputs",
    "runs",
)


def data_root(explicit: str | Path | None = None) -> Path:
    """Resolve the data root without silently depending on the current directory."""

    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    configured = os.environ.get(DATA_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    raise RuntimeError(
        f"robust-ASR data root is not configured; set {DATA_ROOT_ENV} "
        "or pass --data-root"
    )


def require_data_root(explicit: str | Path | None = None) -> Path:
    """Require an existing writable data root and return its resolved path."""

    root = data_root(explicit)
    if not root.is_dir():
        raise FileNotFoundError(
            f"robust-ASR data root does not exist: {root}; set {DATA_ROOT_ENV}"
        )
    if not os.access(root, os.R_OK | os.W_OK | os.X_OK):
        raise PermissionError(f"robust-ASR data root is not writable: {root}")
    return root


def initialize_data_root(explicit: str | Path | None = None) -> Path:
    """Create the frozen external-storage directory layout."""

    root = data_root(explicit)
    for relative in DATA_SUBDIRECTORIES:
        (root / relative).mkdir(parents=True, exist_ok=True)
    return require_data_root(root)
