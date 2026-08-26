#!/usr/bin/env python3
"""Repair macOS XSym placeholders in a Hugging Face cache copied from exFAT."""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path


def _xsym_relative_target(path: Path) -> str | None:
    """Return an XSym's relative blobs target, or ``None`` for a normal file."""

    content = path.read_bytes()
    if not content.startswith(b"XSym\n"):
        return None
    marker = b"../../blobs/"
    start = content.find(marker)
    if start < 0:
        raise ValueError(f"XSym placeholder has no blobs target: {path}")
    relative = content[start:].split(b"\x00", maxsplit=1)[0].splitlines()[0]
    try:
        value = relative.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"XSym placeholder is not ASCII: {path}") from exc
    if not value.startswith("../../blobs/") or "/" in value[len("../../blobs/") :]:
        raise ValueError(f"unsafe XSym blobs target: {path}")
    return value


def repair_xsym(cache_root: Path, *, apply: bool) -> list[Path]:
    """Validate and optionally replace XSym cache placeholders with symlinks."""

    root = cache_root.resolve()
    repaired: list[Path] = []
    for snapshot in root.glob("models--*/snapshots/*"):
        if not snapshot.is_dir():
            continue
        for path in snapshot.iterdir():
            if not path.is_file() or path.is_symlink():
                continue
            relative = _xsym_relative_target(path)
            if relative is None:
                continue
            target = (path.parent / relative).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                raise FileNotFoundError(
                    f"XSym target is absent or outside cache: {path} -> {relative}"
                )
            repaired.append(path)
            if apply:
                temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
                os.symlink(relative, temporary)
                os.replace(temporary, path)
    return repaired


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Replace validated placeholders; without this flag, only audit them.",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    repaired = repair_xsym(args.cache_root, apply=args.apply)
    action = "repaired" if args.apply else "would_repair"
    print({"action": action, "count": len(repaired)})
    for path in repaired:
        print(path)


if __name__ == "__main__":
    main()
