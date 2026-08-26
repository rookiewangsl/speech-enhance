#!/usr/bin/env python3
"""Download AISHELL-1 with TLS verification, resume, size validation, and receipt."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from robust_asr.download import finalize_download, validate_download
from robust_asr.paths import require_data_root


ARCHIVES = {
    "data": {
        "filename": "data_aishell.tgz",
        "bytes": 15_582_913_665,
    },
    "resource": {
        "filename": "resource_aishell.tgz",
        "bytes": 1_246_920,
    },
}
MIRRORS = (
    "https://openslr.trmal.net/resources/33",
    "https://openslr.elda.org/resources/33",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--archive", choices=("data", "resource"), default="data"
    )
    parser.add_argument("--mirror", choices=("trmal", "elda"), default="trmal")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    root = require_data_root(args.data_root)
    specification = ARCHIVES[args.archive]
    filename = str(specification["filename"])
    expected_bytes = specification["bytes"]
    target = root / "downloads" / filename
    partial = target.with_suffix(target.suffix + ".part")
    mirror_index = 0 if args.mirror == "trmal" else 1
    url = f"{MIRRORS[mirror_index]}/{filename}"

    if target.is_file():
        observed_bytes, observed_sha = validate_download(
            target,
            expected_bytes=expected_bytes if isinstance(expected_bytes, int) else None,
        )
        print(
            json.dumps(
                {
                    "status": "reused",
                    "path": str(target),
                    "bytes": observed_bytes,
                    "sha256": observed_sha,
                },
                indent=2,
            )
        )
        return

    partial.parent.mkdir(parents=True, exist_ok=True)
    complete_partial = (
        partial.is_file()
        and isinstance(expected_bytes, int)
        and partial.stat().st_size == expected_bytes
    )
    if not complete_partial:
        if (
            partial.is_file()
            and isinstance(expected_bytes, int)
            and partial.stat().st_size > expected_bytes
        ):
            raise ValueError(f"partial download exceeds expected size: {partial}")
        command = [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "10",
            "--retry-delay",
            "5",
            "--retry-all-errors",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ]
        subprocess.run(command, check=True)
    receipt = finalize_download(
        partial,
        target,
        url=url,
        expected_bytes=expected_bytes if isinstance(expected_bytes, int) else None,
    )
    print(json.dumps({"status": "downloaded", **asdict(receipt)}, indent=2))


if __name__ == "__main__":
    main()
