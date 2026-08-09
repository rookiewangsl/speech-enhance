"""Download official VoiceBank + DEMAND paired archives safely.

The downloader is idempotent and keeps partial files so multi-gigabyte training
archives can be resumed after a network interruption.
"""

from __future__ import annotations

import argparse
import hashlib
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Download:
    split: str
    filename: str
    url: str
    minimum_bytes: int


DOWNLOADS = (
    Download(
        "test",
        "clean_testset_wav.zip",
        "https://datashare.ed.ac.uk/bitstreams/"
        "dec213d3-bf57-4777-9663-c24bdce92d5e/download",
        100 * 1024**2,
    ),
    Download(
        "test",
        "noisy_testset_wav.zip",
        "https://datashare.ed.ac.uk/bitstreams/"
        "13c1bfbf-14a6-41db-9b41-8f7310f01ad5/download",
        100 * 1024**2,
    ),
    Download(
        "train28",
        "clean_trainset_28spk_wav.zip",
        "https://datashare.ed.ac.uk/bitstreams/"
        "245452b6-6235-44b6-a6f9-e7eb19797769/download",
        2 * 1024**3,
    ),
    Download(
        "train28",
        "noisy_trainset_28spk_wav.zip",
        "https://datashare.ed.ac.uk/bitstreams/"
        "ecb5a102-bb00-46d3-8af5-40c79823b837/download",
        2 * 1024**3,
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(item: Download, directory: Path, retries: int = 5) -> Path:
    """Download through a resumable temporary path and verify response length."""

    destination = directory / item.filename
    if destination.exists() and destination.stat().st_size >= item.minimum_bytes:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            headers = {"User-Agent": "speech-frontend-dataset-preparer/0.2"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(item.url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                expected = response.headers.get("Content-Length")
                resumed = offset > 0 and response.status == 206
                if offset and not resumed:
                    offset = 0
                mode = "ab" if resumed else "wb"
                with temporary.open(mode) as output:
                    shutil_copy(response, output)
            expected_total = offset + int(expected) if expected is not None else None
            if expected_total is not None and temporary.stat().st_size != expected_total:
                raise OSError("download length does not match Content-Length")
            if temporary.stat().st_size < item.minimum_bytes:
                raise OSError("download is unexpectedly small")
            temporary.replace(destination)
            return destination
        except (OSError, TimeoutError) as error:
            if attempt == retries:
                raise RuntimeError(f"failed to download {item.filename}") from error
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def shutil_copy(source: object, destination: object) -> None:
    while True:
        chunk = source.read(1024**2)
        if not chunk:
            return
        destination.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Dataset root; raw archives are stored below this directory.",
    )
    parser.add_argument(
        "--subset",
        choices=("test", "train28", "all"),
        default="test",
        help="Archive group to download (default: existing small test subset).",
    )
    arguments = parser.parse_args()
    directory = arguments.data_root / "raw" / "voicebank_demand"
    directory.mkdir(parents=True, exist_ok=True)

    selected = (
        DOWNLOADS
        if arguments.subset == "all"
        else tuple(item for item in DOWNLOADS if item.split == arguments.subset)
    )
    for item in selected:
        path = download(item, directory)
        print(f"{path}: {path.stat().st_size} bytes, sha256={sha256(path)}")


if __name__ == "__main__":
    main()
