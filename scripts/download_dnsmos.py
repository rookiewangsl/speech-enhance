"""Download and verify the frozen official DNSMOS P.835 ONNX model."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

from speech_frontend.dnsmos import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dnsmos_p835.json"))
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    model = config["model"]
    target = args.model_root / str(model["name"])
    expected_sha = str(model["sha256"])
    expected_bytes = int(model["bytes"])
    if target.exists() and not args.force:
        if target.stat().st_size != expected_bytes or sha256_file(target) != expected_sha:
            raise ValueError(f"existing DNSMOS model failed validation: {target}")
        print(json.dumps({"status": "reused", "model": str(target), "sha256": expected_sha}))
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        request = urllib.request.Request(
            str(model["url"]), headers={"User-Agent": "realtime-speech-enhancement-dnsmos/1"}
        )
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.stat().st_size != expected_bytes:
            raise ValueError("downloaded DNSMOS model has unexpected byte length")
        actual_sha = sha256_file(temporary)
        if actual_sha != expected_sha:
            raise ValueError(f"downloaded DNSMOS SHA mismatch: {actual_sha}")
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(json.dumps({"status": "downloaded", "model": str(target), "sha256": expected_sha}))


if __name__ == "__main__":
    main()
