"""Render a transcript JSONL file as a PNG timeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from speech_frontend.asr.timeline import render_transcript_timeline
from speech_frontend.audio import read_audio
from speech_frontend.rnnoise import RNNoiseLibrary, StreamingRNNoise48k
from speech_frontend.rnnoise.resampler import StreamingDownsampler3
from speech_frontend.vad.endpoint import EndpointConfig, EndpointSegment, StreamingEndpointDetector

INPUT_SAMPLE_RATE = 48_000
INPUT_FRAME_SAMPLES = 480
ASR_FRAME_SAMPLES = 160
ASR_FRAME_SECONDS = 0.010


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def recover_endpoint_segments(
    audio_path: Path,
    config: EndpointConfig,
    library: RNNoiseLibrary,
) -> list[EndpointSegment]:
    """Replay the stored microphone stream to recover legacy endpoint times."""

    audio = read_audio(audio_path)
    if audio.sample_rate != INPUT_SAMPLE_RATE:
        raise ValueError(
            f"legacy timestamp recovery needs a {INPUT_SAMPLE_RATE} Hz raw WAV"
        )
    stream = StreamingRNNoise48k(library)
    downsampler = StreamingDownsampler3()
    endpoint = StreamingEndpointDetector(config)
    completed: list[EndpointSegment] = []

    def process_result(samples: np.ndarray, probabilities: np.ndarray) -> None:
        if samples.size == 0:
            if probabilities.size:
                raise RuntimeError("RNNoise returned VAD scores without output audio")
            return
        downsampled = downsampler.process_chunk(samples)
        if downsampled.size != probabilities.size * ASR_FRAME_SAMPLES:
            raise RuntimeError("RNNoise/VAD frames do not align during timestamp recovery")
        for index, probability in enumerate(probabilities):
            start = index * ASR_FRAME_SAMPLES
            segment = endpoint.process_frame(
                downsampled[start : start + ASR_FRAME_SAMPLES],
                float(probability),
            )
            if segment is not None:
                completed.append(segment)

    try:
        for start in range(0, audio.samples.size, INPUT_FRAME_SAMPLES):
            result = stream.process_chunk(
                audio.samples[start : start + INPUT_FRAME_SAMPLES]
            )
            process_result(result.samples, result.vad_probabilities)
        tail = stream.flush()
        # The live demo does not feed the zero-filled RNNoise tail to endpointing.
        downsampler.process_chunk(tail.samples)
        final = endpoint.flush()
        if final is not None:
            completed.append(final)
    finally:
        stream.close()
    return completed


def add_recovered_timestamps(
    rows: list[dict[str, Any]],
    segments: list[EndpointSegment],
) -> list[dict[str, Any]]:
    """Attach recovered timings without rewriting the original legacy JSONL."""

    if len(rows) != len(segments):
        raise ValueError(
            "cannot match legacy transcripts to recovered endpoints: "
            f"{len(rows)} transcript rows but {len(segments)} endpoint segments"
        )
    enriched: list[dict[str, Any]] = []
    for row, segment in zip(rows, segments, strict=True):
        value = dict(row)
        value["start_seconds"] = segment.start_frame * ASR_FRAME_SECONDS
        value["end_seconds"] = segment.end_frame * ASR_FRAME_SECONDS
        enriched.append(value)
    return enriched


def rows_with_timestamps(
    rows: list[dict[str, Any]],
    *,
    audio_path: Path | None,
    report_path: Path | None,
    library_path: Path | None,
) -> list[dict[str, Any]]:
    if all("start_seconds" in row and "end_seconds" in row for row in rows):
        return rows
    if audio_path is None or report_path is None:
        raise ValueError(
            "legacy transcript JSONL has no timestamps; provide its matching "
            "microphone_raw.wav and report.json to recover them"
        )
    report = read_json(report_path)
    endpoint_values = report.get("endpoint")
    if not isinstance(endpoint_values, dict):
        raise ValueError(f"{report_path}: missing endpoint configuration")
    config = EndpointConfig(**endpoint_values)
    segments = recover_endpoint_segments(
        audio_path,
        config,
        RNNoiseLibrary(library_path),
    )
    return add_recovered_timestamps(rows, segments)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--library", type=Path)
    arguments = parser.parse_args()

    output = arguments.output or arguments.input.with_name(
        f"{arguments.input.stem}_timeline.png"
    )
    rows = read_jsonl(arguments.input)
    default_audio = arguments.input.with_name("microphone_raw.wav")
    default_report = arguments.input.with_name("report.json")
    rows = rows_with_timestamps(
        rows,
        audio_path=arguments.audio or (
            default_audio if default_audio.is_file() else None
        ),
        report_path=arguments.report or (
            default_report if default_report.is_file() else None
        ),
        library_path=arguments.library,
    )
    intervals = render_transcript_timeline(
        rows,
        output,
        title=f"ASR transcript timeline: {arguments.input.name}",
    )
    print(f"Wrote {output} ({len(intervals)} segment(s))")


if __name__ == "__main__":
    main()
