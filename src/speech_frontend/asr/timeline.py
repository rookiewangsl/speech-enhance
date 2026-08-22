"""PNG timeline rendering for endpointed ASR transcript JSONL files."""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TranscriptInterval:
    """One transcript with its offset from the start of a microphone run."""

    sequence: int
    start_seconds: float
    end_seconds: float
    text: str


def _finite_nonnegative(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def intervals_from_rows(rows: Iterable[dict[str, Any]]) -> list[TranscriptInterval]:
    """Read explicit timestamps from realtime ASR transcript records."""

    intervals: list[TranscriptInterval] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError("each transcript row must be a JSON object")
        sequence = int(row.get("sequence", index))
        if "start_seconds" not in row or "end_seconds" not in row:
            raise ValueError(
                "transcript rows need start_seconds and end_seconds; "
                "legacy realtime JSONL requires audio-backed timestamp recovery"
            )
        start = _finite_nonnegative(row["start_seconds"], "start_seconds")
        end = _finite_nonnegative(row["end_seconds"], "end_seconds")
        if end < start:
            raise ValueError("end_seconds must not precede start_seconds")
        intervals.append(
            TranscriptInterval(
                sequence=sequence,
                start_seconds=start,
                end_seconds=end,
                text=str(row.get("text", "")).strip() or "(no speech recognized)",
            )
        )
    return intervals


def format_timestamp(seconds: float) -> str:
    """Format a non-negative offset as ``MM:SS.s``."""

    whole_minutes, remainder = divmod(max(seconds, 0.0), 60.0)
    return f"{int(whole_minutes):02d}:{remainder:04.1f}"


def render_transcript_timeline(
    rows: Iterable[dict[str, Any]],
    output: Path,
    *,
    title: str = "ASR transcript timeline",
) -> list[TranscriptInterval]:
    """Render transcript timing and text as a self-contained PNG figure."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter, MaxNLocator
    except ImportError as error:
        raise RuntimeError(
            "ASR timeline rendering requires matplotlib. "
            "Run: ./.venv-asr/bin/pip install -e '.[asr,demo]'"
        ) from error

    intervals = intervals_from_rows(rows)
    height = max(3.2, 1.8 + 1.45 * max(len(intervals), 1))
    figure = plt.figure(figsize=(15, height), constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=(3, 2))
    axis = figure.add_subplot(grid[0, 0])
    transcript_axis = figure.add_subplot(grid[0, 1], sharey=axis)
    figure.suptitle(title)
    axis.set_xlabel("Time from demo start (MM:SS.s)")
    axis.set_ylabel("ASR segment")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=2))
    axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _pos: format_timestamp(value))
    )
    axis.grid(axis="x", alpha=0.25, linewidth=0.8)
    transcript_axis.set_title("ASR transcription")
    transcript_axis.set_xlim(0.0, 1.0)
    transcript_axis.set_xticks([])
    transcript_axis.tick_params(axis="y", left=False, labelleft=False)
    for spine in transcript_axis.spines.values():
        spine.set_visible(False)

    if not intervals:
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(-0.8, 0.8)
        axis.set_yticks([])
        axis.text(0.5, 0.0, "No completed ASR segments", ha="center", va="center")
        transcript_axis.text(
            0.5,
            0.0,
            "No ASR transcription",
            ha="center",
            va="center",
        )
    else:
        maximum_end = max(interval.end_seconds for interval in intervals)
        axis.set_xlim(0.0, max(1.0, maximum_end + 0.4))
        y_positions = list(range(len(intervals) - 1, -1, -1))
        axis.set_ylim(-0.8, len(intervals) - 0.2)
        axis.set_yticks(y_positions, [f"#{item.sequence:03d}" for item in intervals])
        for interval, y_position in zip(intervals, y_positions, strict=True):
            duration = max(interval.end_seconds - interval.start_seconds, 0.02)
            axis.broken_barh(
                [(interval.start_seconds, duration)],
                (y_position - 0.30, 0.60),
                facecolors="#2878b5",
                edgecolors="#174b70",
                linewidth=0.8,
            )
            label = "\n".join(textwrap.wrap(interval.text, width=52))
            axis.annotate(
                format_timestamp(interval.start_seconds),
                xy=(interval.start_seconds, y_position),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                color="white",
                fontsize=10,
                clip_on=True,
            )
            transcript_axis.text(
                0.01,
                y_position,
                label,
                ha="left",
                va="center",
                color="#1a1a1a",
                fontsize=10,
                wrap=True,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return intervals
