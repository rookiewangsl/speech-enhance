from __future__ import annotations

import matplotlib.image as mpimg
import pytest

from speech_frontend.asr.timeline import (
    format_timestamp,
    intervals_from_rows,
    render_transcript_timeline,
)


def test_intervals_require_explicit_timestamps() -> None:
    intervals = intervals_from_rows(
        [
            {
                "sequence": 1,
                "text": "first transcript",
                "duration_seconds": 1.2,
                "start_seconds": 2.3,
                "end_seconds": 3.5,
            }
        ]
    )

    assert (intervals[0].start_seconds, intervals[0].end_seconds) == (2.3, 3.5)
    assert format_timestamp(65.2) == "01:05.2"


def test_intervals_reject_legacy_rows_without_timestamps() -> None:
    with pytest.raises(ValueError, match="legacy"):
        intervals_from_rows(
            [{"sequence": 1, "text": "legacy", "duration_seconds": 1.2}]
        )


def test_timeline_renderer_writes_a_readable_png(tmp_path) -> None:
    output = tmp_path / "timeline.png"

    intervals = render_transcript_timeline(
        [
            {
                "sequence": 1,
                "text": "hello from the first segment",
                "duration_seconds": 1.5,
                "start_seconds": 0.4,
                "end_seconds": 1.9,
            }
        ],
        output,
    )

    assert output.is_file()
    assert output.stat().st_size > 1_000
    assert mpimg.imread(output).ndim == 3
    assert intervals[0].text == "hello from the first segment"
