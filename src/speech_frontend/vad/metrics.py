"""Frame-level VAD ground-truth projection and binary metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BinaryVADMetrics:
    precision: float
    recall: float
    f1: float
    false_alarm_rate: float
    miss_rate: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int


def labels_from_intervals(
    frame_count: int,
    *,
    frame_length: int,
    hop_length: int,
    intervals: tuple[tuple[int, int], ...],
) -> NDArray[np.bool_]:
    """Mark a frame as speech when it overlaps any labeled speech interval."""

    if frame_count < 0 or frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame geometry must be positive")
    labels = np.zeros(frame_count, dtype=np.bool_)
    for start, stop in intervals:
        if not 0 <= start < stop:
            raise ValueError("intervals must satisfy 0 <= start < stop")
        first = max(0, (start - frame_length + hop_length) // hop_length)
        last = min(frame_count, int(np.ceil(stop / hop_length)))
        for index in range(first, last):
            frame_start = index * hop_length
            frame_stop = frame_start + frame_length
            labels[index] |= frame_start < stop and start < frame_stop
    return labels


def binary_metrics(
    target: NDArray[np.bool_],
    prediction: NDArray[np.bool_],
) -> BinaryVADMetrics:
    """Compute stable binary VAD metrics for equally shaped label arrays."""

    target = np.asarray(target, dtype=np.bool_)
    prediction = np.asarray(prediction, dtype=np.bool_)
    if target.shape != prediction.shape or target.ndim != 1:
        raise ValueError("target and prediction must be matching vectors")
    tp = int(np.count_nonzero(target & prediction))
    fp = int(np.count_nonzero(~target & prediction))
    fn = int(np.count_nonzero(target & ~prediction))
    tn = int(np.count_nonzero(~target & ~prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryVADMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        false_alarm_rate=fp / (fp + tn) if fp + tn else 0.0,
        miss_rate=fn / (tp + fn) if tp + fn else 0.0,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
    )
