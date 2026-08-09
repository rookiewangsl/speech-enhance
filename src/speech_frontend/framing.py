"""Framing and weighted overlap-add utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def periodic_hann(length: int) -> NDArray[np.float64]:
    """Return a periodic Hann window of ``length`` samples."""

    if length <= 0:
        raise ValueError("window length must be positive")
    indices = np.arange(length, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * indices / length)


def sqrt_periodic_hann(length: int) -> NDArray[np.float64]:
    """Return the square root of a periodic Hann window."""

    return np.sqrt(periodic_hann(length))


def frame_signal(
    signal: FloatArray,
    frame_length: int,
    hop_length: int,
) -> tuple[NDArray[np.float64], int]:
    """Pad and frame a one-dimensional signal.

    ``frame_length - hop_length`` zeros are prepended so that samples at the
    original left boundary are covered by overlapping non-zero window values.
    Additional zeros are appended to end on an exact frame boundary.

    Returns the frames and the number of prepended samples.
    """

    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if frame_length <= 0:
        raise ValueError("frame_length must be positive")
    if hop_length <= 0 or hop_length >= frame_length:
        raise ValueError("hop_length must be in [1, frame_length)")

    left_padding = frame_length - hop_length
    minimum_length = left_padding + signal.size + left_padding
    if minimum_length <= frame_length:
        frame_count = 1
    else:
        frame_count = (
            int(np.ceil((minimum_length - frame_length) / hop_length)) + 1
        )
    padded_length = (frame_count - 1) * hop_length + frame_length
    right_padding = padded_length - left_padding - signal.size
    padded = np.pad(
        np.asarray(signal, dtype=np.float64),
        (left_padding, right_padding),
    )

    shape = (frame_count, frame_length)
    strides = (padded.strides[0] * hop_length, padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=shape,
        strides=strides,
        writeable=False,
    )
    return frames.copy(), left_padding


def weighted_overlap_add(
    frames: FloatArray,
    window: FloatArray,
    hop_length: int,
    *,
    output_length: int,
    left_padding: int,
    epsilon: float = 1e-12,
) -> NDArray[np.float64]:
    """Synthesize a signal using normalized weighted overlap-add."""

    if frames.ndim != 2:
        raise ValueError("frames must be a two-dimensional array")
    if frames.shape[0] == 0:
        raise ValueError("frames must contain at least one frame")
    if window.ndim != 1 or window.size != frames.shape[1]:
        raise ValueError("window length must match frame length")
    if hop_length <= 0 or hop_length >= frames.shape[1]:
        raise ValueError("invalid hop_length")
    if output_length < 0:
        raise ValueError("output_length cannot be negative")
    if left_padding < 0:
        raise ValueError("left_padding cannot be negative")

    total_length = (frames.shape[0] - 1) * hop_length + frames.shape[1]
    output = np.zeros(total_length, dtype=np.float64)
    normalization = np.zeros(total_length, dtype=np.float64)
    window = np.asarray(window, dtype=np.float64)
    window_power = window**2

    for frame_index, frame in enumerate(frames):
        start = frame_index * hop_length
        stop = start + frames.shape[1]
        output[start:stop] += np.asarray(frame, dtype=np.float64) * window
        normalization[start:stop] += window_power

    valid = normalization > epsilon
    output[valid] /= normalization[valid]
    crop_start = left_padding
    crop_stop = crop_start + output_length
    if crop_stop > output.size:
        raise ValueError("requested output exceeds synthesized signal")
    return output[crop_start:crop_stop]
