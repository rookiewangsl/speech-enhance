"""Controlled room geometry and synthetic reverberation utilities."""

from robust_asr.acoustics.geometry import (
    GeometryProtocol,
    SceneGeometry,
    sample_scene_geometry,
)
from robust_asr.acoustics.rir import (
    ConvolutionResult,
    convolve_multichannel,
    direct_to_reverberant_ratio,
    rt60_within_tolerance,
)

__all__ = [
    "ConvolutionResult",
    "GeometryProtocol",
    "SceneGeometry",
    "convolve_multichannel",
    "direct_to_reverberant_ratio",
    "rt60_within_tolerance",
    "sample_scene_geometry",
]

