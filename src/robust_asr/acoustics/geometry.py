"""Deterministic 3-D room, source, and four-microphone geometry sampling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import cos, pi, sin, sqrt

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class NumericRange:
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.minimum) or not np.isfinite(self.maximum):
            raise ValueError("range bounds must be finite")
        if self.minimum >= self.maximum:
            raise ValueError("range minimum must be smaller than maximum")

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.minimum, self.maximum))


@dataclass(frozen=True)
class GeometryProtocol:
    """Frozen desk-array room geometry distribution."""

    room_length: NumericRange = NumericRange(4.0, 8.0)
    room_width: NumericRange = NumericRange(3.0, 6.0)
    room_height: NumericRange = NumericRange(2.5, 3.5)
    source_height: NumericRange = NumericRange(1.3, 1.8)
    source_distance: NumericRange = NumericRange(1.0, 4.0)
    array_radius_m: float = 0.05
    array_height_m: float = 0.8
    wall_margin_m: float = 0.5
    microphone_count: int = 4
    maximum_attempts: int = 10_000

    def __post_init__(self) -> None:
        if self.microphone_count != 4:
            raise ValueError("v0.1 protocol requires exactly four microphones")
        if self.array_radius_m <= 0:
            raise ValueError("array_radius_m must be positive")
        if self.wall_margin_m <= self.array_radius_m:
            raise ValueError("wall margin must exceed the array radius")
        if self.array_height_m <= 0:
            raise ValueError("array_height_m must be positive")
        if self.maximum_attempts <= 0:
            raise ValueError("maximum_attempts must be positive")
        if self.source_height.maximum >= self.room_height.minimum:
            raise ValueError("source height must fit every sampled room")


@dataclass(frozen=True)
class SceneGeometry:
    """One valid source and array placement in a 3-D shoebox room."""

    room_dimensions_m: tuple[float, float, float]
    array_center_m: tuple[float, float, float]
    microphone_positions_m: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    source_position_m: tuple[float, float, float]
    source_array_distance_m: float
    azimuth_degrees: float
    seed: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def uniform_circular_array(
    center_m: tuple[float, float, float],
    *,
    radius_m: float,
    microphone_count: int = 4,
) -> FloatArray:
    """Return an `(M, 3)` horizontal uniform circular array."""

    if microphone_count <= 0:
        raise ValueError("microphone_count must be positive")
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    center = np.asarray(center_m, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("center_m must be a finite 3-D point")
    angles = np.arange(microphone_count, dtype=np.float64)
    angles *= 2.0 * pi / microphone_count
    positions = np.repeat(center[np.newaxis, :], microphone_count, axis=0)
    positions[:, 0] += radius_m * np.cos(angles)
    positions[:, 1] += radius_m * np.sin(angles)
    return positions


def _inside_with_margin(
    point: np.ndarray,
    room_dimensions: np.ndarray,
    *,
    horizontal_margin: float,
) -> bool:
    return bool(
        horizontal_margin <= point[0] <= room_dimensions[0] - horizontal_margin
        and horizontal_margin
        <= point[1]
        <= room_dimensions[1] - horizontal_margin
        and 0.0 < point[2] < room_dimensions[2]
    )


def sample_room_dimensions(
    seed: int,
    protocol: GeometryProtocol | None = None,
) -> tuple[float, float, float]:
    """Sample one room independently of source/array positions."""

    settings = protocol or GeometryProtocol()
    rng = np.random.default_rng(seed)
    return (
        settings.room_length.sample(rng),
        settings.room_width.sample(rng),
        settings.room_height.sample(rng),
    )


def sample_scene_in_room(
    room_dimensions_m: tuple[float, float, float],
    *,
    seed: int,
    protocol: GeometryProtocol | None = None,
) -> SceneGeometry:
    """Sample a source/array placement in fixed room dimensions."""

    settings = protocol or GeometryProtocol()
    room = np.asarray(room_dimensions_m, dtype=np.float64)
    if room.shape != (3,) or not np.all(np.isfinite(room)):
        raise ValueError("room_dimensions_m must be a finite 3-D vector")
    if not (
        settings.room_length.minimum <= room[0] <= settings.room_length.maximum
        and settings.room_width.minimum <= room[1] <= settings.room_width.maximum
        and settings.room_height.minimum <= room[2] <= settings.room_height.maximum
    ):
        raise ValueError("room dimensions are outside the frozen protocol")
    rng = np.random.default_rng(seed)
    for _ in range(settings.maximum_attempts):
        array_margin = settings.wall_margin_m + settings.array_radius_m
        if (
            room[0] <= 2.0 * array_margin
            or room[1] <= 2.0 * array_margin
            or settings.array_height_m >= room[2]
        ):
            break
        array_center = np.asarray(
            [
                rng.uniform(array_margin, room[0] - array_margin),
                rng.uniform(array_margin, room[1] - array_margin),
                settings.array_height_m,
            ],
            dtype=np.float64,
        )
        distance = settings.source_distance.sample(rng)
        source_height = settings.source_height.sample(rng)
        vertical_distance = source_height - settings.array_height_m
        if distance <= abs(vertical_distance):
            continue
        horizontal_distance = sqrt(distance**2 - vertical_distance**2)
        azimuth_radians = float(rng.uniform(0.0, 2.0 * pi))
        source = np.asarray(
            [
                array_center[0] + horizontal_distance * cos(azimuth_radians),
                array_center[1] + horizontal_distance * sin(azimuth_radians),
                source_height,
            ],
            dtype=np.float64,
        )
        if not _inside_with_margin(
            source,
            room,
            horizontal_margin=settings.wall_margin_m,
        ):
            continue
        microphones = uniform_circular_array(
            tuple(map(float, array_center)),
            radius_m=settings.array_radius_m,
            microphone_count=settings.microphone_count,
        )
        if not all(
            _inside_with_margin(
                microphone,
                room,
                horizontal_margin=settings.wall_margin_m,
            )
            for microphone in microphones
        ):
            continue
        actual_distance = float(np.linalg.norm(source - array_center))
        if not (
            settings.source_distance.minimum
            <= actual_distance
            <= settings.source_distance.maximum
        ):
            continue
        return SceneGeometry(
            room_dimensions_m=tuple(map(float, room)),
            array_center_m=tuple(map(float, array_center)),
            microphone_positions_m=tuple(
                tuple(map(float, row)) for row in microphones
            ),  # type: ignore[arg-type]
            source_position_m=tuple(map(float, source)),
            source_array_distance_m=actual_distance,
            azimuth_degrees=float(np.degrees(azimuth_radians)),
            seed=seed,
        )
    raise RuntimeError(
        f"failed to sample a valid scene after {settings.maximum_attempts} attempts"
    )


def sample_scene_geometry(
    seed: int,
    protocol: GeometryProtocol | None = None,
) -> SceneGeometry:
    """Sample a valid scene deterministically from one integer seed."""

    settings = protocol or GeometryProtocol()
    room = sample_room_dimensions(seed, settings)
    return sample_scene_in_room(
        room,
        seed=seed ^ 0x5DEECE66D,
        protocol=settings,
    )
