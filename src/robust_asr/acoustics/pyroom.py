"""Pyroomacoustics RIR simulation with measured-RT60 calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from .geometry import SceneGeometry
from .rir import direct_to_reverberant_ratio, rt60_within_tolerance


@dataclass(frozen=True)
class SimulatedRIR:
    full: NDArray[np.float32]
    direct: NDArray[np.float32]
    target_rt60_seconds: float
    measured_rt60_seconds: tuple[float, ...]
    design_rt60_seconds: float
    absorption: float
    max_order: int
    drr_db: tuple[float, ...]
    seed: int
    calibration_iterations: int

    def metadata(self) -> dict[str, object]:
        value = asdict(self)
        value.pop("full")
        value.pop("direct")
        return value


def _pad_rirs(values: list[np.ndarray]) -> NDArray[np.float64]:
    length = max(len(value) for value in values)
    output = np.zeros((len(values), length), dtype=np.float64)
    for channel, value in enumerate(values):
        output[channel, : len(value)] = value
    return output


def _build_room(
    scene: SceneGeometry,
    *,
    design_rt60_seconds: float,
    sample_rate: int,
    seed: int,
    direct_only: bool,
):
    try:
        import pyroomacoustics as pra
    except ImportError as exc:  # pragma: no cover - optional data dependency
        raise RuntimeError("RIR generation requires pyroomacoustics") from exc

    absorption, max_order = pra.inverse_sabine(
        design_rt60_seconds, scene.room_dimensions_m
    )
    # Randomized ISM draws in both Python and the libroom C++ extension.
    # Seeding only NumPy is insufficient and yields a different RIR per run.
    pra.random.seed(numpy=seed, libroom=seed)
    room = pra.ShoeBox(
        scene.room_dimensions_m,
        fs=sample_rate,
        materials=pra.Material(absorption),
        max_order=0 if direct_only else max_order,
        air_absorption=True,
        use_rand_ism=not direct_only,
        max_rand_disp=0.08,
    )
    room.add_source(scene.source_position_m)
    microphones = np.asarray(scene.microphone_positions_m, dtype=np.float64).T
    room.add_microphone_array(microphones)
    room.compute_rir()
    rirs = _pad_rirs(
        [np.asarray(room.rir[channel][0], dtype=np.float64) for channel in range(4)]
    )
    return rirs, float(absorption), int(max_order)


def _measure_rt60(rirs: np.ndarray, sample_rate: int) -> tuple[float, ...]:
    try:
        import pyroomacoustics as pra
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("RT60 measurement requires pyroomacoustics") from exc
    return tuple(
        float(
            pra.experimental.measure_rt60(
                channel,
                fs=sample_rate,
                decay_db=30,
            )
        )
        for channel in rirs
    )


def simulate_calibrated_rir(
    scene: SceneGeometry,
    *,
    target_rt60_seconds: float,
    seed: int,
    sample_rate: int = 16_000,
    maximum_calibration_iterations: int = 5,
) -> SimulatedRIR:
    """Simulate 4-channel full/direct RIRs and calibrate measured T30."""

    if target_rt60_seconds <= 0:
        raise ValueError("target_rt60_seconds must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if maximum_calibration_iterations <= 0:
        raise ValueError("maximum_calibration_iterations must be positive")

    design_rt60 = target_rt60_seconds
    full: np.ndarray | None = None
    measured: tuple[float, ...] = ()
    absorption = 0.0
    max_order = 0
    for iteration in range(1, maximum_calibration_iterations + 1):
        full, absorption, max_order = _build_room(
            scene,
            design_rt60_seconds=design_rt60,
            sample_rate=sample_rate,
            seed=seed,
            direct_only=False,
        )
        measured = _measure_rt60(full, sample_rate)
        median_rt60 = float(np.median(measured))
        if rt60_within_tolerance(median_rt60, target_rt60_seconds):
            break
        if median_rt60 <= 0 or not np.isfinite(median_rt60):
            raise RuntimeError("invalid measured RT60 during calibration")
        design_rt60 *= target_rt60_seconds / median_rt60
        design_rt60 = float(np.clip(design_rt60, 0.1, 2.0))
    else:
        raise RuntimeError(
            f"failed RT60 calibration: target={target_rt60_seconds}, "
            f"measured={measured}"
        )

    assert full is not None
    direct, _, _ = _build_room(
        scene,
        design_rt60_seconds=design_rt60,
        sample_rate=sample_rate,
        seed=seed,
        direct_only=True,
    )
    drr = direct_to_reverberant_ratio(full, direct)
    return SimulatedRIR(
        full=full.astype(np.float32),
        direct=direct.astype(np.float32),
        target_rt60_seconds=float(target_rt60_seconds),
        measured_rt60_seconds=measured,
        design_rt60_seconds=float(design_rt60),
        absorption=absorption,
        max_order=max_order,
        drr_db=tuple(map(float, drr)),
        seed=seed,
        calibration_iterations=iteration,
    )
