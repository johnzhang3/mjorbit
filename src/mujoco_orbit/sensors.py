# pyright: reportAttributeAccessIssue=false

"""Sensor discovery, custom truth generation, and noisy measurement helpers."""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, cast

import mujoco
import numpy as np

if TYPE_CHECKING:
    from mujoco_orbit.core.scenario import Scenario

_GYRO_SENSOR_TYPE = int(mujoco.mjtSensor.mjSENS_GYRO)
_USER_SENSOR_TYPE = int(mujoco.mjtSensor.mjSENS_USER)

_AXIS_DATATYPE = int(mujoco.mjtDataType.mjDATATYPE_AXIS)
_POSITIVE_DATATYPE = int(mujoco.mjtDataType.mjDATATYPE_POSITIVE)
_QUATERNION_DATATYPE = int(mujoco.mjtDataType.mjDATATYPE_QUATERNION)
_REAL_DATATYPE = int(mujoco.mjtDataType.mjDATATYPE_REAL)

_POS_STAGE = int(mujoco.mjtStage.mjSTAGE_POS)

_ORBIT_SENSOR_PREFIXES = {
    "orbit_sun_": "sun",
    "orbit_horizon_": "horizon",
    "orbit_star_": "star",
}

_MODEL_SENSOR_SUITES: weakref.WeakKeyDictionary[mujoco.MjModel, "SensorSuite"] = (
    weakref.WeakKeyDictionary()
)
_PREVIOUS_SENSOR_CALLBACK: Callable[[mujoco.MjModel, mujoco.MjData, int], None] | None = None
_SENSOR_DISPATCH_INSTALLED = False


@dataclass
class SensorDescriptor:
    """Resolved metadata for one MuJoCo sensor."""

    sensor_id: int
    name: str
    sensor_type: int
    datatype: int
    objtype: int
    objid: int
    adr: int
    dim: int
    noise: float
    cutoff: float
    needstage: int
    user: np.ndarray
    orbit_kind: str | None = None
    reference_eci: np.ndarray | None = None
    gyro_bias_sigma: float = 0.0
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def data_slice(self) -> slice:
        """Slice into ``mjd.sensordata`` for this sensor."""
        return slice(self.adr, self.adr + self.dim)

    @property
    def is_custom_orbit_sensor(self) -> bool:
        """Whether this sensor is handled by mujoco_orbit's sensor callback."""
        return self.orbit_kind is not None


@dataclass
class SensorSuite:
    """Per-scenario sensor metadata and persistent noise state."""

    scenario: Scenario
    descriptors: list[SensorDescriptor]
    by_name: dict[str, SensorDescriptor]
    custom_descriptors: list[SensorDescriptor]
    rng: np.random.Generator = field(default_factory=np.random.default_rng)


def compile_sensor_suite(scenario: Scenario) -> SensorSuite:
    """Resolve MuJoCo sensors and register custom orbital sensors if needed."""
    suite_rng = np.random.default_rng()
    descriptors: list[SensorDescriptor] = []
    custom_descriptors: list[SensorDescriptor] = []

    for sensor_id in range(scenario.mjm.nsensor):
        name = mujoco.mj_id2name(scenario.mjm, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
        if name is None:
            raise ValueError(f"Sensor id {sensor_id} is missing a name")

        descriptor = SensorDescriptor(
            sensor_id=sensor_id,
            name=name,
            sensor_type=int(scenario.mjm.sensor_type[sensor_id]),
            datatype=int(scenario.mjm.sensor_datatype[sensor_id]),
            objtype=int(scenario.mjm.sensor_objtype[sensor_id]),
            objid=int(scenario.mjm.sensor_objid[sensor_id]),
            adr=int(scenario.mjm.sensor_adr[sensor_id]),
            dim=int(scenario.mjm.sensor_dim[sensor_id]),
            noise=float(scenario.mjm.sensor_noise[sensor_id]),
            cutoff=float(scenario.mjm.sensor_cutoff[sensor_id]),
            needstage=int(scenario.mjm.sensor_needstage[sensor_id]),
            user=scenario.mjm.sensor_user[sensor_id].copy(),
        )

        if descriptor.sensor_type == _GYRO_SENSOR_TYPE and descriptor.user.size > 0:
            descriptor.gyro_bias_sigma = max(float(descriptor.user[0]), 0.0)
            if descriptor.gyro_bias_sigma > 0.0:
                descriptor.gyro_bias = suite_rng.normal(0.0, descriptor.gyro_bias_sigma, size=3)

        orbit_kind = _orbit_sensor_kind(descriptor.name)
        if orbit_kind is not None:
            _validate_orbit_sensor_descriptor(descriptor)
            descriptor.orbit_kind = orbit_kind
            if orbit_kind == "star":
                if descriptor.user.size < 3:
                    raise ValueError(
                        f"Sensor '{descriptor.name}' requires a star vector in sensor user[0:3]"
                    )
                descriptor.reference_eci = _normalized(
                    descriptor.user[:3], f"Sensor '{descriptor.name}' star vector"
                )
            custom_descriptors.append(descriptor)

        descriptors.append(descriptor)

    suite = SensorSuite(
        scenario=scenario,
        descriptors=descriptors,
        by_name={descriptor.name: descriptor for descriptor in descriptors},
        custom_descriptors=custom_descriptors,
        rng=suite_rng,
    )

    if custom_descriptors:
        _register_sensor_suite(suite)

    return suite


def update_sensor_environment(scenario: Scenario) -> None:
    """Update MuJoCo's world-frame magnetic field from the current orbital state."""
    scenario.mjm.opt.magnetic[:] = scenario.frame_cache.C_LI @ scenario.env_cache.mag_field_eci


def measure_sensor(
    scenario: Scenario,
    name: str,
    *,
    noisy: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return one sensor reading by name.

    ``noisy=False`` returns a copy of the current MuJoCo truth value from
    ``mjd.sensordata``. ``noisy=True`` adds mujoco_orbit-managed noise without
    mutating MuJoCo state.
    """
    suite = _require_sensor_suite(scenario)
    descriptor = suite.by_name.get(name)
    if descriptor is None:
        raise ValueError(f"Sensor '{name}' not found in model")

    truth = scenario.mjd.sensordata[descriptor.data_slice].copy()
    if not noisy:
        return truth

    generator = suite.rng if rng is None else rng
    return _apply_sensor_noise(generator, descriptor, truth)


def measure_sensors(
    scenario: Scenario,
    *,
    noisy: bool = True,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Return a dictionary of all current sensor readings keyed by sensor name."""
    suite = _require_sensor_suite(scenario)
    generator = suite.rng if rng is None else rng
    return {
        descriptor.name: measure_sensor(scenario, descriptor.name, noisy=noisy, rng=generator)
        for descriptor in suite.descriptors
    }


def _require_sensor_suite(scenario: Scenario) -> SensorSuite:
    if scenario.sensor_suite is None:
        raise RuntimeError("Scenario does not have a compiled sensor suite")
    return scenario.sensor_suite


def _orbit_sensor_kind(name: str) -> str | None:
    for prefix, kind in _ORBIT_SENSOR_PREFIXES.items():
        if name.startswith(prefix):
            return kind
    if name.startswith("orbit_"):
        raise ValueError(
            f"Unsupported custom sensor '{name}'. Use orbit_sun_, orbit_horizon_, or orbit_star_."
        )
    return None


def _validate_orbit_sensor_descriptor(descriptor: SensorDescriptor) -> None:
    if descriptor.sensor_type != _USER_SENSOR_TYPE:
        raise ValueError(
            f"Sensor '{descriptor.name}' must be declared as <sensor><user .../></sensor>"
        )
    if descriptor.objtype != int(mujoco.mjtObj.mjOBJ_SITE):
        raise ValueError(f"Sensor '{descriptor.name}' must attach to a site")
    if descriptor.datatype != _AXIS_DATATYPE:
        raise ValueError(f"Sensor '{descriptor.name}' must use datatype='axis'")
    if descriptor.needstage != _POS_STAGE:
        raise ValueError(f"Sensor '{descriptor.name}' must use needstage='pos'")
    if descriptor.dim != 3:
        raise ValueError(f"Sensor '{descriptor.name}' must declare dim='3'")


def _register_sensor_suite(suite: SensorSuite) -> None:
    global _PREVIOUS_SENSOR_CALLBACK, _SENSOR_DISPATCH_INSTALLED

    if not _SENSOR_DISPATCH_INSTALLED:
        previous = mujoco.get_mjcb_sensor()
        _PREVIOUS_SENSOR_CALLBACK = cast(
            Callable[[mujoco.MjModel, mujoco.MjData, int], None] | None,
            previous if callable(previous) else None,
        )
        mujoco.set_mjcb_sensor(_sensor_dispatch)
        _SENSOR_DISPATCH_INSTALLED = True

    _MODEL_SENSOR_SUITES[suite.scenario.mjm] = suite


def _sensor_dispatch(model: mujoco.MjModel, data: mujoco.MjData, stage: int) -> None:
    if _PREVIOUS_SENSOR_CALLBACK is not None:
        _PREVIOUS_SENSOR_CALLBACK(model, data, stage)

    suite = _MODEL_SENSOR_SUITES.get(model)
    if suite is None:
        return

    for descriptor in suite.custom_descriptors:
        if descriptor.needstage != stage:
            continue
        data.sensordata[descriptor.data_slice] = _custom_sensor_truth(
            suite.scenario, descriptor, data
        )


def _custom_sensor_truth(
    scenario: Scenario, descriptor: SensorDescriptor, data: mujoco.MjData
) -> np.ndarray:
    if descriptor.orbit_kind == "sun":
        world_vec = scenario.frame_cache.C_LI @ scenario.env_cache.sun_vector_eci
    elif descriptor.orbit_kind == "horizon":
        nadir_eci = -scenario.orbit.R_eci / np.linalg.norm(scenario.orbit.R_eci)
        world_vec = scenario.frame_cache.C_LI @ nadir_eci
    elif descriptor.orbit_kind == "star":
        if descriptor.reference_eci is None:
            raise RuntimeError(f"Sensor '{descriptor.name}' is missing its star reference vector")
        world_vec = scenario.frame_cache.C_LI @ descriptor.reference_eci
    else:
        raise RuntimeError(f"Unknown custom sensor kind for '{descriptor.name}'")

    site_rot = data.site_xmat[descriptor.objid].reshape(3, 3)
    site_vec = site_rot.T @ world_vec
    return _normalized(site_vec, f"Sensor '{descriptor.name}' truth vector")


def _apply_sensor_noise(
    rng: np.random.Generator, descriptor: SensorDescriptor, truth: np.ndarray
) -> np.ndarray:
    measurement = truth.copy()

    if descriptor.sensor_type == _GYRO_SENSOR_TYPE and descriptor.gyro_bias_sigma > 0.0:
        measurement = measurement + descriptor.gyro_bias

    if descriptor.noise <= 0.0:
        return _apply_cutoff(descriptor, measurement)

    if descriptor.datatype == _REAL_DATATYPE:
        measurement = measurement + rng.normal(0.0, descriptor.noise, size=descriptor.dim)
    elif descriptor.datatype == _POSITIVE_DATATYPE:
        measurement = measurement + rng.normal(0.0, descriptor.noise, size=descriptor.dim)
        measurement = np.maximum(measurement, 0.0)
    elif descriptor.datatype == _AXIS_DATATYPE:
        measurement = _rotate_axis(truth, rng.normal(0.0, descriptor.noise, size=3))
    elif descriptor.datatype == _QUATERNION_DATATYPE:
        measurement = _rotate_quaternion(truth, rng.normal(0.0, descriptor.noise, size=3))
    else:
        measurement = measurement + rng.normal(0.0, descriptor.noise, size=descriptor.dim)

    return _apply_cutoff(descriptor, measurement)


def _apply_cutoff(descriptor: SensorDescriptor, measurement: np.ndarray) -> np.ndarray:
    if descriptor.cutoff <= 0.0:
        return measurement
    if descriptor.datatype == _POSITIVE_DATATYPE:
        return np.clip(measurement, 0.0, descriptor.cutoff)
    if descriptor.datatype in (_AXIS_DATATYPE, _QUATERNION_DATATYPE):
        return measurement
    return np.clip(measurement, -descriptor.cutoff, descriptor.cutoff)


def _rotate_axis(axis: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    rotated = _rotation_matrix_from_rotvec(rotvec) @ axis
    return _normalized(rotated, "Noisy axis measurement")


def _rotate_quaternion(quat: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    delta = _quat_from_rotvec(rotvec)
    rotated = _quat_mul(delta, quat)
    return _normalized(rotated, "Noisy quaternion measurement")


def _rotation_matrix_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        return np.eye(3) + _skew(rotvec)

    axis = rotvec / theta
    K = _skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _quat_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        return _normalized(
            np.array([1.0, 0.5 * rotvec[0], 0.5 * rotvec[1], 0.5 * rotvec[2]]),
            "Rotation quaternion",
        )

    axis = rotvec / theta
    half = 0.5 * theta
    return np.array([np.cos(half), *(np.sin(half) * axis)])


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def _normalized(vec: np.ndarray, label: str) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError(f"{label} must be non-zero")
    return vec / norm


__all__ = [
    "SensorDescriptor",
    "SensorSuite",
    "compile_sensor_suite",
    "measure_sensor",
    "measure_sensors",
    "update_sensor_environment",
]
