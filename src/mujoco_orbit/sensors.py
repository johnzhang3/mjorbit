# pyright: reportAttributeAccessIssue=false

"""Sensor discovery, custom truth generation, and noisy measurement helpers."""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, cast

import mujoco
import numpy as np

if TYPE_CHECKING:
    from mujoco_orbit.core.runtime import MjoData, MjoModel

_GYRO_SENSOR_TYPE = int(mujoco.mjtSensor.mjSENS_GYRO)
_ACCELEROMETER_SENSOR_TYPE = int(mujoco.mjtSensor.mjSENS_ACCELEROMETER)
_MAGNETOMETER_SENSOR_TYPE = int(mujoco.mjtSensor.mjSENS_MAGNETOMETER)
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

_DATA_SENSOR_NAMESPACES: dict[int, weakref.ReferenceType["SensorDataNamespace"]] = {}
_PREVIOUS_SENSOR_CALLBACK: Callable[[mujoco.MjModel, mujoco.MjData, int], None] | None = None
_SENSOR_DISPATCH_INSTALLED = False
_ADDITIVE_BIAS_SENSOR_TYPES = {
    _ACCELEROMETER_SENSOR_TYPE,
    _GYRO_SENSOR_TYPE,
    _MAGNETOMETER_SENSOR_TYPE,
}


@dataclass(frozen=True)
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
    additive_bias_sigma: np.ndarray | None = None
    angular_bias_sigma: np.ndarray | None = None

    @property
    def data_slice(self) -> slice:
        """Slice into ``mjd.sensordata`` for this sensor."""
        return slice(self.adr, self.adr + self.dim)

    @property
    def is_custom_orbit_sensor(self) -> bool:
        """Whether this sensor is handled by mujoco_orbit's sensor callback."""
        return self.orbit_kind is not None


@dataclass(frozen=True)
class ModelSensorCatalog:
    """Static sensor metadata stored on ``MjoModel``."""

    descriptors: list[SensorDescriptor]
    by_name: dict[str, SensorDescriptor]
    custom_descriptors: list[SensorDescriptor]


@dataclass
class SensorDataNamespace:
    """Per-data sensor helpers and runtime stochastic state."""

    model: MjoModel
    data: MjoData
    rng: np.random.Generator
    biases: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def gyro_biases(self) -> dict[str, np.ndarray]:
        """Backward-compatible view of additive gyro biases only."""
        return {
            name: bias.copy()
            for name, bias in self.biases.items()
            if self.descriptor(name).sensor_type == _GYRO_SENSOR_TYPE
        }

    def descriptor(self, name: str) -> SensorDescriptor:
        descriptor = self.model.sensors.by_name.get(name)
        if descriptor is None:
            raise ValueError(f"Sensor '{name}' not found in model")
        return descriptor

    def bias(self, name: str) -> np.ndarray:
        """Return the persistent bias state for ``name`` if one exists.

        Real-valued sensors use additive bias in measurement units.
        Axis and quaternion sensors use a fixed small-angle rotation vector in radians.
        """
        descriptor = self.descriptor(name)
        bias = self.biases.get(name)
        if bias is not None:
            return bias.copy()
        if descriptor.datatype in (_AXIS_DATATYPE, _QUATERNION_DATATYPE):
            return np.zeros(3)
        return np.zeros(descriptor.dim)

    def measure(
        self,
        name: str,
        *,
        noisy: bool = True,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Return one sensor measurement by name."""
        descriptor = self.descriptor(name)
        truth = self.data.sensordata[descriptor.data_slice].copy()
        if not noisy:
            return truth

        generator = self.rng if rng is None else rng
        bias = self.biases.get(descriptor.name)
        return _apply_sensor_noise(generator, descriptor, truth, bias=bias)

    def measure_all(
        self,
        *,
        noisy: bool = True,
        rng: np.random.Generator | None = None,
    ) -> dict[str, np.ndarray]:
        """Return all current sensor measurements keyed by name."""
        generator = self.rng if rng is None else rng
        return {
            descriptor.name: self.measure(descriptor.name, noisy=noisy, rng=generator)
            for descriptor in self.model.sensors.descriptors
        }


def compile_sensor_catalog(model: mujoco.MjModel) -> ModelSensorCatalog:
    """Resolve static MuJoCo sensor metadata."""
    descriptors: list[SensorDescriptor] = []
    custom_descriptors: list[SensorDescriptor] = []

    for sensor_id in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
        if name is None:
            raise ValueError(f"Sensor id {sensor_id} is missing a name")

        orbit_kind = _orbit_sensor_kind(name)
        reference_eci: np.ndarray | None = None

        user = model.sensor_user[sensor_id].copy()
        gyro_bias_sigma = 0.0
        sensor_type = int(model.sensor_type[sensor_id])
        additive_bias_sigma: np.ndarray | None = None
        angular_bias_sigma: np.ndarray | None = None
        if sensor_type in _ADDITIVE_BIAS_SENSOR_TYPES:
            additive_bias_sigma = _parse_bias_sigma(user, int(model.sensor_dim[sensor_id]))
            if sensor_type == _GYRO_SENSOR_TYPE and additive_bias_sigma is not None:
                gyro_bias_sigma = float(np.max(additive_bias_sigma))

        if orbit_kind == "star":
            if user.size < 3:
                raise ValueError(
                    f"Sensor '{name}' requires a star vector in sensor user[0:3]"
                )
            reference_eci = _normalized(
                user[:3],
                f"Sensor '{name}' star vector",
            )
            angular_bias_sigma = _parse_bias_sigma(user, 3, offset=3)
        elif int(model.sensor_datatype[sensor_id]) in (_AXIS_DATATYPE, _QUATERNION_DATATYPE):
            angular_bias_sigma = _parse_bias_sigma(user, 3)

        descriptor = SensorDescriptor(
            sensor_id=sensor_id,
            name=name,
            sensor_type=sensor_type,
            datatype=int(model.sensor_datatype[sensor_id]),
            objtype=int(model.sensor_objtype[sensor_id]),
            objid=int(model.sensor_objid[sensor_id]),
            adr=int(model.sensor_adr[sensor_id]),
            dim=int(model.sensor_dim[sensor_id]),
            noise=float(model.sensor_noise[sensor_id]),
            cutoff=float(model.sensor_cutoff[sensor_id]),
            needstage=int(model.sensor_needstage[sensor_id]),
            user=user,
            orbit_kind=orbit_kind,
            reference_eci=reference_eci,
            gyro_bias_sigma=gyro_bias_sigma,
            additive_bias_sigma=additive_bias_sigma,
            angular_bias_sigma=angular_bias_sigma,
        )

        if descriptor.orbit_kind is not None:
            _validate_orbit_sensor_descriptor(descriptor)
            custom_descriptors.append(descriptor)

        descriptors.append(descriptor)

    return ModelSensorCatalog(
        descriptors=descriptors,
        by_name={descriptor.name: descriptor for descriptor in descriptors},
        custom_descriptors=custom_descriptors,
    )


def create_sensor_data_namespace(
    model: MjoModel,
    data: MjoData,
    *,
    rng_seed: int | None = None,
) -> SensorDataNamespace:
    """Create the per-run sensor helper object."""
    rng = np.random.default_rng(rng_seed)
    biases: dict[str, np.ndarray] = {}
    for descriptor in model.sensors.descriptors:
        if descriptor.additive_bias_sigma is not None:
            biases[descriptor.name] = rng.normal(
                0.0,
                descriptor.additive_bias_sigma,
                size=descriptor.additive_bias_sigma.shape,
            )
            continue
        if descriptor.angular_bias_sigma is not None:
            biases[descriptor.name] = rng.normal(
                0.0,
                descriptor.angular_bias_sigma,
                size=descriptor.angular_bias_sigma.shape,
            )

    return SensorDataNamespace(model=model, data=data, rng=rng, biases=biases)


def register_sensor_data_namespace(namespace: SensorDataNamespace) -> None:
    """Register a data instance so custom sensor callbacks can find it."""
    global _PREVIOUS_SENSOR_CALLBACK, _SENSOR_DISPATCH_INSTALLED

    if not _SENSOR_DISPATCH_INSTALLED:
        previous = mujoco.get_mjcb_sensor()
        _PREVIOUS_SENSOR_CALLBACK = cast(
            Callable[[mujoco.MjModel, mujoco.MjData, int], None] | None,
            previous if callable(previous) else None,
        )
        mujoco.set_mjcb_sensor(_sensor_dispatch)
        _SENSOR_DISPATCH_INSTALLED = True

    key = id(namespace.data.mj_data)
    _DATA_SENSOR_NAMESPACES[key] = weakref.ref(namespace)
    weakref.finalize(namespace.data, _DATA_SENSOR_NAMESPACES.pop, key, None)


def update_sensor_environment(model: MjoModel, data: MjoData) -> None:
    """Update MuJoCo's world-frame magnetic field from the current orbital state."""
    model.mj_model.opt.magnetic[:] = data.frame.C_LI @ data.env.mag_field_eci


def _sensor_dispatch(model: mujoco.MjModel, mj_data: mujoco.MjData, stage: int) -> None:
    if _PREVIOUS_SENSOR_CALLBACK is not None:
        _PREVIOUS_SENSOR_CALLBACK(model, mj_data, stage)

    namespace_ref = _DATA_SENSOR_NAMESPACES.get(id(mj_data))
    namespace = None if namespace_ref is None else namespace_ref()
    if namespace is None:
        return

    for descriptor in namespace.model.sensors.custom_descriptors:
        if descriptor.needstage != stage:
            continue
        mj_data.sensordata[descriptor.data_slice] = _custom_sensor_truth(
            namespace.data,
            descriptor,
            mj_data,
        )


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


def _custom_sensor_truth(
    data: MjoData,
    descriptor: SensorDescriptor,
    mj_data: mujoco.MjData,
) -> np.ndarray:
    if descriptor.orbit_kind == "sun":
        world_vec = data.frame.C_LI @ data.env.sun_vector_eci
    elif descriptor.orbit_kind == "horizon":
        nadir_eci = -data.orbit.R_eci / np.linalg.norm(data.orbit.R_eci)
        world_vec = data.frame.C_LI @ nadir_eci
    elif descriptor.orbit_kind == "star":
        if descriptor.reference_eci is None:
            raise RuntimeError(f"Sensor '{descriptor.name}' is missing its star reference vector")
        world_vec = data.frame.C_LI @ descriptor.reference_eci
    else:
        raise RuntimeError(f"Unknown custom sensor kind for '{descriptor.name}'")

    site_rot = mj_data.site_xmat[descriptor.objid].reshape(3, 3)
    site_vec = site_rot.T @ world_vec
    return _normalized(site_vec, f"Sensor '{descriptor.name}' truth vector")


def _apply_sensor_noise(
    rng: np.random.Generator,
    descriptor: SensorDescriptor,
    truth: np.ndarray,
    *,
    bias: np.ndarray | None,
) -> np.ndarray:
    measurement = truth.copy()

    if bias is not None:
        measurement = _apply_sensor_bias(descriptor, measurement, bias)

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


def _apply_sensor_bias(
    descriptor: SensorDescriptor,
    measurement: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    if descriptor.datatype in (_REAL_DATATYPE, _POSITIVE_DATATYPE):
        return measurement + bias[: descriptor.dim]
    if descriptor.datatype == _AXIS_DATATYPE:
        return _rotate_axis(measurement, bias[:3])
    if descriptor.datatype == _QUATERNION_DATATYPE:
        return _rotate_quaternion(measurement, bias[:3])
    return measurement + bias[: descriptor.dim]


def _parse_bias_sigma(user: np.ndarray, dim: int, *, offset: int = 0) -> np.ndarray | None:
    payload = np.asarray(user[offset:], dtype=float)
    if payload.size == 0:
        return None

    if payload.size >= dim and np.any(np.abs(payload[1:dim]) > 0.0):
        sigma = np.maximum(payload[:dim], 0.0)
    else:
        sigma0 = max(float(payload[0]), 0.0)
        if sigma0 <= 0.0:
            return None
        sigma = np.full(dim, sigma0)

    if np.all(sigma <= 0.0):
        return None
    return sigma


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
            w1 * q2[0] - x1 * x2 - y1 * y2 - z1 * z2,
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
    "ModelSensorCatalog",
    "SensorDataNamespace",
    "SensorDescriptor",
    "compile_sensor_catalog",
    "create_sensor_data_namespace",
    "register_sensor_data_namespace",
    "update_sensor_environment",
]
