# pyright: reportAttributeAccessIssue=false

"""Python-facing data wrapper for the C++-first runtime."""

from __future__ import annotations

from typing import Any

import numpy as np

from mjorbit import _bindings
from mjorbit.config import OrbitInit
from mjorbit.model import MjoModel

_REAL_DATATYPE = 0
_POSITIVE_DATATYPE = 1
_AXIS_DATATYPE = 2
_QUATERNION_DATATYPE = 3


class MjoData:
    """Runtime state for one simulation run.

    One ``MjoData`` owns one C++ ``mjData`` plus one thread-local orbit plugin
    instance and associated actuator/sensor buffers.
    """

    backend = "cpu"
    nworld = 1

    __slots__ = ("_initial_orbit", "_native", "model")

    def __init__(
        self,
        model: MjoModel,
        *,
        orbit: OrbitInit | None = None,
        rng_seed: int | None = None,
    ) -> None:
        if orbit is None:
            orbit = OrbitInit(R_eci=np.zeros(3), V_eci=np.zeros(3), t=0.0)
        self._initial_orbit = OrbitInit(orbit.R_eci.copy(), orbit.V_eci.copy(), orbit.t)
        self.model = model
        self._native = _bindings.MjoData(
            model._native,
            np.asarray(orbit.R_eci, dtype=float).tolist(),
            np.asarray(orbit.V_eci, dtype=float).tolist(),
            float(orbit.t),
            rng_seed,
        )

    def __getattr__(self, name: str) -> Any:
        if name in {"mj_model", "mj_data"}:
            raise AttributeError(name)
        return getattr(self._native, name)

    @property
    def sensors(self) -> "_SensorDataNamespace":
        return _SensorDataNamespace(self)

    def reset(self, orbit: OrbitInit | None = None) -> None:
        if orbit is None:
            orbit = self._initial_orbit
        else:
            self._initial_orbit = OrbitInit(orbit.R_eci.copy(), orbit.V_eci.copy(), orbit.t)
        self._native.reset(
            np.asarray(orbit.R_eci, dtype=float).tolist(),
            np.asarray(orbit.V_eci, dtype=float).tolist(),
            float(orbit.t),
        )

    def eci_position_from_world(self, position_world_m: Any) -> np.ndarray:
        return self.orbit.R_eci * 1e3 + np.asarray(position_world_m, dtype=float)

    def eci_velocity_from_world(self, velocity_world_m_s: Any) -> np.ndarray:
        return self.orbit.V_eci * 1e3 + np.asarray(velocity_world_m_s, dtype=float)

    def world_position_from_lvlh(self, position_lvlh_m: Any) -> np.ndarray:
        position = np.asarray(position_lvlh_m, dtype=float).tolist()
        return np.asarray(
            self._native.world_position_from_lvlh(position),
            dtype=np.float64,
        )

    def world_velocity_from_lvlh(
        self,
        position_lvlh_m: Any,
        velocity_lvlh_m_s: Any,
    ) -> np.ndarray:
        return np.asarray(
            self._native.world_velocity_from_lvlh(
                np.asarray(position_lvlh_m, dtype=float).tolist(),
                np.asarray(velocity_lvlh_m_s, dtype=float).tolist(),
            ),
            dtype=np.float64,
        )

    def lvlh_position_from_world(self, position_world_m: Any) -> np.ndarray:
        return np.asarray(
            self._native.lvlh_position_from_world(
                np.asarray(position_world_m, dtype=float).tolist()
            ),
            dtype=np.float64,
        )

    def lvlh_velocity_from_world(
        self,
        position_world_m: Any,
        velocity_world_m_s: Any,
    ) -> np.ndarray:
        return np.asarray(
            self._native.lvlh_velocity_from_world(
                np.asarray(position_world_m, dtype=float).tolist(),
                np.asarray(velocity_world_m_s, dtype=float).tolist(),
            ),
            dtype=np.float64,
        )

    def contact_force(self, contact_id: int) -> np.ndarray:
        return np.asarray(self._native.contact_force(int(contact_id)), dtype=np.float64)

    def contact_force_segments(self, *, force_scale: float) -> np.ndarray | None:
        raw_segments = np.asarray(
            self._native.contact_force_segments(float(force_scale)),
            dtype=np.float64,
        )
        if raw_segments.size == 0:
            return None
        return raw_segments.reshape((-1, 2, 3))


class _SensorDataNamespace:
    def __init__(self, data: MjoData) -> None:
        self._data = data
        self._native = data._native.sensors

    def descriptor(self, name: str):
        return self._native.descriptor(name)

    def bias(self, name: str) -> np.ndarray:
        bias = np.asarray(self._native.bias(name), dtype=np.float64)
        if bias.size > 0:
            return bias.copy()
        descriptor = self.descriptor(name)
        if int(descriptor.datatype) in {2, 3}:  # mjDATATYPE_AXIS / mjDATATYPE_QUATERNION
            return np.zeros(3)
        return np.zeros(int(descriptor.dim))

    @property
    def biases(self) -> dict[str, np.ndarray]:
        return {
            descriptor.name: self.bias(descriptor.name)
            for descriptor in self._data.model.sensors.descriptors
        }

    def measure(
        self,
        name: str,
        *,
        noisy: bool = True,
        rng: Any = None,
    ) -> np.ndarray:
        if not noisy or rng is None:
            return np.asarray(self._native.measure(name, noisy), dtype=np.float64)

        descriptor = self.descriptor(name)
        truth = np.asarray(self._native.measure(name, False), dtype=np.float64)
        return _apply_sensor_noise(
            rng,
            descriptor,
            truth,
            bias=self.bias(name),
        )

    def measure_all(self, *, noisy: bool = True, rng: Any = None) -> dict[str, np.ndarray]:
        return {
            descriptor.name: self.measure(descriptor.name, noisy=noisy, rng=rng)
            for descriptor in self._data.model.sensors.descriptors
        }


__all__ = ["MjoData"]


def _apply_sensor_noise(
    rng: Any,
    descriptor: Any,
    truth: np.ndarray,
    *,
    bias: np.ndarray,
) -> np.ndarray:
    measurement = truth.copy()
    datatype = int(descriptor.datatype)
    dim = int(descriptor.dim)

    if bias.size > 0:
        measurement = _apply_sensor_bias(datatype, dim, measurement, bias)

    noise = float(descriptor.noise)
    if noise > 0.0:
        if datatype == _REAL_DATATYPE:
            measurement = measurement + rng.normal(0.0, noise, size=dim)
        elif datatype == _POSITIVE_DATATYPE:
            measurement = measurement + rng.normal(0.0, noise, size=dim)
            measurement = np.maximum(measurement, 0.0)
        elif datatype == _AXIS_DATATYPE:
            measurement = _rotate_axis(measurement, rng.normal(0.0, noise, size=3))
        elif datatype == _QUATERNION_DATATYPE:
            measurement = _rotate_quaternion(measurement, rng.normal(0.0, noise, size=3))
        else:
            measurement = measurement + rng.normal(0.0, noise, size=dim)

    return _apply_cutoff(datatype, float(descriptor.cutoff), measurement)


def _apply_sensor_bias(
    datatype: int,
    dim: int,
    measurement: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    if datatype in {_REAL_DATATYPE, _POSITIVE_DATATYPE}:
        return measurement + bias[:dim]
    if datatype == _AXIS_DATATYPE:
        return _rotate_axis(measurement, bias[:3])
    if datatype == _QUATERNION_DATATYPE:
        return _rotate_quaternion(measurement, bias[:3])
    return measurement + bias[:dim]


def _apply_cutoff(datatype: int, cutoff: float, measurement: np.ndarray) -> np.ndarray:
    if cutoff <= 0.0 or datatype in {_AXIS_DATATYPE, _QUATERNION_DATATYPE}:
        return measurement
    lower = 0.0 if datatype == _POSITIVE_DATATYPE else -cutoff
    return np.clip(measurement, lower, cutoff)


def _rotate_axis(axis: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    rotated = _rotation_matrix_from_rotvec(rotvec) @ axis
    norm = np.linalg.norm(rotated)
    if norm < 1.0e-12:
        raise ValueError("Noisy axis measurement must be non-zero")
    return rotated / norm


def _rotate_quaternion(quat: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    rotated = _quat_mul(_quat_from_rotvec(rotvec), quat)
    norm = np.linalg.norm(rotated)
    if norm < 1.0e-12:
        raise ValueError("Noisy quaternion measurement must be non-zero")
    return rotated / norm


def _rotation_matrix_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rotvec)
    if theta < 1.0e-12:
        return np.eye(3) + _skew(rotvec)
    axis = rotvec / theta
    k = _skew(axis)
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def _quat_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rotvec)
    if theta < 1.0e-12:
        quat = np.array([1.0, 0.5 * rotvec[0], 0.5 * rotvec[1], 0.5 * rotvec[2]])
    else:
        half = 0.5 * theta
        scale = np.sin(half) / theta
        quat = np.array(
            [
                np.cos(half),
                scale * rotvec[0],
                scale * rotvec[1],
                scale * rotvec[2],
            ]
        )
    return quat / np.linalg.norm(quat)


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def _skew(value: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -value[2], value[1]],
            [value[2], 0.0, -value[0]],
            [-value[1], value[0], 0.0],
        ]
    )
