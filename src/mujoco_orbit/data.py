# pyright: reportAttributeAccessIssue=false

"""Python-facing data wrapper for the C++-first runtime."""

from __future__ import annotations

from typing import Any

import numpy as np

from mujoco_orbit import _bindings
from mujoco_orbit.config import OrbitInit
from mujoco_orbit.model import MjoModel


class MjoData:
    """Runtime state for one simulation run.

    One ``MjoData`` owns one C++ ``mjData`` plus one thread-local orbit plugin
    instance and associated actuator/sensor buffers.
    """

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
        del rng
        return np.asarray(self._native.measure(name, noisy), dtype=np.float64)

    def measure_all(self, *, noisy: bool = True, rng: Any = None) -> dict[str, np.ndarray]:
        return {
            descriptor.name: self.measure(descriptor.name, noisy=noisy, rng=rng)
            for descriptor in self._data.model.sensors.descriptors
        }


__all__ = ["MjoData"]
