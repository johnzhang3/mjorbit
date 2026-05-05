"""Public dataclasses callers see on ``data.orbit / frame / env / actuators``.

These are pure host-side numpy buffers; the device-side equivalents live in
``device.types.DeviceCoreData`` and the host shadow in ``host_shadow``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _world_shape(nworld: int, *shape: int) -> tuple[int, ...]:
    return shape if nworld == 1 else (nworld, *shape)

def _zeros_world(nworld: int, *shape: int, dtype: Any = float) -> np.ndarray:
    return np.zeros(_world_shape(nworld, *shape), dtype=dtype)

@dataclass
class OrbitBatchState:
    """Public orbital state buffers for the warp backend."""

    R_eci: np.ndarray
    V_eci: np.ndarray
    t: float | np.ndarray

@dataclass
class FrameBatchCache:
    """Public frame cache buffers for the warp backend."""

    C_LI: np.ndarray
    C_IL: np.ndarray
    omega_lvlh: np.ndarray
    omega_dot_lvlh: np.ndarray

@dataclass
class EnvironmentBatchCache:
    """Public environment cache buffers for the warp backend."""

    sun_vector_eci: np.ndarray
    eclipse: float | np.ndarray
    mag_field_eci: np.ndarray
    atmosphere_omega_eci: np.ndarray
    atm_density: float | np.ndarray

@dataclass
class BatchedActuatorData:
    """Mutable actuator command/state arrays for one or more worlds."""

    rw_speed: np.ndarray
    rw_momentum: np.ndarray
    rw_inertia: np.ndarray
    rw_torque_cmd: np.ndarray
    mtq_dipole_cmd: np.ndarray
    thr_force_cmd: np.ndarray

    @classmethod
    def zeros(
        cls,
        nworld: int,
        n_rw: int,
        rw_inertia: np.ndarray,
        n_mtq: int,
        n_thr: int,
    ) -> "BatchedActuatorData":
        return cls(
            rw_speed=_zeros_world(nworld, n_rw),
            rw_momentum=_zeros_world(nworld, n_rw),
            rw_inertia=np.asarray(rw_inertia, dtype=float).copy(),
            rw_torque_cmd=_zeros_world(nworld, n_rw),
            mtq_dipole_cmd=_zeros_world(nworld, n_mtq),
            thr_force_cmd=_zeros_world(nworld, n_thr),
        )

    def update_rw_momentum(self, rw_inertia: np.ndarray | None = None) -> None:
        """Recompute wheel angular momentum from speed and inertia."""
        inertia = self.rw_inertia if rw_inertia is None else np.asarray(rw_inertia, dtype=float)
        self.rw_inertia[:] = inertia
        np.multiply(self.rw_speed, inertia, out=self.rw_momentum)


__all__ = [
    'BatchedActuatorData',
    'EnvironmentBatchCache',
    'FrameBatchCache',
    'OrbitBatchState',
]