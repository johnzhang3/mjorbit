"""Host ↔ device buffer plumbing for the orbit overlay.

This module owns the small numpy ↔ Warp-array helpers (``_array_*``,
``_world_*``, ``_copy_*``) and the public sync/pull entry points
``sync_core_device_from_public`` and ``pull_core_device_to_public`` used by
``mjo_upload`` / ``mjo_pull``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp


def _array_f64(values: np.ndarray | list[float]) -> Any:
    return wp.array(np.asarray(values, dtype=np.float64), dtype=wp.float32)


def _array_i32(values: np.ndarray | list[int]) -> Any:
    return wp.array(np.asarray(values, dtype=np.int32), dtype=int)


def _array_vec3d(values: np.ndarray | list[list[float]]) -> Any:
    array = np.asarray(values, dtype=np.float64).reshape((-1, 3))
    return wp.array(array, dtype=wp.vec3, shape=(array.shape[0],))


def _world_scalar(values: np.ndarray | float, nworld: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if nworld == 1:
        return array.reshape(1)
    return array.reshape(nworld)


def _world_vec3(values: np.ndarray, nworld: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if nworld == 1:
        return array.reshape((1, 3))
    return array.reshape((nworld, 3))


def _world_array(values: np.ndarray, nworld: int, width: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if nworld == 1:
        return array.reshape((1, width))
    return array.reshape((nworld, width))


def _copy_vec3d(dest: Any, values: np.ndarray, nworld: int) -> None:
    src_values = _world_vec3(values, nworld)
    wp.copy(dest, wp.array(src_values, dtype=wp.vec3, shape=(nworld,)))


def _copy_f64_1d(dest: Any, values: np.ndarray | float, nworld: int) -> None:
    wp.copy(dest, wp.array(_world_scalar(values, nworld), dtype=wp.float32))


def _copy_f64_2d(dest: Any, values: np.ndarray, nworld: int, width: int) -> None:
    if width == 0:
        return
    wp.copy(dest, wp.array(_world_array(values, nworld, width), dtype=wp.float32))


def _sync_field_enabled(fields: frozenset[str] | None, *names: str) -> bool:
    return fields is None or "core" in fields or "actuators" in fields or any(
        name in fields for name in names
    )


def sync_core_device_from_public(data: Any, fields: frozenset[str] | None = None) -> None:
    """Upload user-mutated public orbit/actuator state to device core buffers."""

    core = data.core_data
    if fields is None or "orbit" in fields or "core" in fields:
        _copy_vec3d(core.orbit_R_eci, data.orbit.R_eci, data.nworld)
        _copy_vec3d(core.orbit_V_eci, data.orbit.V_eci, data.nworld)
        _copy_f64_1d(core.orbit_t, data.orbit.t, data.nworld)

    if _sync_field_enabled(fields, "rw_speed"):
        _copy_f64_2d(
            core.rw_speed,
            data.actuators.rw_speed,
            data.nworld,
            len(data.model.reaction_wheels),
        )
    if _sync_field_enabled(fields, "rw_torque_cmd"):
        _copy_f64_2d(
            core.rw_torque_cmd,
            data.actuators.rw_torque_cmd,
            data.nworld,
            len(data.model.reaction_wheels),
        )
    if _sync_field_enabled(fields, "mtq_dipole_cmd"):
        _copy_f64_2d(
            core.mtq_dipole_cmd,
            data.actuators.mtq_dipole_cmd,
            data.nworld,
            len(data.model.magnetorquers),
        )
    if _sync_field_enabled(fields, "thr_force_cmd"):
        _copy_f64_2d(
            core.thr_force_cmd,
            data.actuators.thr_force_cmd,
            data.nworld,
            len(data.model.thrusters),
        )


def _pull_field_enabled(fields: frozenset[str] | None, *names: str) -> bool:
    return fields is None or "core" in fields or any(name in fields for name in names)


def pull_core_device_to_public(data: Any, fields: frozenset[str] | None = None) -> None:
    """Download device orbit/coupling state into public NumPy buffers."""

    core = data.core_data

    if _pull_field_enabled(fields, "orbit"):
        orbit_R = core.orbit_R_eci.numpy()
        orbit_V = core.orbit_V_eci.numpy()
        orbit_t = core.orbit_t.numpy()
        if data.nworld == 1:
            np.copyto(data.orbit.R_eci, orbit_R[0])
            np.copyto(data.orbit.V_eci, orbit_V[0])
            data.orbit.t = float(orbit_t[0])
        else:
            np.copyto(data.orbit.R_eci, orbit_R)
            np.copyto(data.orbit.V_eci, orbit_V)
            np.copyto(data.orbit.t, orbit_t)

    if _pull_field_enabled(fields, "frame"):
        frame_C_LI = core.frame_C_LI.numpy()
        frame_C_IL = core.frame_C_IL.numpy()
        frame_omega = core.frame_omega_lvlh.numpy()
        frame_omega_dot = core.frame_omega_dot_lvlh.numpy()
        if data.nworld == 1:
            np.copyto(data.frame.C_LI, frame_C_LI[0])
            np.copyto(data.frame.C_IL, frame_C_IL[0])
            np.copyto(data.frame.omega_lvlh, frame_omega[0])
            np.copyto(data.frame.omega_dot_lvlh, frame_omega_dot[0])
        else:
            np.copyto(data.frame.C_LI, frame_C_LI)
            np.copyto(data.frame.C_IL, frame_C_IL)
            np.copyto(data.frame.omega_lvlh, frame_omega)
            np.copyto(data.frame.omega_dot_lvlh, frame_omega_dot)

    if _pull_field_enabled(fields, "env"):
        env_sun = core.env_sun_vector_eci.numpy()
        env_eclipse = core.env_eclipse.numpy()
        env_mag = core.env_mag_field_eci.numpy()
        env_atm_omega = core.env_atmosphere_omega_eci.numpy()
        env_density = core.env_atm_density.numpy()
        if data.nworld == 1:
            np.copyto(data.env.sun_vector_eci, env_sun[0])
            data.env.eclipse = float(env_eclipse[0])
            np.copyto(data.env.mag_field_eci, env_mag[0])
            np.copyto(data.env.atmosphere_omega_eci, env_atm_omega[0])
            data.env.atm_density = float(env_density[0])
        else:
            np.copyto(data.env.sun_vector_eci, env_sun)
            np.copyto(data.env.eclipse, env_eclipse)
            np.copyto(data.env.mag_field_eci, env_mag)
            np.copyto(data.env.atmosphere_omega_eci, env_atm_omega)
            np.copyto(data.env.atm_density, env_density)

    if _pull_field_enabled(fields, "wrench_buffer"):
        wrench = core.wrench_buffer.numpy()
        if data.nworld == 1:
            np.copyto(data.wrench_buffer, wrench[0])
        else:
            np.copyto(data.wrench_buffer, wrench)

    if len(data.model.reaction_wheels) and _pull_field_enabled(fields, "actuators"):
        rw_speed = core.rw_speed.numpy()
        rw_momentum = core.rw_momentum.numpy()
        if data.nworld == 1:
            np.copyto(data.actuators.rw_speed, rw_speed[0])
            np.copyto(data.actuators.rw_momentum, rw_momentum[0])
        else:
            np.copyto(data.actuators.rw_speed, rw_speed)
            np.copyto(data.actuators.rw_momentum, rw_momentum)


__all__ = [
    "_array_f64",
    "_array_i32",
    "_array_vec3d",
    "_copy_f64_1d",
    "_copy_f64_2d",
    "_copy_vec3d",
    "_world_array",
    "_world_scalar",
    "_world_vec3",
    "pull_core_device_to_public",
    "sync_core_device_from_public",
]
