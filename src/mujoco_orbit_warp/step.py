# pyright: reportAttributeAccessIssue=false

"""Forward and step functions for the MJWarp-backed API."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from mujoco_orbit.coupling.actuators import (
    _apply_magnetorquers,
    _apply_reaction_wheels,
    _apply_thrusters,
    command_rw_torques,
)
from mujoco_orbit.coupling.apply import assemble_and_apply_wrenches
from mujoco_orbit.coupling.feedback import compute_net_external_wrench, compute_orbit_feedback_accel
from mujoco_orbit.orbit.environment import update_environment_cache
from mujoco_orbit.orbit.lvlh import update_frame_cache
from mujoco_orbit.orbit.propagator import propagate_rk4
from mujoco_orbit.sensors import update_sensor_environment

from ._deps import require_mjwarp
from .runtime import MjoData, MjoModel


def _as_batched(values: np.ndarray | float, *, nworld: int) -> np.ndarray:
    array = np.asarray(values)
    if nworld == 1:
        return np.expand_dims(array, axis=0)
    return array


def _copy_to_device(
    wp: Any,
    dest: Any,
    values: np.ndarray | float,
    *,
    dtype: Any,
    shape: tuple[int, ...] | None = None,
) -> None:
    array = np.asarray(values)
    if array.size == 0:
        return

    if shape is None:
        src = wp.array(array, dtype=dtype)
    else:
        src = wp.array(array, shape=shape, dtype=dtype)
    wp.copy(dest, src)


def _refresh_host_orbit_caches(model: MjoModel, run) -> None:
    run.frame = update_frame_cache(run.orbit, use_j2=model.use_j2)
    run.env = update_environment_cache(run.orbit, run.frame)
    run.actuators.update_rw_momentum(model.rw_inertia)
    update_sensor_environment(model.host_model, run)


def _sync_device_from_public(model: MjoModel, data: MjoData) -> None:
    _, wp = require_mjwarp()

    _copy_to_device(
        wp,
        data.warp_data.qpos,
        _as_batched(data.qpos, nworld=data.nworld),
        dtype=float,
    )
    _copy_to_device(
        wp,
        data.warp_data.qvel,
        _as_batched(data.qvel, nworld=data.nworld),
        dtype=float,
    )
    _copy_to_device(
        wp,
        data.warp_data.qacc_warmstart,
        _as_batched(data.qacc_warmstart, nworld=data.nworld),
        dtype=float,
    )
    _copy_to_device(
        wp,
        data.warp_data.qfrc_applied,
        _as_batched(data.qfrc_applied, nworld=data.nworld),
        dtype=float,
    )
    _copy_to_device(
        wp,
        data.warp_data.ctrl,
        _as_batched(data.ctrl, nworld=data.nworld),
        dtype=float,
    )
    _copy_to_device(
        wp,
        data.warp_data.time,
        _as_batched(np.asarray(data.time), nworld=data.nworld).reshape(data.nworld),
        dtype=float,
    )

    if model.na > 0:
        _copy_to_device(
            wp,
            data.warp_data.act,
            _as_batched(data.act, nworld=data.nworld),
            dtype=float,
        )

    _copy_to_device(
        wp,
        data.warp_data.xfrc_applied,
        _as_batched(data.xfrc_applied, nworld=data.nworld),
        dtype=wp.spatial_vector,
        shape=(data.nworld, model.nbody),
    )

    if model.nmocap > 0:
        _copy_to_device(
            wp,
            data.warp_data.mocap_pos,
            _as_batched(data.mocap_pos, nworld=data.nworld),
            dtype=wp.vec3,
            shape=(data.nworld, model.nmocap),
        )
        _copy_to_device(
            wp,
            data.warp_data.mocap_quat,
            _as_batched(data.mocap_quat, nworld=data.nworld),
            dtype=wp.quat,
            shape=(data.nworld, model.nmocap),
        )

    if model.neq > 0:
        _copy_to_device(
            wp,
            data.warp_data.eq_active,
            _as_batched(data.eq_active, nworld=data.nworld),
            dtype=bool,
        )


def _pull_device_into_host(model: MjoModel, data: MjoData) -> None:
    mjw, _ = require_mjwarp()

    for world_id, run in enumerate(data._host_runs):
        mjw.get_data_into(run.mj_data, model.mj_model, data.warp_data, world_id=world_id)
        _refresh_host_orbit_caches(model, run)
        mujoco.mj_forward(model.mj_model, run.mj_data)


def mjo_forward(model: MjoModel, data: MjoData) -> None:
    """Synchronize derived runtime state after direct mutation."""
    mjw, _ = require_mjwarp()

    data._push_to_host()
    for run in data._host_runs:
        from mujoco_orbit.core.step import mjo_forward as cpu_forward

        cpu_forward(model.host_model, run)

    data._pull_from_host()
    _sync_device_from_public(model, data)
    mjw.forward(model.warp_model, data.warp_data)
    _pull_device_into_host(model, data)
    data._pull_from_host()


def mjo_step(model: MjoModel, data: MjoData) -> None:
    """Advance one fully coupled MJWarp simulation step in-place."""
    mjw, _ = require_mjwarp()

    mj_dt = model.opt.timestep
    orbit_dt = model.orbit_dt if model.orbit_dt is not None else mj_dt

    data._push_to_host()
    for run in data._host_runs:
        _refresh_host_orbit_caches(model, run)
        mujoco.mj_forward(model.mj_model, run.mj_data)

        run.clear_wrench_buffer()
        assemble_and_apply_wrenches(model.host_model, run)
        _apply_reaction_wheels(model.host_model, run)
        command_rw_torques(model.host_model, run, run.actuators.rw_torque_cmd, mj_dt)
        _apply_magnetorquers(model.host_model, run)
        _apply_thrusters(model.host_model, run)
        np.copyto(run.xfrc_applied, run.wrench_buffer)

        net_force, _ = compute_net_external_wrench(run)
        a_feedback = compute_orbit_feedback_accel(model.host_model, run, net_force)
        orbit_next = propagate_rk4(
            run.orbit,
            orbit_dt,
            use_j2=model.use_j2,
            a_external=a_feedback,
        )
        run.orbit.R_eci[:] = orbit_next.R_eci
        run.orbit.V_eci[:] = orbit_next.V_eci
        run.orbit.t = orbit_next.t
        _refresh_host_orbit_caches(model, run)

    data._pull_from_host()
    _sync_device_from_public(model, data)
    mjw.step(model.warp_model, data.warp_data)
    _pull_device_into_host(model, data)
    data._pull_from_host()


__all__ = ["mjo_forward", "mjo_step"]
