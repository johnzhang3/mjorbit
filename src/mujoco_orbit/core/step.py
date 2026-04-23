# pyright: reportAttributeAccessIssue=false

"""Forward and step functions for the MuJoCo-style API."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_orbit.core.runtime import MjoData, MjoModel
from mujoco_orbit.coupling.actuators import (
    _apply_cmgs,
    _apply_magnetorquers,
    _apply_reaction_wheels,
    _apply_thrusters,
    command_cmg_gimbal_rates,
    command_rw_torques,
)
from mujoco_orbit.coupling.apply import assemble_and_apply_wrenches
from mujoco_orbit.coupling.feedback import compute_net_external_wrench, compute_orbit_feedback_accel
from mujoco_orbit.orbit.propagator import propagate_rk4
from mujoco_orbit.sensors import update_sensor_environment


def _clear_wrench_buffer(data: MjoData) -> None:
    data.wrench_buffer[:] = 0.0
    data.xfrc_applied[:] = 0.0


def _refresh_orbit_caches(model: MjoModel, data: MjoData) -> None:
    data.refresh_orbit_caches(model.use_j2)
    data.actuators.update_rw_momentum(model.rw_inertia)
    update_sensor_environment(model, data)


def _subtract_origin_acceleration(model: MjoModel, data: MjoData, a_origin_eci: np.ndarray) -> None:
    """Apply the fictitious force from chief-frame translational acceleration."""
    a_origin_m_s2 = a_origin_eci * 1e3
    if not np.any(a_origin_m_s2):
        return

    for body_id in range(1, model.nbody):
        mass = model.body_mass[body_id]
        if mass <= 0.0:
            continue
        data.wrench_buffer[body_id, :3] -= mass * a_origin_m_s2


def mjo_forward(model: MjoModel, data: MjoData) -> None:
    """Synchronize derived runtime state after direct mutation."""
    _refresh_orbit_caches(model, data)
    mujoco.mj_forward(model.mj_model, data.mj_data)

    _clear_wrench_buffer(data)
    assemble_and_apply_wrenches(model, data)
    _apply_reaction_wheels(model, data)
    _apply_cmgs(model, data)
    _apply_magnetorquers(model, data)
    _apply_thrusters(model, data)
    np.copyto(data.xfrc_applied, data.wrench_buffer)

    mujoco.mj_forward(model.mj_model, data.mj_data)


def mjo_step(model: MjoModel, data: MjoData) -> None:
    """Advance one fully coupled simulation step in-place."""
    mj_dt = model.opt.timestep
    orbit_dt = model.orbit_dt if model.orbit_dt is not None else mj_dt

    _clear_wrench_buffer(data)
    assemble_and_apply_wrenches(model, data)
    _apply_reaction_wheels(model, data)
    command_rw_torques(model, data, data.actuators.rw_torque_cmd, mj_dt)
    _apply_cmgs(model, data)
    command_cmg_gimbal_rates(model, data, data.actuators.cmg_gimbal_rate_cmd, mj_dt)
    _apply_magnetorquers(model, data)
    _apply_thrusters(model, data)
    np.copyto(data.xfrc_applied, data.wrench_buffer)

    net_force, _ = compute_net_external_wrench(data)
    a_feedback = compute_orbit_feedback_accel(model, data, net_force)
    _subtract_origin_acceleration(model, data, a_feedback)
    np.copyto(data.xfrc_applied, data.wrench_buffer)

    orbit_next = propagate_rk4(data.orbit, orbit_dt, use_j2=model.use_j2, a_external=a_feedback)
    data.orbit.R_eci[:] = orbit_next.R_eci
    data.orbit.V_eci[:] = orbit_next.V_eci
    data.orbit.t = orbit_next.t

    _refresh_orbit_caches(model, data)
    mujoco.mj_step(model.mj_model, data.mj_data)


__all__ = ["mjo_forward", "mjo_step"]
