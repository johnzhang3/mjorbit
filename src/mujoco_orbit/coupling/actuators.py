# pyright: reportAttributeAccessIssue=false

"""External actuator wrench assembly (Phase 7).

Handles reaction wheels, magnetorquers, and thrusters.
All are modeled outside MuJoCo rigid-body state.

Reaction wheels:
  - Integrate wheel speed from commanded torque
  - Apply equal-and-opposite torque to host body
  - No translational force

Magnetorquers:
  - Commanded dipole moment → tau = m × B
  - No translational force

Thrusters:
  - Commanded force at a body-fixed application point
  - Produces force + torque on body
  - Contributes to orbit feedback through net external force
"""

from __future__ import annotations

import numpy as np

from mujoco_orbit.core.runtime import MjoData, MjoModel


def apply_actuator_wrenches(
    model: MjoModel,
    data: MjoData,
    dt: float,
) -> None:
    """Compute and apply all external actuator wrenches.

    Also integrates reaction wheel speeds (modifies ``data.actuators`` in-place).
    """
    _apply_reaction_wheels(model, data)
    command_rw_torques(model, data, data.actuators.rw_torque_cmd, dt)
    _apply_magnetorquers(model, data)
    _apply_thrusters(model, data)


def _apply_reaction_wheels(model: MjoModel, data: MjoData) -> None:
    """Apply gyroscopic coupling torque from stored reaction wheel momentum.

    MuJoCo integrates the rigid-body Euler equations:
        J·ω̇ + ω × J·ω = τ_ext

    For a gyrostat with rotor momentum h in the body frame, the correct
    equation of motion is:
        J·ω̇ + ω × (J·ω + h) = τ_ext

    The extra term  -ω × h  must be applied as an external torque so that
    MuJoCo's integrator produces the correct gyrostat dynamics.

    Torque commands (wheel acceleration) are handled separately by
    ``command_rw_torques``.
    """
    if not model.reaction_wheels:
        return

    act = data.actuators
    mjd = data.mj_data

    for i, rw_cfg in enumerate(model.reaction_wheels):
        bid = rw_cfg.body_id

        # Wheel angular momentum in body frame: h_i = I_w * Ω_w * axis
        h_body = model.rw_inertia[i] * act.rw_speed[i] * rw_cfg.axis_body  # kg·m²/s

        # Free-joint angular velocity is in body frame
        w_body = mjd.qvel[3:6]

        # Gyroscopic coupling torque: τ = -ω × h  (body frame)
        tau_body = -np.cross(w_body, h_body)

        # Rotate to world frame for xfrc_applied
        # Use xmat (body frame), NOT ximat (inertia frame)
        R_body = mjd.xmat[bid].reshape(3, 3)
        tau_world = R_body @ tau_body

        data.wrench_buffer[bid, 3:] += tau_world


def command_rw_torques(
    model: MjoModel,
    data: MjoData,
    torque_cmds: np.ndarray,
    dt: float,
) -> None:
    """Apply reaction wheel torque commands, integrate wheel speeds, and write body torques.

    Args:
        model: compiled model
        data: runtime state
        torque_cmds: shape (n_rw,) torque commands on each wheel, N·m
            Positive = accelerate wheel in +axis direction
        dt: timestep for speed integration
    """
    if not model.reaction_wheels:
        return

    act = data.actuators
    mjd = data.mj_data

    for i, rw_cfg in enumerate(model.reaction_wheels):
        bid = rw_cfg.body_id
        inertia = model.rw_inertia[i]
        if inertia <= 0.0:
            continue

        # Clamp torque command
        tau = torque_cmds[i]
        if rw_cfg.torque_limit is not None:
            tau = np.clip(tau, -rw_cfg.torque_limit, rw_cfg.torque_limit)

        # Compute wheel acceleration
        alpha = tau / inertia

        # Speed saturation: if at limit, don't accelerate further in that direction
        if rw_cfg.speed_limit is not None:
            if act.rw_speed[i] >= rw_cfg.speed_limit and alpha > 0:
                alpha = 0.0
            elif act.rw_speed[i] <= -rw_cfg.speed_limit and alpha < 0:
                alpha = 0.0

        # Integrate wheel speed
        act.rw_speed[i] += alpha * dt
        if rw_cfg.speed_limit is not None:
            act.rw_speed[i] = np.clip(act.rw_speed[i], -rw_cfg.speed_limit, rw_cfg.speed_limit)

        # Reaction torque on body: equal and opposite
        # tau_body = -tau_wheel (in body frame along axis)
        reaction_tau = -inertia * alpha  # N·m
        tau_body = reaction_tau * rw_cfg.axis_body

        # Rotate to world frame
        R_body = mjd.xmat[bid].reshape(3, 3)
        tau_world = R_body @ tau_body

        data.wrench_buffer[bid, 3:] += tau_world

    act.update_rw_momentum(model.rw_inertia)


def _apply_magnetorquers(model: MjoModel, data: MjoData) -> None:
    """Apply magnetorquer torques: tau = m × B."""
    if not model.magnetorquers:
        return
    if not model.use_magnetic:
        return

    act = data.actuators
    mjd = data.mj_data
    fc = data.frame
    env = data.env

    # B field in world (LVLH) frame
    B_world = fc.C_LI @ env.mag_field_eci

    for i, mtq_cfg in enumerate(model.magnetorquers):
        bid = mtq_cfg.body_id
        # Clamp dipole command
        m_cmd = np.clip(act.mtq_dipole_cmd[i], -mtq_cfg.dipole_limit, mtq_cfg.dipole_limit)
        dipole_body = m_cmd * mtq_cfg.axis_body  # A·m^2 in body frame

        R_body = mjd.xmat[bid].reshape(3, 3)
        B_body = R_body.T @ B_world

        tau_body = np.cross(dipole_body, B_body)
        tau_world = R_body @ tau_body

        data.wrench_buffer[bid, 3:] += tau_world


def _apply_thrusters(model: MjoModel, data: MjoData) -> None:
    """Apply thruster forces and torques."""
    if not model.thrusters:
        return

    act = data.actuators
    mjd = data.mj_data

    for i, thr_cfg in enumerate(model.thrusters):
        bid = thr_cfg.body_id
        # Clamp thrust command
        f_cmd = np.clip(act.thr_force_cmd[i], 0.0, thr_cfg.force_limit)

        F_body = f_cmd * thr_cfg.direction_body  # N in body frame

        R_body = mjd.xmat[bid].reshape(3, 3)
        F_world = R_body @ F_body  # N in world frame

        # xfrc_applied torques are about the body COM, not the body-frame origin.
        r_com_body = thr_cfg.position_body - model.body_ipos[bid]
        tau_body = np.cross(r_com_body, F_body)  # N·m in body frame
        tau_world = R_body @ tau_body

        data.wrench_buffer[bid, :3] += F_world
        data.wrench_buffer[bid, 3:] += tau_world
