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
    _apply_cmgs(model, data)
    command_cmg_gimbal_rates(model, data, data.actuators.cmg_gimbal_rate_cmd, dt)
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

        # cvel stores angular velocity in the world frame for every body, including
        # downstream articulated links. Rotate it back into the host body frame.
        R_body = mjd.xmat[bid].reshape(3, 3)
        w_body = R_body.T @ mjd.cvel[bid, :3]

        # Gyroscopic coupling torque: τ = -ω × h  (body frame)
        tau_body = -np.cross(w_body, h_body)

        # Rotate to world frame for xfrc_applied
        # Use xmat (body frame), NOT ximat (inertia frame)
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
    env = data.env

    # MuJoCo world axes are parallel to ECI, so cached B is already in world axes.
    B_world = env.mag_field_eci

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


def _cmg_momentum_body(
    rotor_momentum: float,
    gimbal_angle: float,
    spin_axis_0: np.ndarray,
    torque_axis_0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute rotor momentum vector and torque axis in the body frame.

    Returns:
        h_body: rotor angular momentum in body frame (kg·m^2/s)
        torque_axis_body: unit vector ``t̂(θ) = ĝ × ŝ(θ)`` in body frame
    """
    c = np.cos(gimbal_angle)
    s = np.sin(gimbal_angle)
    spin_axis = c * spin_axis_0 + s * torque_axis_0
    # torque_axis(θ) = ĝ × ŝ(θ) = c·(ĝ × ŝ0) + s·(ĝ × t̂0) = c·t̂0 − s·ŝ0
    torque_axis = c * torque_axis_0 - s * spin_axis_0
    h_body = rotor_momentum * spin_axis
    return h_body, torque_axis


def _apply_cmgs(model: MjoModel, data: MjoData) -> None:
    """Apply gyroscopic coupling torque from stored CMG rotor momentum.

    For a gyrostat with rotor momentum ``h`` (body frame) on top of a rigid body with
    inertia ``J``, the Euler equation is::

        J·ω̇ + ω × (J·ω + h) = τ_ext

    MuJoCo integrates ``J·ω̇ + ω × J·ω = τ_ext``, so the extra term ``-ω × h`` must be
    applied externally. The gimbal-rate-induced output torque is handled separately
    by ``command_cmg_gimbal_rates``.
    """
    if not model.cmgs:
        return

    act = data.actuators
    mjd = data.mj_data

    for i, cmg_cfg in enumerate(model.cmgs):
        bid = cmg_cfg.body_id

        h_body, _ = _cmg_momentum_body(
            act.cmg_rotor_momentum[i],
            float(act.cmg_gimbal_angle[i]),
            cmg_cfg.spin_axis_body_0,
            cmg_cfg.torque_axis_body_0,
        )

        R_body = mjd.xmat[bid].reshape(3, 3)
        w_body = R_body.T @ mjd.cvel[bid, :3]

        tau_body = -np.cross(w_body, h_body)
        tau_world = R_body @ tau_body

        data.wrench_buffer[bid, 3:] += tau_world


def command_cmg_gimbal_rates(
    model: MjoModel,
    data: MjoData,
    gimbal_rate_cmds: np.ndarray,
    dt: float,
) -> None:
    """Apply CMG gimbal-rate commands, integrate gimbal angles, and write body torques.

    The controllable output torque of an SGCMG is ``τ = -h · θ̇ · t̂(θ)`` on the
    spacecraft body, where ``h`` is the rotor momentum magnitude, ``θ̇`` is the
    gimbal rate, and ``t̂(θ) = ĝ × ŝ(θ)`` is the (body-frame) torque axis. This
    function clips the commanded rate, respects gimbal-angle saturation, integrates
    the gimbal angle with forward Euler, and accumulates the reaction torque into
    ``data.wrench_buffer``.
    """
    if not model.cmgs:
        return

    act = data.actuators
    mjd = data.mj_data

    for i, cmg_cfg in enumerate(model.cmgs):
        bid = cmg_cfg.body_id
        h_rotor = act.cmg_rotor_momentum[i]
        if h_rotor <= 0.0:
            continue

        theta_old = float(act.cmg_gimbal_angle[i])

        theta_dot = float(gimbal_rate_cmds[i])
        if cmg_cfg.gimbal_rate_limit is not None:
            theta_dot = float(
                np.clip(theta_dot, -cmg_cfg.gimbal_rate_limit, cmg_cfg.gimbal_rate_limit)
            )

        # Gimbal-angle saturation: freeze rate if we're against a hard stop
        if cmg_cfg.gimbal_angle_limit is not None:
            limit = cmg_cfg.gimbal_angle_limit
            if theta_old >= limit and theta_dot > 0:
                theta_dot = 0.0
            elif theta_old <= -limit and theta_dot < 0:
                theta_dot = 0.0

        # Compute torque axis at current gimbal angle
        _, t_axis_body = _cmg_momentum_body(
            h_rotor,
            theta_old,
            cmg_cfg.spin_axis_body_0,
            cmg_cfg.torque_axis_body_0,
        )

        # Integrate gimbal angle (forward Euler)
        theta_new = theta_old + theta_dot * dt
        if cmg_cfg.gimbal_angle_limit is not None:
            theta_new = float(
                np.clip(theta_new, -cmg_cfg.gimbal_angle_limit, cmg_cfg.gimbal_angle_limit)
            )
        act.cmg_gimbal_angle[i] = theta_new
        theta_dot_effective = (theta_new - theta_old) / dt

        # Reaction torque on body: τ_body = -h · θ̇ · t̂(θ)
        tau_body = -h_rotor * theta_dot_effective * t_axis_body
        R_body = mjd.xmat[bid].reshape(3, 3)
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
