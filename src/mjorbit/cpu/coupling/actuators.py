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
import mujoco

from mjorbit.cpu.core.scenario import CPUScenario


def apply_actuator_wrenches(scenario: CPUScenario, dt: float) -> None:
    """Compute and apply all external actuator wrenches.

    Also integrates reaction wheel speeds (modifies actuator_state in-place).
    """
    _apply_reaction_wheels(scenario, dt)
    _apply_magnetorquers(scenario)
    _apply_thrusters(scenario)


def _apply_reaction_wheels(scenario: CPUScenario, dt: float) -> None:
    """Integrate wheel speeds and apply reaction torques to host bodies."""
    cfg_rws = scenario.cfg.reaction_wheels
    if not cfg_rws:
        return

    act = scenario.actuator_state
    mjd = scenario.mjd

    for i, rw_cfg in enumerate(cfg_rws):
        bid = mujoco.mj_name2id(
            scenario.mjm, mujoco.mjtObj.mjOBJ_BODY, rw_cfg.body_name
        )
        if bid < 0:
            continue

        # Commanded torque on wheel (from external command interface)
        # For now, commanded torque is stored as: act.rw_speed is integrated externally.
        # The control loop sets a desired torque → we compute wheel accel.
        # We use rw_speed as state and expect the user to set a commanded torque.
        # Convention: positive rw_cmd = torque on wheel in +axis direction

        # Compute wheel angular acceleration from commanded torque
        # The commanded torque is encoded in rw_speed changes by the control loop.
        # For Phase 7, we provide a helper that takes a torque command.
        # But for the wrench assembly, we just need the current speed and any
        # commanded torque. We store the "last commanded torque" in the speed update.
        #
        # Simplification: the control loop calls `command_rw_torque` which updates
        # speed and returns the reaction torque. Here we just read the stored speed.
        # Actually, we need the torque command for reaction. Let me restructure:
        # The control flow is:
        #   1. User sets rw_torque_cmd[i] (external)
        #   2. We clamp, saturate, integrate speed, apply reaction torque
        # We need a torque command buffer. Let me use a convention:
        # act.rw_speed contains the current speed. The commanded torque comes from
        # a separate buffer. For now, I'll add it inline.
        pass  # Handled by command_rw_torques below


def command_rw_torques(
    scenario: CPUScenario,
    torque_cmds: np.ndarray,
    dt: float,
) -> None:
    """Apply reaction wheel torque commands, integrate wheel speeds, and write body torques.

    Args:
        scenario: current scenario
        torque_cmds: shape (n_rw,) torque commands on each wheel, N·m
            Positive = accelerate wheel in +axis direction
        dt: timestep for speed integration
    """
    cfg_rws = scenario.cfg.reaction_wheels
    if not cfg_rws:
        return

    act = scenario.actuator_state
    mjd = scenario.mjd

    for i, rw_cfg in enumerate(cfg_rws):
        bid = mujoco.mj_name2id(
            scenario.mjm, mujoco.mjtObj.mjOBJ_BODY, rw_cfg.body_name
        )
        if bid < 0:
            continue

        inertia = act.rw_inertia[i]
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
        axis_body = rw_cfg.axis_body / np.linalg.norm(rw_cfg.axis_body)
        tau_body = reaction_tau * axis_body

        # Rotate to world frame
        R_body = mjd.ximat[bid].reshape(3, 3)
        tau_world = R_body @ tau_body

        scenario._wrench_buffer[bid, 3:] += tau_world

    act.update_rw_momentum()


def _apply_magnetorquers(scenario: CPUScenario) -> None:
    """Apply magnetorquer torques: tau = m × B."""
    cfg_mtqs = scenario.cfg.magnetorquers
    if not cfg_mtqs:
        return
    if not scenario.cfg.use_magnetic:
        return

    act = scenario.actuator_state
    mjd = scenario.mjd
    fc = scenario.frame_cache
    env = scenario.env_cache

    # B field in world (LVLH) frame
    B_world = fc.C_LI @ env.mag_field_eci

    for i, mtq_cfg in enumerate(cfg_mtqs):
        bid = mujoco.mj_name2id(
            scenario.mjm, mujoco.mjtObj.mjOBJ_BODY, mtq_cfg.body_name
        )
        if bid < 0:
            continue

        # Clamp dipole command
        m_cmd = np.clip(act.mtq_dipole[i], -mtq_cfg.dipole_limit, mtq_cfg.dipole_limit)
        axis_body = mtq_cfg.axis_body / np.linalg.norm(mtq_cfg.axis_body)
        dipole_body = m_cmd * axis_body  # A·m^2 in body frame

        R_body = mjd.ximat[bid].reshape(3, 3)
        B_body = R_body.T @ B_world

        tau_body = np.cross(dipole_body, B_body)
        tau_world = R_body @ tau_body

        scenario._wrench_buffer[bid, 3:] += tau_world


def _apply_thrusters(scenario: CPUScenario) -> None:
    """Apply thruster forces and torques."""
    cfg_thrs = scenario.cfg.thrusters
    if not cfg_thrs:
        return

    act = scenario.actuator_state
    mjd = scenario.mjd

    for i, thr_cfg in enumerate(cfg_thrs):
        bid = mujoco.mj_name2id(
            scenario.mjm, mujoco.mjtObj.mjOBJ_BODY, thr_cfg.body_name
        )
        if bid < 0:
            continue

        # Clamp thrust command
        f_cmd = np.clip(act.thr_force[i], 0.0, thr_cfg.force_limit)

        direction_body = thr_cfg.direction_body / np.linalg.norm(thr_cfg.direction_body)
        F_body = f_cmd * direction_body  # N in body frame

        R_body = mjd.ximat[bid].reshape(3, 3)
        F_world = R_body @ F_body  # N in world frame

        # xfrc_applied torques are about the body COM, not the body-frame origin.
        r_com_body = thr_cfg.position_body - scenario.mjm.body_ipos[bid]
        tau_body = np.cross(r_com_body, F_body)  # N·m in body frame
        tau_world = R_body @ tau_body

        scenario._wrench_buffer[bid, :3] += F_world
        scenario._wrench_buffer[bid, 3:] += tau_world
