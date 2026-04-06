# pyright: reportAttributeAccessIssue=false

"""step — advance Scenario by one timestep (Phase 8)."""

from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np

from mujoco_orbit.core.scenario import Scenario
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


def step(
    scenario: Scenario,
    ctrl: Optional[np.ndarray] = None,
    rw_torques: Optional[np.ndarray] = None,
) -> None:
    """Advance scenario by one coupled simulation step (in-place).

    Order of operations (Phase 8):
    1. Assemble per-body/per-surface wrenches at current state
    2. Apply actuator wrenches (RW, MTQ, thrusters)
    3. Compute net external wrench for orbit feedback
    4. Propagate chief orbit with gravity + feedback perturbation
    5. Update frame and environment caches
    6. Write xfrc_applied
    7. Write MuJoCo controls if provided
    8. Call mujoco.mj_step

    Args:
        scenario: simulation state (modified in-place)
        ctrl: optional MuJoCo control vector
        rw_torques: optional reaction wheel torque commands, shape (n_rw,) N·m

    Notes:
        MuJoCo advances by ``mjm.opt.timestep`` every call. The chief orbit uses
        ``cfg.orbit_dt`` when provided, otherwise it falls back to the MuJoCo step.
    """
    mj_dt = scenario.mjm.opt.timestep
    orbit_dt = scenario.cfg.orbit_dt if scenario.cfg.orbit_dt is not None else mj_dt

    # 1. Compute environment wrenches (inertial, surfaces, magnetic residual)
    scenario.clear_wrench_buffer()
    assemble_and_apply_wrenches(scenario)

    # 2. Apply external actuator wrenches
    _apply_reaction_wheels(scenario, mj_dt)
    if rw_torques is not None:
        command_rw_torques(scenario, rw_torques, mj_dt)
    _apply_magnetorquers(scenario)
    _apply_thrusters(scenario)

    # Re-copy after actuator contributions
    np.copyto(scenario.mjd.xfrc_applied, scenario._wrench_buffer)

    # 3. Compute orbit feedback from net external force
    net_force, _net_torque = compute_net_external_wrench(scenario)
    a_feedback = compute_orbit_feedback_accel(scenario, net_force)

    # 4. Propagate chief orbit
    scenario.orbit = propagate_rk4(
        scenario.orbit, orbit_dt, use_j2=scenario.cfg.use_j2, a_external=a_feedback
    )

    # 5. Update caches
    scenario.frame_cache = update_frame_cache(
        scenario.orbit, use_j2=scenario.cfg.use_j2
    )
    scenario.env_cache = update_environment_cache(scenario.orbit, scenario.frame_cache)

    # 6. Write MuJoCo controls
    if ctrl is not None:
        np.copyto(scenario.mjd.ctrl, ctrl)

    # 7. Step MuJoCo
    mujoco.mj_step(scenario.mjm, scenario.mjd)
