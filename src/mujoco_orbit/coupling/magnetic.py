"""Magnetic torque coupling: tau = m × B (Phase 6).

Handles fixed residual dipoles from MagneticBodyCfg.
Magnetorquer actuator coupling is in coupling/actuators.py.

Frame convention:
  - B field is in ECI from the environment cache
  - Dipoles are specified in body frame
  - Torques are computed in body frame and rotated to world frame for xfrc_applied
"""

from __future__ import annotations

import numpy as np

from mujoco_orbit.core.scenario import Scenario


def apply_magnetic_wrenches(scenario: Scenario) -> None:
    """Apply torques from fixed residual magnetic dipoles (tau = m × B)."""
    if not scenario.magnetic_bodies:
        return
    if not scenario.cfg.use_magnetic:
        return

    fc = scenario.frame_cache
    env = scenario.env_cache
    mjd = scenario.mjd

    # B field in world (LVLH) frame
    B_world = fc.C_LI @ env.mag_field_eci

    for mag in scenario.magnetic_bodies:
        bid = mag.body_id

        # World-from-body rotation
        R_body = mjd.xmat[bid].reshape(3, 3)

        # Body-from-world rotation
        R_inv = R_body.T

        # Magnetic field in body frame
        B_body = R_inv @ B_world

        # Torque in body frame: tau = m × B
        tau_body = np.cross(mag.dipole_body, B_body)

        # Rotate to world frame for xfrc_applied
        tau_world = R_body @ tau_body

        # No translational force from magnetic dipole
        scenario._wrench_buffer[bid, 3:] += tau_world
