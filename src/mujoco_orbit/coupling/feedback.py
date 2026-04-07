"""Reduce per-body/per-surface loads to net system external wrench for orbit feedback.

The net external force feeds back into chief orbit propagation.
Internal MuJoCo forces (contacts, joints, constraints) do NOT feed back.

What feeds back:
  - Thruster forces (translational)
  - Drag forces (translational)
  - SRP forces (translational)
  - Net gravity/J2 over the distributed mass model (usually near-zero for close bodies)

What does NOT feed back:
  - Reaction wheel torques (internal angular momentum exchange)
  - Magnetorquer torques (no translational component)
  - Inertial coupling forces (these are fictitious forces that cancel at the system level)

Units: N, N·m (SI).
"""

from __future__ import annotations

import numpy as np

from mujoco_orbit.core.runtime import MjoData, MjoModel


def compute_net_external_wrench(data: MjoData) -> tuple[np.ndarray, np.ndarray]:
    """Sum all body wrenches to get net external force and torque on the system.

    Returns:
        (net_force_world, net_torque_world) each shape (3,), in N and N·m.
    """
    # Sum forces and torques over all bodies
    net_force = np.sum(data.wrench_buffer[:, :3], axis=0)
    net_torque = np.sum(data.wrench_buffer[:, 3:], axis=0)
    return net_force, net_torque


def compute_orbit_feedback_accel(
    model: MjoModel,
    data: MjoData,
    net_force_world: np.ndarray,
) -> np.ndarray:
    """Convert net external force in world (LVLH) frame to ECI acceleration for orbit feedback.

    Args:
        model: compiled model (provides mass)
        data: runtime state (provides frame cache)
        net_force_world: net external force in LVLH/world frame, N

    Returns:
        acceleration in ECI, km/s^2
    """
    # Total system mass
    total_mass = np.sum(model.body_mass[1:])  # skip world body
    if total_mass <= 0.0:
        return np.zeros(3)

    # Force (N) -> acceleration (m/s^2) -> km/s^2
    a_world = net_force_world / total_mass  # m/s^2
    a_world_km = a_world * 1e-3  # km/s^2

    # Rotate from LVLH to ECI
    a_eci = data.frame.C_IL @ a_world_km
    return a_eci
