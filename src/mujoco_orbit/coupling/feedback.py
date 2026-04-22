"""Reduce per-body/per-surface loads to net system external wrench for orbit feedback.

The net external force feeds back into chief orbit propagation.
Internal MuJoCo forces (contacts, joints, constraints) do NOT feed back.

What feeds back:
  - Thruster forces (translational)
  - Drag forces (translational)
  - SRP forces (translational)

What does NOT feed back:
  - Reaction wheel torques (internal angular momentum exchange)
  - Magnetorquer torques (no translational component)
  - Gravity forces already represented by the reference orbit propagator

Units: N, N·m (SI).
"""

from __future__ import annotations

import numpy as np

from mujoco_orbit.core.runtime import MjoData, MjoModel
from mujoco_orbit.orbit.gravity import total_accel


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
    """Convert net external force in world (ECI) frame to ECI acceleration for orbit feedback.

    Args:
        model: compiled model (provides mass)
        data: runtime state (provides frame cache)
        net_force_world: net external force in ECI/world frame, N

    Returns:
        acceleration in ECI, km/s^2
    """
    # Total system mass
    total_mass = np.sum(model.body_mass[1:])  # skip world body
    if total_mass <= 0.0:
        return np.zeros(3)

    gravity_force = np.zeros(3)
    for body_id in range(1, model.nbody):
        mass = model.body_mass[body_id]
        if mass <= 0.0:
            continue
        r_eci_km = data.xipos[body_id] * 1e-3
        gravity_force += mass * total_accel(r_eci_km, use_j2=model.use_j2) * 1e3

    # Force (N) -> acceleration (m/s^2) -> km/s^2. The wrench buffer includes
    # full ECI gravity applied to MuJoCo bodies, while the reference orbit
    # propagator already applies gravity. Subtract all body gravity here so
    # only non-gravitational external loads feed back into the chief orbit.
    non_gravity_force = net_force_world - gravity_force
    a_feedback_eci = (non_gravity_force / total_mass) * 1e-3

    return a_feedback_eci
