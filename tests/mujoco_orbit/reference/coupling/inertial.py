"""Per-body gravity forcing in the chief-centered inertial MuJoCo world frame.

The MuJoCo model lives in SI coordinates relative to the chief, with axes
parallel to ECI. Each body's absolute ECI position is reconstructed before
evaluating gravity, and the chief gravitational acceleration is subtracted so
MuJoCo integrates the relative translational dynamics.

Units convention:
  MuJoCo uses SI (m, s, kg, N).
  Orbit layer uses km, km/s, km/s^2.
  Forces written to xfrc_applied must be in N.

  Conversion: 1 km/s^2 = 1e3 m/s^2
  Force (N) = mass (kg) * accel (m/s^2)
  Orbit accel (km/s^2) * 1e3 -> m/s^2
"""

from __future__ import annotations

import numpy as np

from mujoco_orbit.core.runtime import MjoData, MjoModel
from tests.mujoco_orbit.reference.orbit.gravity import total_accel

_M_TO_KM = 1e-3
_KM_S2_TO_M_S2 = 1e3  # km/s^2 -> m/s^2


def body_eci_position_km(data: MjoData, position_world_m: np.ndarray) -> np.ndarray:
    """Convert a MuJoCo world position to absolute ECI km."""
    return data.orbit.R_eci + np.asarray(position_world_m, dtype=float) * _M_TO_KM


def body_eci_velocity_km_s(data: MjoData, velocity_world_m_s: np.ndarray) -> np.ndarray:
    """Convert a MuJoCo world velocity to absolute ECI km/s."""
    return data.orbit.V_eci + np.asarray(velocity_world_m_s, dtype=float) * _M_TO_KM


def chief_gravity(data: MjoData, model: MjoModel) -> np.ndarray:
    """Return chief/reference gravitational acceleration in km/s^2."""
    return total_accel(data.orbit.R_eci, use_j2=model.use_j2)


def differential_gravity_force(
    model: MjoModel,
    data: MjoData,
    body_id: int,
    *,
    chief_accel: np.ndarray | None = None,
) -> np.ndarray:
    """Return a body's chief-relative gravity force in ECI-parallel world axes."""
    mass = model.body_mass[body_id]
    if mass <= 0.0:
        return np.zeros(3)

    g_chief = chief_gravity(data, model) if chief_accel is None else chief_accel
    r_body_eci = body_eci_position_km(data, data.xipos[body_id])
    g_body = total_accel(r_body_eci, use_j2=model.use_j2)
    return mass * (g_body - g_chief) * _KM_S2_TO_M_S2


def apply_inertial_wrenches(model: MjoModel, data: MjoData) -> None:
    """Compute per-body chief-relative gravity accelerations and accumulate forces.

    Writes force contributions (N) into ``data.wrench_buffer[:, :3]``.
    Torque contributions are zero for translational forcing.
    """
    g_chief = chief_gravity(data, model)

    for body_id in range(1, model.nbody):  # skip world body (id=0)
        mass = model.body_mass[body_id]  # kg
        if mass <= 0.0:
            continue

        data.wrench_buffer[body_id, :3] += differential_gravity_force(
            model,
            data,
            body_id,
            chief_accel=g_chief,
        )
