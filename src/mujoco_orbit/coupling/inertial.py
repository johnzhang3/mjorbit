"""Per-body inertial and gravity forcing in the LVLH rotating frame.

The MuJoCo model lives in a local LVLH frame. Each body's position in the LVLH
frame represents a relative offset from the chief. We compute the apparent
acceleration acting on each body in that frame and apply it as a force.

Apparent acceleration in LVLH (rotating frame):
    a_body_lvlh = C_LI @ (g_eci(r_body_eci) - g_eci(R_ref))
                  - 2 * omega x v_body_lvlh
                  - omega_dot x r_body_lvlh
                  - omega x (omega x r_body_lvlh)

This is the exact formulation — no CW linearization.

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

from mujoco_orbit.core.scenario import Scenario
from mujoco_orbit.orbit.gravity import total_accel

# MuJoCo world frame = LVLH frame (by construction of the model)
# Body COM positions in MuJoCo are in meters (SI).
# Chief reference position is in km.
# Convert: r_body_lvlh_km = r_body_lvlh_m * 1e-3

_M_TO_KM = 1e-3
_KM_S2_TO_M_S2 = 1e3  # km/s^2 -> m/s^2


def apply_inertial_wrenches(scenario: Scenario) -> None:
    """Compute per-body apparent accelerations and accumulate forces.

    Writes force contributions (N) into scenario._wrench_buffer[:, :3].
    Torque contributions are zero for translational forcing.
    """
    orbit = scenario.orbit
    fc = scenario.frame_cache
    mjm = scenario.mjm
    mjd = scenario.mjd

    R_ref = orbit.R_eci  # km
    g_ref = total_accel(R_ref, use_j2=scenario.cfg.use_j2)  # km/s^2

    omega = fc.omega_lvlh  # rad/s, in LVLH
    omega_dot = fc.omega_dot_lvlh  # rad/s^2, in LVLH
    C_LI = fc.C_LI

    for body_id in range(1, mjm.nbody):  # skip world body (id=0)
        mass = mjm.body_mass[body_id]  # kg
        if mass <= 0.0:
            continue

        # Body COM position in MuJoCo world (= LVLH) frame, meters
        r_m = mjd.xipos[body_id].copy()  # shape (3,) m

        # Convert to km for orbit-layer computation
        r_lvlh_km = r_m * _M_TO_KM

        # Absolute ECI position of body
        r_body_eci = R_ref + fc.C_IL @ r_lvlh_km

        # Gravity at body position
        g_body = total_accel(r_body_eci, use_j2=scenario.cfg.use_j2)  # km/s^2

        # Relative gravity gradient term
        dg = C_LI @ (g_body - g_ref)  # km/s^2, in LVLH

        # Body COM velocity in LVLH/world frame (from MuJoCo cvel)
        # cvel[i] is 6D spatial velocity [angular(3), linear(3)] at body COM, in world frame.
        v_lvlh_m_s = mjd.cvel[body_id, 3:].copy()  # linear velocity, m/s, world frame
        v_lvlh_km_s = v_lvlh_m_s * 1e-3  # km/s

        # Rotating frame fictitious accelerations (km/s^2)
        a_coriolis = -2.0 * np.cross(omega, v_lvlh_km_s)
        a_euler = -np.cross(omega_dot, r_lvlh_km)
        a_centripetal = -np.cross(omega, np.cross(omega, r_lvlh_km))

        a_total_km_s2 = dg + a_coriolis + a_euler + a_centripetal

        # Convert to m/s^2 and multiply by mass for force in N
        F_N = mass * a_total_km_s2 * _KM_S2_TO_M_S2

        scenario._wrench_buffer[body_id, :3] += F_N
