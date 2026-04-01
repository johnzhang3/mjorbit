"""Drag and SRP surface load computation (Phase 5).

For each configured flat-plate surface:
1. Rotate center of pressure and surface normal to world (LVLH) frame
2. Compute atmosphere-relative velocity at the surface point
3. Compute drag force from projected area
4. Compute SRP force from projected area
5. Convert force-at-point to body wrench (force + torque)
6. Accumulate into scenario wrench buffer

Units: forces in N, torques in N·m (SI, MuJoCo convention).
"""

from __future__ import annotations

import numpy as np
from mjorbit.cpu.core.scenario import CPUScenario
from mjorbit.constants import P_SUN, OMEGA_EARTH


def apply_surface_wrenches(scenario: CPUScenario) -> None:
    """Compute drag and SRP loads for all configured surfaces."""
    if not scenario.surfaces:
        return

    orbit = scenario.orbit
    fc = scenario.frame_cache
    env = scenario.env_cache
    mjd = scenario.mjd

    # Chief's atmosphere-relative velocity in ECI, then LVLH (km/s)
    omega_earth = np.array([0.0, 0.0, OMEGA_EARTH])
    v_rel_chief_eci = orbit.V_eci - np.cross(omega_earth, orbit.R_eci)
    v_rel_chief_lvlh = fc.C_LI @ v_rel_chief_eci  # km/s

    # Sun direction in world (LVLH) frame
    sun_world = fc.C_LI @ env.sun_vector_eci

    # Earth-rotation angular velocity in LVLH frame
    omega_earth_lvlh = fc.C_LI @ omega_earth

    rho = env.atm_density  # kg/m^3

    for surf in scenario.surfaces:
        bid = surf.body_id

        # Body world-from-body rotation
        R_body = mjd.ximat[bid].reshape(3, 3)

        # Surface point in world frame (meters, relative to body COM)
        r_cop_world = R_body @ surf.center_of_pressure_body  # m
        n_world = R_body @ surf.normal_body  # unit vector

        # Body COM velocity and angular velocity (world frame, m/s, rad/s)
        v_com = mjd.cvel[bid, 3:].copy()  # m/s
        omega_body = mjd.cvel[bid, :3].copy()  # rad/s

        # Velocity at surface point in world frame (m/s)
        v_point = v_com + np.cross(omega_body, r_cop_world)

        # Atmosphere-relative velocity at surface point in world frame (m/s)
        # v_rel = v_chief_rel + v_body_in_lvlh + rotation corrections
        r_point_lvlh_km = (mjd.xipos[bid] + r_cop_world) * 1e-3  # km
        correction_km_s = np.cross(fc.omega_lvlh - omega_earth_lvlh, r_point_lvlh_km)
        v_rel_m_s = v_rel_chief_lvlh * 1e3 + v_point + correction_km_s * 1e3

        speed = np.linalg.norm(v_rel_m_s)

        F_total = np.zeros(3)

        # --- Drag ---
        if surf.use_drag and scenario.cfg.use_drag and speed > 1e-10:
            v_hat = v_rel_m_s / speed
            # Projected area: only when the panel normal points into the flow.
            cos_angle = np.dot(n_world, v_hat)
            if cos_angle > 0.0:
                projected_area = surf.area * cos_angle
                F_drag = -0.5 * rho * surf.drag_coeff * projected_area * speed**2 * v_hat
                F_total += F_drag

        # --- SRP ---
        if surf.use_srp and scenario.cfg.use_srp and env.eclipse > 0.0:
            cos_sun = np.dot(n_world, sun_world)
            if cos_sun > 0.0:
                projected_area = surf.area * cos_sun
                F_srp = -env.eclipse * P_SUN * surf.srp_coeff * projected_area * sun_world
                F_total += F_srp

        # Accumulate force and torque into wrench buffer
        scenario._wrench_buffer[bid, :3] += F_total
        tau = np.cross(r_cop_world, F_total)  # N·m
        scenario._wrench_buffer[bid, 3:] += tau
