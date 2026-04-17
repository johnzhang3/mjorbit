"""Drag and SRP surface load computation (Phase 5).

For each configured flat-plate surface:
1. Rotate center of pressure and surface normal to world (LVLH) frame
2. Compute atmosphere-relative velocity at the surface point
3. Compute drag force from projected area
4. Compute SRP force from projected area
5. Convert force-at-point to body wrench (force + torque)
6. Accumulate into the runtime wrench buffer

Units: forces in N, torques in N·m (SI, MuJoCo convention).
"""

from __future__ import annotations

import numpy as np

from mujoco_orbit.constants import OMEGA_EARTH, P_SUN
from mujoco_orbit.core.runtime import MjoData, MjoModel
from mujoco_orbit.orbit.environment import atm_density, eclipse_factor


def apply_surface_wrenches(model: MjoModel, data: MjoData) -> None:
    """Compute drag and SRP loads for all configured surfaces.

    Atmospheric density and eclipse are evaluated at each surface's ECI
    position so that bodies offset from the chief see the correct altitude
    and shadow state. Sun direction uses the chief's cached value, since
    parallax over LVLH separations is negligible.
    """
    if not model.surfaces:
        return

    orbit = data.orbit
    fc = data.frame
    env = data.env
    mjd = data.mj_data

    # Chief's atmosphere-relative velocity in ECI, then LVLH (km/s)
    omega_earth = np.array([0.0, 0.0, OMEGA_EARTH])
    v_rel_chief_eci = orbit.V_eci - np.cross(omega_earth, orbit.R_eci)
    v_rel_chief_lvlh = fc.C_LI @ v_rel_chief_eci  # km/s

    # Sun direction in world (LVLH) frame
    sun_hat_eci = env.sun_vector_eci
    sun_world = fc.C_LI @ sun_hat_eci

    # Earth-rotation angular velocity in LVLH frame
    omega_earth_lvlh = fc.C_LI @ omega_earth

    for surf in model.surfaces:
        bid = surf.body_id

        # Body world-from-body rotation
        R_body = mjd.xmat[bid].reshape(3, 3)

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
        r_point_eci = orbit.R_eci + fc.C_IL @ r_point_lvlh_km
        correction_km_s = np.cross(fc.omega_lvlh - omega_earth_lvlh, r_point_lvlh_km)
        v_rel_m_s = v_rel_chief_lvlh * 1e3 + v_point + correction_km_s * 1e3

        speed = np.linalg.norm(v_rel_m_s)

        F_total = np.zeros(3)

        # --- Drag ---
        if surf.use_drag and model.use_drag and speed > 1e-10:
            v_hat = v_rel_m_s / speed
            # Projected area: only when the panel normal points into the flow.
            cos_angle = np.dot(n_world, v_hat)
            if cos_angle > 0.0:
                rho = atm_density(r_point_eci)
                projected_area = surf.area * cos_angle
                F_drag = -0.5 * rho * surf.drag_coeff * projected_area * speed**2 * v_hat
                F_total += F_drag

        # --- SRP ---
        if surf.use_srp and model.use_srp:
            cos_sun = np.dot(n_world, sun_world)
            if cos_sun > 0.0:
                eclipse = eclipse_factor(r_point_eci, sun_hat_eci)
                if eclipse > 0.0:
                    projected_area = surf.area * cos_sun
                    F_srp = -eclipse * P_SUN * surf.srp_coeff * projected_area * sun_world
                    F_total += F_srp

        # Accumulate force and torque into wrench buffer
        data.wrench_buffer[bid, :3] += F_total
        tau = np.cross(r_cop_world, F_total)  # N·m
        data.wrench_buffer[bid, 3:] += tau
