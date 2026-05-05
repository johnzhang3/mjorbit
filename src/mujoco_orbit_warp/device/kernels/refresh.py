# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false, reportInvalidTypeForm=false
# pyright: reportOperatorIssue=false, reportReturnType=false

"""Per-step refresh of the chief-derived caches (frame, sun, eclipse, B-field, atmosphere)."""

from __future__ import annotations

import warp as wp

from mujoco_orbit.constants import OMEGA_EARTH

from .environment import (
    _atm_density,
    _dipole_field_eci,
    _eclipse_factor,
    _sun_vector_eci,
)
from .frame import _frame_from_orbit


@wp.func
def _refresh_core(
    world_id: int,
    use_j2: int,
    atm_h0_km: wp.float32,
    atm_rho0: wp.float32,
    atm_h_scale_km: wp.float32,
    orbit_R_eci: wp.array(dtype=wp.vec3),
    orbit_V_eci: wp.array(dtype=wp.vec3),
    orbit_t: wp.array(dtype=wp.float32),
    frame_C_LI: wp.array(dtype=wp.mat33),
    frame_C_IL: wp.array(dtype=wp.mat33),
    frame_omega_lvlh: wp.array(dtype=wp.vec3),
    frame_omega_dot_lvlh: wp.array(dtype=wp.vec3),
    env_sun_vector_eci: wp.array(dtype=wp.vec3),
    env_eclipse: wp.array(dtype=wp.float32),
    env_mag_field_eci: wp.array(dtype=wp.vec3),
    env_atmosphere_omega_eci: wp.array(dtype=wp.vec3),
    env_atm_density: wp.array(dtype=wp.float32),
):
    R = orbit_R_eci[world_id]
    V = orbit_V_eci[world_id]
    C_LI, C_IL, omega_lvlh, omega_dot_lvlh = _frame_from_orbit(R, V, use_j2)
    sun_hat = _sun_vector_eci(orbit_t[world_id])
    frame_C_LI[world_id] = C_LI
    frame_C_IL[world_id] = C_IL
    frame_omega_lvlh[world_id] = omega_lvlh
    frame_omega_dot_lvlh[world_id] = omega_dot_lvlh
    env_sun_vector_eci[world_id] = sun_hat
    env_eclipse[world_id] = _eclipse_factor(R, sun_hat)
    env_mag_field_eci[world_id] = _dipole_field_eci(R)
    env_atmosphere_omega_eci[world_id] = wp.vec3(
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(OMEGA_EARTH),
    )
    env_atm_density[world_id] = _atm_density(R, atm_h0_km, atm_rho0, atm_h_scale_km)
@wp.kernel
def _refresh_core_kernel(
    use_j2: int,
    atm_h0_km: wp.float32,
    atm_rho0: wp.float32,
    atm_h_scale_km: wp.float32,
    orbit_R_eci: wp.array(dtype=wp.vec3),
    orbit_V_eci: wp.array(dtype=wp.vec3),
    orbit_t: wp.array(dtype=wp.float32),
    frame_C_LI: wp.array(dtype=wp.mat33),
    frame_C_IL: wp.array(dtype=wp.mat33),
    frame_omega_lvlh: wp.array(dtype=wp.vec3),
    frame_omega_dot_lvlh: wp.array(dtype=wp.vec3),
    env_sun_vector_eci: wp.array(dtype=wp.vec3),
    env_eclipse: wp.array(dtype=wp.float32),
    env_mag_field_eci: wp.array(dtype=wp.vec3),
    env_atmosphere_omega_eci: wp.array(dtype=wp.vec3),
    env_atm_density: wp.array(dtype=wp.float32),
    rw_inertia: wp.array(dtype=wp.float32),
    rw_speed: wp.array2d(dtype=wp.float32),
    rw_momentum: wp.array2d(dtype=wp.float32),
    nrw: int,
):
    world_id = wp.tid()
    _refresh_core(
        world_id,
        use_j2,
        atm_h0_km,
        atm_rho0,
        atm_h_scale_km,
        orbit_R_eci,
        orbit_V_eci,
        orbit_t,
        frame_C_LI,
        frame_C_IL,
        frame_omega_lvlh,
        frame_omega_dot_lvlh,
        env_sun_vector_eci,
        env_eclipse,
        env_mag_field_eci,
        env_atmosphere_omega_eci,
        env_atm_density,
    )
    for i in range(nrw):
        rw_momentum[world_id, i] = rw_inertia[i] * rw_speed[world_id, i]


__all__ = [
    '_refresh_core',
    '_refresh_core_kernel',
]