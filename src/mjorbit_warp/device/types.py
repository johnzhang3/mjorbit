"""Dataclasses describing what lives on the Warp device.

`DeviceCoreModel` is static metadata uploaded once at compile time;
`DeviceCoreData` is mutable per-step state. Both are simple containers — see
`device.build` for how they are populated and `device.kernels.*` for how their
fields flow through compiled Warp kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DeviceCoreModel:
    """Static orbit/coupling metadata stored on the Warp device."""

    body_mass: Any
    body_ipos: Any
    body_inertia: Any

    surface_body_id: Any
    surface_cop_body: Any
    surface_normal_body: Any
    surface_area: Any
    surface_drag_coeff: Any
    surface_srp_coeff: Any
    surface_use_drag: Any
    surface_use_srp: Any

    magnetic_body_id: Any
    magnetic_dipole_body: Any

    rw_body_id: Any
    rw_axis_body: Any
    rw_inertia: Any
    rw_speed_limit: Any
    rw_has_speed_limit: Any
    rw_torque_limit: Any
    rw_has_torque_limit: Any

    mtq_body_id: Any
    mtq_axis_body: Any
    mtq_dipole_limit: Any

    thr_body_id: Any
    thr_position_body: Any
    thr_direction_body: Any
    thr_force_limit: Any

    nbody: int
    nsurface: int
    nmagnetic: int
    nrw: int
    nmtq: int
    nthr: int
    total_mass: float
    radius_km: float
    magnetic_b0: float
    magnetic_axis: Any  # normalized dipole-axis direction, ECEF components (wp.vec3)
    spin_axis: Any  # normalized central-body spin axis (wp.vec3)
    omega_mag: float  # |omega| of the central body, rad/s
    atm_h0_km: float
    atm_rho0: float
    atm_h_scale_km: float


@dataclass
class DeviceCoreData:
    """Mutable orbit/coupling state stored on the Warp device."""

    orbit_R_eci: Any
    orbit_V_eci: Any
    # Device clock split: orbit_t is float32 time since the float64 per-world
    # anchor orbit_t0 (s since J2000). Absolute time = orbit_t0 + orbit_t; the
    # relative clock keeps float32 rounding small even for epoch-anchored runs.
    orbit_t: Any
    orbit_t0: Any
    feedback_force_world: Any  # accumulated non-gravitational force on chief, in N
    orbit_segment_start_R_eci: Any
    orbit_segment_start_V_eci: Any
    orbit_segment_start_t: Any
    orbit_segment_end_R_eci: Any
    orbit_segment_end_V_eci: Any
    orbit_segment_duration: Any
    orbit_segment_elapsed: Any
    orbit_feedback_int_eci: Any
    orbit_feedback_int_dt: Any
    frame_C_LI: Any
    frame_C_IL: Any
    frame_omega_lvlh: Any
    frame_omega_dot_lvlh: Any
    env_sun_vector_eci: Any
    env_eclipse: Any
    env_mag_field_eci: Any
    env_atmosphere_omega_eci: Any
    env_atm_density: Any
    rw_speed: Any
    rw_momentum: Any
    rw_torque_cmd: Any
    mtq_dipole_cmd: Any
    thr_force_cmd: Any
    wrench_buffer: Any


__all__ = ["DeviceCoreData", "DeviceCoreModel"]
