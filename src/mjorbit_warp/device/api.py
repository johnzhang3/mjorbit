# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false

"""Python-side launchers (``wp.launch``) for the Warp kernels.

Each function takes the public ``MjoModel`` / ``MjoData`` wrappers and
extracts ``model.core_model`` / ``data.core_data`` / ``data.warp_data``
to feed the kernel parameter list.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from .kernels.assembly import (
    _assemble_forward_kernel,
    _assemble_step_kernel,
    _reset_orbit_schedule_kernel,
)
from .kernels.refresh import _refresh_core_kernel


def reset_orbit_schedule(data: Any) -> None:
    """Mark the multirate orbit schedule as needing re-initialization.

    Call after directly mutating ``data.orbit.R_eci``, ``orbit.V_eci``, or
    ``orbit.t`` so the next step rebuilds the segment endpoints from current
    state.
    """

    cd = data.core_data
    wp.launch(
        _reset_orbit_schedule_kernel,
        dim=(data.nworld,),
        inputs=[cd.orbit_segment_duration],
    )
def refresh_core(model: Any, data: Any) -> None:
    """Update device frame/environment caches from device orbit state."""

    cm = model.core_model
    cd = data.core_data
    wp.launch(
        _refresh_core_kernel,
        dim=(data.nworld,),
        inputs=[
            int(model.use_j2),
            cm.atm_h0_km,
            cm.atm_rho0,
            cm.atm_h_scale_km,
            cm.radius_km,
            cm.magnetic_b0,
            cm.magnetic_axis,
            cd.orbit_R_eci,
            cd.orbit_V_eci,
            cd.orbit_t,
            cd.frame_C_LI,
            cd.frame_C_IL,
            cd.frame_omega_lvlh,
            cd.frame_omega_dot_lvlh,
            cd.env_sun_vector_eci,
            cd.env_eclipse,
            cd.env_mag_field_eci,
            cd.env_atmosphere_omega_eci,
            cd.env_atm_density,
            cm.rw_inertia,
            cd.rw_speed,
            cd.rw_momentum,
            cm.nrw,
        ],
    )
def assemble_forward_wrenches(model: Any, data: Any) -> None:
    """Assemble device wrenches without integrating actuator/orbit state."""

    cm = model.core_model
    cd = data.core_data
    wd = data.warp_data
    wp.launch(
        _assemble_forward_kernel,
        dim=(data.nworld,),
        inputs=[
            cm.body_mass,
            cm.body_ipos,
            cm.body_inertia,
            cm.surface_body_id,
            cm.surface_cop_body,
            cm.surface_normal_body,
            cm.surface_area,
            cm.surface_drag_coeff,
            cm.surface_srp_coeff,
            cm.surface_use_drag,
            cm.surface_use_srp,
            cm.magnetic_body_id,
            cm.magnetic_dipole_body,
            cm.rw_body_id,
            cm.rw_axis_body,
            cm.rw_inertia,
            cm.rw_speed_limit,
            cm.rw_has_speed_limit,
            cm.rw_torque_limit,
            cm.rw_has_torque_limit,
            cm.mtq_body_id,
            cm.mtq_axis_body,
            cm.mtq_dipole_limit,
            cm.thr_body_id,
            cm.thr_position_body,
            cm.thr_direction_body,
            cm.thr_force_limit,
            cm.nbody,
            cm.nsurface,
            cm.nmagnetic,
            cm.nrw,
            cm.nmtq,
            cm.nthr,
            int(model.use_j2),
            int(model.use_drag),
            int(model.use_srp),
            int(model.use_magnetic),
            int(model.use_gravity_gradient),
            cm.atm_h0_km,
            cm.atm_rho0,
            cm.atm_h_scale_km,
            cm.radius_km,
            cd.orbit_R_eci,
            cd.orbit_V_eci,
            cd.frame_C_LI,
            cd.frame_C_IL,
            cd.frame_omega_lvlh,
            cd.frame_omega_dot_lvlh,
            cd.env_sun_vector_eci,
            cd.env_eclipse,
            cd.env_mag_field_eci,
            cd.env_atmosphere_omega_eci,
            cd.env_atm_density,
            cd.feedback_force_world,
            cd.rw_speed,
            cd.rw_momentum,
            cd.rw_torque_cmd,
            cd.mtq_dipole_cmd,
            cd.thr_force_cmd,
            cd.wrench_buffer,
            wd.xipos,
            wd.xmat,
            wd.ximat,
            wd.cvel,
            wd.xfrc_applied,
        ],
    )
def assemble_step_and_propagate(model: Any, data: Any, *, mj_dt: float, orbit_dt: float) -> None:
    """Assemble device wrenches, integrate RW commands, and propagate orbit state."""

    cm = model.core_model
    cd = data.core_data
    wd = data.warp_data
    wp.launch(
        _assemble_step_kernel,
        dim=(data.nworld,),
        inputs=[
            cm.body_mass,
            cm.body_ipos,
            cm.body_inertia,
            cm.surface_body_id,
            cm.surface_cop_body,
            cm.surface_normal_body,
            cm.surface_area,
            cm.surface_drag_coeff,
            cm.surface_srp_coeff,
            cm.surface_use_drag,
            cm.surface_use_srp,
            cm.magnetic_body_id,
            cm.magnetic_dipole_body,
            cm.rw_body_id,
            cm.rw_axis_body,
            cm.rw_inertia,
            cm.rw_speed_limit,
            cm.rw_has_speed_limit,
            cm.rw_torque_limit,
            cm.rw_has_torque_limit,
            cm.mtq_body_id,
            cm.mtq_axis_body,
            cm.mtq_dipole_limit,
            cm.thr_body_id,
            cm.thr_position_body,
            cm.thr_direction_body,
            cm.thr_force_limit,
            cm.nbody,
            cm.nsurface,
            cm.nmagnetic,
            cm.nrw,
            cm.nmtq,
            cm.nthr,
            int(model.use_j2),
            int(model.use_drag),
            int(model.use_srp),
            int(model.use_magnetic),
            int(model.use_gravity_gradient),
            cm.atm_h0_km,
            cm.atm_rho0,
            cm.atm_h_scale_km,
            cm.radius_km,
            cm.magnetic_b0,
            cm.magnetic_axis,
            cm.total_mass,
            float(mj_dt),
            float(orbit_dt),
            cd.orbit_R_eci,
            cd.orbit_V_eci,
            cd.orbit_t,
            cd.orbit_segment_start_R_eci,
            cd.orbit_segment_start_V_eci,
            cd.orbit_segment_start_t,
            cd.orbit_segment_end_R_eci,
            cd.orbit_segment_end_V_eci,
            cd.orbit_segment_duration,
            cd.orbit_segment_elapsed,
            cd.orbit_feedback_int_eci,
            cd.orbit_feedback_int_dt,
            cd.frame_C_LI,
            cd.frame_C_IL,
            cd.frame_omega_lvlh,
            cd.frame_omega_dot_lvlh,
            cd.env_sun_vector_eci,
            cd.env_eclipse,
            cd.env_mag_field_eci,
            cd.env_atmosphere_omega_eci,
            cd.env_atm_density,
            cd.feedback_force_world,
            cd.rw_speed,
            cd.rw_momentum,
            cd.rw_torque_cmd,
            cd.mtq_dipole_cmd,
            cd.thr_force_cmd,
            cd.wrench_buffer,
            wd.xipos,
            wd.xmat,
            wd.ximat,
            wd.cvel,
            wd.xfrc_applied,
        ],
    )


__all__ = [
    'assemble_forward_wrenches',
    'assemble_step_and_propagate',
    'refresh_core',
    'reset_orbit_schedule',
]