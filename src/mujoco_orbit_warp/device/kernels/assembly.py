# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false, reportInvalidTypeForm=false
# pyright: reportOperatorIssue=false, reportReturnType=false

"""@wp.kernel entry points: forward-only wrench assembly, full step, and orbit-schedule reset.

These are the kernels Python actually launches via ``wp.launch`` in
``device.api``. They wrap the body-coupling and actuator @wp.func helpers
together with the multirate orbit propagator.
"""

from __future__ import annotations

import warp as wp

from ..shared import _apply_origin_compensation
from .actuators import _command_rw_torques
from .coupling import _assemble_wrenches
from .propagator import _propagate_rk4
from .refresh import _refresh_core


@wp.kernel
def _assemble_forward_kernel(
    # model
    body_mass: wp.array(dtype=wp.float32),
    body_ipos: wp.array(dtype=wp.vec3),
    body_inertia: wp.array(dtype=wp.vec3),
    surface_body_id: wp.array(dtype=int),
    surface_cop_body: wp.array(dtype=wp.vec3),
    surface_normal_body: wp.array(dtype=wp.vec3),
    surface_area: wp.array(dtype=wp.float32),
    surface_drag_coeff: wp.array(dtype=wp.float32),
    surface_srp_coeff: wp.array(dtype=wp.float32),
    surface_use_drag: wp.array(dtype=int),
    surface_use_srp: wp.array(dtype=int),
    magnetic_body_id: wp.array(dtype=int),
    magnetic_dipole_body: wp.array(dtype=wp.vec3),
    rw_body_id: wp.array(dtype=int),
    rw_axis_body: wp.array(dtype=wp.vec3),
    rw_inertia: wp.array(dtype=wp.float32),
    rw_speed_limit: wp.array(dtype=wp.float32),
    rw_has_speed_limit: wp.array(dtype=int),
    rw_torque_limit: wp.array(dtype=wp.float32),
    rw_has_torque_limit: wp.array(dtype=int),
    mtq_body_id: wp.array(dtype=int),
    mtq_axis_body: wp.array(dtype=wp.vec3),
    mtq_dipole_limit: wp.array(dtype=wp.float32),
    thr_body_id: wp.array(dtype=int),
    thr_position_body: wp.array(dtype=wp.vec3),
    thr_direction_body: wp.array(dtype=wp.vec3),
    thr_force_limit: wp.array(dtype=wp.float32),
    nbody: int,
    nsurface: int,
    nmagnetic: int,
    nrw: int,
    nmtq: int,
    nthr: int,
    use_j2: int,
    use_drag: int,
    use_srp: int,
    use_magnetic: int,
    use_gravity_gradient: int,
    atm_h0_km: wp.float32,
    atm_rho0: wp.float32,
    atm_h_scale_km: wp.float32,
    # core data
    orbit_R_eci: wp.array(dtype=wp.vec3),
    orbit_V_eci: wp.array(dtype=wp.vec3),
    frame_C_LI: wp.array(dtype=wp.mat33),
    frame_C_IL: wp.array(dtype=wp.mat33),
    frame_omega_lvlh: wp.array(dtype=wp.vec3),
    frame_omega_dot_lvlh: wp.array(dtype=wp.vec3),
    env_sun_vector_eci: wp.array(dtype=wp.vec3),
    env_eclipse: wp.array(dtype=wp.float32),
    env_mag_field_eci: wp.array(dtype=wp.vec3),
    env_atmosphere_omega_eci: wp.array(dtype=wp.vec3),
    env_atm_density: wp.array(dtype=wp.float32),
    feedback_force_world: wp.array(dtype=wp.vec3),
    rw_speed: wp.array2d(dtype=wp.float32),
    rw_momentum: wp.array2d(dtype=wp.float32),
    rw_torque_cmd: wp.array2d(dtype=wp.float32),
    mtq_dipole_cmd: wp.array2d(dtype=wp.float32),
    thr_force_cmd: wp.array2d(dtype=wp.float32),
    wrench_buffer: wp.array2d(dtype=wp.spatial_vector),
    # MJWarp data
    xipos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    ximat: wp.array2d(dtype=wp.mat33),
    cvel: wp.array2d(dtype=wp.spatial_vector),
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector),
):
    world_id = wp.tid()
    _assemble_wrenches(
        world_id,
        body_mass,
        body_ipos,
        body_inertia,
        surface_body_id,
        surface_cop_body,
        surface_normal_body,
        surface_area,
        surface_drag_coeff,
        surface_srp_coeff,
        surface_use_drag,
        surface_use_srp,
        magnetic_body_id,
        magnetic_dipole_body,
        rw_body_id,
        rw_axis_body,
        rw_inertia,
        mtq_body_id,
        mtq_axis_body,
        mtq_dipole_limit,
        thr_body_id,
        thr_position_body,
        thr_direction_body,
        thr_force_limit,
        nbody,
        nsurface,
        nmagnetic,
        nrw,
        nmtq,
        nthr,
        use_j2,
        use_drag,
        use_srp,
        use_magnetic,
        use_gravity_gradient,
        atm_h0_km,
        atm_rho0,
        atm_h_scale_km,
        orbit_R_eci,
        orbit_V_eci,
        frame_C_LI,
        frame_C_IL,
        frame_omega_lvlh,
        frame_omega_dot_lvlh,
        env_sun_vector_eci,
        env_eclipse,
        env_mag_field_eci,
        env_atmosphere_omega_eci,
        env_atm_density,
        feedback_force_world,
        rw_speed,
        rw_momentum,
        rw_torque_cmd,
        rw_torque_limit,
        rw_has_torque_limit,
        rw_speed_limit,
        rw_has_speed_limit,
        mtq_dipole_cmd,
        thr_force_cmd,
        wrench_buffer,
        xipos,
        xmat,
        ximat,
        cvel,
        xfrc_applied,
    )

    # Origin-acceleration compensation so that mjo_forward leaves xfrc_applied
    # in the same state as a CPU forward pass (otherwise downstream qacc and
    # wrench_buffer parity tests see uncompensated thrust/drag forces).
    total_mass = wp.float32(0.0)
    for bid in range(nbody):
        total_mass = total_mass + body_mass[bid]
    if total_mass > wp.float32(0.0):
        a_chief_m_s2 = feedback_force_world[world_id] / total_mass
        _apply_origin_compensation(
            world_id, nbody, body_mass, a_chief_m_s2, wrench_buffer, xfrc_applied
        )
@wp.kernel
def _assemble_step_kernel(
    # model
    body_mass: wp.array(dtype=wp.float32),
    body_ipos: wp.array(dtype=wp.vec3),
    body_inertia: wp.array(dtype=wp.vec3),
    surface_body_id: wp.array(dtype=int),
    surface_cop_body: wp.array(dtype=wp.vec3),
    surface_normal_body: wp.array(dtype=wp.vec3),
    surface_area: wp.array(dtype=wp.float32),
    surface_drag_coeff: wp.array(dtype=wp.float32),
    surface_srp_coeff: wp.array(dtype=wp.float32),
    surface_use_drag: wp.array(dtype=int),
    surface_use_srp: wp.array(dtype=int),
    magnetic_body_id: wp.array(dtype=int),
    magnetic_dipole_body: wp.array(dtype=wp.vec3),
    rw_body_id: wp.array(dtype=int),
    rw_axis_body: wp.array(dtype=wp.vec3),
    rw_inertia: wp.array(dtype=wp.float32),
    rw_speed_limit: wp.array(dtype=wp.float32),
    rw_has_speed_limit: wp.array(dtype=int),
    rw_torque_limit: wp.array(dtype=wp.float32),
    rw_has_torque_limit: wp.array(dtype=int),
    mtq_body_id: wp.array(dtype=int),
    mtq_axis_body: wp.array(dtype=wp.vec3),
    mtq_dipole_limit: wp.array(dtype=wp.float32),
    thr_body_id: wp.array(dtype=int),
    thr_position_body: wp.array(dtype=wp.vec3),
    thr_direction_body: wp.array(dtype=wp.vec3),
    thr_force_limit: wp.array(dtype=wp.float32),
    nbody: int,
    nsurface: int,
    nmagnetic: int,
    nrw: int,
    nmtq: int,
    nthr: int,
    use_j2: int,
    use_drag: int,
    use_srp: int,
    use_magnetic: int,
    use_gravity_gradient: int,
    atm_h0_km: wp.float32,
    atm_rho0: wp.float32,
    atm_h_scale_km: wp.float32,
    total_mass: wp.float32,
    mj_dt: wp.float32,
    orbit_dt: wp.float32,
    # core data
    orbit_R_eci: wp.array(dtype=wp.vec3),
    orbit_V_eci: wp.array(dtype=wp.vec3),
    orbit_t: wp.array(dtype=wp.float32),
    orbit_segment_start_R_eci: wp.array(dtype=wp.vec3),
    orbit_segment_start_V_eci: wp.array(dtype=wp.vec3),
    orbit_segment_start_t: wp.array(dtype=wp.float32),
    orbit_segment_end_R_eci: wp.array(dtype=wp.vec3),
    orbit_segment_end_V_eci: wp.array(dtype=wp.vec3),
    orbit_segment_duration: wp.array(dtype=wp.float32),
    orbit_segment_elapsed: wp.array(dtype=wp.float32),
    orbit_feedback_int_eci: wp.array(dtype=wp.vec3),
    orbit_feedback_int_dt: wp.array(dtype=wp.float32),
    frame_C_LI: wp.array(dtype=wp.mat33),
    frame_C_IL: wp.array(dtype=wp.mat33),
    frame_omega_lvlh: wp.array(dtype=wp.vec3),
    frame_omega_dot_lvlh: wp.array(dtype=wp.vec3),
    env_sun_vector_eci: wp.array(dtype=wp.vec3),
    env_eclipse: wp.array(dtype=wp.float32),
    env_mag_field_eci: wp.array(dtype=wp.vec3),
    env_atmosphere_omega_eci: wp.array(dtype=wp.vec3),
    env_atm_density: wp.array(dtype=wp.float32),
    feedback_force_world: wp.array(dtype=wp.vec3),
    rw_speed: wp.array2d(dtype=wp.float32),
    rw_momentum: wp.array2d(dtype=wp.float32),
    rw_torque_cmd: wp.array2d(dtype=wp.float32),
    mtq_dipole_cmd: wp.array2d(dtype=wp.float32),
    thr_force_cmd: wp.array2d(dtype=wp.float32),
    wrench_buffer: wp.array2d(dtype=wp.spatial_vector),
    # MJWarp data
    xipos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    ximat: wp.array2d(dtype=wp.mat33),
    cvel: wp.array2d(dtype=wp.spatial_vector),
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector),
):
    world_id = wp.tid()
    _assemble_wrenches(
        world_id,
        body_mass,
        body_ipos,
        body_inertia,
        surface_body_id,
        surface_cop_body,
        surface_normal_body,
        surface_area,
        surface_drag_coeff,
        surface_srp_coeff,
        surface_use_drag,
        surface_use_srp,
        magnetic_body_id,
        magnetic_dipole_body,
        rw_body_id,
        rw_axis_body,
        rw_inertia,
        mtq_body_id,
        mtq_axis_body,
        mtq_dipole_limit,
        thr_body_id,
        thr_position_body,
        thr_direction_body,
        thr_force_limit,
        nbody,
        nsurface,
        nmagnetic,
        nrw,
        nmtq,
        nthr,
        use_j2,
        use_drag,
        use_srp,
        use_magnetic,
        use_gravity_gradient,
        atm_h0_km,
        atm_rho0,
        atm_h_scale_km,
        orbit_R_eci,
        orbit_V_eci,
        frame_C_LI,
        frame_C_IL,
        frame_omega_lvlh,
        frame_omega_dot_lvlh,
        env_sun_vector_eci,
        env_eclipse,
        env_mag_field_eci,
        env_atmosphere_omega_eci,
        env_atm_density,
        feedback_force_world,
        rw_speed,
        rw_momentum,
        rw_torque_cmd,
        rw_torque_limit,
        rw_has_torque_limit,
        rw_speed_limit,
        rw_has_speed_limit,
        mtq_dipole_cmd,
        thr_force_cmd,
        wrench_buffer,
        xipos,
        xmat,
        ximat,
        cvel,
        xfrc_applied,
    )

    _command_rw_torques(
        world_id,
        rw_body_id,
        rw_axis_body,
        rw_inertia,
        rw_speed_limit,
        rw_has_speed_limit,
        rw_torque_limit,
        rw_has_torque_limit,
        nrw,
        mj_dt,
        rw_speed,
        rw_momentum,
        rw_torque_cmd,
        wrench_buffer,
        xmat,
        xfrc_applied,
    )

    a_feedback = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    if total_mass > wp.float32(0.0):
        # External (drag + SRP + thruster) force on chief, accumulated by
        # _assemble_wrenches above. Mirrors CPU compute_feedback_accel.
        a_chief_m_s2 = feedback_force_world[world_id] / total_mass
        a_feedback = a_chief_m_s2 * wp.float32(1.0e-3)
        _apply_origin_compensation(
            world_id, nbody, body_mass, a_chief_m_s2, wrench_buffer, xfrc_applied
        )

    # Multirate orbit advance, mirroring src/cpp/src/orbit_schedule.cc.
    # When orbit_dt <= mj_dt: RK4 substeps over mj_dt with current feedback.
    # When orbit_dt > mj_dt:  walk mj_dt across the open segment, accumulating
    # a time-averaged feedback acceleration; commit each segment with one RK4
    # call from segment_start using that average, and linearly interpolate
    # R/V at intermediate mj_dt ticks.
    zero64 = wp.float32(0.0)
    one64 = wp.float32(1.0)
    eps_dt = wp.float32(1.0e-12)
    R_curr = orbit_R_eci[world_id]
    V_curr = orbit_V_eci[world_id]
    t_curr = orbit_t[world_id]

    # Lazy init: triggered on first step or after a state edit that zeros
    # orbit_segment_duration (see init_orbit_schedule).
    if orbit_segment_duration[world_id] <= eps_dt:
        orbit_segment_start_R_eci[world_id] = R_curr
        orbit_segment_start_V_eci[world_id] = V_curr
        orbit_segment_start_t[world_id] = t_curr
        orbit_segment_duration[world_id] = orbit_dt
        orbit_segment_elapsed[world_id] = zero64
        orbit_feedback_int_eci[world_id] = wp.vec3(zero64, zero64, zero64)
        orbit_feedback_int_dt[world_id] = zero64
        R_e0, V_e0 = _propagate_rk4(
            R_curr, V_curr, orbit_dt, use_j2,
            wp.vec3(zero64, zero64, zero64),
        )
        orbit_segment_end_R_eci[world_id] = R_e0
        orbit_segment_end_V_eci[world_id] = V_e0

    if orbit_dt <= mj_dt + eps_dt:
        # Substep regime — chief integrates faster than MuJoCo, so do
        # ceil(mj_dt / orbit_dt) RK4 substeps with the current feedback.
        R = R_curr
        V = V_curr
        t = t_curr
        remaining = mj_dt
        while remaining > eps_dt:
            sub_dt = wp.min(remaining, orbit_dt)
            R, V = _propagate_rk4(R, V, sub_dt, use_j2, a_feedback)
            t = t + sub_dt
            remaining = remaining - sub_dt
        orbit_R_eci[world_id] = R
        orbit_V_eci[world_id] = V
        orbit_t[world_id] = t
        orbit_segment_start_R_eci[world_id] = R
        orbit_segment_start_V_eci[world_id] = V
        orbit_segment_start_t[world_id] = t
        orbit_segment_duration[world_id] = orbit_dt
        orbit_segment_elapsed[world_id] = zero64
        orbit_feedback_int_eci[world_id] = wp.vec3(zero64, zero64, zero64)
        orbit_feedback_int_dt[world_id] = zero64
        R_e1, V_e1 = _propagate_rk4(R, V, orbit_dt, use_j2, a_feedback)
        orbit_segment_end_R_eci[world_id] = R_e1
        orbit_segment_end_V_eci[world_id] = V_e1
    else:
        # Multirate regime — orbit_dt > mj_dt.
        R_start = orbit_segment_start_R_eci[world_id]
        V_start = orbit_segment_start_V_eci[world_id]
        t_start = orbit_segment_start_t[world_id]
        R_end = orbit_segment_end_R_eci[world_id]
        V_end = orbit_segment_end_V_eci[world_id]
        duration = orbit_segment_duration[world_id]
        elapsed = orbit_segment_elapsed[world_id]
        feedback_int = orbit_feedback_int_eci[world_id]
        feedback_int_dt_local = orbit_feedback_int_dt[world_id]

        R_out = R_curr
        V_out = V_curr
        t_out = t_curr
        remaining = mj_dt
        while remaining > eps_dt:
            seg_remaining = duration - elapsed
            dt_chunk = wp.min(remaining, seg_remaining)
            feedback_int = feedback_int + a_feedback * dt_chunk
            feedback_int_dt_local = feedback_int_dt_local + dt_chunk
            elapsed = elapsed + dt_chunk
            remaining = remaining - dt_chunk

            if elapsed + eps_dt >= duration:
                avg = feedback_int / feedback_int_dt_local
                R_committed, V_committed = _propagate_rk4(
                    R_start, V_start, duration, use_j2, avg
                )
                R_out = R_committed
                V_out = V_committed
                t_out = t_start + duration
                R_start = R_out
                V_start = V_out
                t_start = t_out
                duration = orbit_dt
                elapsed = zero64
                feedback_int = wp.vec3(zero64, zero64, zero64)
                feedback_int_dt_local = zero64
                R_pred, V_pred = _propagate_rk4(
                    R_start, V_start, duration, use_j2, avg
                )
                R_end = R_pred
                V_end = V_pred
            else:
                alpha = elapsed / duration
                R_out = R_start * (one64 - alpha) + R_end * alpha
                V_out = V_start * (one64 - alpha) + V_end * alpha
                t_out = t_start + elapsed

        orbit_R_eci[world_id] = R_out
        orbit_V_eci[world_id] = V_out
        orbit_t[world_id] = t_out
        orbit_segment_start_R_eci[world_id] = R_start
        orbit_segment_start_V_eci[world_id] = V_start
        orbit_segment_start_t[world_id] = t_start
        orbit_segment_end_R_eci[world_id] = R_end
        orbit_segment_end_V_eci[world_id] = V_end
        orbit_segment_duration[world_id] = duration
        orbit_segment_elapsed[world_id] = elapsed
        orbit_feedback_int_eci[world_id] = feedback_int
        orbit_feedback_int_dt[world_id] = feedback_int_dt_local

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
@wp.kernel
def _reset_orbit_schedule_kernel(
    orbit_segment_duration: wp.array(dtype=wp.float32),
):
    world_id = wp.tid()
    orbit_segment_duration[world_id] = wp.float32(0.0)


__all__ = [
    '_assemble_forward_kernel',
    '_assemble_step_kernel',
    '_reset_orbit_schedule_kernel',
]