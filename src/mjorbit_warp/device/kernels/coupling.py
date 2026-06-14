# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false, reportInvalidTypeForm=false
# pyright: reportOperatorIssue=false, reportReturnType=false

"""Body-coupling kernel.

Includes differential gravity, gravity-gradient torque, surface drag/SRP,
magnetic dipole + magnetorquer, RW gyroscopic + command torque, and
thrusters.
"""

from __future__ import annotations

import warp as wp

from mjorbit.constants import P_SUN

from ..shared import (
    _mat33d_from_mat33,
    _spatial_add,
    _spatial_ang,
    _spatial_lin,
    _vec3d_from_vec3,
)
from .environment import _atm_density, _eclipse_factor
from .gravity import _gravity_gradient_torque, _relative_accel


@wp.func
def _assemble_wrenches(
    world_id: int,
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
    radius_km: wp.float32,
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
    rw_torque_limit: wp.array(dtype=wp.float32),
    rw_has_torque_limit: wp.array(dtype=int),
    rw_speed_limit: wp.array(dtype=wp.float32),
    rw_has_speed_limit: wp.array(dtype=int),
    mtq_dipole_cmd: wp.array2d(dtype=wp.float32),
    thr_force_cmd: wp.array2d(dtype=wp.float32),
    wrench_buffer: wp.array2d(dtype=wp.spatial_vector),
    xipos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    ximat: wp.array2d(dtype=wp.mat33),
    cvel: wp.array2d(dtype=wp.spatial_vector),
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector),
):
    zero = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    zero64 = wp.float32(0.0)
    m_to_km = wp.float32(1.0e-3)
    km_to_m = wp.float32(1.0e3)
    for body_id in range(nbody):
        wrench_buffer[world_id, body_id] = zero
        xfrc_applied[world_id, body_id] = zero

    R_ref = orbit_R_eci[world_id]
    V_ref = orbit_V_eci[world_id]
    # frame_C_LI / frame_C_IL are no longer read here: with MJ-world ≡ ECI
    # orientation, body coupling needs no LVLH↔ECI rotation. The arrays remain
    # part of the public host shadow (data.frame) for analysis tooling.

    # Reset the per-world non-gravitational force accumulator. Surface drag/SRP
    # and thruster loops below add into it; differential gravity (Encke) and
    # internal torques (RW gyro/cmd, MTQ, GG) do not contribute. Mirrors CPU
    # apply_passive_wrenches's zeroing of inst->feedback_force_world at the top
    # of each pass.
    feedback_force_world[world_id] = wp.vec3(zero64, zero64, zero64)

    # MJ-world frame is the chief-centered local inertial frame with axes
    # parallel to ECI (per CLAUDE.md / project convention). xipos is therefore
    # the chief-relative offset already in ECI orientation; no LVLH rotation
    # is needed and no rotating-frame pseudo-forces apply. Mirrors CPU's
    # apply_inertial_wrenches (src/cpp/src/coupling_passive.cc:140-167).
    for body_id in range(1, nbody):
        mass = body_mass[body_id]
        if mass <= zero64:
            continue

        rho_km = _vec3d_from_vec3(xipos[world_id, body_id]) * m_to_km
        dg = _relative_accel(rho_km, R_ref, use_j2)
        force = dg * (mass * km_to_m)

        tau_gg = wp.vec3(zero64, zero64, zero64)
        if use_gravity_gradient != 0:
            r_body_eci = R_ref + rho_km
            r_mag_km = wp.length(r_body_eci)
            r_hat_world = r_body_eci / r_mag_km
            ximat_world = _mat33d_from_mat33(ximat[world_id, body_id])
            tau_gg = _gravity_gradient_torque(
                r_hat_world, r_mag_km, ximat_world, body_inertia[body_id]
            )

        wrench_buffer[world_id, body_id] = _spatial_add(
            wrench_buffer[world_id, body_id],
            force,
            tau_gg,
        )

    # Surface drag + SRP. Mirrors CPU apply_surface_wrenches at
    # src/cpp/src/coupling_passive.cc:246-340. All math is in MJ-world (≡ ECI
    # orientation): v_point in MJ-world, r_point in absolute ECI = R_chief +
    # body offset (no LVLH rotation), v_rel against the rotating atmosphere.
    omega_earth_eci = env_atmosphere_omega_eci[world_id]
    sun_eci = env_sun_vector_eci[world_id]

    for surface_id in range(nsurface):
        bid = surface_body_id[surface_id]
        R_body = _mat33d_from_mat33(xmat[world_id, bid])
        r_cop_world = R_body @ surface_cop_body[surface_id]
        n_world = R_body @ surface_normal_body[surface_id]
        vel = cvel[world_id, bid]
        v_com_world = _spatial_lin(vel)
        omega_body_world = _spatial_ang(vel)
        v_point_world = v_com_world + wp.cross(omega_body_world, r_cop_world)

        r_point_eci_km = R_ref + (_vec3d_from_vec3(xipos[world_id, bid]) + r_cop_world) * m_to_km
        v_point_eci_m_s = V_ref * km_to_m + v_point_world
        v_atm_eci_m_s = wp.cross(omega_earth_eci, r_point_eci_km) * km_to_m
        v_rel_m_s = v_point_eci_m_s - v_atm_eci_m_s
        speed = wp.length(v_rel_m_s)
        force = wp.vec3(zero64, zero64, zero64)

        if surface_use_drag[surface_id] != 0 and use_drag != 0 and speed > wp.float32(1.0e-10):
            v_hat = v_rel_m_s / speed
            cos_angle = wp.dot(n_world, v_hat)
            if cos_angle > zero64:
                rho_local = _atm_density(
                    r_point_eci_km, atm_h0_km, atm_rho0, atm_h_scale_km, radius_km
                )
                projected_area = surface_area[surface_id] * cos_angle
                drag_scale = -wp.float32(0.5) * rho_local
                drag_scale = drag_scale * surface_drag_coeff[surface_id]
                drag_scale = drag_scale * projected_area * speed * speed
                force = force + v_hat * drag_scale

        if surface_use_srp[surface_id] != 0 and use_srp != 0:
            cos_sun = wp.dot(n_world, sun_eci)
            if cos_sun > zero64:
                eclipse_local = _eclipse_factor(r_point_eci_km, sun_eci, radius_km)
                if eclipse_local > zero64:
                    projected_area = surface_area[surface_id] * cos_sun
                    srp_scale = -eclipse_local * wp.float32(P_SUN)
                    srp_scale = srp_scale * surface_srp_coeff[surface_id]
                    srp_scale = srp_scale * projected_area
                    force = force + sun_eci * srp_scale

        torque = wp.cross(r_cop_world, force)
        wrench_buffer[world_id, bid] = _spatial_add(wrench_buffer[world_id, bid], force, torque)
        feedback_force_world[world_id] = feedback_force_world[world_id] + force

    # Magnetic field is given in ECI. With MJ-world ≡ ECI orientation, no
    # rotation is needed to bring it into the simulation frame.
    if use_magnetic != 0:
        B_world = env_mag_field_eci[world_id]

        for magnetic_id in range(nmagnetic):
            bid = magnetic_body_id[magnetic_id]
            R_body = _mat33d_from_mat33(xmat[world_id, bid])
            B_body = wp.transpose(R_body) @ B_world
            tau_body = wp.cross(magnetic_dipole_body[magnetic_id], B_body)
            tau_world = R_body @ tau_body
            wrench_buffer[world_id, bid] = _spatial_add(
                wrench_buffer[world_id, bid],
                wp.vec3(zero64, zero64, zero64),
                tau_world,
            )

        for mtq_id in range(nmtq):
            bid = mtq_body_id[mtq_id]
            m_cmd = wp.clamp(
                mtq_dipole_cmd[world_id, mtq_id],
                -mtq_dipole_limit[mtq_id],
                mtq_dipole_limit[mtq_id],
            )
            R_body = _mat33d_from_mat33(xmat[world_id, bid])
            B_body = wp.transpose(R_body) @ B_world
            tau_body = wp.cross(mtq_axis_body[mtq_id] * m_cmd, B_body)
            tau_world = R_body @ tau_body
            wrench_buffer[world_id, bid] = _spatial_add(
                wrench_buffer[world_id, bid],
                wp.vec3(zero64, zero64, zero64),
                tau_world,
            )

    # Reaction wheels: apply both gyroscopic τ = -ω_body × h_body and command
    # τ_cmd = -I·α, mirroring CPU apply_reaction_wheel_wrenches at
    # src/cpp/src/coupling_passive.cc:363. Speed advance lives in
    # _command_rw_torques (only called from the step kernel).
    for rw_id in range(nrw):
        bid = rw_body_id[rw_id]
        inertia_rw = rw_inertia[rw_id]
        speed_rw = rw_speed[world_id, rw_id]
        rw_momentum[world_id, rw_id] = inertia_rw * speed_rw

        R_body = _mat33d_from_mat33(xmat[world_id, bid])
        w_body = wp.transpose(R_body) @ _spatial_ang(cvel[world_id, bid])
        h_body = rw_axis_body[rw_id] * (inertia_rw * speed_rw)
        gyro_tau_body = -wp.cross(w_body, h_body)

        cmd_tau_body = wp.vec3(zero64, zero64, zero64)
        if inertia_rw > zero64:
            tau_cmd = rw_torque_cmd[world_id, rw_id]
            if rw_has_torque_limit[rw_id] != 0:
                lim = rw_torque_limit[rw_id]
                tau_cmd = wp.clamp(tau_cmd, -lim, lim)
            alpha = tau_cmd / inertia_rw
            if rw_has_speed_limit[rw_id] != 0:
                slim = rw_speed_limit[rw_id]
                if speed_rw >= slim and alpha > zero64:
                    alpha = zero64
                elif speed_rw <= -slim and alpha < zero64:
                    alpha = zero64
            cmd_tau_body = rw_axis_body[rw_id] * (-inertia_rw * alpha)

        tau_world = R_body @ (gyro_tau_body + cmd_tau_body)
        wrench_buffer[world_id, bid] = _spatial_add(
            wrench_buffer[world_id, bid],
            wp.vec3(zero64, zero64, zero64),
            tau_world,
        )

    for thr_id in range(nthr):
        bid = thr_body_id[thr_id]
        f_cmd = wp.clamp(thr_force_cmd[world_id, thr_id], zero64, thr_force_limit[thr_id])
        R_body = _mat33d_from_mat33(xmat[world_id, bid])
        force_body = thr_direction_body[thr_id] * f_cmd
        force_world = R_body @ force_body
        r_com_body = thr_position_body[thr_id] - body_ipos[bid]
        tau_body = wp.cross(r_com_body, force_body)
        tau_world = R_body @ tau_body
        wrench_buffer[world_id, bid] = _spatial_add(
            wrench_buffer[world_id, bid],
            force_world,
            tau_world,
        )
        feedback_force_world[world_id] = feedback_force_world[world_id] + force_world

    for body_id in range(nbody):
        xfrc_applied[world_id, body_id] = wrench_buffer[world_id, body_id]


__all__ = [
    '_assemble_wrenches',
]