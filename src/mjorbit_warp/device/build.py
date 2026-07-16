"""Constructors for the device-side dataclasses.

`make_device_core_model` uploads compile-time metadata (body inertia, surface
tables, actuator catalogs, central-body atmosphere parameters); this is called
once when an `MjoModel` is built. `make_device_core_data` allocates mutable
runtime state (chief orbit, frame caches, env caches, actuator state, multirate
scheduler buffers, wrench accumulator) — once per `MjoData`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from mjorbit.config import OrbitInit

from .io import _array_f64, _array_i32, _array_vec3d
from .types import DeviceCoreData, DeviceCoreModel


def make_device_core_model(model: Any) -> DeviceCoreModel:
    """Upload static orbit/coupling metadata for a compiled CPU model."""

    # Control moment gyros are not yet implemented on the device core: there is
    # no CMG loop in _assemble_wrenches and no device advance_cmgs, so a
    # CMG-equipped model would silently run with zero CMG gyroscopic/command
    # torque and a frozen gimbal angle, diverging from the CPU backend
    # (apply_cmg_wrenches / advance_cmgs in src/cpp/src/coupling_passive.cc).
    # Fail loudly rather than silently mis-simulating until the device kernels
    # land. The other actuators (reaction wheels, magnetorquers, thrusters) and
    # surfaces are fully supported below.
    cmgs = getattr(model, "cmgs", ())
    if len(cmgs) > 0:
        raise NotImplementedError(
            f"The MJWarp backend does not yet implement control moment gyros "
            f"(found {len(cmgs)} CMG(s) on the model). CMG gyroscopic and "
            f"command torques and gimbal integration are CPU-only for now; use "
            f"the mjorbit (CPU) backend for CMG-equipped models, or remove "
            f"the CMGs to run on the GPU."
        )

    empty_vec3 = np.zeros((0, 3), dtype=np.float64)
    empty_float = np.zeros(0, dtype=np.float64)
    empty_int = np.zeros(0, dtype=np.int32)

    surfaces = model.surfaces
    magnetic_bodies = model.magnetic_bodies
    reaction_wheels = model.reaction_wheels
    magnetorquers = model.magnetorquers
    thrusters = model.thrusters

    rw_speed_limit = np.array(
        [
            wheel.speed_limit
            if getattr(wheel, "has_speed_limit", wheel.speed_limit is not None)
            else 0.0
            for wheel in reaction_wheels
        ],
        dtype=np.float64,
    )
    rw_has_speed_limit = np.array(
        [
            getattr(wheel, "has_speed_limit", wheel.speed_limit is not None)
            for wheel in reaction_wheels
        ],
        dtype=np.int32,
    )
    rw_torque_limit = np.array(
        [
            wheel.torque_limit
            if getattr(wheel, "has_torque_limit", wheel.torque_limit is not None)
            else 0.0
            for wheel in reaction_wheels
        ],
        dtype=np.float64,
    )
    rw_has_torque_limit = np.array(
        [
            getattr(wheel, "has_torque_limit", wheel.torque_limit is not None)
            for wheel in reaction_wheels
        ],
        dtype=np.int32,
    )

    total_mass = float(np.sum(model.body_mass[1:]))
    central_body = model.central_body

    # Normalize the dipole axis here (the CPU dipole_field_eci normalizes it too)
    # so the device kernel can treat it as a unit moment direction.
    mag_axis = np.asarray(central_body.magnetic_axis, dtype=np.float64)
    mag_axis_norm = float(np.linalg.norm(mag_axis))
    if mag_axis_norm > 0.0:
        mag_axis = mag_axis / mag_axis_norm

    # Central-body spin (for dipole co-rotation): unit axis + rate magnitude.
    spin = np.asarray(central_body.omega, dtype=np.float64)
    omega_mag = float(np.linalg.norm(spin))
    spin_axis = spin / omega_mag if omega_mag > 0.0 else np.array([0.0, 0.0, 1.0])

    return DeviceCoreModel(
        body_mass=_array_f64(model.body_mass),
        body_ipos=_array_vec3d(model.body_ipos),
        body_inertia=_array_vec3d(model.body_inertia),
        surface_body_id=_array_i32([surface.body_id for surface in surfaces] or empty_int),
        surface_cop_body=_array_vec3d(
            [surface.center_of_pressure_body for surface in surfaces] or empty_vec3
        ),
        surface_normal_body=_array_vec3d(
            [surface.normal_body for surface in surfaces] or empty_vec3
        ),
        surface_area=_array_f64([surface.area for surface in surfaces] or empty_float),
        surface_drag_coeff=_array_f64([surface.drag_coeff for surface in surfaces] or empty_float),
        surface_srp_coeff=_array_f64([surface.srp_coeff for surface in surfaces] or empty_float),
        surface_use_drag=_array_i32([surface.use_drag for surface in surfaces] or empty_int),
        surface_use_srp=_array_i32([surface.use_srp for surface in surfaces] or empty_int),
        magnetic_body_id=_array_i32([mag.body_id for mag in magnetic_bodies] or empty_int),
        magnetic_dipole_body=_array_vec3d(
            [mag.dipole_body for mag in magnetic_bodies] or empty_vec3
        ),
        rw_body_id=_array_i32([wheel.body_id for wheel in reaction_wheels] or empty_int),
        rw_axis_body=_array_vec3d([wheel.axis_body for wheel in reaction_wheels] or empty_vec3),
        rw_inertia=_array_f64([wheel.inertia for wheel in reaction_wheels] or empty_float),
        rw_speed_limit=_array_f64(rw_speed_limit),
        rw_has_speed_limit=_array_i32(rw_has_speed_limit),
        rw_torque_limit=_array_f64(rw_torque_limit),
        rw_has_torque_limit=_array_i32(rw_has_torque_limit),
        mtq_body_id=_array_i32([mtq.body_id for mtq in magnetorquers] or empty_int),
        mtq_axis_body=_array_vec3d([mtq.axis_body for mtq in magnetorquers] or empty_vec3),
        mtq_dipole_limit=_array_f64([mtq.dipole_limit for mtq in magnetorquers] or empty_float),
        thr_body_id=_array_i32([thr.body_id for thr in thrusters] or empty_int),
        thr_position_body=_array_vec3d([thr.position_body for thr in thrusters] or empty_vec3),
        thr_direction_body=_array_vec3d([thr.direction_body for thr in thrusters] or empty_vec3),
        thr_force_limit=_array_f64([thr.force_limit for thr in thrusters] or empty_float),
        nbody=model.nbody,
        nsurface=len(surfaces),
        nmagnetic=len(magnetic_bodies),
        nrw=len(reaction_wheels),
        nmtq=len(magnetorquers),
        nthr=len(thrusters),
        total_mass=total_mass,
        radius_km=float(central_body.radius),
        magnetic_b0=float(central_body.magnetic_b0),
        magnetic_axis=wp.vec3(
            float(mag_axis[0]), float(mag_axis[1]), float(mag_axis[2])
        ),
        spin_axis=wp.vec3(
            float(spin_axis[0]), float(spin_axis[1]), float(spin_axis[2])
        ),
        omega_mag=omega_mag,
        atm_h0_km=float(central_body.atmosphere_h0),
        atm_rho0=float(central_body.atmosphere_rho0),
        atm_h_scale_km=float(central_body.atmosphere_scale_height),
    )


def make_device_core_data(
    orbit_inits: list[OrbitInit],
    *,
    nworld: int,
    model: Any,
) -> DeviceCoreData:
    """Allocate mutable device state for orbit/coupling kernels."""

    orbit_R = np.stack([init.R_eci for init in orbit_inits], axis=0)
    orbit_V = np.stack([init.V_eci for init in orbit_inits], axis=0)
    orbit_t = np.array([init.t for init in orbit_inits], dtype=np.float64)

    return DeviceCoreData(
        orbit_R_eci=wp.array(orbit_R, dtype=wp.vec3, shape=(nworld,)),
        orbit_V_eci=wp.array(orbit_V, dtype=wp.vec3, shape=(nworld,)),
        # Split clock: float64 absolute anchor + float32 device-relative time
        # (starts at 0), so epoch-anchored t (~1e9 s since J2000) keeps full
        # precision in the time-keyed environment models.
        orbit_t=wp.zeros((nworld,), dtype=wp.float32),
        orbit_t0=wp.array(orbit_t, dtype=wp.float64),
        # External (non-gravitational) net force on chief: sum of drag/SRP/thrust
        # forces on all bodies. Used to drive chief feedback acceleration AND
        # to apply origin-acceleration compensation -m·a_chief to each body.
        # Mirrors CPU OrbitInstance::feedback_force_world.
        feedback_force_world=wp.zeros((nworld,), dtype=wp.vec3),
        # Multirate orbit schedule: lazily initialized in the step kernel when
        # ``orbit_segment_duration[w] <= 0``.
        orbit_segment_start_R_eci=wp.zeros((nworld,), dtype=wp.vec3),
        orbit_segment_start_V_eci=wp.zeros((nworld,), dtype=wp.vec3),
        orbit_segment_start_t=wp.zeros((nworld,), dtype=wp.float32),
        orbit_segment_end_R_eci=wp.zeros((nworld,), dtype=wp.vec3),
        orbit_segment_end_V_eci=wp.zeros((nworld,), dtype=wp.vec3),
        orbit_segment_duration=wp.zeros((nworld,), dtype=wp.float32),
        orbit_segment_elapsed=wp.zeros((nworld,), dtype=wp.float32),
        orbit_feedback_int_eci=wp.zeros((nworld,), dtype=wp.vec3),
        orbit_feedback_int_dt=wp.zeros((nworld,), dtype=wp.float32),
        frame_C_LI=wp.zeros((nworld,), dtype=wp.mat33),
        frame_C_IL=wp.zeros((nworld,), dtype=wp.mat33),
        frame_omega_lvlh=wp.zeros((nworld,), dtype=wp.vec3),
        frame_omega_dot_lvlh=wp.zeros((nworld,), dtype=wp.vec3),
        env_sun_vector_eci=wp.zeros((nworld,), dtype=wp.vec3),
        env_eclipse=wp.zeros((nworld,), dtype=wp.float32),
        env_mag_field_eci=wp.zeros((nworld,), dtype=wp.vec3),
        env_atmosphere_omega_eci=wp.zeros((nworld,), dtype=wp.vec3),
        env_atm_density=wp.zeros((nworld,), dtype=wp.float32),
        rw_speed=wp.zeros((nworld, len(model.reaction_wheels)), dtype=wp.float32),
        rw_momentum=wp.zeros((nworld, len(model.reaction_wheels)), dtype=wp.float32),
        rw_torque_cmd=wp.zeros((nworld, len(model.reaction_wheels)), dtype=wp.float32),
        mtq_dipole_cmd=wp.zeros((nworld, len(model.magnetorquers)), dtype=wp.float32),
        thr_force_cmd=wp.zeros((nworld, len(model.thrusters)), dtype=wp.float32),
        wrench_buffer=wp.zeros((nworld, model.nbody), dtype=wp.spatial_vector),
    )


__all__ = ["make_device_core_data", "make_device_core_model"]
