# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false, reportInvalidTypeForm=false
# pyright: reportOperatorIssue=false, reportReturnType=false

"""Warp kernels for the orbit/coupling part of the MJWarp backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from mujoco_orbit.constants import (
    B0_EARTH,
    GM_EARTH,
    J2_EARTH,
    OMEGA_EARTH,
    P_SUN,
    R_EARTH,
)
from mujoco_orbit.core.config import OrbitInit

_DEG_TO_RAD = np.pi / 180.0
_ATM_H0_KM = 400.0
_ATM_RHO0 = 2.62e-13
_ATM_H_SCALE = 58.2


@dataclass
class DeviceCoreModel:
    """Static orbit/coupling metadata stored on the Warp device."""

    body_mass: Any
    body_ipos: Any

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


@dataclass
class DeviceCoreData:
    """Mutable orbit/coupling state stored on the Warp device."""

    orbit_R_eci: Any
    orbit_V_eci: Any
    orbit_t: Any
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


def _array_f64(values: np.ndarray | list[float]) -> Any:
    return wp.array(np.asarray(values, dtype=np.float64), dtype=wp.float64)


def _array_i32(values: np.ndarray | list[int]) -> Any:
    return wp.array(np.asarray(values, dtype=np.int32), dtype=int)


def _array_vec3d(values: np.ndarray | list[list[float]]) -> Any:
    array = np.asarray(values, dtype=np.float64).reshape((-1, 3))
    return wp.array(array, dtype=wp.vec3d, shape=(array.shape[0],))


def _world_scalar(values: np.ndarray | float, nworld: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if nworld == 1:
        return array.reshape(1)
    return array.reshape(nworld)


def _world_vec3(values: np.ndarray, nworld: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if nworld == 1:
        return array.reshape((1, 3))
    return array.reshape((nworld, 3))


def _world_array(values: np.ndarray, nworld: int, width: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if nworld == 1:
        return array.reshape((1, width))
    return array.reshape((nworld, width))


def _copy_vec3d(dest: Any, values: np.ndarray, nworld: int) -> None:
    src_values = _world_vec3(values, nworld)
    wp.copy(dest, wp.array(src_values, dtype=wp.vec3d, shape=(nworld,)))


def _copy_f64_1d(dest: Any, values: np.ndarray | float, nworld: int) -> None:
    wp.copy(dest, wp.array(_world_scalar(values, nworld), dtype=wp.float64))


def _copy_f64_2d(dest: Any, values: np.ndarray, nworld: int, width: int) -> None:
    if width == 0:
        return
    wp.copy(dest, wp.array(_world_array(values, nworld, width), dtype=wp.float64))


def make_device_core_model(model: Any) -> DeviceCoreModel:
    """Upload static orbit/coupling metadata for a compiled CPU model."""

    empty_vec3 = np.zeros((0, 3), dtype=np.float64)
    empty_float = np.zeros(0, dtype=np.float64)
    empty_int = np.zeros(0, dtype=np.int32)

    surfaces = model.surfaces
    magnetic_bodies = model.magnetic_bodies
    reaction_wheels = model.reaction_wheels
    magnetorquers = model.magnetorquers
    thrusters = model.thrusters

    rw_speed_limit = np.array(
        [0.0 if wheel.speed_limit is None else wheel.speed_limit for wheel in reaction_wheels],
        dtype=np.float64,
    )
    rw_has_speed_limit = np.array(
        [wheel.speed_limit is not None for wheel in reaction_wheels],
        dtype=np.int32,
    )
    rw_torque_limit = np.array(
        [0.0 if wheel.torque_limit is None else wheel.torque_limit for wheel in reaction_wheels],
        dtype=np.float64,
    )
    rw_has_torque_limit = np.array(
        [wheel.torque_limit is not None for wheel in reaction_wheels],
        dtype=np.int32,
    )

    total_mass = float(np.sum(model.body_mass[1:]))

    return DeviceCoreModel(
        body_mass=_array_f64(model.body_mass),
        body_ipos=_array_vec3d(model.body_ipos),
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
        orbit_R_eci=wp.array(orbit_R, dtype=wp.vec3d, shape=(nworld,)),
        orbit_V_eci=wp.array(orbit_V, dtype=wp.vec3d, shape=(nworld,)),
        orbit_t=wp.array(orbit_t, dtype=wp.float64),
        frame_C_LI=wp.zeros((nworld,), dtype=wp.mat33d),
        frame_C_IL=wp.zeros((nworld,), dtype=wp.mat33d),
        frame_omega_lvlh=wp.zeros((nworld,), dtype=wp.vec3d),
        frame_omega_dot_lvlh=wp.zeros((nworld,), dtype=wp.vec3d),
        env_sun_vector_eci=wp.zeros((nworld,), dtype=wp.vec3d),
        env_eclipse=wp.zeros((nworld,), dtype=wp.float64),
        env_mag_field_eci=wp.zeros((nworld,), dtype=wp.vec3d),
        env_atmosphere_omega_eci=wp.zeros((nworld,), dtype=wp.vec3d),
        env_atm_density=wp.zeros((nworld,), dtype=wp.float64),
        rw_speed=wp.zeros((nworld, len(model.reaction_wheels)), dtype=wp.float64),
        rw_momentum=wp.zeros((nworld, len(model.reaction_wheels)), dtype=wp.float64),
        rw_torque_cmd=wp.zeros((nworld, len(model.reaction_wheels)), dtype=wp.float64),
        mtq_dipole_cmd=wp.zeros((nworld, len(model.magnetorquers)), dtype=wp.float64),
        thr_force_cmd=wp.zeros((nworld, len(model.thrusters)), dtype=wp.float64),
        wrench_buffer=wp.zeros((nworld, model.nbody), dtype=wp.spatial_vector),
    )


def _sync_field_enabled(fields: frozenset[str] | None, *names: str) -> bool:
    return fields is None or "core" in fields or "actuators" in fields or any(
        name in fields for name in names
    )


def sync_core_device_from_public(data: Any, fields: frozenset[str] | None = None) -> None:
    """Upload user-mutated public orbit/actuator state to device core buffers."""

    core = data.core_data
    if fields is None or "orbit" in fields or "core" in fields:
        _copy_vec3d(core.orbit_R_eci, data.orbit.R_eci, data.nworld)
        _copy_vec3d(core.orbit_V_eci, data.orbit.V_eci, data.nworld)
        _copy_f64_1d(core.orbit_t, data.orbit.t, data.nworld)

    if _sync_field_enabled(fields, "rw_speed"):
        _copy_f64_2d(
            core.rw_speed,
            data.actuators.rw_speed,
            data.nworld,
            len(data.model.reaction_wheels),
        )
    if _sync_field_enabled(fields, "rw_torque_cmd"):
        _copy_f64_2d(
            core.rw_torque_cmd,
            data.actuators.rw_torque_cmd,
            data.nworld,
            len(data.model.reaction_wheels),
        )
    if _sync_field_enabled(fields, "mtq_dipole_cmd"):
        _copy_f64_2d(
            core.mtq_dipole_cmd,
            data.actuators.mtq_dipole_cmd,
            data.nworld,
            len(data.model.magnetorquers),
        )
    if _sync_field_enabled(fields, "thr_force_cmd"):
        _copy_f64_2d(
            core.thr_force_cmd,
            data.actuators.thr_force_cmd,
            data.nworld,
            len(data.model.thrusters),
        )


def _pull_field_enabled(fields: frozenset[str] | None, *names: str) -> bool:
    return fields is None or "core" in fields or any(name in fields for name in names)


def pull_core_device_to_public(data: Any, fields: frozenset[str] | None = None) -> None:
    """Download device orbit/coupling state into public NumPy buffers."""

    core = data.core_data

    if _pull_field_enabled(fields, "orbit"):
        orbit_R = core.orbit_R_eci.numpy()
        orbit_V = core.orbit_V_eci.numpy()
        orbit_t = core.orbit_t.numpy()
        if data.nworld == 1:
            np.copyto(data.orbit.R_eci, orbit_R[0])
            np.copyto(data.orbit.V_eci, orbit_V[0])
            data.orbit.t = float(orbit_t[0])
        else:
            np.copyto(data.orbit.R_eci, orbit_R)
            np.copyto(data.orbit.V_eci, orbit_V)
            np.copyto(data.orbit.t, orbit_t)

    if _pull_field_enabled(fields, "frame"):
        frame_C_LI = core.frame_C_LI.numpy()
        frame_C_IL = core.frame_C_IL.numpy()
        frame_omega = core.frame_omega_lvlh.numpy()
        frame_omega_dot = core.frame_omega_dot_lvlh.numpy()
        if data.nworld == 1:
            np.copyto(data.frame.C_LI, frame_C_LI[0])
            np.copyto(data.frame.C_IL, frame_C_IL[0])
            np.copyto(data.frame.omega_lvlh, frame_omega[0])
            np.copyto(data.frame.omega_dot_lvlh, frame_omega_dot[0])
        else:
            np.copyto(data.frame.C_LI, frame_C_LI)
            np.copyto(data.frame.C_IL, frame_C_IL)
            np.copyto(data.frame.omega_lvlh, frame_omega)
            np.copyto(data.frame.omega_dot_lvlh, frame_omega_dot)

    if _pull_field_enabled(fields, "env"):
        env_sun = core.env_sun_vector_eci.numpy()
        env_eclipse = core.env_eclipse.numpy()
        env_mag = core.env_mag_field_eci.numpy()
        env_atm_omega = core.env_atmosphere_omega_eci.numpy()
        env_density = core.env_atm_density.numpy()
        if data.nworld == 1:
            np.copyto(data.env.sun_vector_eci, env_sun[0])
            data.env.eclipse = float(env_eclipse[0])
            np.copyto(data.env.mag_field_eci, env_mag[0])
            np.copyto(data.env.atmosphere_omega_eci, env_atm_omega[0])
            data.env.atm_density = float(env_density[0])
        else:
            np.copyto(data.env.sun_vector_eci, env_sun)
            np.copyto(data.env.eclipse, env_eclipse)
            np.copyto(data.env.mag_field_eci, env_mag)
            np.copyto(data.env.atmosphere_omega_eci, env_atm_omega)
            np.copyto(data.env.atm_density, env_density)

    if _pull_field_enabled(fields, "wrench_buffer"):
        wrench = core.wrench_buffer.numpy()
        if data.nworld == 1:
            np.copyto(data.wrench_buffer, wrench[0])
        else:
            np.copyto(data.wrench_buffer, wrench)

    if len(data.model.reaction_wheels) and _pull_field_enabled(fields, "actuators"):
        rw_speed = core.rw_speed.numpy()
        rw_momentum = core.rw_momentum.numpy()
        if data.nworld == 1:
            np.copyto(data.actuators.rw_speed, rw_speed[0])
            np.copyto(data.actuators.rw_momentum, rw_momentum[0])
        else:
            np.copyto(data.actuators.rw_speed, rw_speed)
            np.copyto(data.actuators.rw_momentum, rw_momentum)


@wp.func
def _vec3d_from_vec3(value: wp.vec3) -> wp.vec3d:
    return wp.vec3d(wp.float64(value[0]), wp.float64(value[1]), wp.float64(value[2]))


@wp.func
def _spatial_ang(value: wp.spatial_vector) -> wp.vec3d:
    return wp.vec3d(wp.float64(value[0]), wp.float64(value[1]), wp.float64(value[2]))


@wp.func
def _spatial_lin(value: wp.spatial_vector) -> wp.vec3d:
    return wp.vec3d(wp.float64(value[3]), wp.float64(value[4]), wp.float64(value[5]))


@wp.func
def _mat33d_from_mat33(value: wp.mat33) -> wp.mat33d:
    return wp.mat33d(
        wp.float64(value[0, 0]),
        wp.float64(value[0, 1]),
        wp.float64(value[0, 2]),
        wp.float64(value[1, 0]),
        wp.float64(value[1, 1]),
        wp.float64(value[1, 2]),
        wp.float64(value[2, 0]),
        wp.float64(value[2, 1]),
        wp.float64(value[2, 2]),
    )


@wp.func
def _spatial_from_parts(force: wp.vec3d, torque: wp.vec3d) -> wp.spatial_vector:
    return wp.spatial_vector(
        wp.float32(force[0]),
        wp.float32(force[1]),
        wp.float32(force[2]),
        wp.float32(torque[0]),
        wp.float32(torque[1]),
        wp.float32(torque[2]),
    )


@wp.func
def _spatial_add(value: wp.spatial_vector, force: wp.vec3d, torque: wp.vec3d) -> wp.spatial_vector:
    return wp.spatial_vector(
        value[0] + wp.float32(force[0]),
        value[1] + wp.float32(force[1]),
        value[2] + wp.float32(force[2]),
        value[3] + wp.float32(torque[0]),
        value[4] + wp.float32(torque[1]),
        value[5] + wp.float32(torque[2]),
    )


@wp.func
def _total_accel(R: wp.vec3d, use_j2: int) -> wp.vec3d:
    one = wp.float64(1.0)
    gm = wp.float64(GM_EARTH)
    earth_radius = wp.float64(R_EARTH)
    j2 = wp.float64(J2_EARTH)
    r = wp.length(R)
    inv_r3 = one / (r * r * r)
    accel = R * (-gm * inv_r3)

    if use_j2 != 0:
        x = R[0]
        y = R[1]
        z = R[2]
        factor = wp.float64(1.5) * j2 * gm * earth_radius * earth_radius
        factor = factor / (r * r * r * r * r)
        z_r2 = (z / r) * (z / r)
        accel = accel + wp.vec3d(
            factor * x * (wp.float64(5.0) * z_r2 - one),
            factor * y * (wp.float64(5.0) * z_r2 - one),
            factor * z * (wp.float64(5.0) * z_r2 - wp.float64(3.0)),
        )

    return accel


@wp.func
def _frame_from_orbit(
    R: wp.vec3d,
    V: wp.vec3d,
    use_j2: int,
) -> tuple[wp.mat33d, wp.mat33d, wp.vec3d, wp.vec3d]:
    r = wp.length(R)
    x_hat = R / r
    h = wp.cross(R, V)
    z_hat = h / wp.length(h)
    y_hat = wp.cross(z_hat, x_hat)

    C_LI = wp.mat33d(
        x_hat[0],
        x_hat[1],
        x_hat[2],
        y_hat[0],
        y_hat[1],
        y_hat[2],
        z_hat[0],
        z_hat[1],
        z_hat[2],
    )
    C_IL = wp.transpose(C_LI)

    omega_eci = h / (r * r)
    omega_lvlh = C_LI @ omega_eci

    a_eci = _total_accel(R, use_j2)
    dh_dt = wp.cross(R, a_eci)
    dr_dt = wp.dot(R, V) / r
    domega_dt_eci = dh_dt / (r * r) - h * (wp.float64(2.0) * dr_dt / (r * r * r))
    omega_dot_lvlh = C_LI @ domega_dt_eci
    return C_LI, C_IL, omega_lvlh, omega_dot_lvlh


@wp.func
def _sun_vector_eci(t: wp.float64) -> wp.vec3d:
    deg_to_rad = wp.float64(_DEG_TO_RAD)
    T_jc = t / (wp.float64(36525.0) * wp.float64(86400.0))
    lambda_sun = (wp.float64(280.460) + wp.float64(36000.771) * T_jc) * deg_to_rad
    M_sun = (wp.float64(357.528) + wp.float64(35999.050) * T_jc) * deg_to_rad
    lambda_ecl = lambda_sun + wp.float64(1.915) * deg_to_rad * wp.sin(M_sun)
    lambda_ecl = lambda_ecl + wp.float64(0.020) * deg_to_rad * wp.sin(wp.float64(2.0) * M_sun)
    eps = (wp.float64(23.439) - wp.float64(0.013) * T_jc) * deg_to_rad
    return wp.vec3d(
        wp.cos(lambda_ecl),
        wp.sin(lambda_ecl) * wp.cos(eps),
        wp.sin(lambda_ecl) * wp.sin(eps),
    )


@wp.func
def _eclipse_factor(R: wp.vec3d, sun_hat: wp.vec3d) -> wp.float64:
    proj = -wp.dot(R, sun_hat)
    if proj < wp.float64(0.0):
        return wp.float64(1.0)

    d_perp = wp.length(R - sun_hat * wp.dot(R, sun_hat))
    if d_perp < wp.float64(R_EARTH):
        return wp.float64(0.0)
    return wp.float64(1.0)


@wp.func
def _dipole_field_eci(R: wp.vec3d) -> wp.vec3d:
    r = wp.length(R)
    r_hat = R / r
    m_hat = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(-1.0))
    radius_ratio = wp.float64(R_EARTH) / r
    factor = wp.float64(B0_EARTH) * radius_ratio * radius_ratio * radius_ratio
    return (r_hat * (wp.float64(3.0) * wp.dot(m_hat, r_hat)) - m_hat) * factor


@wp.func
def _atm_density(R: wp.vec3d) -> wp.float64:
    alt_km = wp.length(R) - wp.float64(R_EARTH)
    rho = wp.float64(_ATM_RHO0) * wp.exp(
        -(alt_km - wp.float64(_ATM_H0_KM)) / wp.float64(_ATM_H_SCALE)
    )
    return wp.max(rho, wp.float64(0.0))


@wp.func
def _refresh_core(
    world_id: int,
    use_j2: int,
    orbit_R_eci: wp.array(dtype=wp.vec3d),
    orbit_V_eci: wp.array(dtype=wp.vec3d),
    orbit_t: wp.array(dtype=wp.float64),
    frame_C_LI: wp.array(dtype=wp.mat33d),
    frame_C_IL: wp.array(dtype=wp.mat33d),
    frame_omega_lvlh: wp.array(dtype=wp.vec3d),
    frame_omega_dot_lvlh: wp.array(dtype=wp.vec3d),
    env_sun_vector_eci: wp.array(dtype=wp.vec3d),
    env_eclipse: wp.array(dtype=wp.float64),
    env_mag_field_eci: wp.array(dtype=wp.vec3d),
    env_atmosphere_omega_eci: wp.array(dtype=wp.vec3d),
    env_atm_density: wp.array(dtype=wp.float64),
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
    env_atmosphere_omega_eci[world_id] = wp.vec3d(
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(OMEGA_EARTH),
    )
    env_atm_density[world_id] = _atm_density(R)


@wp.kernel
def _refresh_core_kernel(
    use_j2: int,
    orbit_R_eci: wp.array(dtype=wp.vec3d),
    orbit_V_eci: wp.array(dtype=wp.vec3d),
    orbit_t: wp.array(dtype=wp.float64),
    frame_C_LI: wp.array(dtype=wp.mat33d),
    frame_C_IL: wp.array(dtype=wp.mat33d),
    frame_omega_lvlh: wp.array(dtype=wp.vec3d),
    frame_omega_dot_lvlh: wp.array(dtype=wp.vec3d),
    env_sun_vector_eci: wp.array(dtype=wp.vec3d),
    env_eclipse: wp.array(dtype=wp.float64),
    env_mag_field_eci: wp.array(dtype=wp.vec3d),
    env_atmosphere_omega_eci: wp.array(dtype=wp.vec3d),
    env_atm_density: wp.array(dtype=wp.float64),
    rw_inertia: wp.array(dtype=wp.float64),
    rw_speed: wp.array2d(dtype=wp.float64),
    rw_momentum: wp.array2d(dtype=wp.float64),
    nrw: int,
):
    world_id = wp.tid()
    _refresh_core(
        world_id,
        use_j2,
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


@wp.kernel
def _assemble_forward_kernel(
    # model
    body_mass: wp.array(dtype=wp.float64),
    body_ipos: wp.array(dtype=wp.vec3d),
    surface_body_id: wp.array(dtype=int),
    surface_cop_body: wp.array(dtype=wp.vec3d),
    surface_normal_body: wp.array(dtype=wp.vec3d),
    surface_area: wp.array(dtype=wp.float64),
    surface_drag_coeff: wp.array(dtype=wp.float64),
    surface_srp_coeff: wp.array(dtype=wp.float64),
    surface_use_drag: wp.array(dtype=int),
    surface_use_srp: wp.array(dtype=int),
    magnetic_body_id: wp.array(dtype=int),
    magnetic_dipole_body: wp.array(dtype=wp.vec3d),
    rw_body_id: wp.array(dtype=int),
    rw_axis_body: wp.array(dtype=wp.vec3d),
    rw_inertia: wp.array(dtype=wp.float64),
    mtq_body_id: wp.array(dtype=int),
    mtq_axis_body: wp.array(dtype=wp.vec3d),
    mtq_dipole_limit: wp.array(dtype=wp.float64),
    thr_body_id: wp.array(dtype=int),
    thr_position_body: wp.array(dtype=wp.vec3d),
    thr_direction_body: wp.array(dtype=wp.vec3d),
    thr_force_limit: wp.array(dtype=wp.float64),
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
    # core data
    orbit_R_eci: wp.array(dtype=wp.vec3d),
    orbit_V_eci: wp.array(dtype=wp.vec3d),
    frame_C_LI: wp.array(dtype=wp.mat33d),
    frame_C_IL: wp.array(dtype=wp.mat33d),
    frame_omega_lvlh: wp.array(dtype=wp.vec3d),
    frame_omega_dot_lvlh: wp.array(dtype=wp.vec3d),
    env_sun_vector_eci: wp.array(dtype=wp.vec3d),
    env_eclipse: wp.array(dtype=wp.float64),
    env_mag_field_eci: wp.array(dtype=wp.vec3d),
    env_atm_density: wp.array(dtype=wp.float64),
    rw_speed: wp.array2d(dtype=wp.float64),
    rw_momentum: wp.array2d(dtype=wp.float64),
    mtq_dipole_cmd: wp.array2d(dtype=wp.float64),
    thr_force_cmd: wp.array2d(dtype=wp.float64),
    wrench_buffer: wp.array2d(dtype=wp.spatial_vector),
    # MJWarp data
    xipos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    cvel: wp.array2d(dtype=wp.spatial_vector),
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector),
):
    world_id = wp.tid()
    _assemble_wrenches(
        world_id,
        body_mass,
        body_ipos,
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
        orbit_R_eci,
        orbit_V_eci,
        frame_C_LI,
        frame_C_IL,
        frame_omega_lvlh,
        frame_omega_dot_lvlh,
        env_sun_vector_eci,
        env_eclipse,
        env_mag_field_eci,
        env_atm_density,
        rw_speed,
        rw_momentum,
        mtq_dipole_cmd,
        thr_force_cmd,
        wrench_buffer,
        xipos,
        xmat,
        cvel,
        xfrc_applied,
    )


@wp.kernel
def _assemble_step_kernel(
    # model
    body_mass: wp.array(dtype=wp.float64),
    body_ipos: wp.array(dtype=wp.vec3d),
    surface_body_id: wp.array(dtype=int),
    surface_cop_body: wp.array(dtype=wp.vec3d),
    surface_normal_body: wp.array(dtype=wp.vec3d),
    surface_area: wp.array(dtype=wp.float64),
    surface_drag_coeff: wp.array(dtype=wp.float64),
    surface_srp_coeff: wp.array(dtype=wp.float64),
    surface_use_drag: wp.array(dtype=int),
    surface_use_srp: wp.array(dtype=int),
    magnetic_body_id: wp.array(dtype=int),
    magnetic_dipole_body: wp.array(dtype=wp.vec3d),
    rw_body_id: wp.array(dtype=int),
    rw_axis_body: wp.array(dtype=wp.vec3d),
    rw_inertia: wp.array(dtype=wp.float64),
    rw_speed_limit: wp.array(dtype=wp.float64),
    rw_has_speed_limit: wp.array(dtype=int),
    rw_torque_limit: wp.array(dtype=wp.float64),
    rw_has_torque_limit: wp.array(dtype=int),
    mtq_body_id: wp.array(dtype=int),
    mtq_axis_body: wp.array(dtype=wp.vec3d),
    mtq_dipole_limit: wp.array(dtype=wp.float64),
    thr_body_id: wp.array(dtype=int),
    thr_position_body: wp.array(dtype=wp.vec3d),
    thr_direction_body: wp.array(dtype=wp.vec3d),
    thr_force_limit: wp.array(dtype=wp.float64),
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
    total_mass: wp.float64,
    mj_dt: wp.float64,
    orbit_dt: wp.float64,
    # core data
    orbit_R_eci: wp.array(dtype=wp.vec3d),
    orbit_V_eci: wp.array(dtype=wp.vec3d),
    orbit_t: wp.array(dtype=wp.float64),
    frame_C_LI: wp.array(dtype=wp.mat33d),
    frame_C_IL: wp.array(dtype=wp.mat33d),
    frame_omega_lvlh: wp.array(dtype=wp.vec3d),
    frame_omega_dot_lvlh: wp.array(dtype=wp.vec3d),
    env_sun_vector_eci: wp.array(dtype=wp.vec3d),
    env_eclipse: wp.array(dtype=wp.float64),
    env_mag_field_eci: wp.array(dtype=wp.vec3d),
    env_atmosphere_omega_eci: wp.array(dtype=wp.vec3d),
    env_atm_density: wp.array(dtype=wp.float64),
    rw_speed: wp.array2d(dtype=wp.float64),
    rw_momentum: wp.array2d(dtype=wp.float64),
    rw_torque_cmd: wp.array2d(dtype=wp.float64),
    mtq_dipole_cmd: wp.array2d(dtype=wp.float64),
    thr_force_cmd: wp.array2d(dtype=wp.float64),
    wrench_buffer: wp.array2d(dtype=wp.spatial_vector),
    # MJWarp data
    xipos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    cvel: wp.array2d(dtype=wp.spatial_vector),
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector),
):
    world_id = wp.tid()
    _assemble_wrenches(
        world_id,
        body_mass,
        body_ipos,
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
        orbit_R_eci,
        orbit_V_eci,
        frame_C_LI,
        frame_C_IL,
        frame_omega_lvlh,
        frame_omega_dot_lvlh,
        env_sun_vector_eci,
        env_eclipse,
        env_mag_field_eci,
        env_atm_density,
        rw_speed,
        rw_momentum,
        mtq_dipole_cmd,
        thr_force_cmd,
        wrench_buffer,
        xipos,
        xmat,
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

    net_force = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    for body_id in range(nbody):
        wrench = wrench_buffer[world_id, body_id]
        net_force = net_force + wp.vec3d(
            wp.float64(wrench[0]),
            wp.float64(wrench[1]),
            wp.float64(wrench[2]),
        )

    a_feedback = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    if total_mass > wp.float64(0.0):
        a_world_km = (net_force / total_mass) * wp.float64(1.0e-3)
        a_feedback = frame_C_IL[world_id] @ a_world_km

    R_next, V_next = _propagate_rk4(
        orbit_R_eci[world_id],
        orbit_V_eci[world_id],
        orbit_dt,
        use_j2,
        a_feedback,
    )
    orbit_R_eci[world_id] = R_next
    orbit_V_eci[world_id] = V_next
    orbit_t[world_id] = orbit_t[world_id] + orbit_dt

    _refresh_core(
        world_id,
        use_j2,
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


@wp.func
def _assemble_wrenches(
    world_id: int,
    body_mass: wp.array(dtype=wp.float64),
    body_ipos: wp.array(dtype=wp.vec3d),
    surface_body_id: wp.array(dtype=int),
    surface_cop_body: wp.array(dtype=wp.vec3d),
    surface_normal_body: wp.array(dtype=wp.vec3d),
    surface_area: wp.array(dtype=wp.float64),
    surface_drag_coeff: wp.array(dtype=wp.float64),
    surface_srp_coeff: wp.array(dtype=wp.float64),
    surface_use_drag: wp.array(dtype=int),
    surface_use_srp: wp.array(dtype=int),
    magnetic_body_id: wp.array(dtype=int),
    magnetic_dipole_body: wp.array(dtype=wp.vec3d),
    rw_body_id: wp.array(dtype=int),
    rw_axis_body: wp.array(dtype=wp.vec3d),
    rw_inertia: wp.array(dtype=wp.float64),
    mtq_body_id: wp.array(dtype=int),
    mtq_axis_body: wp.array(dtype=wp.vec3d),
    mtq_dipole_limit: wp.array(dtype=wp.float64),
    thr_body_id: wp.array(dtype=int),
    thr_position_body: wp.array(dtype=wp.vec3d),
    thr_direction_body: wp.array(dtype=wp.vec3d),
    thr_force_limit: wp.array(dtype=wp.float64),
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
    orbit_R_eci: wp.array(dtype=wp.vec3d),
    orbit_V_eci: wp.array(dtype=wp.vec3d),
    frame_C_LI: wp.array(dtype=wp.mat33d),
    frame_C_IL: wp.array(dtype=wp.mat33d),
    frame_omega_lvlh: wp.array(dtype=wp.vec3d),
    frame_omega_dot_lvlh: wp.array(dtype=wp.vec3d),
    env_sun_vector_eci: wp.array(dtype=wp.vec3d),
    env_eclipse: wp.array(dtype=wp.float64),
    env_mag_field_eci: wp.array(dtype=wp.vec3d),
    env_atm_density: wp.array(dtype=wp.float64),
    rw_speed: wp.array2d(dtype=wp.float64),
    rw_momentum: wp.array2d(dtype=wp.float64),
    mtq_dipole_cmd: wp.array2d(dtype=wp.float64),
    thr_force_cmd: wp.array2d(dtype=wp.float64),
    wrench_buffer: wp.array2d(dtype=wp.spatial_vector),
    xipos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    cvel: wp.array2d(dtype=wp.spatial_vector),
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector),
):
    zero = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    zero64 = wp.float64(0.0)
    m_to_km = wp.float64(1.0e-3)
    km_to_m = wp.float64(1.0e3)
    for body_id in range(nbody):
        wrench_buffer[world_id, body_id] = zero
        xfrc_applied[world_id, body_id] = zero

    R_ref = orbit_R_eci[world_id]
    V_ref = orbit_V_eci[world_id]
    C_LI = frame_C_LI[world_id]
    C_IL = frame_C_IL[world_id]
    omega = frame_omega_lvlh[world_id]
    omega_dot = frame_omega_dot_lvlh[world_id]
    g_ref = _total_accel(R_ref, use_j2)

    for body_id in range(1, nbody):
        mass = body_mass[body_id]
        if mass <= zero64:
            continue

        r_lvlh_km = _vec3d_from_vec3(xipos[world_id, body_id]) * m_to_km
        r_body_eci = R_ref + C_IL @ r_lvlh_km
        g_body = _total_accel(r_body_eci, use_j2)
        dg = C_LI @ (g_body - g_ref)
        v_lvlh_km_s = _spatial_lin(cvel[world_id, body_id]) * m_to_km
        a_total = dg
        a_total = a_total - wp.cross(omega, v_lvlh_km_s) * wp.float64(2.0)
        a_total = a_total - wp.cross(omega_dot, r_lvlh_km)
        a_total = a_total - wp.cross(omega, wp.cross(omega, r_lvlh_km))
        force = a_total * (mass * km_to_m)
        wrench_buffer[world_id, body_id] = _spatial_add(
            wrench_buffer[world_id, body_id],
            force,
            wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)),
        )

    omega_earth = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(OMEGA_EARTH))
    v_rel_chief_lvlh = C_LI @ (V_ref - wp.cross(omega_earth, R_ref))
    sun_world = C_LI @ env_sun_vector_eci[world_id]
    omega_earth_lvlh = C_LI @ omega_earth

    sun_hat_eci = env_sun_vector_eci[world_id]
    for surface_id in range(nsurface):
        bid = surface_body_id[surface_id]
        R_body = _mat33d_from_mat33(xmat[world_id, bid])
        r_cop_world = R_body @ surface_cop_body[surface_id]
        n_world = R_body @ surface_normal_body[surface_id]
        vel = cvel[world_id, bid]
        v_com = _spatial_lin(vel)
        omega_body = _spatial_ang(vel)
        v_point = v_com + wp.cross(omega_body, r_cop_world)
        r_point_lvlh_km = (_vec3d_from_vec3(xipos[world_id, bid]) + r_cop_world) * m_to_km
        r_point_eci = R_ref + C_IL @ r_point_lvlh_km
        correction_km_s = wp.cross(omega - omega_earth_lvlh, r_point_lvlh_km)
        v_rel_m_s = v_rel_chief_lvlh * km_to_m + v_point + correction_km_s * km_to_m
        speed = wp.length(v_rel_m_s)
        force = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))

        if surface_use_drag[surface_id] != 0 and use_drag != 0 and speed > wp.float64(1.0e-10):
            v_hat = v_rel_m_s / speed
            cos_angle = wp.dot(n_world, v_hat)
            if cos_angle > zero64:
                rho_local = _atm_density(r_point_eci)
                projected_area = surface_area[surface_id] * cos_angle
                drag_scale = -wp.float64(0.5) * rho_local
                drag_scale = drag_scale * surface_drag_coeff[surface_id]
                drag_scale = drag_scale * projected_area * speed * speed
                force = force + v_hat * drag_scale

        if surface_use_srp[surface_id] != 0 and use_srp != 0:
            cos_sun = wp.dot(n_world, sun_world)
            if cos_sun > zero64:
                eclipse_local = _eclipse_factor(r_point_eci, sun_hat_eci)
                if eclipse_local > zero64:
                    projected_area = surface_area[surface_id] * cos_sun
                    srp_scale = -eclipse_local * wp.float64(P_SUN)
                    srp_scale = srp_scale * surface_srp_coeff[surface_id]
                    srp_scale = srp_scale * projected_area
                    force = force + sun_world * srp_scale

        torque = wp.cross(r_cop_world, force)
        wrench_buffer[world_id, bid] = _spatial_add(wrench_buffer[world_id, bid], force, torque)

    if use_magnetic != 0:
        B_world = C_LI @ env_mag_field_eci[world_id]

        for magnetic_id in range(nmagnetic):
            bid = magnetic_body_id[magnetic_id]
            R_body = _mat33d_from_mat33(xmat[world_id, bid])
            B_body = wp.transpose(R_body) @ B_world
            tau_body = wp.cross(magnetic_dipole_body[magnetic_id], B_body)
            tau_world = R_body @ tau_body
            wrench_buffer[world_id, bid] = _spatial_add(
                wrench_buffer[world_id, bid],
                wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)),
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
                wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)),
                tau_world,
            )

    for rw_id in range(nrw):
        bid = rw_body_id[rw_id]
        R_body = _mat33d_from_mat33(xmat[world_id, bid])
        w_body = wp.transpose(R_body) @ _spatial_ang(cvel[world_id, bid])
        h_body = rw_axis_body[rw_id] * (rw_inertia[rw_id] * rw_speed[world_id, rw_id])
        tau_world = R_body @ (-wp.cross(w_body, h_body))
        wrench_buffer[world_id, bid] = _spatial_add(
            wrench_buffer[world_id, bid],
            wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)),
            tau_world,
        )
        rw_momentum[world_id, rw_id] = rw_inertia[rw_id] * rw_speed[world_id, rw_id]

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

    for body_id in range(nbody):
        xfrc_applied[world_id, body_id] = wrench_buffer[world_id, body_id]


@wp.func
def _command_rw_torques(
    world_id: int,
    rw_body_id: wp.array(dtype=int),
    rw_axis_body: wp.array(dtype=wp.vec3d),
    rw_inertia: wp.array(dtype=wp.float64),
    rw_speed_limit: wp.array(dtype=wp.float64),
    rw_has_speed_limit: wp.array(dtype=int),
    rw_torque_limit: wp.array(dtype=wp.float64),
    rw_has_torque_limit: wp.array(dtype=int),
    nrw: int,
    dt: wp.float64,
    rw_speed: wp.array2d(dtype=wp.float64),
    rw_momentum: wp.array2d(dtype=wp.float64),
    rw_torque_cmd: wp.array2d(dtype=wp.float64),
    wrench_buffer: wp.array2d(dtype=wp.spatial_vector),
    xmat: wp.array2d(dtype=wp.mat33),
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector),
):
    for rw_id in range(nrw):
        inertia = rw_inertia[rw_id]
        if inertia <= wp.float64(0.0):
            continue

        tau = rw_torque_cmd[world_id, rw_id]
        if rw_has_torque_limit[rw_id] != 0:
            tau = wp.clamp(tau, -rw_torque_limit[rw_id], rw_torque_limit[rw_id])

        alpha = tau / inertia
        speed = rw_speed[world_id, rw_id]
        if rw_has_speed_limit[rw_id] != 0:
            limit = rw_speed_limit[rw_id]
            if speed >= limit and alpha > wp.float64(0.0):
                alpha = wp.float64(0.0)
            elif speed <= -limit and alpha < wp.float64(0.0):
                alpha = wp.float64(0.0)

        speed = speed + alpha * dt
        if rw_has_speed_limit[rw_id] != 0:
            speed = wp.clamp(speed, -rw_speed_limit[rw_id], rw_speed_limit[rw_id])
        rw_speed[world_id, rw_id] = speed
        rw_momentum[world_id, rw_id] = inertia * speed

        bid = rw_body_id[rw_id]
        reaction_tau = -inertia * alpha
        tau_body = rw_axis_body[rw_id] * reaction_tau
        R_body = _mat33d_from_mat33(xmat[world_id, bid])
        tau_world = R_body @ tau_body
        wrench_buffer[world_id, bid] = _spatial_add(
            wrench_buffer[world_id, bid],
            wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)),
            tau_world,
        )
        xfrc_applied[world_id, bid] = wrench_buffer[world_id, bid]


@wp.func
def _prop_deriv(
    R: wp.vec3d,
    V: wp.vec3d,
    use_j2: int,
    a_external: wp.vec3d,
) -> tuple[wp.vec3d, wp.vec3d]:
    return V, _total_accel(R, use_j2) + a_external


@wp.func
def _propagate_rk4(
    R: wp.vec3d,
    V: wp.vec3d,
    dt: wp.float64,
    use_j2: int,
    a_external: wp.vec3d,
) -> tuple[wp.vec3d, wp.vec3d]:
    k1R, k1V = _prop_deriv(R, V, use_j2, a_external)
    half_dt = wp.float64(0.5) * dt
    k2R, k2V = _prop_deriv(R + k1R * half_dt, V + k1V * half_dt, use_j2, a_external)
    k3R, k3V = _prop_deriv(R + k2R * half_dt, V + k2V * half_dt, use_j2, a_external)
    k4R, k4V = _prop_deriv(R + k3R * dt, V + k3V * dt, use_j2, a_external)
    two = wp.float64(2.0)
    sixth_dt = dt / wp.float64(6.0)
    R_next = R + (k1R + k2R * two + k3R * two + k4R) * sixth_dt
    V_next = V + (k1V + k2V * two + k3V * two + k4V) * sixth_dt
    return R_next, V_next


def refresh_core(model: Any, data: Any) -> None:
    """Update device frame/environment caches from device orbit state."""

    cm = model.core_model
    cd = data.core_data
    wp.launch(
        _refresh_core_kernel,
        dim=(data.nworld,),
        inputs=[
            int(model.use_j2),
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
            cd.orbit_R_eci,
            cd.orbit_V_eci,
            cd.frame_C_LI,
            cd.frame_C_IL,
            cd.frame_omega_lvlh,
            cd.frame_omega_dot_lvlh,
            cd.env_sun_vector_eci,
            cd.env_eclipse,
            cd.env_mag_field_eci,
            cd.env_atm_density,
            cd.rw_speed,
            cd.rw_momentum,
            cd.mtq_dipole_cmd,
            cd.thr_force_cmd,
            cd.wrench_buffer,
            wd.xipos,
            wd.xmat,
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
            cm.total_mass,
            float(mj_dt),
            float(orbit_dt),
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
            cd.rw_speed,
            cd.rw_momentum,
            cd.rw_torque_cmd,
            cd.mtq_dipole_cmd,
            cd.thr_force_cmd,
            cd.wrench_buffer,
            wd.xipos,
            wd.xmat,
            wd.cvel,
            wd.xfrc_applied,
        ],
    )


__all__ = [
    "DeviceCoreData",
    "DeviceCoreModel",
    "assemble_forward_wrenches",
    "assemble_step_and_propagate",
    "make_device_core_data",
    "make_device_core_model",
    "pull_core_device_to_public",
    "refresh_core",
    "sync_core_device_from_public",
]
