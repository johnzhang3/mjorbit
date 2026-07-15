# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false, reportInvalidTypeForm=false
# pyright: reportOperatorIssue=false, reportReturnType=false

"""Sun direction, eclipse, dipole magnetic field, and exponential atmosphere."""

from __future__ import annotations

import numpy as np
import warp as wp

from mjorbit.constants import ERA_J2000 as _ERA_J2000

_DEG_TO_RAD = np.pi / 180.0
_TWO_PI = 2.0 * np.pi


@wp.func
def _sun_vector_eci(t: wp.float64) -> wp.vec3:
    # Epoch-anchored t is ~1e9 s since J2000: the angle accumulation must run in
    # float64 (float32 resolution at that magnitude is ~64 s of time, i.e. degrees
    # of solar longitude). Only the final unit vector is narrowed to float32.
    deg_to_rad = wp.float64(_DEG_TO_RAD)
    T_jc = t / (wp.float64(36525.0) * wp.float64(86400.0))
    lambda_sun = (wp.float64(280.460) + wp.float64(36000.771) * T_jc) * deg_to_rad
    M_sun = (wp.float64(357.528) + wp.float64(35999.050) * T_jc) * deg_to_rad
    lambda_ecl = lambda_sun + wp.float64(1.915) * deg_to_rad * wp.sin(M_sun)
    lambda_ecl = lambda_ecl + wp.float64(0.020) * deg_to_rad * wp.sin(wp.float64(2.0) * M_sun)
    eps = (wp.float64(23.439) - wp.float64(0.013) * T_jc) * deg_to_rad
    return wp.vec3(
        wp.float32(wp.cos(lambda_ecl)),
        wp.float32(wp.sin(lambda_ecl) * wp.cos(eps)),
        wp.float32(wp.sin(lambda_ecl) * wp.sin(eps)),
    )
@wp.func
def _magnetic_axis_eci(
    t: wp.float64,
    magnetic_axis: wp.vec3,
    spin_axis: wp.vec3,
    omega_mag: wp.float64,
) -> wp.vec3:
    # ``magnetic_axis`` is body-fixed (ECEF components): co-rotate it about the
    # spin axis by theta(t) = ERA_J2000 + |omega| * t (t in s since J2000). With
    # the default axis parallel to the spin axis this is the identity. Matches
    # CPU magnetic_axis_eci (src/cpp/src/environment.cc). theta accumulates in
    # float64 and is reduced mod 2*pi before narrowing.
    if omega_mag == wp.float64(0.0):
        return magnetic_axis
    two_pi = wp.float64(_TWO_PI)
    theta64 = wp.float64(_ERA_J2000) + omega_mag * t
    theta64 = theta64 - two_pi * wp.floor(theta64 / two_pi)
    c = wp.float32(wp.cos(theta64))
    s = wp.float32(wp.sin(theta64))
    # Rodrigues rotation of magnetic_axis about spin_axis by theta.
    cross = wp.cross(spin_axis, magnetic_axis)
    dot = wp.dot(spin_axis, magnetic_axis)
    return magnetic_axis * c + cross * s + spin_axis * (dot * (wp.float32(1.0) - c))
@wp.func
def _eclipse_factor(R: wp.vec3, sun_hat: wp.vec3, radius_km: wp.float32) -> wp.float32:
    proj = -wp.dot(R, sun_hat)
    if proj < wp.float32(0.0):
        return wp.float32(1.0)

    d_perp = wp.length(R - sun_hat * wp.dot(R, sun_hat))
    if d_perp < radius_km:
        return wp.float32(0.0)
    return wp.float32(1.0)
@wp.func
def _dipole_field_eci(
    R: wp.vec3,
    magnetic_b0: wp.float32,
    m_hat_eci: wp.vec3,
    radius_km: wp.float32,
) -> wp.vec3:
    # ``m_hat_eci`` is the unit dipole-axis direction already rotated into ECI
    # (see _magnetic_axis_eci); the configured axis is pre-normalized at build
    # time (device/build.py), matching the CPU dipole_field_eci.
    r = wp.length(R)
    r_hat = R / r
    m_hat = m_hat_eci
    radius_ratio = radius_km / r
    factor = magnetic_b0 * radius_ratio * radius_ratio * radius_ratio
    return (r_hat * (wp.float32(3.0) * wp.dot(m_hat, r_hat)) - m_hat) * factor
@wp.func
def _atm_density(
    R: wp.vec3,
    atm_h0_km: wp.float32,
    atm_rho0: wp.float32,
    atm_h_scale_km: wp.float32,
    radius_km: wp.float32,
) -> wp.float32:
    # Guard degenerate atmosphere params (matches CPU atm_density): a zero or
    # negative scale height / reference density yields no drag, not inf/NaN.
    if atm_rho0 <= wp.float32(0.0) or atm_h_scale_km <= wp.float32(0.0):
        return wp.float32(0.0)
    alt_km = wp.length(R) - radius_km
    rho = atm_rho0 * wp.exp(-(alt_km - atm_h0_km) / atm_h_scale_km)
    return wp.max(rho, wp.float32(0.0))


__all__ = [
    '_atm_density',
    '_dipole_field_eci',
    '_eclipse_factor',
    '_magnetic_axis_eci',
    '_sun_vector_eci',
]