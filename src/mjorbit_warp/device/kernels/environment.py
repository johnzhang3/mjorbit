# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false, reportInvalidTypeForm=false
# pyright: reportOperatorIssue=false, reportReturnType=false

"""Sun direction, eclipse, dipole magnetic field, and exponential atmosphere."""

from __future__ import annotations

import numpy as np
import warp as wp

_DEG_TO_RAD = np.pi / 180.0


@wp.func
def _sun_vector_eci(t: wp.float32) -> wp.vec3:
    deg_to_rad = wp.float32(_DEG_TO_RAD)
    T_jc = t / (wp.float32(36525.0) * wp.float32(86400.0))
    lambda_sun = (wp.float32(280.460) + wp.float32(36000.771) * T_jc) * deg_to_rad
    M_sun = (wp.float32(357.528) + wp.float32(35999.050) * T_jc) * deg_to_rad
    lambda_ecl = lambda_sun + wp.float32(1.915) * deg_to_rad * wp.sin(M_sun)
    lambda_ecl = lambda_ecl + wp.float32(0.020) * deg_to_rad * wp.sin(wp.float32(2.0) * M_sun)
    eps = (wp.float32(23.439) - wp.float32(0.013) * T_jc) * deg_to_rad
    return wp.vec3(
        wp.cos(lambda_ecl),
        wp.sin(lambda_ecl) * wp.cos(eps),
        wp.sin(lambda_ecl) * wp.sin(eps),
    )
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
    magnetic_axis: wp.vec3,
    radius_km: wp.float32,
) -> wp.vec3:
    # ``magnetic_axis`` is pre-normalized at build time (see device/build.py),
    # matching the CPU dipole_field_eci which normalizes the configured axis.
    r = wp.length(R)
    r_hat = R / r
    m_hat = magnetic_axis
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
    '_sun_vector_eci',
]