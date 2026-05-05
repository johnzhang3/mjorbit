# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false, reportInvalidTypeForm=false
# pyright: reportOperatorIssue=false, reportReturnType=false

"""Two-body and J2 gravitational accelerations, plus the gravity-gradient torque.

``_relative_accel`` is the cancellation-safe Encke form used for body-relative
differential gravity. The pure-J2 differential is too small to suffer from
cancellation, so it is computed by direct subtraction.
"""

from __future__ import annotations

import warp as wp

from mujoco_orbit.constants import GM_EARTH, J2_EARTH, R_EARTH


@wp.func
def _j2_accel(R: wp.vec3) -> wp.vec3:
    one = wp.float32(1.0)
    gm = wp.float32(GM_EARTH)
    earth_radius = wp.float32(R_EARTH)
    j2 = wp.float32(J2_EARTH)
    r = wp.length(R)
    factor = wp.float32(1.5) * j2 * gm * earth_radius * earth_radius
    factor = factor / (r * r * r * r * r)
    x = R[0]
    y = R[1]
    z = R[2]
    z_r2 = (z / r) * (z / r)
    return wp.vec3(
        factor * x * (wp.float32(5.0) * z_r2 - one),
        factor * y * (wp.float32(5.0) * z_r2 - one),
        factor * z * (wp.float32(5.0) * z_r2 - wp.float32(3.0)),
    )
@wp.func
def _total_accel(R: wp.vec3, use_j2: int) -> wp.vec3:
    one = wp.float32(1.0)
    gm = wp.float32(GM_EARTH)
    r = wp.length(R)
    inv_r3 = one / (r * r * r)
    accel = R * (-gm * inv_r3)
    if use_j2 != 0:
        accel = accel + _j2_accel(R)
    return accel
@wp.func
def _encke_point_mass_relative_accel(
    rho: wp.vec3,
    R_chief: wp.vec3,
) -> wp.vec3:
    """Encke's identity for the two-body differential ``g_pm(R+rho) - g_pm(R)``.

    Cancellation-safe at single precision when ||rho|| << ||R_chief||.
    Mirrors src/cpp/src/gravity.cc:57.
    """
    one = wp.float32(1.0)
    two = wp.float32(2.0)
    three = wp.float32(3.0)
    gm = wp.float32(GM_EARTH)
    rc2 = wp.dot(R_chief, R_chief)
    rc = wp.sqrt(rc2)
    sigma = (two * wp.dot(rho, R_chief) + wp.dot(rho, rho)) / rc2
    one_plus_sigma = one + sigma
    one_plus_sigma_3_2 = one_plus_sigma * wp.sqrt(one_plus_sigma)
    f = (
        sigma
        * (three + three * sigma + sigma * sigma)
        / ((one + one_plus_sigma_3_2) * one_plus_sigma_3_2)
    )
    scale = -gm / (rc2 * rc)
    return (rho - (R_chief + rho) * f) * scale
@wp.func
def _relative_accel(
    rho: wp.vec3,
    R_chief: wp.vec3,
    use_j2: int,
) -> wp.vec3:
    """Differential gravity ``g(R+rho) - g(R)``: Encke for point-mass, direct
    subtraction for J2 (J2 itself is ~3 orders smaller, no cancellation issue)."""
    a = _encke_point_mass_relative_accel(rho, R_chief)
    if use_j2 != 0:
        a = a + _j2_accel(R_chief + rho) - _j2_accel(R_chief)
    return a
@wp.func
def _gravity_gradient_torque(
    r_hat: wp.vec3,
    r_mag_km: wp.float32,
    ximat_target: wp.mat33,
    inertia_principal: wp.vec3,
) -> wp.vec3:
    """Body gravity-gradient torque ``tau = 3 GM / r^3 * r_hat x (J r_hat)``.

    ``r_hat`` and ``ximat_target`` must be in the same frame; the returned torque
    is in that frame. Pass r_hat in MuJoCo-world (LVLH for the Warp backend) and
    ``ximat`` from MJWarp data, since ``ximat`` is world-from-principal-axes so
    ``J = ximat @ diag(I) @ ximat^T`` directly. Inertia is in kg·m², r in km, GM
    in km³/s², so the resulting torque is in N·m. Mirrors
    src/cpp/src/coupling_passive.cc:170.
    """
    three = wp.float32(3.0)
    gm = wp.float32(GM_EARTH)
    a = wp.transpose(ximat_target) @ r_hat
    Ja = wp.vec3(
        inertia_principal[0] * a[0],
        inertia_principal[1] * a[1],
        inertia_principal[2] * a[2],
    )
    j_rhat = ximat_target @ Ja
    coeff = three * gm / (r_mag_km * r_mag_km * r_mag_km)
    return wp.cross(r_hat, j_rhat) * coeff


__all__ = [
    '_encke_point_mass_relative_accel',
    '_gravity_gradient_torque',
    '_j2_accel',
    '_relative_accel',
    '_total_accel',
]