# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false, reportInvalidTypeForm=false
# pyright: reportOperatorIssue=false, reportReturnType=false

"""LVLH frame construction from the chief orbit (R_eci, V_eci)."""

from __future__ import annotations

import warp as wp

from .gravity import _total_accel


@wp.func
def _frame_from_orbit(
    R: wp.vec3,
    V: wp.vec3,
    use_j2: int,
) -> tuple[wp.mat33, wp.mat33, wp.vec3, wp.vec3]:
    r = wp.length(R)
    x_hat = R / r
    h = wp.cross(R, V)
    z_hat = h / wp.length(h)
    y_hat = wp.cross(z_hat, x_hat)

    C_LI = wp.mat33(
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
    domega_dt_eci = dh_dt / (r * r) - h * (wp.float32(2.0) * dr_dt / (r * r * r))
    omega_dot_lvlh = C_LI @ domega_dt_eci
    return C_LI, C_IL, omega_lvlh, omega_dot_lvlh


__all__ = [
    '_frame_from_orbit',
]