# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false, reportInvalidTypeForm=false
# pyright: reportOperatorIssue=false, reportReturnType=false

"""Chief-orbit RK4 propagator with optional external acceleration injection."""

from __future__ import annotations

import warp as wp

from .gravity import _total_accel


@wp.func
def _prop_deriv(
    R: wp.vec3,
    V: wp.vec3,
    use_j2: int,
    a_external: wp.vec3,
) -> tuple[wp.vec3, wp.vec3]:
    return V, _total_accel(R, use_j2) + a_external
@wp.func
def _propagate_rk4(
    R: wp.vec3,
    V: wp.vec3,
    dt: wp.float32,
    use_j2: int,
    a_external: wp.vec3,
) -> tuple[wp.vec3, wp.vec3]:
    k1R, k1V = _prop_deriv(R, V, use_j2, a_external)
    half_dt = wp.float32(0.5) * dt
    k2R, k2V = _prop_deriv(R + k1R * half_dt, V + k1V * half_dt, use_j2, a_external)
    k3R, k3V = _prop_deriv(R + k2R * half_dt, V + k2V * half_dt, use_j2, a_external)
    k4R, k4V = _prop_deriv(R + k3R * dt, V + k3V * dt, use_j2, a_external)
    two = wp.float32(2.0)
    sixth_dt = dt / wp.float32(6.0)
    R_next = R + (k1R + k2R * two + k3R * two + k4R) * sixth_dt
    V_next = V + (k1V + k2V * two + k3V * two + k4V) * sixth_dt
    return R_next, V_next


__all__ = [
    '_prop_deriv',
    '_propagate_rk4',
]