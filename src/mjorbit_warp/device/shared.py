# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false, reportInvalidTypeForm=false
# pyright: reportOperatorIssue=false, reportReturnType=false

"""Small `@wp.func` helpers shared by multiple kernels.

These were the fp64↔fp32 conversion helpers when the orbit overlay ran in
fp64; now that everything is fp32, most are identity wrappers. They are kept
as named helpers because individual call sites read better with named
operations than with raw vec/spatial indexing. ``_net_linear_force`` and
``_apply_origin_compensation`` live here because they are used by both
``_assemble_forward_kernel`` and ``_assemble_step_kernel``.
"""

from __future__ import annotations

import warp as wp


@wp.func
def _vec3d_from_vec3(value: wp.vec3) -> wp.vec3:
    return value
@wp.func
def _spatial_ang(value: wp.spatial_vector) -> wp.vec3:
    return wp.vec3(value[0], value[1], value[2])
@wp.func
def _spatial_lin(value: wp.spatial_vector) -> wp.vec3:
    return wp.vec3(value[3], value[4], value[5])
@wp.func
def _mat33d_from_mat33(value: wp.mat33) -> wp.mat33:
    return value
@wp.func
def _spatial_from_parts(force: wp.vec3, torque: wp.vec3) -> wp.spatial_vector:
    return wp.spatial_vector(
        force[0], force[1], force[2], torque[0], torque[1], torque[2]
    )
@wp.func
def _spatial_add(value: wp.spatial_vector, force: wp.vec3, torque: wp.vec3) -> wp.spatial_vector:
    return wp.spatial_vector(
        value[0] + force[0],
        value[1] + force[1],
        value[2] + force[2],
        value[3] + torque[0],
        value[4] + torque[1],
        value[5] + torque[2],
    )
@wp.func
def _net_linear_force(
    world_id: int,
    nbody: int,
    wrench_buffer: wp.array2d(dtype=wp.spatial_vector),
) -> wp.vec3:
    net = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    for body_id in range(nbody):
        wrench = wrench_buffer[world_id, body_id]
        net = net + wp.vec3(
            wp.float32(wrench[0]),
            wp.float32(wrench[1]),
            wp.float32(wrench[2]),
        )
    return net
@wp.func
def _apply_origin_compensation(
    world_id: int,
    nbody: int,
    body_mass: wp.array(dtype=wp.float32),
    a_chief_m_s2: wp.vec3,
    wrench_buffer: wp.array2d(dtype=wp.spatial_vector),
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector),
):
    """Apply -m_body * a_chief to each body. Mirrors CPU
    apply_origin_acceleration_wrenches (src/cpp/src/coupling_passive.cc:593).

    A chief-centered translating frame is non-inertial when the chief experiences
    non-gravitational acceleration; each body in MJ-world feels a corresponding
    pseudo-force so its absolute-frame motion comes out correct."""
    zero64 = wp.float32(0.0)
    for body_id in range(1, nbody):
        mass_b = body_mass[body_id]
        if mass_b > zero64:
            comp_force = a_chief_m_s2 * (-mass_b)
            wrench_buffer[world_id, body_id] = _spatial_add(
                wrench_buffer[world_id, body_id],
                comp_force,
                wp.vec3(zero64, zero64, zero64),
            )
            xfrc_applied[world_id, body_id] = wrench_buffer[world_id, body_id]


__all__ = [
    '_apply_origin_compensation',
    '_mat33d_from_mat33',
    '_net_linear_force',
    '_spatial_add',
    '_spatial_ang',
    '_spatial_from_parts',
    '_spatial_lin',
    '_vec3d_from_vec3',
]