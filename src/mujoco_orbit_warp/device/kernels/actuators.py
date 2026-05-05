# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false, reportInvalidTypeForm=false
# pyright: reportOperatorIssue=false, reportReturnType=false

"""Reaction-wheel speed-advance kernel.

The corresponding torque (gyroscopic + reaction) is applied inside
``_assemble_wrenches`` (in coupling.py) so the forward path also sees it;
this kernel only integrates wheel speed during the Advance phase.
"""

from __future__ import annotations

import warp as wp


@wp.func
def _command_rw_torques(
    world_id: int,
    rw_body_id: wp.array(dtype=int),
    rw_axis_body: wp.array(dtype=wp.vec3),
    rw_inertia: wp.array(dtype=wp.float32),
    rw_speed_limit: wp.array(dtype=wp.float32),
    rw_has_speed_limit: wp.array(dtype=int),
    rw_torque_limit: wp.array(dtype=wp.float32),
    rw_has_torque_limit: wp.array(dtype=int),
    nrw: int,
    dt: wp.float32,
    rw_speed: wp.array2d(dtype=wp.float32),
    rw_momentum: wp.array2d(dtype=wp.float32),
    rw_torque_cmd: wp.array2d(dtype=wp.float32),
    wrench_buffer: wp.array2d(dtype=wp.spatial_vector),
    xmat: wp.array2d(dtype=wp.mat33),
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector),
):
    # Reaction torque is applied in _assemble_wrenches so the forward path also
    # sees it. This kernel only advances the wheel speed (Advance phase).
    # Mirrors CPU advance_reaction_wheels (src/cpp/src/coupling_passive.cc:612).
    for rw_id in range(nrw):
        inertia = rw_inertia[rw_id]
        if inertia <= wp.float32(0.0):
            continue

        tau = rw_torque_cmd[world_id, rw_id]
        if rw_has_torque_limit[rw_id] != 0:
            tau = wp.clamp(tau, -rw_torque_limit[rw_id], rw_torque_limit[rw_id])

        alpha = tau / inertia
        speed = rw_speed[world_id, rw_id]
        if rw_has_speed_limit[rw_id] != 0:
            limit = rw_speed_limit[rw_id]
            if speed >= limit and alpha > wp.float32(0.0):
                alpha = wp.float32(0.0)
            elif speed <= -limit and alpha < wp.float32(0.0):
                alpha = wp.float32(0.0)

        speed = speed + alpha * dt
        if rw_has_speed_limit[rw_id] != 0:
            speed = wp.clamp(speed, -rw_speed_limit[rw_id], rw_speed_limit[rw_id])
        rw_speed[world_id, rw_id] = speed
        rw_momentum[world_id, rw_id] = inertia * speed


__all__ = [
    '_command_rw_torques',
]