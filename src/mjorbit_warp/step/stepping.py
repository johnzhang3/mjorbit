# pyright: reportAttributeAccessIssue=false

"""``mjo_step`` — advance one fully coupled MJWarp simulation step in-place."""

from __future__ import annotations

import mujoco

from .._deps import require_mjwarp
from ..data import MjoData
from ..model import MjoModel
from .field_specs import _CORE_COMMAND_FIELDS
from .pull import mjo_pull
from .upload import mjo_upload


def mjo_step(model: MjoModel, data: MjoData, *, sync: bool = False) -> None:
    """Advance one fully coupled MJWarp simulation step in-place.

    By default the MJWarp device state is authoritative: large public NumPy
    buffers (``qpos``/``qvel``/``ctrl``) and integrated state (``orbit``,
    ``rw_speed``) are not uploaded before the step and host mirrors are not
    refreshed afterward.  As a convenience the cheap actuator *command* inputs
    (``rw_torque_cmd``, ``mtq_dipole_cmd``, ``thr_force_cmd``) are always synced
    from the public buffers so the documented ``data.actuators.* = cmd;
    mjo_step(model, data)`` pattern produces torque on this backend exactly as
    it does on the CPU backend (issue #10).  Call ``mjo_upload`` for explicit
    full-buffer uploads, pass ``sync=True`` for the full upload+pull, and call
    ``mjo_pull`` before reading public buffers or host ``MjData``.
    """
    mjw, _ = require_mjwarp()
    from ..core_gpu import assemble_step_and_propagate, refresh_core, sync_core_device_from_public

    mj_dt = model.opt.timestep
    # The native binding stores an unset orbit_dt as 0.0 (not None), so treat
    # any non-positive value as "use mj_dt" — otherwise the kernel ends up
    # substepping with sub_dt=0 and loops forever.
    orbit_dt = model.orbit_dt if (model.orbit_dt is not None and model.orbit_dt > 0.0) else mj_dt

    if sync:
        mjo_upload(model, data)
    else:
        # Always push the pure command inputs (never the device-integrated
        # orbit/rw_speed state) so set-cmd-then-step works without an explicit
        # upload while keeping the device authoritative for everything else.
        sync_core_device_from_public(data, fields=_CORE_COMMAND_FIELDS)
    refresh_core(model, data)
    # MJWarp's step2 falls back to Euler for RK4, so keep the one-piece
    # step path when RK4 semantics are requested.
    if model.opt.integrator == int(mujoco.mjtIntegrator.mjINT_RK4):
        mjw.forward(model.warp_model, data.warp_data)
        assemble_step_and_propagate(model, data, mj_dt=mj_dt, orbit_dt=orbit_dt)
        mjw.step(model.warp_model, data.warp_data)
    else:
        mjw.step1(model.warp_model, data.warp_data)
        assemble_step_and_propagate(model, data, mj_dt=mj_dt, orbit_dt=orbit_dt)
        mjw.step2(model.warp_model, data.warp_data)
    if sync:
        mjo_pull(model, data)


__all__ = [
    'mjo_step',
]
