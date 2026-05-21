# pyright: reportAttributeAccessIssue=false

"""``mjo_step`` — advance one fully coupled MJWarp simulation step in-place."""

from __future__ import annotations

import mujoco

from .._deps import require_mjwarp
from ..data import MjoData
from ..model import MjoModel
from .pull import mjo_pull
from .upload import mjo_upload


def mjo_step(model: MjoModel, data: MjoData, *, sync: bool = False) -> None:
    """Advance one fully coupled MJWarp simulation step in-place.

    By default the MJWarp device state is authoritative: public NumPy buffers
    are not uploaded before the step and host mirrors are not refreshed
    afterward.  Call ``mjo_upload`` for explicit public-buffer uploads and
    ``mjo_pull`` before reading public buffers or host ``MjData``.
    """
    mjw, _ = require_mjwarp()
    from ..core_gpu import assemble_step_and_propagate, refresh_core

    mj_dt = model.opt.timestep
    # The native binding stores an unset orbit_dt as 0.0 (not None), so treat
    # any non-positive value as "use mj_dt" — otherwise the kernel ends up
    # substepping with sub_dt=0 and loops forever.
    orbit_dt = model.orbit_dt if (model.orbit_dt is not None and model.orbit_dt > 0.0) else mj_dt

    if sync:
        mjo_upload(model, data)
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
