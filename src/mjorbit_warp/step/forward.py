"""``mjo_forward`` — synchronize derived runtime state after direct mutation."""

from __future__ import annotations

from .._deps import require_mjwarp
from ..data import MjoData
from ..model import MjoModel
from .pull import mjo_pull
from .upload import mjo_upload, sync_command_inputs


def mjo_forward(model: MjoModel, data: MjoData, *, sync: bool = False) -> None:
    """Synchronize derived runtime state after direct mutation.

    By default the MJWarp device state is authoritative; large public NumPy
    buffers and integrated state are uploaded and host mirrors refreshed only
    when ``sync=True``.  The cheap actuator *command* inputs
    (``rw_torque_cmd``, ``mtq_dipole_cmd``, ``thr_force_cmd``) are synced from
    the public buffers each call (except during CUDA-graph capture), so editing
    a command and calling ``mjo_forward`` takes effect without an explicit
    upload, matching the CPU backend (issue #10).
    """
    mjw, _ = require_mjwarp()
    from ..core_gpu import (
        assemble_forward_wrenches,
        refresh_core,
        reset_orbit_schedule,
    )

    if sync:
        mjo_upload(model, data)
    else:
        sync_command_inputs(data)
    # Direct mutation may have invalidated the multirate segment endpoints;
    # the next step will rebuild the schedule from current state.
    reset_orbit_schedule(data)
    refresh_core(model, data)
    mjw.forward(model.warp_model, data.warp_data)
    assemble_forward_wrenches(model, data)
    mjw.forward(model.warp_model, data.warp_data)
    if sync:
        mjo_pull(model, data)


__all__ = [
    'mjo_forward',
]