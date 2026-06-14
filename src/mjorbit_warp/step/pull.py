# pyright: reportAttributeAccessIssue=false, reportIndexIssue=false

"""``mjo_pull`` and the device→host plumbing it relies on."""

from __future__ import annotations

from collections.abc import Iterable

import mujoco
import numpy as np

from .._deps import require_mjwarp
from ..core_gpu import pull_core_device_to_public
from ..data import MjoData
from ..field_registry import _MIRRORED_ARRAY_FIELDS, _MIRRORED_SCALAR_FIELDS
from ..model import MjoModel
from .field_specs import _CORE_PULL_FIELDS, _normalize_pull_fields


def _copy_device_field_to_public(data: MjoData, field: str) -> None:
    device_value = getattr(data.warp_data, field)
    values = device_value.numpy()

    if field == "time":
        if data.nworld == 1:
            data.time = float(values[0])
        else:
            time = data.time
            if not isinstance(time, np.ndarray):
                raise TypeError("batched warp data time must be a NumPy array")
            np.copyto(time, values)
        return

    if field in _MIRRORED_SCALAR_FIELDS:
        if data.nworld == 1:
            setattr(data, field, int(values[0]))
        else:
            np.copyto(getattr(data, field), values)
        return

    target = getattr(data, field)
    source = values[0] if data.nworld == 1 else values
    if source.shape != target.shape:
        if source.size != target.size:
            raise ValueError(
                f"Cannot pull MJWarp field {field!r}: "
                f"device shape {source.shape} is incompatible with public shape {target.shape}"
            )
        source = source.reshape(target.shape)
    np.copyto(target, source)

def _pull_device_fields_to_public(data: MjoData, fields: frozenset[str]) -> None:

    for field in _MIRRORED_ARRAY_FIELDS:
        if field in fields:
            _copy_device_field_to_public(data, field)

    for field in _MIRRORED_SCALAR_FIELDS:
        if field in fields:
            _copy_device_field_to_public(data, field)

    if "time" in fields:
        _copy_device_field_to_public(data, "time")

    core_fields = fields & _CORE_PULL_FIELDS
    if core_fields:
        pull_core_device_to_public(data, fields=core_fields)

def _refresh_host_orbit_caches(model: MjoModel, run) -> None:
    run.actuators.update_rw_momentum(model.rw_inertia)
    run.refresh_native(model)

def _copy_public_core_to_host(data: MjoData, run, world_id: int) -> None:
    if data.nworld == 1:
        np.copyto(run.orbit.R_eci, data.orbit.R_eci)
        np.copyto(run.orbit.V_eci, data.orbit.V_eci)
        run.orbit.t = float(data.orbit.t)
        np.copyto(run.actuators.rw_speed, data.actuators.rw_speed)
        np.copyto(run.actuators.rw_momentum, data.actuators.rw_momentum)
        np.copyto(run.actuators.rw_torque_cmd, data.actuators.rw_torque_cmd)
        np.copyto(run.actuators.mtq_dipole_cmd, data.actuators.mtq_dipole_cmd)
        np.copyto(run.actuators.thr_force_cmd, data.actuators.thr_force_cmd)
        np.copyto(run.wrench_buffer, data.wrench_buffer)
        return

    np.copyto(run.orbit.R_eci, data.orbit.R_eci[world_id])
    np.copyto(run.orbit.V_eci, data.orbit.V_eci[world_id])
    run.orbit.t = float(data.orbit.t[world_id])
    np.copyto(run.actuators.rw_speed, data.actuators.rw_speed[world_id])
    np.copyto(run.actuators.rw_momentum, data.actuators.rw_momentum[world_id])
    np.copyto(run.actuators.rw_torque_cmd, data.actuators.rw_torque_cmd[world_id])
    np.copyto(run.actuators.mtq_dipole_cmd, data.actuators.mtq_dipole_cmd[world_id])
    np.copyto(run.actuators.thr_force_cmd, data.actuators.thr_force_cmd[world_id])
    np.copyto(run.wrench_buffer, data.wrench_buffer[world_id])

def _pull_device_into_host(model: MjoModel, data: MjoData) -> None:
    mjw, _ = require_mjwarp()

    pull_core_device_to_public(data)

    for world_id, run in enumerate(data._host_runs):
        mjw.get_data_into(run.mj_data, model.mj_model, data.warp_data, world_id=world_id)
        _copy_public_core_to_host(data, run, world_id)
        _refresh_host_orbit_caches(model, run)
        mujoco.mj_forward(model.mj_model, run.mj_data)

def mjo_pull(
    model: MjoModel,
    data: MjoData,
    *,
    fields: Iterable[str] | str | None = None,
) -> None:
    """Refresh public NumPy buffers and host shadows from device state.

    ``fields=None`` performs the full compatibility pull through host shadows.
    Passing field names refreshes only those public NumPy buffers and leaves host
    shadows untouched.  This does not upload public buffers to the device first,
    so it is safe to use after a device-resident ``mjo_step(..., sync=False)``
    loop.
    """
    if fields is not None:
        _pull_device_fields_to_public(data, _normalize_pull_fields(fields))
        return

    _pull_device_into_host(model, data)
    data._pull_from_host()


__all__ = [
    'mjo_pull',
]
