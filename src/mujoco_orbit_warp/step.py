# pyright: reportAttributeAccessIssue=false, reportIndexIssue=false

"""Forward and step functions for the MJWarp-backed API."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import mujoco
import numpy as np

from mujoco_orbit.orbit.environment import update_environment_cache
from mujoco_orbit.orbit.lvlh import update_frame_cache
from mujoco_orbit.sensors import update_sensor_environment

from ._deps import require_mjwarp
from .runtime import (
    _MIRRORED_ARRAY_FIELDS,
    _MIRRORED_SCALAR_FIELDS,
    MjoData,
    MjoModel,
)

_CORE_PULL_FIELDS = frozenset({"orbit", "frame", "env", "actuators", "wrench_buffer"})
_CORE_UPLOAD_FIELDS = frozenset(
    {"orbit", "actuators", "rw_speed", "rw_torque_cmd", "mtq_dipole_cmd", "thr_force_cmd"}
)
_PULL_ALIASES = {
    "environment": "env",
    "wrench": "wrench_buffer",
    "wrenches": "wrench_buffer",
    "rw": "actuators",
}
_PULL_GROUPS = {
    "state": ("time", "qpos", "qvel"),
    "integration": ("time", "qpos", "qvel", "act", "qacc_warmstart", "ctrl"),
    "inputs": ("ctrl", "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat", "eq_active"),
    "core": tuple(_CORE_PULL_FIELDS),
}
_DEVICE_PULL_FIELDS = frozenset(_MIRRORED_ARRAY_FIELDS) | frozenset(_MIRRORED_SCALAR_FIELDS) | {
    "time"
}
_SELECTIVE_PULL_FIELDS = _DEVICE_PULL_FIELDS | _CORE_PULL_FIELDS
_DEVICE_UPLOAD_FIELDS = frozenset(
    {
        "time",
        "qpos",
        "qvel",
        "act",
        "qacc_warmstart",
        "ctrl",
        "qfrc_applied",
        "xfrc_applied",
        "mocap_pos",
        "mocap_quat",
        "eq_active",
    }
)
_UPLOAD_GROUPS = {
    "state": ("time", "qpos", "qvel"),
    "integration": ("time", "qpos", "qvel", "act", "qacc_warmstart", "ctrl"),
    "inputs": ("ctrl", "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat", "eq_active"),
    "core": tuple(_CORE_UPLOAD_FIELDS),
}
_SELECTIVE_UPLOAD_FIELDS = _DEVICE_UPLOAD_FIELDS | _CORE_UPLOAD_FIELDS


def _as_batched(values: np.ndarray | float, *, nworld: int) -> np.ndarray:
    array = np.asarray(values)
    if nworld == 1:
        return np.expand_dims(array, axis=0)
    return array


def _copy_to_device(
    wp: Any,
    dest: Any,
    values: np.ndarray | float,
    *,
    dtype: Any,
    shape: tuple[int, ...] | None = None,
) -> None:
    array = np.asarray(values)
    if array.size == 0:
        return

    if shape is None:
        src = wp.array(array, dtype=dtype)
    else:
        src = wp.array(array, shape=shape, dtype=dtype)
    wp.copy(dest, src)


def _normalize_pull_fields(fields: Iterable[str] | str) -> frozenset[str]:
    raw_fields = (fields,) if isinstance(fields, str) else tuple(fields)
    selected: set[str] = set()
    for raw_field in raw_fields:
        raw_name = raw_field.lower()
        field = _PULL_ALIASES.get(raw_name, raw_name)
        group = _PULL_GROUPS.get(field)
        if group is None:
            selected.add(field)
        else:
            selected.update(group)

    unknown = selected - _SELECTIVE_PULL_FIELDS
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown MJWarp pull field(s): {unknown_list}")
    return frozenset(selected)


def _normalize_upload_fields(fields: Iterable[str] | str) -> frozenset[str]:
    raw_fields = (fields,) if isinstance(fields, str) else tuple(fields)
    selected: set[str] = set()
    for raw_field in raw_fields:
        raw_name = raw_field.lower()
        field = _PULL_ALIASES.get(raw_name, raw_name)
        group = _UPLOAD_GROUPS.get(field)
        if group is None:
            selected.add(field)
        else:
            selected.update(group)

    unknown = selected - _SELECTIVE_UPLOAD_FIELDS
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown MJWarp upload field(s): {unknown_list}")
    return frozenset(selected)


def _field_enabled(fields: frozenset[str] | None, field: str) -> bool:
    return fields is None or field in fields


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
    if data.nworld == 1:
        np.copyto(target, values[0])
    else:
        np.copyto(target, values)


def _pull_device_fields_to_public(data: MjoData, fields: frozenset[str]) -> None:
    from .core_gpu import pull_core_device_to_public

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
    run.frame = update_frame_cache(run.orbit, use_j2=model.use_j2)
    run.env = update_environment_cache(run.orbit, run.frame)
    run.actuators.update_rw_momentum(model.rw_inertia)
    update_sensor_environment(model.host_model, run)


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


def _sync_device_from_public(
    model: MjoModel,
    data: MjoData,
    fields: frozenset[str] | None = None,
) -> None:
    _, wp = require_mjwarp()

    if _field_enabled(fields, "qpos"):
        _copy_to_device(
            wp,
            data.warp_data.qpos,
            _as_batched(data.qpos, nworld=data.nworld),
            dtype=float,
        )
    if _field_enabled(fields, "qvel"):
        _copy_to_device(
            wp,
            data.warp_data.qvel,
            _as_batched(data.qvel, nworld=data.nworld),
            dtype=float,
        )
    if _field_enabled(fields, "qacc_warmstart"):
        _copy_to_device(
            wp,
            data.warp_data.qacc_warmstart,
            _as_batched(data.qacc_warmstart, nworld=data.nworld),
            dtype=float,
        )
    if _field_enabled(fields, "qfrc_applied"):
        _copy_to_device(
            wp,
            data.warp_data.qfrc_applied,
            _as_batched(data.qfrc_applied, nworld=data.nworld),
            dtype=float,
        )
    if _field_enabled(fields, "ctrl"):
        _copy_to_device(
            wp,
            data.warp_data.ctrl,
            _as_batched(data.ctrl, nworld=data.nworld),
            dtype=float,
        )
    if _field_enabled(fields, "time"):
        _copy_to_device(
            wp,
            data.warp_data.time,
            _as_batched(np.asarray(data.time), nworld=data.nworld).reshape(data.nworld),
            dtype=float,
        )

    if model.na > 0 and _field_enabled(fields, "act"):
        _copy_to_device(
            wp,
            data.warp_data.act,
            _as_batched(data.act, nworld=data.nworld),
            dtype=float,
        )

    if _field_enabled(fields, "xfrc_applied"):
        _copy_to_device(
            wp,
            data.warp_data.xfrc_applied,
            _as_batched(data.xfrc_applied, nworld=data.nworld),
            dtype=wp.spatial_vector,
            shape=(data.nworld, model.nbody),
        )

    if model.nmocap > 0 and _field_enabled(fields, "mocap_pos"):
        _copy_to_device(
            wp,
            data.warp_data.mocap_pos,
            _as_batched(data.mocap_pos, nworld=data.nworld),
            dtype=wp.vec3,
            shape=(data.nworld, model.nmocap),
        )
    if model.nmocap > 0 and _field_enabled(fields, "mocap_quat"):
        _copy_to_device(
            wp,
            data.warp_data.mocap_quat,
            _as_batched(data.mocap_quat, nworld=data.nworld),
            dtype=wp.quat,
            shape=(data.nworld, model.nmocap),
        )

    if model.neq > 0 and _field_enabled(fields, "eq_active"):
        _copy_to_device(
            wp,
            data.warp_data.eq_active,
            _as_batched(data.eq_active, nworld=data.nworld),
            dtype=bool,
        )


def _pull_device_into_host(model: MjoModel, data: MjoData) -> None:
    mjw, _ = require_mjwarp()
    from .core_gpu import pull_core_device_to_public

    pull_core_device_to_public(data)

    for world_id, run in enumerate(data._host_runs):
        mjw.get_data_into(run.mj_data, model.mj_model, data.warp_data, world_id=world_id)
        _copy_public_core_to_host(data, run, world_id)
        _refresh_host_orbit_caches(model, run)
        mujoco.mj_forward(model.mj_model, run.mj_data)


def mjo_upload(
    model: MjoModel,
    data: MjoData,
    *,
    fields: Iterable[str] | str | None = None,
) -> None:
    """Upload explicitly selected public NumPy inputs to device state.

    ``fields=None`` preserves the old broad upload behavior.  Passing field
    names or groups uploads only those public buffers.
    """
    from .core_gpu import sync_core_device_from_public

    selected = None if fields is None else _normalize_upload_fields(fields)
    device_fields = None if selected is None else selected & _DEVICE_UPLOAD_FIELDS
    core_fields = None if selected is None else selected & _CORE_UPLOAD_FIELDS

    if selected is None or device_fields:
        _sync_device_from_public(model, data, fields=device_fields)
    if selected is None or core_fields:
        sync_core_device_from_public(data, fields=core_fields)


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


def mjo_forward(model: MjoModel, data: MjoData, *, sync: bool = False) -> None:
    """Synchronize derived runtime state after direct mutation.

    By default the MJWarp device state is authoritative.  Public NumPy buffers
    are uploaded and host mirrors are refreshed only when ``sync=True``.
    """
    mjw, _ = require_mjwarp()
    from .core_gpu import assemble_forward_wrenches, refresh_core

    if sync:
        mjo_upload(model, data)
    refresh_core(model, data)
    mjw.forward(model.warp_model, data.warp_data)
    assemble_forward_wrenches(model, data)
    mjw.forward(model.warp_model, data.warp_data)
    if sync:
        mjo_pull(model, data)


def mjo_step(model: MjoModel, data: MjoData, *, sync: bool = False) -> None:
    """Advance one fully coupled MJWarp simulation step in-place.

    By default the MJWarp device state is authoritative: public NumPy buffers
    are not uploaded before the step and host mirrors are not refreshed
    afterward.  Call ``mjo_upload`` for explicit public-buffer uploads and
    ``mjo_pull`` before reading public buffers or host ``MjData``.
    """
    mjw, _ = require_mjwarp()
    from .core_gpu import assemble_step_and_propagate, refresh_core

    mj_dt = model.opt.timestep
    orbit_dt = model.orbit_dt if model.orbit_dt is not None else mj_dt

    if sync:
        mjo_upload(model, data)
    refresh_core(model, data)
    mjw.forward(model.warp_model, data.warp_data)
    assemble_step_and_propagate(model, data, mj_dt=mj_dt, orbit_dt=orbit_dt)
    mjw.step(model.warp_model, data.warp_data)
    if sync:
        mjo_pull(model, data)


__all__ = ["mjo_forward", "mjo_pull", "mjo_step", "mjo_upload"]
