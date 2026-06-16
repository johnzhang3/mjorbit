"""``mjo_upload`` and the device-side sync of MJWarp-mirrored public buffers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .._deps import require_mjwarp
from ..core_gpu import sync_core_device_from_public
from ..data import MjoData
from ..model import MjoModel
from .field_specs import (
    _CORE_COMMAND_FIELDS,
    _CORE_UPLOAD_FIELDS,
    _DEVICE_UPLOAD_FIELDS,
    _field_enabled,
    _normalize_upload_fields,
)


def _device_is_capturing(data: MjoData) -> bool:
    """True when the data's device stream is recording a CUDA graph.

    Recording a host->device command copy inside ``wp.ScopedCapture`` ties the
    captured hot loop to allocator behavior: depending on the Warp/CUDA version
    the replayed copy can re-apply the capture-time host values on every
    ``wp.capture_launch`` and clobber commands uploaded between launches.  We
    sidestep that entirely by skipping the convenience command auto-sync during
    capture, leaving the captured step identical to the pre-#10 behavior (no
    command sync) so the device buffers stay authoritative.
    """
    try:
        device = data.core_data.mtq_dipole_cmd.device
    except AttributeError:
        return False
    if not getattr(device, "is_cuda", False):
        return False
    _, wp = require_mjwarp()
    return bool(wp.get_stream(device).is_capturing)


def sync_command_inputs(data: MjoData) -> None:
    """Push the pure actuator command inputs (``*_cmd``) host->device.

    Called on every eager ``mjo_step`` / ``mjo_forward`` so the documented
    ``data.actuators.*_cmd = cmd; mjo_step(...)`` pattern produces torque on the
    warp backend without an explicit upload.  These buffers are device inputs
    the coupling kernel reads but never integrates, so re-copying them each step
    is safe — unlike ``orbit`` / ``rw_speed``, which the device advances.

    No-op when the model has no orbit actuators, and while a CUDA graph is being
    captured: a captured step must not record a host->device copy (it would tie
    the captured hot loop to allocator behavior), so under capture the device
    buffers stay authoritative and commands are uploaded explicitly between
    ``wp.capture_launch`` calls, exactly as for ``ctrl``.
    """
    model = data.model
    if not (len(model.reaction_wheels) or len(model.magnetorquers) or len(model.thrusters)):
        return
    if _device_is_capturing(data):
        return
    sync_core_device_from_public(data, fields=_CORE_COMMAND_FIELDS)


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

    selected = None if fields is None else _normalize_upload_fields(fields)
    device_fields = None if selected is None else selected & _DEVICE_UPLOAD_FIELDS
    core_fields = None if selected is None else selected & _CORE_UPLOAD_FIELDS

    if selected is None or device_fields:
        _sync_device_from_public(model, data, fields=device_fields)
    if selected is None or core_fields:
        sync_core_device_from_public(data, fields=core_fields)


__all__ = [
    'mjo_upload',
    'sync_command_inputs',
]
