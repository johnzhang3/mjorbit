"""Field-name registries and normalization for selective ``mjo_upload`` / ``mjo_pull``.

Defines the canonical set of fields each entry point understands, the
user-facing aliases / group names (``state``, ``integration``, ``inputs``,
``core``), and the normalization helpers that expand them into a flat
frozenset for downstream upload/pull machinery.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..field_registry import _MIRRORED_ARRAY_FIELDS, _MIRRORED_SCALAR_FIELDS

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
