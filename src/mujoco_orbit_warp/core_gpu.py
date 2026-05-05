# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false, reportInvalidTypeForm=false
# pyright: reportOperatorIssue=false, reportReturnType=false

"""Backward-compatibility shim.

The orbit-overlay kernels and dataclasses now live under ``device/`` (split
from this single file in the refactor of 2026-05). This module re-exports
the public surface so external imports of ``mujoco_orbit_warp.core_gpu``
continue to work.
"""

from __future__ import annotations

from .device.api import (
    assemble_forward_wrenches,
    assemble_step_and_propagate,
    refresh_core,
    reset_orbit_schedule,
)
from .device.build import make_device_core_data, make_device_core_model
from .device.io import (
    pull_core_device_to_public,
    sync_core_device_from_public,
)
from .device.types import DeviceCoreData, DeviceCoreModel

__all__ = [
    "DeviceCoreData",
    "DeviceCoreModel",
    "assemble_forward_wrenches",
    "assemble_step_and_propagate",
    "make_device_core_data",
    "make_device_core_model",
    "pull_core_device_to_public",
    "refresh_core",
    "reset_orbit_schedule",
    "sync_core_device_from_public",
]
