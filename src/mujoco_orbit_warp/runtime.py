"""Backward-compatibility shim.

The original ``runtime.py`` was split into ``model``, ``data``,
``state``, ``host_shadow``, and ``field_registry`` modules in the refactor
of 2026-05. This module re-exports the public surface so external imports
of ``mujoco_orbit_warp.runtime`` continue to work.
"""

from __future__ import annotations

from .data import MjoData
from .model import MjoModel
from .state import (
    BatchedActuatorData,
    EnvironmentBatchCache,
    FrameBatchCache,
    OrbitBatchState,
)

__all__ = [
    "BatchedActuatorData",
    "EnvironmentBatchCache",
    "FrameBatchCache",
    "MjoData",
    "MjoModel",
    "OrbitBatchState",
]
