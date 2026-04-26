# pyright: reportAttributeAccessIssue=false

"""Forward and step functions for the C++-first API."""

from __future__ import annotations

from mujoco_orbit import _bindings
from mujoco_orbit.core.runtime import MjoData, MjoModel


def mjo_forward(model: MjoModel, data: MjoData) -> None:
    """Synchronize derived runtime state after direct mutation."""
    _bindings.mjo_forward(model._native, data._native)


def mjo_step(model: MjoModel, data: MjoData) -> None:
    """Advance one fully coupled simulation step in-place."""
    _bindings.mjo_step(model._native, data._native)


__all__ = ["mjo_forward", "mjo_step"]
