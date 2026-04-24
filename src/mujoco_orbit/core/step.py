# pyright: reportAttributeAccessIssue=false

"""Forward and step functions for the MuJoCo-style API."""

from __future__ import annotations

import mujoco

from mujoco_orbit.core.runtime import MjoData, MjoModel


def _clear_wrench_buffer(data: MjoData) -> None:
    data.wrench_buffer[:] = 0.0
    data.xfrc_applied[:] = 0.0


def mjo_forward(model: MjoModel, data: MjoData) -> None:
    """Synchronize derived runtime state after direct mutation."""
    _clear_wrench_buffer(data)
    mujoco.mj_forward(model.mj_model, data.mj_data)
    data.xfrc_applied[:] = data.wrench_buffer


def mjo_step(model: MjoModel, data: MjoData) -> None:
    """Advance one fully coupled simulation step in-place."""
    _clear_wrench_buffer(data)
    mujoco.mj_step(model.mj_model, data.mj_data)
    data.xfrc_applied[:] = data.wrench_buffer


__all__ = ["mjo_forward", "mjo_step"]
