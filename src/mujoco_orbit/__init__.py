"""MuJoCo-style coupled orbital and multibody dynamics."""

from __future__ import annotations

from mujoco_orbit._native import load_native_plugin as _load_native_plugin

_load_native_plugin()


from mujoco_orbit.core.config import (  # noqa: E402
    ControlMomentGyroSpec,
    MagneticBodySpec,
    MagnetorquerSpec,
    OrbitInit,
    ReactionWheelSpec,
    SurfaceSpec,
    ThrusterSpec,
)
from mujoco_orbit.core.rollout import (  # noqa: E402
    mjo_control_size,
    mjo_get_state,
    mjo_set_state,
    mjo_state_size,
    rollout,
)
from mujoco_orbit.core.runtime import MjoData, MjoModel  # noqa: E402
from mujoco_orbit.core.step import mjo_forward, mjo_step  # noqa: E402

__all__ = [
    "ControlMomentGyroSpec",
    "MagneticBodySpec",
    "MagnetorquerSpec",
    "MjoData",
    "MjoModel",
    "OrbitInit",
    "ReactionWheelSpec",
    "SurfaceSpec",
    "ThrusterSpec",
    "mjo_control_size",
    "mjo_forward",
    "mjo_get_state",
    "mjo_set_state",
    "mjo_step",
    "mjo_state_size",
    "rollout",
]
