"""MuJoCo-style coupled orbital and multibody dynamics."""

from __future__ import annotations

from mujoco_orbit._native import load_native_bindings as _load_native_bindings
from mujoco_orbit._native import load_native_plugin as _load_native_plugin

_load_native_plugin()
_load_native_bindings()


from mujoco_orbit.config import (  # noqa: E402
    ControlMomentGyroSpec,
    MagneticBodySpec,
    MagnetorquerSpec,
    OrbitInit,
    ReactionWheelSpec,
    SurfaceSpec,
    ThrusterSpec,
)
from mujoco_orbit.data import MjoData  # noqa: E402
from mujoco_orbit.model import MjoModel  # noqa: E402
from mujoco_orbit.rollout import (  # noqa: E402
    mjo_control_size,
    mjo_get_state,
    mjo_set_state,
    mjo_state_size,
    rollout,
)
from mujoco_orbit.spec import CentralBodySpec, MjoOrbitSpec, MjoSpec  # noqa: E402
from mujoco_orbit.step import mjo_forward, mjo_step  # noqa: E402

__all__ = [
    "CentralBodySpec",
    "ControlMomentGyroSpec",
    "MagneticBodySpec",
    "MagnetorquerSpec",
    "MjoData",
    "MjoModel",
    "MjoOrbitSpec",
    "MjoSpec",
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
