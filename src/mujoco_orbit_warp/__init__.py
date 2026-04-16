# pyright: reportUnsupportedDunderAll=false

"""MJWarp-backed coupled orbital and multibody dynamics.

The package exposes the orbit-aware ``MjoModel`` / ``MjoData`` API and lazily
adapts the standard ``mujoco_warp`` public functions.  Users can therefore use
``mujoco_orbit_warp.step(model, data)`` like MJWarp; when ``model`` and ``data``
are orbit wrappers, the coupled orbital step is used.
"""

from __future__ import annotations

from typing import Any

from mujoco_orbit.core.config import (
    MagneticBodySpec,
    MagnetorquerSpec,
    OrbitInit,
    ReactionWheelSpec,
    SurfaceSpec,
    ThrusterSpec,
)

from ._deps import require_mjwarp
from .runtime import MjoData, MjoModel
from .step import mjo_forward, mjo_step

_MJWARP_FACTORY_NAMES = {
    "create_render_context",
    "get_data_into",
    "make_data",
    "put_data",
    "put_model",
}

_MJWARP_SYNCED_MODEL_DATA_NAMES = {
    "camlight",
    "collision",
    "com_pos",
    "com_vel",
    "contact_force",
    "crb",
    "deriv_smooth_vel",
    "energy_pos",
    "energy_vel",
    "euler",
    "factor_m",
    "flex",
    "forward",
    "fwd_acceleration",
    "fwd_actuation",
    "fwd_position",
    "fwd_velocity",
    "get_state",
    "implicit",
    "inverse",
    "island",
    "jac",
    "kinematics",
    "make_constraint",
    "mul_m",
    "nxn_broadphase",
    "passive",
    "primitive_narrowphase",
    "ray",
    "rays",
    "refit_bvh",
    "render",
    "reset_data",
    "rne",
    "rne_postconstraint",
    "rungekutta4",
    "sap_broadphase",
    "sdf_narrowphase",
    "sensor_acc",
    "sensor_pos",
    "sensor_vel",
    "set_const",
    "set_const_0",
    "set_const_fixed",
    "set_length_range",
    "solve",
    "solve_m",
    "step",
    "step1",
    "step2",
    "subtree_vel",
    "tendon",
    "transmission",
    "xfrc_accumulate",
}

_MJWARP_PUBLIC_NAMES = _MJWARP_FACTORY_NAMES | _MJWARP_SYNCED_MODEL_DATA_NAMES | {
    "BiasType",
    "BroadphaseFilter",
    "BroadphaseType",
    "ConeType",
    "Constraint",
    "Contact",
    "Data",
    "DisableBit",
    "DynType",
    "EnableBit",
    "GainType",
    "GeomType",
    "IntegratorType",
    "JointType",
    "Model",
    "Option",
    "RenderContext",
    "SolverType",
    "State",
    "Statistic",
    "TrnType",
    "get_depth",
    "get_rgb",
    "metadata",
}


def _is_wrapped_pair(args: tuple[Any, ...]) -> bool:
    return len(args) >= 2 and isinstance(args[0], MjoModel) and isinstance(args[1], MjoData)


def _unwrap_device_arg(value: Any) -> Any:
    if isinstance(value, MjoModel):
        return value.warp_model
    if isinstance(value, MjoData):
        return value.warp_data
    return value


def _unwrap_host_model_arg(value: Any) -> Any:
    if isinstance(value, MjoModel):
        return value.mj_model
    return value


def _unwrap_host_data_arg(value: Any) -> Any:
    if isinstance(value, MjoData):
        return value.mj_data
    return value


def _unwrap_kwargs(kwargs: dict[str, Any], unwrap) -> dict[str, Any]:
    return {key: unwrap(value) for key, value in kwargs.items()}


def _sync_device_from_wrapper(model: MjoModel, data: MjoData) -> None:
    from .step import _sync_device_from_public

    data._push_to_host()
    data._pull_from_host()
    _sync_device_from_public(model, data)


def _pull_device_into_wrapper(model: MjoModel, data: MjoData) -> None:
    mjw, _ = require_mjwarp()
    for world_id, run in enumerate(data._host_runs):
        mjw.get_data_into(run.mj_data, model.mj_model, data.warp_data, world_id=world_id)
    data._pull_from_host()


def _call_mjwarp(name: str, *args: Any, **kwargs: Any) -> Any:
    mjw, _ = require_mjwarp()
    fn = getattr(mjw, name)

    if name in _MJWARP_SYNCED_MODEL_DATA_NAMES and _is_wrapped_pair(args):
        model = args[0]
        data = args[1]
        _sync_device_from_wrapper(model, data)
        result = fn(
            model.warp_model,
            data.warp_data,
            *(_unwrap_device_arg(arg) for arg in args[2:]),
            **_unwrap_kwargs(kwargs, _unwrap_device_arg),
        )
        _pull_device_into_wrapper(model, data)
        return result

    return fn(
        *(_unwrap_device_arg(arg) for arg in args),
        **_unwrap_kwargs(kwargs, _unwrap_device_arg),
    )


def forward(model: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
    """Run MJWarp ``forward`` or the orbit-aware wrapper forward."""
    if isinstance(model, MjoModel) and isinstance(data, MjoData):
        if args or kwargs:
            raise TypeError("orbit-aware forward accepts only (model, data)")
        return mjo_forward(model, data)
    return _call_mjwarp("forward", model, data, *args, **kwargs)


def step(model: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
    """Run MJWarp ``step`` or the orbit-aware wrapper step."""
    if isinstance(model, MjoModel) and isinstance(data, MjoData):
        if args or kwargs:
            raise TypeError("orbit-aware step accepts only (model, data)")
        return mjo_step(model, data)
    return _call_mjwarp("step", model, data, *args, **kwargs)


def put_model(model: Any) -> Any:
    """Return an MJWarp device model for a host or orbit wrapper model."""
    if isinstance(model, MjoModel):
        return model.warp_model
    mjw, _ = require_mjwarp()
    return mjw.put_model(model)


def make_data(model: Any, *args: Any, **kwargs: Any) -> Any:
    """Create data using MJWarp semantics, or orbit wrapper data when ``orbit`` is supplied."""
    if isinstance(model, MjoModel):
        orbit = kwargs.pop("orbit", None)
        rng_seed = kwargs.pop("rng_seed", None)
        if orbit is not None:
            return model.make_data(*args, orbit=orbit, rng_seed=rng_seed, **kwargs)
        mjw, _ = require_mjwarp()
        return mjw.make_data(model.mj_model, *args, **kwargs)

    mjw, _ = require_mjwarp()
    return mjw.make_data(model, *args, **kwargs)


def put_data(model: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
    """Upload host data to MJWarp, accepting orbit wrapper arguments when provided."""
    mjw, _ = require_mjwarp()
    return mjw.put_data(
        _unwrap_host_model_arg(model),
        _unwrap_host_data_arg(data),
        *args,
        **kwargs,
    )


def get_data_into(result: Any, model: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
    """Download MJWarp data into a host ``mujoco.MjData``."""
    mjw, _ = require_mjwarp()
    return mjw.get_data_into(
        _unwrap_host_data_arg(result),
        _unwrap_host_model_arg(model),
        _unwrap_device_arg(data),
        *args,
        **kwargs,
    )


def create_render_context(model: Any, *args: Any, **kwargs: Any) -> Any:
    """Create a standard MJWarp render context from a host or orbit wrapper model."""
    mjw, _ = require_mjwarp()
    return mjw.create_render_context(_unwrap_host_model_arg(model), *args, **kwargs)


def __getattr__(name: str) -> Any:
    if name not in _MJWARP_PUBLIC_NAMES:
        raise AttributeError(f"module 'mujoco_orbit_warp' has no attribute {name!r}")

    mjw, _ = require_mjwarp()
    obj = getattr(mjw, name)
    if callable(obj) and name in _MJWARP_SYNCED_MODEL_DATA_NAMES:
        return lambda *args, **kwargs: _call_mjwarp(name, *args, **kwargs)
    return obj

__all__ = [
    "BiasType",
    "BroadphaseFilter",
    "BroadphaseType",
    "ConeType",
    "Constraint",
    "Contact",
    "Data",
    "DisableBit",
    "DynType",
    "EnableBit",
    "GainType",
    "GeomType",
    "IntegratorType",
    "JointType",
    "MagneticBodySpec",
    "MagnetorquerSpec",
    "Model",
    "MjoData",
    "MjoModel",
    "OrbitInit",
    "Option",
    "ReactionWheelSpec",
    "RenderContext",
    "SolverType",
    "State",
    "Statistic",
    "SurfaceSpec",
    "ThrusterSpec",
    "TrnType",
    "camlight",
    "collision",
    "com_pos",
    "com_vel",
    "contact_force",
    "crb",
    "create_render_context",
    "deriv_smooth_vel",
    "energy_pos",
    "energy_vel",
    "euler",
    "factor_m",
    "flex",
    "forward",
    "fwd_acceleration",
    "fwd_actuation",
    "fwd_position",
    "fwd_velocity",
    "get_data_into",
    "get_depth",
    "get_rgb",
    "get_state",
    "implicit",
    "inverse",
    "island",
    "jac",
    "kinematics",
    "make_constraint",
    "make_data",
    "metadata",
    "mul_m",
    "mjo_forward",
    "mjo_step",
    "nxn_broadphase",
    "passive",
    "primitive_narrowphase",
    "put_data",
    "put_model",
    "ray",
    "rays",
    "refit_bvh",
    "render",
    "reset_data",
    "rne",
    "rne_postconstraint",
    "rungekutta4",
    "sap_broadphase",
    "sdf_narrowphase",
    "sensor_acc",
    "sensor_pos",
    "sensor_vel",
    "set_const",
    "set_const_0",
    "set_const_fixed",
    "set_length_range",
    "solve",
    "solve_m",
    "step",
    "step1",
    "step2",
    "subtree_vel",
    "tendon",
    "transmission",
    "xfrc_accumulate",
]
