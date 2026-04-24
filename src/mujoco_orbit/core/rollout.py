# pyright: reportAttributeAccessIssue=false

"""Batched open-loop rollout for the MuJoCo-style orbit API."""

from __future__ import annotations

import ctypes
from typing import Any

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from mujoco_orbit._native import load_native_library
from mujoco_orbit.core.runtime import MjoData, MjoModel

_FULLPHYSICS = mujoco.mjtState.mjSTATE_FULLPHYSICS.value
_DEFAULT_CONTROL_SPEC = mujoco.mjtState.mjSTATE_CTRL.value
_USER_STATE_MASK = mujoco.mjtState.mjSTATE_USER.value
_DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)
_ROLLOUT_ERROR_MESSAGES = {
    -1: "mjo_rollout received a null model, data, or initial_state pointer",
    -2: "mjo_rollout could not resolve the mujoco_orbit plugin instance",
    -3: "mjo_rollout received an incompatible state size",
    -4: "mjo_rollout received an incompatible control size",
    -5: "mjo_rollout received a negative batch or step count",
}


def _mj_model(model: MjoModel) -> mujoco.MjModel:
    return model.mj_model


def _full_state_size(model: MjoModel) -> int:
    return int(mujoco.mj_stateSize(_mj_model(model), _FULLPHYSICS))


def _control_prefix_size(model: MjoModel, control_spec: int) -> int:
    return int(mujoco.mj_stateSize(_mj_model(model), int(control_spec)))


def _mjo_state_tail_size(model: MjoModel) -> int:
    # R_eci[3], V_eci[3], t, rw_speed, cmg_gimbal_angle, cmg_rotor_momentum.
    return 7 + len(model.reaction_wheels) + 2 * len(model.cmgs)


def _mjo_control_tail_size(model: MjoModel) -> int:
    # rw_torque_cmd, mtq_dipole_cmd, thr_force_cmd, cmg_gimbal_rate_cmd.
    return (
        len(model.reaction_wheels)
        + len(model.magnetorquers)
        + len(model.thrusters)
        + len(model.cmgs)
    )


def mjo_state_size(model: MjoModel) -> int:
    """Return the length of a full ``mjo`` rollout state vector.

    The vector starts with MuJoCo ``mjSTATE_FULLPHYSICS`` and appends the
    chief orbit ``R_eci, V_eci, t`` plus external actuator runtime state.
    """
    return _full_state_size(model) + _mjo_state_tail_size(model)


def mjo_control_size(
    model: MjoModel,
    control_spec: int = _DEFAULT_CONTROL_SPEC,
) -> int:
    """Return the length of one open-loop ``mjo`` control vector."""
    _validate_control_spec(control_spec)
    return _control_prefix_size(model, control_spec) + _mjo_control_tail_size(model)


def mjo_get_state(
    model: MjoModel,
    data: MjoData,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Pack ``data`` into a full ``mjo`` rollout state vector."""
    size = mjo_state_size(model)
    if out is None:
        out = np.empty(size, dtype=np.float64)
    else:
        out = np.asarray(out, dtype=np.float64)
        if out.shape != (size,):
            raise ValueError(f"out must have shape ({size},), got {out.shape}")

    nfull = _full_state_size(model)
    mujoco.mj_getState(model.mj_model, data.mj_data, out[:nfull], _FULLPHYSICS)
    idx = nfull
    out[idx : idx + 3] = data.orbit.R_eci
    idx += 3
    out[idx : idx + 3] = data.orbit.V_eci
    idx += 3
    out[idx] = data.orbit.t
    idx += 1
    nrw = len(model.reaction_wheels)
    out[idx : idx + nrw] = data.actuators.rw_speed
    idx += nrw
    ncmg = len(model.cmgs)
    out[idx : idx + ncmg] = data.actuators.cmg_gimbal_angle
    idx += ncmg
    out[idx : idx + ncmg] = data.actuators.cmg_rotor_momentum
    return out


def mjo_set_state(model: MjoModel, data: MjoData, state: ArrayLike) -> None:
    """Unpack a full ``mjo`` rollout state vector into ``data``."""
    state_arr = np.asarray(state, dtype=np.float64)
    size = mjo_state_size(model)
    if state_arr.shape != (size,):
        raise ValueError(f"state must have shape ({size},), got {state_arr.shape}")

    nfull = _full_state_size(model)
    mujoco.mj_setState(model.mj_model, data.mj_data, state_arr[:nfull], _FULLPHYSICS)
    idx = nfull
    data.orbit.R_eci[:] = state_arr[idx : idx + 3]
    idx += 3
    data.orbit.V_eci[:] = state_arr[idx : idx + 3]
    idx += 3
    data.orbit.t = float(state_arr[idx])
    idx += 1
    nrw = len(model.reaction_wheels)
    data.actuators.rw_speed[:] = state_arr[idx : idx + nrw]
    idx += nrw
    ncmg = len(model.cmgs)
    data.actuators.cmg_gimbal_angle[:] = state_arr[idx : idx + ncmg]
    idx += ncmg
    data.actuators.cmg_rotor_momentum[:] = state_arr[idx : idx + ncmg]
    data.actuators.update_rw_momentum(model.rw_inertia)


def rollout(
    model: MjoModel,
    data: MjoData,
    initial_state: ArrayLike | None = None,
    control: ArrayLike | None = None,
    *,
    control_spec: int = _DEFAULT_CONTROL_SPEC,
    nstep: int | None = None,
    initial_warmstart: ArrayLike | None = None,
    state: ArrayLike | None = None,
    sensordata: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out batched open-loop trajectories and return states and sensors.

    ``initial_state`` has shape ``([nbatch or 1], mjo_state_size(model))``.
    ``control`` has shape
    ``([nbatch or 1], [nstep or 1], mjo_control_size(model, control_spec))``.
    Singleton batch and time dimensions are expanded like MuJoCo's rollout.
    """
    _validate_control_spec(control_spec)
    if nstep is not None and not isinstance(nstep, int):
        raise ValueError("nstep must be an integer")

    if initial_state is None:
        initial_state = mjo_get_state(model, data)

    _check_must_be_numeric(
        initial_state=initial_state,
        initial_warmstart=initial_warmstart,
        control=control,
        state=state,
        sensordata=sensordata,
    )
    _check_number_of_dimensions(
        2, initial_state=initial_state, initial_warmstart=initial_warmstart
    )
    _check_number_of_dimensions(3, control=control, state=state, sensordata=sensordata)

    initial_state_arr = _ensure_2d(initial_state)
    assert initial_state_arr is not None
    initial_warmstart_arr = _ensure_2d(initial_warmstart)
    control_arr = _ensure_3d(control)
    state_arr = _ensure_3d(state)
    sensordata_arr = _ensure_3d(sensordata)

    nbatch = _infer_dimension(
        0,
        1,
        initial_state=initial_state_arr,
        initial_warmstart=initial_warmstart_arr,
        control=control_arr,
        state=state_arr,
        sensordata=sensordata_arr,
    )
    inferred_nstep = _infer_dimension(
        1,
        nstep or 1,
        control=control_arr,
        state=state_arr,
        sensordata=sensordata_arr,
    )
    nstep = inferred_nstep

    nstate = mjo_state_size(model)
    ncontrol = mjo_control_size(model, control_spec)
    nsensordata = int(model.nsensordata)
    nv = int(model.nv)

    _check_trailing_dimension(nstate, initial_state=initial_state_arr, state=state_arr)
    _check_trailing_dimension(ncontrol, control=control_arr)
    _check_trailing_dimension(nv, initial_warmstart=initial_warmstart_arr)
    _check_trailing_dimension(nsensordata, sensordata=sensordata_arr)

    initial_state_arr = _tile_if_required(initial_state_arr, nbatch)
    assert initial_state_arr is not None
    initial_warmstart_arr = _tile_if_required(initial_warmstart_arr, nbatch)
    control_arr = _tile_if_required(control_arr, nbatch, nstep)

    if state_arr is None:
        state_arr = np.empty((nbatch, nstep, nstate), dtype=np.float64)
    if sensordata_arr is None:
        sensordata_arr = np.empty((nbatch, nstep, nsensordata), dtype=np.float64)

    _native_rollout(
        model,
        data,
        nbatch=nbatch,
        nstep=nstep,
        control_spec=control_spec,
        initial_state=initial_state_arr,
        initial_warmstart=initial_warmstart_arr,
        control=control_arr,
        state=state_arr,
        sensordata=sensordata_arr,
    )
    return state_arr, sensordata_arr


def _validate_control_spec(control_spec: int) -> None:
    if int(control_spec) & ~_USER_STATE_MASK:
        raise ValueError("control_spec can only contain bits in mjSTATE_USER")


def _check_must_be_numeric(**kwargs: Any) -> None:
    for key, value in kwargs.items():
        if value is None:
            continue
        try:
            np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be numeric") from exc


def _check_number_of_dimensions(ndim: int, **kwargs: Any) -> None:
    for key, value in kwargs.items():
        if value is None:
            continue
        if np.asarray(value).ndim > ndim:
            raise ValueError(f"{key} can have at most {ndim} dimensions")


def _check_trailing_dimension(dim: int, **kwargs: np.ndarray | None) -> None:
    for key, value in kwargs.items():
        if value is None:
            continue
        if value.shape[-1] != dim:
            raise ValueError(
                f"trailing dimension of {key} must be {dim}, got {value.shape[-1]}"
            )


def _ensure_2d(arg: ArrayLike | None) -> np.ndarray | None:
    if arg is None:
        return None
    return np.ascontiguousarray(np.atleast_2d(arg), dtype=np.float64)


def _ensure_3d(arg: ArrayLike | None) -> np.ndarray | None:
    if arg is None:
        return None
    arr = np.asarray(arg)
    if arr.ndim == 0:
        arr = arr[np.newaxis, np.newaxis, np.newaxis, ...]
    elif arr.ndim == 1:
        arr = arr[np.newaxis, np.newaxis, ...]
    elif arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    return np.ascontiguousarray(arr, dtype=np.float64)


def _infer_dimension(dim: int, value: int, **kwargs: np.ndarray | None) -> int:
    for name, array in kwargs.items():
        if array is None:
            continue
        if array.shape[dim] != value:
            if value == 1:
                value = int(array.shape[dim])
            elif array.shape[dim] != 1:
                raise ValueError(
                    f"dimension {dim} inferred as {value} but {name} has {array.shape[dim]}"
                )
    return value


def _tile_if_required(
    array: np.ndarray | None,
    dim0: int,
    dim1: int | None = None,
) -> np.ndarray | None:
    if array is None:
        return None
    reps = [1] * array.ndim
    if array.shape[0] == 1:
        reps[0] = dim0
    if dim1 is not None and array.shape[1] == 1:
        reps[1] = dim1
    return np.ascontiguousarray(np.tile(array, tuple(reps)), dtype=np.float64)


def _as_double_ptr(array: np.ndarray | None):
    if array is None:
        return None
    return array.ctypes.data_as(_DOUBLE_PTR)


def _native_rollout(
    model: MjoModel,
    data: MjoData,
    *,
    nbatch: int,
    nstep: int,
    control_spec: int,
    initial_state: np.ndarray,
    initial_warmstart: np.ndarray | None,
    control: np.ndarray | None,
    state: np.ndarray,
    sensordata: np.ndarray,
) -> None:
    lib = load_native_library()
    fn = lib.mjo_rollout
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_int,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
    ]
    fn.restype = ctypes.c_int

    result = fn(
        ctypes.c_void_p(int(model.mj_model._address)),
        ctypes.c_void_p(int(data.mj_data._address)),
        ctypes.c_int(int(model.orbit_plugin_instance)),
        ctypes.c_int(int(nbatch)),
        ctypes.c_int(int(nstep)),
        ctypes.c_uint(int(control_spec)),
        ctypes.c_int(int(mjo_state_size(model))),
        ctypes.c_int(int(mjo_control_size(model, control_spec))),
        _as_double_ptr(initial_state),
        _as_double_ptr(initial_warmstart),
        _as_double_ptr(control),
        _as_double_ptr(state),
        _as_double_ptr(sensordata),
    )
    if result != 0:
        message = _ROLLOUT_ERROR_MESSAGES.get(result, f"mjo_rollout failed with code {result}")
        raise RuntimeError(message)


__all__ = [
    "mjo_control_size",
    "mjo_get_state",
    "mjo_set_state",
    "mjo_state_size",
    "rollout",
]
