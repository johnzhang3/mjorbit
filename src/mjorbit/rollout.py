# pyright: reportAttributeAccessIssue=false

"""Batched open-loop rollout for the C++-first orbit API."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from mjorbit import _bindings
from mjorbit.config import OrbitInit
from mjorbit.data import MjoData
from mjorbit.model import MjoModel

_DEFAULT_CONTROL_SPEC = mujoco.mjtState.mjSTATE_CTRL.value
_USER_STATE_MASK = mujoco.mjtState.mjSTATE_USER.value
_ROLLOUT_ERROR_MESSAGES = {
    -1: "mjo_rollout received a null model, data, or initial_state pointer",
    -2: "mjo_rollout could not resolve the mjorbit plugin instance",
    -3: "mjo_rollout received an incompatible state size",
    -4: "mjo_rollout received an incompatible control size",
    -5: "mjo_rollout received a negative batch or step count",
}


def mjo_state_size(model: MjoModel) -> int:
    """Return the length of a full ``mjo`` rollout state vector."""
    return int(_bindings.mjo_state_size(model._native))


def mjo_control_size(model: MjoModel, control_spec: int = _DEFAULT_CONTROL_SPEC) -> int:
    """Return the length of one open-loop ``mjo`` control vector."""
    _validate_control_spec(control_spec)
    return int(_bindings.mjo_control_size(model._native, int(control_spec)))


def mjo_get_state(model: MjoModel, data: MjoData, out: np.ndarray | None = None) -> np.ndarray:
    """Pack ``data`` into a full ``mjo`` rollout state vector."""
    state = np.asarray(_bindings.mjo_get_state(model._native, data._native), dtype=np.float64)
    if out is None:
        return state
    out = np.asarray(out, dtype=np.float64)
    if out.shape != state.shape:
        raise ValueError(f"out must have shape {state.shape}, got {out.shape}")
    out[:] = state
    return out


def mjo_set_state(model: MjoModel, data: MjoData, state: ArrayLike) -> None:
    """Unpack a full ``mjo`` rollout state vector into ``data``."""
    state_arr = np.ascontiguousarray(state, dtype=np.float64)
    expected = mjo_state_size(model)
    if state_arr.shape != (expected,):
        raise ValueError(f"state must have shape ({expected},), got {state_arr.shape}")
    _bindings.mjo_set_state(model._native, data._native, state_arr)


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
    nthread: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out batched open-loop trajectories and return states and sensors.

    When ``nthread`` is ``None`` or ``1``, the rollout runs on the calling thread
    using the supplied ``data`` as workspace. When ``nthread > 1``, the function
    allocates ``nthread - 1`` additional ``MjoData`` instances sharing ``model``,
    splits the batch across a ``ThreadPoolExecutor``, and joins the results.
    """
    _validate_control_spec(control_spec)
    if nstep is not None and not isinstance(nstep, int):
        raise ValueError("nstep must be an integer")
    if nthread is not None and (not isinstance(nthread, int) or nthread < 1):
        raise ValueError("nthread must be a positive integer")
    if initial_state is None:
        initial_state = mjo_get_state(model, data)

    _check_must_be_numeric(
        initial_state=initial_state,
        initial_warmstart=initial_warmstart,
        control=control,
        state=state,
        sensordata=sensordata,
    )
    _check_number_of_dimensions(2, initial_state=initial_state, initial_warmstart=initial_warmstart)
    _check_number_of_dimensions(3, control=control, state=state, sensordata=sensordata)

    initial_state_arr = _ensure_2d(initial_state)
    assert initial_state_arr is not None
    initial_warmstart_arr = _ensure_2d(initial_warmstart)
    control_arr = _ensure_3d(control)
    # Output arrays are filled in place by the native rollout. ``_ensure_3d``
    # would silently return a COPY for a non-float64 or non-C-contiguous buffer,
    # leaving the caller's array unfilled; reject those loudly instead.
    for _label, _out in (("state", state), ("sensordata", sensordata)):
        if _out is None:
            continue
        if (
            not isinstance(_out, np.ndarray)
            or _out.dtype != np.float64
            or not _out.flags["C_CONTIGUOUS"]
        ):
            raise ValueError(
                f"{_label} output array must be a float64 C-contiguous numpy array so "
                f"results are written in place; allocate it with "
                f"np.empty((nbatch, nstep, n), dtype=np.float64) or omit it to receive "
                f"a freshly allocated array"
            )
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
    nstep = _infer_dimension(
        1,
        nstep or 1,
        control=control_arr,
        state=state_arr,
        sensordata=sensordata_arr,
    )

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

    initial_state_c = np.ascontiguousarray(initial_state_arr, dtype=np.float64)
    initial_warmstart_c = (
        _empty_array()
        if initial_warmstart_arr is None
        else np.ascontiguousarray(initial_warmstart_arr)
    )
    control_c = (
        _empty_array() if control_arr is None else np.ascontiguousarray(control_arr)
    )

    nworkers = max(1, int(nthread)) if nthread is not None else 1
    nworkers = min(nworkers, int(nbatch))

    if nworkers <= 1:
        result = _bindings.mjo_rollout_native(
            model._native,
            data._native,
            int(nbatch),
            int(nstep),
            int(control_spec),
            int(nstate),
            int(ncontrol),
            initial_state_c,
            initial_warmstart_c,
            control_c,
            state_arr,
            sensordata_arr,
        )
        if result != 0:
            raise RuntimeError(
                _ROLLOUT_ERROR_MESSAGES.get(result, f"mjo_rollout failed: {result}")
            )
        return state_arr, sensordata_arr

    chunks = _split_batches(int(nbatch), nworkers)
    workers = [data] + [_make_worker_data(model) for _ in range(nworkers - 1)]
    nv_eff = int(nv)

    def _run(worker_idx: int, start: int, count: int) -> int:
        if count == 0:
            return 0
        worker = workers[worker_idx]
        is_warmstart = initial_warmstart_arr is not None
        ws = (
            initial_warmstart_c[start : start + count]
            if is_warmstart
            else _empty_array()
        )
        ctrl = (
            control_c[start : start + count] if control_arr is not None else _empty_array()
        )
        return _bindings.mjo_rollout_native(
            model._native,
            worker._native,
            int(count),
            int(nstep),
            int(control_spec),
            int(nstate),
            int(ncontrol),
            np.ascontiguousarray(initial_state_c[start : start + count]),
            np.ascontiguousarray(ws) if ws.size else ws,
            np.ascontiguousarray(ctrl) if ctrl.size else ctrl,
            state_arr[start : start + count],
            sensordata_arr[start : start + count],
        )

    _ = nv_eff  # nv kept for trailing-dim checks above
    futures = []
    with ThreadPoolExecutor(max_workers=nworkers) as pool:
        offset = 0
        for worker_idx, count in enumerate(chunks):
            futures.append(pool.submit(_run, worker_idx, offset, count))
            offset += count
        for fut in futures:
            result = fut.result()
            if result != 0:
                raise RuntimeError(
                    _ROLLOUT_ERROR_MESSAGES.get(result, f"mjo_rollout failed: {result}")
                )
    return state_arr, sensordata_arr


def _split_batches(nbatch: int, nworkers: int) -> list[int]:
    base, extra = divmod(nbatch, nworkers)
    return [base + (1 if i < extra else 0) for i in range(nworkers)]


def _make_worker_data(model: MjoModel) -> MjoData:
    """Allocate a fresh ``MjoData`` workspace for a rollout worker thread."""
    return MjoData(model, orbit=OrbitInit(R_eci=np.zeros(3), V_eci=np.zeros(3), t=0.0))


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
        if value is not None and np.asarray(value).ndim > ndim:
            raise ValueError(f"{key} can have at most {ndim} dimensions")


def _check_trailing_dimension(dim: int, **kwargs: np.ndarray | None) -> None:
    for key, value in kwargs.items():
        if value is not None and value.shape[-1] != dim:
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


def _empty_array() -> np.ndarray:
    return np.empty((0,), dtype=np.float64)


__all__ = [
    "mjo_control_size",
    "mjo_get_state",
    "mjo_set_state",
    "mjo_state_size",
    "rollout",
]
