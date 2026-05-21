# pyright: reportAttributeAccessIssue=false

"""Per-world CPU mirror that bridges the orbit-aware MJWarp data with raw MuJoCo.

``_HostRun`` keeps a CPU ``MjoData`` and ``mujoco.MjData`` in sync so the
existing CPU sensor stack and contact/inertia helpers can read the latest
state. Forward/step on the GPU writes to the device buffers; pulling for
analysis (or full host-side compatibility mode) syncs through here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mujoco_orbit.config import OrbitInit
from mujoco_orbit.data import MjoData as CpuMjoData
from mujoco_orbit.step import mjo_forward as cpu_mjo_forward

from .state import BatchedActuatorData

if TYPE_CHECKING:
    from .model import MjoModel


def _copy_world_value(
    target: np.ndarray,
    world_id: int,
    source: np.ndarray,
    *,
    nworld: int,
) -> None:
    if nworld == 1:
        np.copyto(target, source)
        return
    np.copyto(target[world_id], source)

def _copy_host_value(target: np.ndarray, source: np.ndarray) -> None:
    np.copyto(target, source)

def _copy_raw_to_native_data(raw_data: mujoco.MjData, native_data: CpuMjoData) -> None:
    for field in ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied"):
        target = getattr(native_data, field, None)
        source = getattr(raw_data, field, None)
        if target is not None and source is not None and np.shape(target) == np.shape(source):
            np.copyto(target, source)

def _copy_native_to_raw_data(native_data: CpuMjoData, raw_data: mujoco.MjData) -> None:
    for field in ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied"):
        target = getattr(raw_data, field, None)
        source = getattr(native_data, field, None)
        if target is not None and source is not None and np.shape(target) == np.shape(source):
            np.copyto(target, source)

def _world_view(array: np.ndarray, world_id: int, *, nworld: int) -> np.ndarray:
    return array if nworld == 1 else array[world_id]

@dataclass
class _HostRun:
    """Single-world host shadow used for MuJoCo interop and sensor readback."""

    native_data: CpuMjoData
    mj_data: mujoco.MjData
    actuators: BatchedActuatorData
    wrench_buffer: np.ndarray

    @classmethod
    def create(
        cls,
        model: "MjoModel",
        *,
        orbit: OrbitInit,
        rng_seed: int | None,
    ) -> "_HostRun":
        native_data = CpuMjoData(model.host_model, orbit=orbit, rng_seed=rng_seed)
        mj_data = mujoco.MjData(model.mj_model)
        run = cls(
            native_data=native_data,
            mj_data=mj_data,
            actuators=BatchedActuatorData.zeros(
                1,
                len(model.reaction_wheels),
                model.rw_inertia,
                len(model.magnetorquers),
                len(model.thrusters),
            ),
            wrench_buffer=np.zeros((model.nbody, 6), dtype=float),
        )
        run.refresh_native(model)
        return run

    @property
    def orbit(self):
        return self.native_data.orbit

    @property
    def frame(self):
        return self.native_data.frame

    @property
    def env(self):
        return self.native_data.env

    @property
    def sensors(self):
        return self.native_data.sensors

    def __getattr__(self, name: str):  # pragma: no cover - trivial delegation
        return getattr(self.mj_data, name)

    def clear_wrench_buffer(self) -> None:
        self.wrench_buffer[:] = 0.0
        self.mj_data.xfrc_applied[:] = 0.0
        self.native_data.clear_wrench_buffer()

    def copy_core_to_native(self) -> None:
        np.copyto(self.native_data.orbit.R_eci, self.orbit.R_eci)
        np.copyto(self.native_data.orbit.V_eci, self.orbit.V_eci)
        self.native_data.orbit.t = float(self.orbit.t)
        np.copyto(self.native_data.actuators.rw_speed, self.actuators.rw_speed)
        np.copyto(self.native_data.actuators.rw_torque_cmd, self.actuators.rw_torque_cmd)
        np.copyto(self.native_data.actuators.mtq_dipole_cmd, self.actuators.mtq_dipole_cmd)
        np.copyto(self.native_data.actuators.thr_force_cmd, self.actuators.thr_force_cmd)
        np.copyto(self.native_data.wrench_buffer, self.wrench_buffer)

    def refresh_native(self, model: "MjoModel") -> None:
        _copy_raw_to_native_data(self.mj_data, self.native_data)
        self.copy_core_to_native()
        cpu_mjo_forward(model.host_model, self.native_data)
        np.copyto(self.mj_data.sensordata, self.native_data.sensordata)


__all__: list[str] = []
