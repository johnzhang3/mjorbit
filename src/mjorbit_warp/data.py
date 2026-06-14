# pyright: reportAttributeAccessIssue=false, reportIndexIssue=false

"""``MjoData`` — orbit-aware MJWarp runtime state for one or more worlds.

Owns:
  * Device-side buffers (``core_data``, ``warp_data``)
  * Per-world CPU host shadows (``_host_runs``)
  * Public NumPy views into all mirrored MJWarp fields
  * Public state dataclasses (``orbit``, ``frame``, ``env``, ``actuators``)

Step-time orchestration (``mjo_forward``, ``mjo_step``, ``mjo_upload``,
``mjo_pull``) is split into the ``step`` package; this module only owns the
data structure itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import mujoco
import numpy as np

from mjorbit.config import OrbitInit
from mjorbit.data import MjoData as CpuMjoData

from ._deps import require_mjwarp
from .field_registry import (
    _HOST_MUTABLE_FIELDS,
    _MIRRORED_ARRAY_FIELDS,
    _MIRRORED_SCALAR_FIELDS,
)
from .host_shadow import (
    _copy_host_value,
    _copy_native_to_raw_data,
    _copy_world_value,
    _HostRun,
    _world_view,
)
from .model import MjoModel
from .state import (
    BatchedActuatorData,
    EnvironmentBatchCache,
    FrameBatchCache,
    OrbitBatchState,
    _zeros_world,
)


def _normalize_orbit_inits(
    orbit: OrbitInit | Sequence[OrbitInit],
    *,
    nworld: int,
) -> list[OrbitInit]:
    if isinstance(orbit, OrbitInit):
        return [
            OrbitInit(R_eci=orbit.R_eci.copy(), V_eci=orbit.V_eci.copy(), t=orbit.t)
            for _ in range(nworld)
        ]

    orbit_list = list(orbit)
    if len(orbit_list) != nworld:
        raise ValueError(f"Expected {nworld} orbit initializations, got {len(orbit_list)}")

    return [
        OrbitInit(R_eci=init.R_eci.copy(), V_eci=init.V_eci.copy(), t=init.t)
        for init in orbit_list
    ]

def _normalize_rng_seeds(
    rng_seed: int | Sequence[int | None] | None,
    *,
    nworld: int,
) -> list[int | None]:
    if rng_seed is None or isinstance(rng_seed, int):
        return [rng_seed for _ in range(nworld)]

    seed_list = list(rng_seed)
    if len(seed_list) != nworld:
        raise ValueError(f"Expected {nworld} RNG seeds, got {len(seed_list)}")
    return seed_list

class MjoData:
    """Runtime state for one or more MJWarp worlds."""

    backend = "warp"

    def __init__(
        self,
        model: MjoModel,
        *,
        orbit: OrbitInit | Sequence[OrbitInit],
        rng_seed: int | Sequence[int | None] | None = None,
        nworld: int = 1,
        nconmax: int | None = None,
        nccdmax: int | None = None,
        njmax: int | None = None,
        naconmax: int | None = None,
        naccdmax: int | None = None,
    ) -> None:
        if nworld < 1:
            raise ValueError("nworld must be >= 1")

        mjw, _ = require_mjwarp()

        self.model = model
        self.nworld = nworld

        orbit_inits = _normalize_orbit_inits(orbit, nworld=nworld)
        seeds = _normalize_rng_seeds(rng_seed, nworld=nworld)
        self._host_runs = [
            _HostRun.create(model, orbit=orbit_init, rng_seed=seed)
            for orbit_init, seed in zip(orbit_inits, seeds, strict=True)
        ]

        if nworld == 1:
            self.warp_data = mjw.put_data(
                model.mj_model,
                self._host_runs[0].mj_data,
                nworld=1,
                nconmax=nconmax,
                nccdmax=nccdmax,
                njmax=njmax,
                naconmax=naconmax,
                naccdmax=naccdmax,
            )
        else:
            self.warp_data = mjw.make_data(
                model.mj_model,
                nworld=nworld,
                nconmax=nconmax,
                nccdmax=nccdmax,
                njmax=njmax,
                naconmax=naconmax,
                naccdmax=naccdmax,
            )

        first = self._host_runs[0].mj_data
        for field in _MIRRORED_ARRAY_FIELDS:
            value = getattr(first, field)
            setattr(self, field, _zeros_world(nworld, *value.shape, dtype=value.dtype))

        self.time = 0.0 if nworld == 1 else np.zeros(nworld, dtype=float)
        for field in _MIRRORED_SCALAR_FIELDS:
            setattr(
                self,
                field,
                int(getattr(first, field)) if nworld == 1 else np.zeros(nworld, dtype=np.int32),
            )
        self.orbit = OrbitBatchState(
            R_eci=_zeros_world(nworld, 3),
            V_eci=_zeros_world(nworld, 3),
            t=0.0 if nworld == 1 else np.zeros(nworld, dtype=float),
        )
        self.frame = FrameBatchCache(
            C_LI=_zeros_world(nworld, 3, 3),
            C_IL=_zeros_world(nworld, 3, 3),
            omega_lvlh=_zeros_world(nworld, 3),
            omega_dot_lvlh=_zeros_world(nworld, 3),
        )
        self.env = EnvironmentBatchCache(
            sun_vector_eci=_zeros_world(nworld, 3),
            eclipse=0.0 if nworld == 1 else np.zeros(nworld, dtype=float),
            mag_field_eci=_zeros_world(nworld, 3),
            atmosphere_omega_eci=_zeros_world(nworld, 3),
            atm_density=0.0 if nworld == 1 else np.zeros(nworld, dtype=float),
        )
        self.actuators = BatchedActuatorData.zeros(
            nworld,
            len(model.reaction_wheels),
            model.rw_inertia,
            len(model.magnetorquers),
            len(model.thrusters),
        )
        self.wrench_buffer = _zeros_world(nworld, model.nbody, 6)
        # TODO(sensors): re-attach a GPU-native sensors namespace here.
        from .core_gpu import make_device_core_data

        self.core_data = make_device_core_data(orbit_inits, nworld=nworld, model=model.host_model)

        self._pull_from_host()

        from mjorbit_warp.step import mjo_forward

        mjo_forward(model, self)

    @classmethod
    def from_host_data(
        cls,
        model: MjoModel,
        host_data: CpuMjoData,
        *,
        nworld: int = 1,
        nconmax: int | None = None,
        nccdmax: int | None = None,
        njmax: int | None = None,
        naconmax: int | None = None,
        naccdmax: int | None = None,
    ) -> "MjoData":
        """Upload a CPU mjorbit data object into the orbit-aware MJWarp wrapper."""
        if nworld < 1:
            raise ValueError("nworld must be >= 1")

        orbit = OrbitInit(
            R_eci=host_data.orbit.R_eci.copy(),
            V_eci=host_data.orbit.V_eci.copy(),
            t=host_data.orbit.t,
        )
        data = cls(
            model,
            orbit=orbit,
            nworld=nworld,
            nconmax=nconmax,
            nccdmax=nccdmax,
            njmax=njmax,
            naconmax=naconmax,
            naccdmax=naccdmax,
        )

        for run in data._host_runs:
            _copy_native_to_raw_data(host_data, run.mj_data)
            np.copyto(run.orbit.R_eci, host_data.orbit.R_eci)
            np.copyto(run.orbit.V_eci, host_data.orbit.V_eci)
            run.orbit.t = float(host_data.orbit.t)
            np.copyto(run.actuators.rw_speed, host_data.actuators.rw_speed)
            np.copyto(run.actuators.rw_torque_cmd, host_data.actuators.rw_torque_cmd)
            np.copyto(run.actuators.mtq_dipole_cmd, host_data.actuators.mtq_dipole_cmd)
            np.copyto(run.actuators.thr_force_cmd, host_data.actuators.thr_force_cmd)
            np.copyto(run.wrench_buffer, host_data.wrench_buffer)
            run.actuators.update_rw_momentum(model.rw_inertia)
            run.refresh_native(model)

        data._pull_from_host()
        data.upload(fields=("state", "inputs", "core"))
        return data

    @property
    def mj_data(self) -> mujoco.MjData:
        """Return the sole host shadow ``MjData`` for single-world use."""
        if self.nworld != 1:
            raise AttributeError(
                "Batched warp data has multiple host shadows. Use `host_data(world_id)` instead."
            )
        return self._host_runs[0].mj_data

    def __getattr__(self, name: str):  # pragma: no cover - trivial delegation
        if self.nworld == 1:
            return getattr(self._host_runs[0].mj_data, name)
        raise AttributeError(
            f"'MjoData' has no attribute '{name}'. "
            "Use explicit mirrored buffers or `host_data(world_id)` for batched access."
        )

    def host_data(self, world_id: int = 0) -> mujoco.MjData:
        """Return one host shadow ``MjData`` instance."""
        return self._host_runs[world_id].mj_data

    @property
    def device_data(self) -> Any:
        """Return the underlying ``mujoco_warp.Data``."""
        return self.warp_data

    @property
    def device_core_data(self) -> Any:
        """Return the underlying orbit/coupling device state."""
        return self.core_data

    def pull(self, fields: Iterable[str] | str | None = None) -> None:
        """Refresh public buffers from device state.

        Passing ``fields`` performs a selective public-buffer readback.  Leaving
        it as ``None`` performs the full compatibility pull into host shadows.
        """
        from mjorbit_warp.step import mjo_pull

        mjo_pull(self.model, self, fields=fields)

    def upload(self, fields: Iterable[str] | str | None = None) -> None:
        """Upload explicitly selected public input buffers to device state."""
        from mjorbit_warp.step import mjo_upload

        mjo_upload(self.model, self, fields=fields)

    def clear_wrench_buffer(self) -> None:
        """Reset the assembled external wrench buffer and applied wrench."""
        self.wrench_buffer[:] = 0.0
        self.xfrc_applied[:] = 0.0
        for run in self._host_runs:
            run.clear_wrench_buffer()

    def contact_force(
        self,
        contact_id: int,
        *,
        world_id: int = 0,
        to_world_frame: bool = False,
    ) -> np.ndarray:
        """Return one contact wrench from the host shadow data."""
        mjd = self.host_data(world_id)
        if contact_id < 0 or contact_id >= mjd.ncon:
            raise IndexError(f"contact_id {contact_id} out of range for world {world_id}")

        force = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(self.model.mj_model, mjd, contact_id, force)
        if not to_world_frame:
            return force

        frame = mjd.contact[contact_id].frame.reshape(3, 3)
        world_force = np.zeros(6, dtype=float)
        world_force[:3] = frame @ force[:3]
        world_force[3:] = frame @ force[3:]
        return world_force

    def _push_to_host(self) -> None:
        for world_id, run in enumerate(self._host_runs):
            mjd = run.mj_data
            for field in _HOST_MUTABLE_FIELDS:
                _copy_host_value(
                    getattr(mjd, field),
                    _world_view(getattr(self, field), world_id, nworld=self.nworld),
                )

            if self.nworld == 1:
                mjd.time = float(self.time)
                np.copyto(run.orbit.R_eci, self.orbit.R_eci)
                np.copyto(run.orbit.V_eci, self.orbit.V_eci)
                run.orbit.t = float(self.orbit.t)
                np.copyto(run.actuators.rw_speed, self.actuators.rw_speed)
                np.copyto(run.actuators.rw_torque_cmd, self.actuators.rw_torque_cmd)
                np.copyto(run.actuators.mtq_dipole_cmd, self.actuators.mtq_dipole_cmd)
                np.copyto(run.actuators.thr_force_cmd, self.actuators.thr_force_cmd)
                np.copyto(run.wrench_buffer, self.wrench_buffer)
            else:
                mjd.time = float(self.time[world_id])
                np.copyto(run.orbit.R_eci, self.orbit.R_eci[world_id])
                np.copyto(run.orbit.V_eci, self.orbit.V_eci[world_id])
                run.orbit.t = float(self.orbit.t[world_id])
                np.copyto(run.actuators.rw_speed, self.actuators.rw_speed[world_id])
                np.copyto(run.actuators.rw_torque_cmd, self.actuators.rw_torque_cmd[world_id])
                np.copyto(
                    run.actuators.mtq_dipole_cmd,
                    self.actuators.mtq_dipole_cmd[world_id],
                )
                np.copyto(run.actuators.thr_force_cmd, self.actuators.thr_force_cmd[world_id])
                np.copyto(run.wrench_buffer, self.wrench_buffer[world_id])

            np.copyto(run.actuators.rw_inertia, self.actuators.rw_inertia)
            run.actuators.update_rw_momentum(self.actuators.rw_inertia)

    def _pull_from_host(self) -> None:
        for world_id, run in enumerate(self._host_runs):
            mjd = run.mj_data
            for field in _MIRRORED_ARRAY_FIELDS:
                _copy_world_value(
                    getattr(self, field),
                    world_id,
                    getattr(mjd, field),
                    nworld=self.nworld,
                )

            if self.nworld == 1:
                self.time = float(mjd.time)
                for field in _MIRRORED_SCALAR_FIELDS:
                    setattr(self, field, int(getattr(mjd, field)))
                np.copyto(self.orbit.R_eci, run.orbit.R_eci)
                np.copyto(self.orbit.V_eci, run.orbit.V_eci)
                self.orbit.t = float(run.orbit.t)
                np.copyto(self.frame.C_LI, run.frame.C_LI)
                np.copyto(self.frame.C_IL, run.frame.C_IL)
                np.copyto(self.frame.omega_lvlh, run.frame.omega_lvlh)
                np.copyto(self.frame.omega_dot_lvlh, run.frame.omega_dot_lvlh)
                np.copyto(self.env.sun_vector_eci, run.env.sun_vector_eci)
                self.env.eclipse = float(run.env.eclipse)
                np.copyto(self.env.mag_field_eci, run.env.mag_field_eci)
                np.copyto(self.env.atmosphere_omega_eci, run.env.atmosphere_omega_eci)
                self.env.atm_density = float(run.env.atm_density)
                np.copyto(self.actuators.rw_speed, run.actuators.rw_speed)
                np.copyto(self.actuators.rw_momentum, run.actuators.rw_momentum)
                np.copyto(self.actuators.rw_torque_cmd, run.actuators.rw_torque_cmd)
                np.copyto(self.actuators.mtq_dipole_cmd, run.actuators.mtq_dipole_cmd)
                np.copyto(self.actuators.thr_force_cmd, run.actuators.thr_force_cmd)
                np.copyto(self.wrench_buffer, run.wrench_buffer)
            else:
                self.time[world_id] = float(mjd.time)
                for field in _MIRRORED_SCALAR_FIELDS:
                    getattr(self, field)[world_id] = int(getattr(mjd, field))
                np.copyto(self.orbit.R_eci[world_id], run.orbit.R_eci)
                np.copyto(self.orbit.V_eci[world_id], run.orbit.V_eci)
                self.orbit.t[world_id] = float(run.orbit.t)
                np.copyto(self.frame.C_LI[world_id], run.frame.C_LI)
                np.copyto(self.frame.C_IL[world_id], run.frame.C_IL)
                np.copyto(self.frame.omega_lvlh[world_id], run.frame.omega_lvlh)
                np.copyto(self.frame.omega_dot_lvlh[world_id], run.frame.omega_dot_lvlh)
                np.copyto(self.env.sun_vector_eci[world_id], run.env.sun_vector_eci)
                self.env.eclipse[world_id] = float(run.env.eclipse)
                np.copyto(self.env.mag_field_eci[world_id], run.env.mag_field_eci)
                np.copyto(
                    self.env.atmosphere_omega_eci[world_id],
                    run.env.atmosphere_omega_eci,
                )
                self.env.atm_density[world_id] = float(run.env.atm_density)
                np.copyto(self.actuators.rw_speed[world_id], run.actuators.rw_speed)
                np.copyto(self.actuators.rw_momentum[world_id], run.actuators.rw_momentum)
                np.copyto(
                    self.actuators.rw_torque_cmd[world_id],
                    run.actuators.rw_torque_cmd,
                )
                np.copyto(
                    self.actuators.mtq_dipole_cmd[world_id],
                    run.actuators.mtq_dipole_cmd,
                )
                np.copyto(
                    self.actuators.thr_force_cmd[world_id],
                    run.actuators.thr_force_cmd,
                )
                np.copyto(self.wrench_buffer[world_id], run.wrench_buffer)

        self.actuators.rw_inertia[:] = self.model.rw_inertia
        self.actuators.update_rw_momentum(self.model.rw_inertia)


__all__ = [
    'MjoData',
]
