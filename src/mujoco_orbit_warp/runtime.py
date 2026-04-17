# pyright: reportAttributeAccessIssue=false, reportIndexIssue=false

"""MJWarp-backed model/data wrappers with MuJoCo-style names."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from mujoco_orbit.core.config import (
    MagneticBodySpec,
    MagnetorquerSpec,
    OrbitInit,
    ReactionWheelSpec,
    SurfaceSpec,
    ThrusterSpec,
)
from mujoco_orbit.core.runtime import MjoData as CpuMjoData
from mujoco_orbit.core.runtime import MjoModel as CpuMjoModel

from ._deps import require_mjwarp

_MIRRORED_ARRAY_FIELDS = (
    "solver_niter",
    "energy",
    "qpos",
    "qvel",
    "qacc",
    "act_dot",
    "act",
    "ctrl",
    "qacc_warmstart",
    "qfrc_applied",
    "xfrc_applied",
    "sensordata",
    "xpos",
    "xipos",
    "xquat",
    "xmat",
    "ximat",
    "xanchor",
    "xaxis",
    "geom_xpos",
    "geom_xmat",
    "site_xpos",
    "site_xmat",
    "cam_xpos",
    "cam_xmat",
    "light_xpos",
    "light_xdir",
    "subtree_com",
    "cdof",
    "cinert",
    "flexvert_xpos",
    "flexedge_J",
    "flexedge_length",
    "flexedge_velocity",
    "actuator_length",
    "actuator_moment",
    "actuator_velocity",
    "moment_rownnz",
    "moment_rowadr",
    "moment_colind",
    "crb",
    "qM",
    "qLD",
    "qLDiagInv",
    "ten_wrapadr",
    "ten_wrapnum",
    "ten_J",
    "ten_length",
    "ten_velocity",
    "wrap_obj",
    "wrap_xpos",
    "cvel",
    "cdof_dot",
    "qfrc_bias",
    "qfrc_spring",
    "qfrc_damper",
    "qfrc_gravcomp",
    "qfrc_fluid",
    "qfrc_passive",
    "subtree_linvel",
    "subtree_angmom",
    "actuator_force",
    "qfrc_actuator",
    "qfrc_smooth",
    "qacc_smooth",
    "qfrc_constraint",
    "qfrc_inverse",
    "cacc",
    "cfrc_int",
    "cfrc_ext",
    "tree_island",
    "mocap_pos",
    "mocap_quat",
    "eq_active",
)

_MIRRORED_SCALAR_FIELDS = ("ncon", "ne", "nf", "nl", "nefc", "nisland")

_HOST_MUTABLE_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "qacc_warmstart",
    "qfrc_applied",
    "xfrc_applied",
    "mocap_pos",
    "mocap_quat",
    "eq_active",
)


def _world_shape(nworld: int, *shape: int) -> tuple[int, ...]:
    return shape if nworld == 1 else (nworld, *shape)


def _zeros_world(nworld: int, *shape: int, dtype: Any = float) -> np.ndarray:
    return np.zeros(_world_shape(nworld, *shape), dtype=dtype)


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


def _world_view(array: np.ndarray, world_id: int, *, nworld: int) -> np.ndarray:
    return array if nworld == 1 else array[world_id]


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


@dataclass
class OrbitBatchState:
    """Public orbital state buffers for the warp backend."""

    R_eci: np.ndarray
    V_eci: np.ndarray
    t: float | np.ndarray


@dataclass
class FrameBatchCache:
    """Public frame cache buffers for the warp backend."""

    C_LI: np.ndarray
    C_IL: np.ndarray
    omega_lvlh: np.ndarray
    omega_dot_lvlh: np.ndarray


@dataclass
class EnvironmentBatchCache:
    """Public environment cache buffers for the warp backend."""

    sun_vector_eci: np.ndarray
    eclipse: float | np.ndarray
    mag_field_eci: np.ndarray
    atmosphere_omega_eci: np.ndarray
    atm_density: float | np.ndarray


@dataclass
class BatchedActuatorData:
    """Mutable actuator command/state arrays for one or more worlds."""

    rw_speed: np.ndarray
    rw_momentum: np.ndarray
    rw_inertia: np.ndarray
    rw_torque_cmd: np.ndarray
    mtq_dipole_cmd: np.ndarray
    thr_force_cmd: np.ndarray

    @classmethod
    def zeros(
        cls,
        nworld: int,
        n_rw: int,
        rw_inertia: np.ndarray,
        n_mtq: int,
        n_thr: int,
    ) -> "BatchedActuatorData":
        return cls(
            rw_speed=_zeros_world(nworld, n_rw),
            rw_momentum=_zeros_world(nworld, n_rw),
            rw_inertia=np.asarray(rw_inertia, dtype=float).copy(),
            rw_torque_cmd=_zeros_world(nworld, n_rw),
            mtq_dipole_cmd=_zeros_world(nworld, n_mtq),
            thr_force_cmd=_zeros_world(nworld, n_thr),
        )

    def update_rw_momentum(self, rw_inertia: np.ndarray | None = None) -> None:
        """Recompute wheel angular momentum from speed and inertia."""
        inertia = self.rw_inertia if rw_inertia is None else np.asarray(rw_inertia, dtype=float)
        self.rw_inertia[:] = inertia
        np.multiply(self.rw_speed, inertia, out=self.rw_momentum)


@dataclass
class WarpSensorDataNamespace:
    """Sensor helper that delegates per-world behavior to host shadow data."""

    model: "MjoModel"
    data: "MjoData"

    @property
    def gyro_biases(self) -> dict[str, np.ndarray]:
        names = [
            descriptor.name
            for descriptor in self.model.sensors.descriptors
            if descriptor.sensor_type == int(mujoco.mjtSensor.mjSENS_GYRO)
        ]
        if self.data.nworld == 1:
            return self.data._host_runs[0].sensors.gyro_biases
        return {
            name: np.stack([run.sensors.bias(name) for run in self.data._host_runs], axis=0)
            for name in names
        }

    def descriptor(self, name: str):
        return self.model.sensor(name)

    def bias(self, name: str, *, world_id: int = 0) -> np.ndarray:
        return self.data._host_runs[world_id].sensors.bias(name)

    def measure(
        self,
        name: str,
        *,
        noisy: bool = True,
        rng: np.random.Generator | None = None,
        world_id: int = 0,
    ) -> np.ndarray:
        return self.data._host_runs[world_id].sensors.measure(name, noisy=noisy, rng=rng)

    def measure_all(
        self,
        *,
        noisy: bool = True,
        rng: np.random.Generator | None = None,
        world_id: int = 0,
    ) -> dict[str, np.ndarray]:
        return self.data._host_runs[world_id].sensors.measure_all(noisy=noisy, rng=rng)


@dataclass
class MjoModel:
    """MJWarp-backed model wrapper with host metadata retained."""

    host_model: CpuMjoModel
    warp_model: Any
    core_model: Any
    backend: str = "warp"

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self.host_model.mj_model

    @classmethod
    def from_host_model(cls, host_model: CpuMjoModel) -> "MjoModel":
        """Upload a CPU mjorbit model into the orbit-aware MJWarp wrapper."""
        mjw, _ = require_mjwarp()
        from .core_gpu import make_device_core_model

        return cls(
            host_model=host_model,
            warp_model=mjw.put_model(host_model.mj_model),
            core_model=make_device_core_model(host_model),
        )

    @classmethod
    def from_xml_path(
        cls,
        xml_path: str,
        *,
        surfaces: Iterable[SurfaceSpec] = (),
        magnetic_bodies: Iterable[MagneticBodySpec] = (),
        reaction_wheels: Iterable[ReactionWheelSpec] = (),
        magnetorquers: Iterable[MagnetorquerSpec] = (),
        thrusters: Iterable[ThrusterSpec] = (),
        mj_timestep: float | None = 0.01,
        orbit_dt: float | None = None,
        use_j2: bool = True,
        use_drag: bool = True,
        use_srp: bool = True,
        use_magnetic: bool = True,
    ) -> "MjoModel":
        """Compile the host model and upload an MJWarp device model."""
        host_model = CpuMjoModel.from_xml_path(
            xml_path,
            surfaces=surfaces,
            magnetic_bodies=magnetic_bodies,
            reaction_wheels=reaction_wheels,
            magnetorquers=magnetorquers,
            thrusters=thrusters,
            mj_timestep=mj_timestep,
            orbit_dt=orbit_dt,
            use_j2=use_j2,
            use_drag=use_drag,
            use_srp=use_srp,
            use_magnetic=use_magnetic,
        )
        return cls.from_host_model(host_model)

    def __getattr__(self, name: str):  # pragma: no cover - trivial delegation
        return getattr(self.host_model, name)

    @property
    def device_model(self) -> Any:
        """Return the underlying ``mujoco_warp.Model``."""
        return self.warp_model

    def body_id(self, name: str) -> int:
        return self.host_model.body_id(name)

    def sensor(self, name: str):
        return self.host_model.sensor(name)

    def make_data(
        self,
        *,
        orbit: OrbitInit | Sequence[OrbitInit],
        rng_seed: int | Sequence[int | None] | None = None,
        nworld: int = 1,
        nconmax: int | None = None,
        nccdmax: int | None = None,
        njmax: int | None = None,
        naconmax: int | None = None,
        naccdmax: int | None = None,
    ) -> "MjoData":
        """Construct one or more runtime state objects for this compiled model."""
        return MjoData(
            self,
            orbit=orbit,
            rng_seed=rng_seed,
            nworld=nworld,
            nconmax=nconmax,
            nccdmax=nccdmax,
            njmax=njmax,
            naconmax=naconmax,
            naccdmax=naccdmax,
        )


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
            CpuMjoData(model.host_model, orbit=orbit_init, rng_seed=seed)
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
        self.sensors = WarpSensorDataNamespace(model=model, data=self)
        from .core_gpu import make_device_core_data

        self.core_data = make_device_core_data(orbit_inits, nworld=nworld, model=model.host_model)

        self._pull_from_host()

        from mujoco_orbit_warp.step import mjo_forward

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

        mjw, _ = require_mjwarp()
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
        data.warp_data = mjw.put_data(
            model.mj_model,
            host_data.mj_data,
            nworld=nworld,
            nconmax=nconmax,
            nccdmax=nccdmax,
            njmax=njmax,
            naconmax=naconmax,
            naccdmax=naccdmax,
        )

        for run in data._host_runs:
            mujoco.mj_copyData(run.mj_data, model.mj_model, host_data.mj_data)
            np.copyto(run.orbit.R_eci, host_data.orbit.R_eci)
            np.copyto(run.orbit.V_eci, host_data.orbit.V_eci)
            run.orbit.t = float(host_data.orbit.t)
            run.frame = host_data.frame
            run.env = host_data.env
            np.copyto(run.actuators.rw_speed, host_data.actuators.rw_speed)
            np.copyto(run.actuators.rw_momentum, host_data.actuators.rw_momentum)
            np.copyto(run.actuators.rw_torque_cmd, host_data.actuators.rw_torque_cmd)
            np.copyto(run.actuators.mtq_dipole_cmd, host_data.actuators.mtq_dipole_cmd)
            np.copyto(run.actuators.thr_force_cmd, host_data.actuators.thr_force_cmd)
            np.copyto(run.wrench_buffer, host_data.wrench_buffer)

        data._pull_from_host()
        data.upload(fields="core")
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
        from mujoco_orbit_warp.step import mjo_pull

        mjo_pull(self.model, self, fields=fields)

    def upload(self, fields: Iterable[str] | str | None = None) -> None:
        """Upload explicitly selected public input buffers to device state."""
        from mujoco_orbit_warp.step import mjo_upload

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
    "BatchedActuatorData",
    "EnvironmentBatchCache",
    "FrameBatchCache",
    "MjoData",
    "MjoModel",
    "OrbitBatchState",
    "WarpSensorDataNamespace",
]
