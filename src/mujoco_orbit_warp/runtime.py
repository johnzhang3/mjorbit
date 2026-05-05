# pyright: reportAttributeAccessIssue=false, reportIndexIssue=false

"""MJWarp-backed model/data wrappers with MuJoCo-style names."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from mujoco_orbit.config import (
    MagneticBodySpec,
    MagnetorquerSpec,
    OrbitInit,
    ReactionWheelSpec,
    SurfaceSpec,
    ThrusterSpec,
)
from mujoco_orbit.data import MjoData as CpuMjoData
from mujoco_orbit.model import MjoModel as CpuMjoModel
from mujoco_orbit.spec import MjoSpec, _raw_mujoco_xml
from mujoco_orbit.step import mjo_forward as cpu_mjo_forward

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


def _compile_raw_mujoco_model(raw_xml: str, *, mj_timestep: float | None) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_string(raw_xml)
    if mj_timestep is not None:
        model.opt.timestep = float(mj_timestep)
    model.opt.gravity[:] = 0.0
    return model


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


# TODO(sensors): Warp-side sensor implementation removed pending GPU-native
# port. The previous WarpSensorDataNamespace delegated per-world reads to the
# CPU host shadow, which obscured Warp-side bugs and made the sensor parity
# test misleading. The CPU sensor stack (mujoco_orbit.data.MjoData.sensors,
# src/cpp/src/sensors_plugin.cc, tests/mujoco_orbit/test_sensors.py) is
# untouched and remains the reference. Reintroduce here when GPU sensor
# kernels are written. mj_data.sensordata is still allocated below because
# MJWarp's solver writes to it during step.


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


@dataclass
class MjoModel:
    """MJWarp-backed model wrapper with host metadata retained."""

    host_model: CpuMjoModel
    mj_model: mujoco.MjModel
    warp_model: Any
    core_model: Any
    backend: str = "warp"

    @classmethod
    def from_host_model(cls, host_model: CpuMjoModel) -> "MjoModel":
        """Upload a CPU mjorbit model into the orbit-aware MJWarp wrapper."""
        mjw, _ = require_mjwarp()
        from .core_gpu import make_device_core_model

        raw_xml = getattr(host_model, "_raw_xml", None)
        if raw_xml is None:
            raise TypeError(
                "Cannot upload this CPU MjoModel to MJWarp because it does not retain raw XML"
            )
        mj_model = _compile_raw_mujoco_model(raw_xml, mj_timestep=float(host_model.opt.timestep))
        return cls(
            host_model=host_model,
            mj_model=mj_model,
            warp_model=mjw.put_model(mj_model),
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
        use_gravity_gradient: bool = True,
    ) -> "MjoModel":
        """Compile the host model and upload an MJWarp device model."""
        spec = MjoSpec.from_xml_path(xml_path)
        spec.mjorbit.use_j2 = use_j2
        spec.mjorbit.use_drag = use_drag
        spec.mjorbit.use_srp = use_srp
        spec.mjorbit.use_magnetic = use_magnetic
        spec.mjorbit.use_gravity_gradient = use_gravity_gradient
        spec.mjorbit.orbit_dt = orbit_dt
        for surface in surfaces:
            spec.mjorbit.add_surface(surface)
        for magnetic_body in magnetic_bodies:
            spec.mjorbit.add_magnetic_body(magnetic_body)
        for wheel in reaction_wheels:
            spec.mjorbit.add_reaction_wheel(wheel)
        for magnetorquer in magnetorquers:
            spec.mjorbit.add_magnetorquer(magnetorquer)
        for thruster in thrusters:
            spec.mjorbit.add_thruster(thruster)

        host_model = spec.compile(mj_timestep=mj_timestep)
        raw_xml = _raw_mujoco_xml(spec.to_xml())
        mj_model = _compile_raw_mujoco_model(raw_xml, mj_timestep=mj_timestep)

        mjw, _ = require_mjwarp()
        from .core_gpu import make_device_core_model

        return cls(
            host_model=host_model,
            mj_model=mj_model,
            warp_model=mjw.put_model(mj_model),
            core_model=make_device_core_model(host_model),
        )

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
]
