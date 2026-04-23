# pyright: reportAttributeAccessIssue=false

"""Core model/data types for the MuJoCo-style mujoco_orbit API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mujoco
import numpy as np

from mujoco_orbit.core.actuators import ActuatorData
from mujoco_orbit.core.config import (
    ControlMomentGyroSpec,
    MagneticBodySpec,
    MagnetorquerSpec,
    OrbitInit,
    ReactionWheelSpec,
    SurfaceSpec,
    ThrusterSpec,
)
from mujoco_orbit.orbit.environment import update_environment_cache
from mujoco_orbit.orbit.lvlh import update_frame_cache
from mujoco_orbit.orbit.state import EnvironmentCache, FrameCache, OrbitState
from mujoco_orbit.sensors import (
    ModelSensorCatalog,
    SensorDataNamespace,
    compile_sensor_catalog,
    create_sensor_data_namespace,
    register_sensor_data_namespace,
)


@dataclass(frozen=True)
class SurfaceMetadata:
    """Resolved surface metadata after compile-time body lookup."""

    body_name: str
    body_id: int
    center_of_pressure_body: np.ndarray
    normal_body: np.ndarray
    area: float
    drag_coeff: float
    srp_coeff: float
    use_drag: bool
    use_srp: bool


@dataclass(frozen=True)
class MagneticMetadata:
    """Resolved residual magnetic dipole metadata."""

    body_name: str
    body_id: int
    dipole_body: np.ndarray


@dataclass(frozen=True)
class ReactionWheelMetadata:
    """Resolved reaction wheel metadata."""

    body_name: str
    body_id: int
    axis_body: np.ndarray
    inertia: float
    speed_limit: float | None
    torque_limit: float | None


@dataclass(frozen=True)
class MagnetorquerMetadata:
    """Resolved magnetorquer metadata."""

    body_name: str
    body_id: int
    axis_body: np.ndarray
    dipole_limit: float


@dataclass(frozen=True)
class ControlMomentGyroMetadata:
    """Resolved control moment gyro metadata."""

    body_name: str
    body_id: int
    gimbal_axis_body: np.ndarray
    spin_axis_body_0: np.ndarray
    torque_axis_body_0: np.ndarray  # gimbal_axis × spin_axis_0 (unit)
    rotor_momentum: float
    gimbal_rate_limit: float | None
    gimbal_angle_limit: float | None


@dataclass(frozen=True)
class ThrusterMetadata:
    """Resolved thruster metadata."""

    body_name: str
    body_id: int
    position_body: np.ndarray
    direction_body: np.ndarray
    force_limit: float


def _normalized(vec: np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if norm < 1e-12:
        raise ValueError(f"{label} must be non-zero")
    return arr / norm


def _resolve_body_id(mj_model: mujoco.MjModel, body_name: str) -> int:
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in MuJoCo model")
    return body_id


def _resolve_surfaces(
    mj_model: mujoco.MjModel, surfaces: Iterable[SurfaceSpec]
) -> list[SurfaceMetadata]:
    resolved: list[SurfaceMetadata] = []
    for surface in surfaces:
        resolved.append(
            SurfaceMetadata(
                body_name=surface.body_name,
                body_id=_resolve_body_id(mj_model, surface.body_name),
                center_of_pressure_body=np.asarray(surface.center_of_pressure_body, dtype=float),
                normal_body=_normalized(
                    surface.normal_body,
                    f"Surface '{surface.body_name}' normal",
                ),
                area=surface.area,
                drag_coeff=surface.drag_coeff,
                srp_coeff=surface.srp_coeff,
                use_drag=surface.use_drag,
                use_srp=surface.use_srp,
            )
        )
    return resolved


def _resolve_magnetic_bodies(
    mj_model: mujoco.MjModel, magnetic_bodies: Iterable[MagneticBodySpec]
) -> list[MagneticMetadata]:
    resolved: list[MagneticMetadata] = []
    for magnetic_body in magnetic_bodies:
        resolved.append(
            MagneticMetadata(
                body_name=magnetic_body.body_name,
                body_id=_resolve_body_id(mj_model, magnetic_body.body_name),
                dipole_body=np.asarray(magnetic_body.dipole_body, dtype=float),
            )
        )
    return resolved


def _resolve_reaction_wheels(
    mj_model: mujoco.MjModel, reaction_wheels: Iterable[ReactionWheelSpec]
) -> list[ReactionWheelMetadata]:
    resolved: list[ReactionWheelMetadata] = []
    for reaction_wheel in reaction_wheels:
        resolved.append(
            ReactionWheelMetadata(
                body_name=reaction_wheel.body_name,
                body_id=_resolve_body_id(mj_model, reaction_wheel.body_name),
                axis_body=_normalized(
                    reaction_wheel.axis_body,
                    f"Reaction wheel '{reaction_wheel.body_name}' axis",
                ),
                inertia=reaction_wheel.inertia,
                speed_limit=reaction_wheel.speed_limit,
                torque_limit=reaction_wheel.torque_limit,
            )
        )
    return resolved


def _resolve_magnetorquers(
    mj_model: mujoco.MjModel, magnetorquers: Iterable[MagnetorquerSpec]
) -> list[MagnetorquerMetadata]:
    resolved: list[MagnetorquerMetadata] = []
    for magnetorquer in magnetorquers:
        resolved.append(
            MagnetorquerMetadata(
                body_name=magnetorquer.body_name,
                body_id=_resolve_body_id(mj_model, magnetorquer.body_name),
                axis_body=_normalized(
                    magnetorquer.axis_body,
                    f"Magnetorquer '{magnetorquer.body_name}' axis",
                ),
                dipole_limit=magnetorquer.dipole_limit,
            )
        )
    return resolved


def _resolve_cmgs(
    mj_model: mujoco.MjModel, cmgs: Iterable[ControlMomentGyroSpec]
) -> list[ControlMomentGyroMetadata]:
    resolved: list[ControlMomentGyroMetadata] = []
    for cmg in cmgs:
        gimbal_axis = _normalized(
            cmg.gimbal_axis_body, f"CMG '{cmg.body_name}' gimbal axis"
        )
        spin_axis_0 = _normalized(
            cmg.spin_axis_body_0, f"CMG '{cmg.body_name}' spin axis"
        )
        dot = float(np.dot(gimbal_axis, spin_axis_0))
        if abs(dot) > 1e-8:
            raise ValueError(
                f"CMG '{cmg.body_name}' spin axis must be orthogonal to gimbal axis "
                f"(dot product = {dot:.3e})"
            )
        torque_axis_0 = np.cross(gimbal_axis, spin_axis_0)
        if cmg.rotor_momentum <= 0.0:
            raise ValueError(
                f"CMG '{cmg.body_name}' rotor_momentum must be positive "
                f"(got {cmg.rotor_momentum})"
            )
        resolved.append(
            ControlMomentGyroMetadata(
                body_name=cmg.body_name,
                body_id=_resolve_body_id(mj_model, cmg.body_name),
                gimbal_axis_body=gimbal_axis,
                spin_axis_body_0=spin_axis_0,
                torque_axis_body_0=torque_axis_0,
                rotor_momentum=float(cmg.rotor_momentum),
                gimbal_rate_limit=cmg.gimbal_rate_limit,
                gimbal_angle_limit=cmg.gimbal_angle_limit,
            )
        )
    return resolved


def _resolve_thrusters(
    mj_model: mujoco.MjModel, thrusters: Iterable[ThrusterSpec]
) -> list[ThrusterMetadata]:
    resolved: list[ThrusterMetadata] = []
    for thruster in thrusters:
        resolved.append(
            ThrusterMetadata(
                body_name=thruster.body_name,
                body_id=_resolve_body_id(mj_model, thruster.body_name),
                position_body=np.asarray(thruster.position_body, dtype=float),
                direction_body=_normalized(
                    thruster.direction_body,
                    f"Thruster '{thruster.body_name}' direction",
                ),
                force_limit=thruster.force_limit,
            )
        )
    return resolved


@dataclass
class MjoModel:
    """Compiled, mostly-static state shared across runs."""

    mj_model: mujoco.MjModel
    surfaces: list[SurfaceMetadata]
    magnetic_bodies: list[MagneticMetadata]
    reaction_wheels: list[ReactionWheelMetadata]
    magnetorquers: list[MagnetorquerMetadata]
    thrusters: list[ThrusterMetadata]
    cmgs: list[ControlMomentGyroMetadata]
    sensors: ModelSensorCatalog
    use_j2: bool = True
    use_drag: bool = True
    use_srp: bool = True
    use_magnetic: bool = True
    use_gravity_gradient: bool = True
    orbit_dt: float | None = None

    def __post_init__(self) -> None:
        self.rw_inertia = np.array([wheel.inertia for wheel in self.reaction_wheels], dtype=float)
        self.cmg_rotor_momentum = np.array(
            [cmg.rotor_momentum for cmg in self.cmgs], dtype=float
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
        cmgs: Iterable[ControlMomentGyroSpec] = (),
        mj_timestep: float | None = 0.01,
        orbit_dt: float | None = None,
        use_j2: bool = True,
        use_drag: bool = True,
        use_srp: bool = True,
        use_magnetic: bool = True,
        use_gravity_gradient: bool = True,
    ) -> "MjoModel":
        """Compile a MuJoCo model plus static orbital coupling metadata."""
        sensor_callback = mujoco.get_mjcb_sensor()
        if sensor_callback is not None:
            mujoco.set_mjcb_sensor(None)
        try:
            mj_model = mujoco.MjModel.from_xml_path(xml_path)
        finally:
            if sensor_callback is not None:
                mujoco.set_mjcb_sensor(sensor_callback)

        if mj_timestep is not None:
            mj_model.opt.timestep = mj_timestep

        # Orbital dynamics supply the gravity model.
        mj_model.opt.gravity[:] = 0.0

        return cls(
            mj_model=mj_model,
            surfaces=_resolve_surfaces(mj_model, surfaces),
            magnetic_bodies=_resolve_magnetic_bodies(mj_model, magnetic_bodies),
            reaction_wheels=_resolve_reaction_wheels(mj_model, reaction_wheels),
            magnetorquers=_resolve_magnetorquers(mj_model, magnetorquers),
            thrusters=_resolve_thrusters(mj_model, thrusters),
            cmgs=_resolve_cmgs(mj_model, cmgs),
            sensors=compile_sensor_catalog(mj_model),
            use_j2=use_j2,
            use_drag=use_drag,
            use_srp=use_srp,
            use_magnetic=use_magnetic,
            use_gravity_gradient=use_gravity_gradient,
            orbit_dt=orbit_dt,
        )

    def __getattr__(self, name: str):  # pragma: no cover - trivial delegation
        return getattr(self.mj_model, name)

    def body_id(self, name: str) -> int:
        """Resolve a MuJoCo body name to its integer id."""
        return _resolve_body_id(self.mj_model, name)

    def sensor(self, name: str):
        """Return sensor metadata by name."""
        descriptor = self.sensors.by_name.get(name)
        if descriptor is None:
            raise ValueError(f"Sensor '{name}' not found in model")
        return descriptor


class MjoData:
    """Runtime state for one simulation run."""

    def __init__(self, model: MjoModel, *, orbit: OrbitInit, rng_seed: int | None = None) -> None:
        self.model = model
        self.mj_data = mujoco.MjData(model.mj_model)
        self.orbit = OrbitState(
            R_eci=np.asarray(orbit.R_eci, dtype=float).copy(),
            V_eci=np.asarray(orbit.V_eci, dtype=float).copy(),
            t=orbit.t,
        )
        self.frame: FrameCache = update_frame_cache(self.orbit, use_j2=model.use_j2)
        self.env: EnvironmentCache = update_environment_cache(self.orbit, self.frame)
        self.actuators = ActuatorData.zeros(
            len(model.reaction_wheels),
            model.rw_inertia,
            len(model.magnetorquers),
            len(model.thrusters),
            len(model.cmgs),
            model.cmg_rotor_momentum,
        )
        self.actuators.update_rw_momentum(model.rw_inertia)
        self.wrench_buffer = np.zeros((model.nbody, 6))
        self.sensors: SensorDataNamespace = create_sensor_data_namespace(
            model,
            self,
            rng_seed=rng_seed,
        )
        register_sensor_data_namespace(self.sensors)
        self._initialize_freejoints_in_chief_inertial()

        from mujoco_orbit.core.step import mjo_forward

        mjo_forward(model, self)

    def __getattr__(self, name: str):  # pragma: no cover - trivial delegation
        return getattr(self.mj_data, name)

    def clear_wrench_buffer(self) -> None:
        """Reset the assembled external wrench buffer and applied MuJoCo wrench."""
        self.wrench_buffer[:] = 0.0
        self.xfrc_applied[:] = 0.0

    def world_position_from_lvlh(self, position_lvlh_m: np.ndarray) -> np.ndarray:
        """Convert a chief-relative LVLH position to MuJoCo world meters.

        The MuJoCo world frame is chief-centered with axes parallel to ECI.
        """
        position_lvlh_km = np.asarray(position_lvlh_m, dtype=float) * 1e-3
        return 1000.0 * (self.frame.C_IL @ position_lvlh_km)

    def world_velocity_from_lvlh(
        self,
        position_lvlh_m: np.ndarray,
        velocity_lvlh_m_s: np.ndarray,
    ) -> np.ndarray:
        """Convert chief-relative LVLH velocity to MuJoCo world m/s."""
        position_lvlh_km = np.asarray(position_lvlh_m, dtype=float) * 1e-3
        velocity_lvlh_km_s = np.asarray(velocity_lvlh_m_s, dtype=float) * 1e-3
        relative_eci_km_s = self.frame.C_IL @ (
            velocity_lvlh_km_s + np.cross(self.frame.omega_lvlh, position_lvlh_km)
        )
        return 1000.0 * relative_eci_km_s

    def lvlh_position_from_world(self, position_world_m: np.ndarray) -> np.ndarray:
        """Convert MuJoCo world meters to chief-relative LVLH meters."""
        position_world_km = np.asarray(position_world_m, dtype=float) * 1e-3
        return 1000.0 * (self.frame.C_LI @ position_world_km)

    def lvlh_velocity_from_world(
        self,
        position_world_m: np.ndarray,
        velocity_world_m_s: np.ndarray,
    ) -> np.ndarray:
        """Convert MuJoCo world velocity to chief-relative LVLH m/s."""
        position_lvlh_km = self.lvlh_position_from_world(position_world_m) * 1e-3
        velocity_world_km_s = np.asarray(velocity_world_m_s, dtype=float) * 1e-3
        velocity_lvlh_km_s = (
            self.frame.C_LI @ velocity_world_km_s
            - np.cross(self.frame.omega_lvlh, position_lvlh_km)
        )
        return 1000.0 * velocity_lvlh_km_s

    def eci_position_from_world(self, position_world_m: np.ndarray) -> np.ndarray:
        """Convert MuJoCo chief-inertial world meters to absolute ECI meters."""
        position_world_km = np.asarray(position_world_m, dtype=float) * 1e-3
        return 1000.0 * (self.orbit.R_eci + position_world_km)

    def eci_velocity_from_world(self, velocity_world_m_s: np.ndarray) -> np.ndarray:
        """Convert MuJoCo chief-inertial world velocity to absolute ECI m/s."""
        velocity_world_km_s = np.asarray(velocity_world_m_s, dtype=float) * 1e-3
        return 1000.0 * (self.orbit.V_eci + velocity_world_km_s)

    def world_position_from_eci(self, position_eci_m: np.ndarray) -> np.ndarray:
        """Convert absolute ECI meters to MuJoCo chief-inertial world meters."""
        position_eci_km = np.asarray(position_eci_m, dtype=float) * 1e-3
        return 1000.0 * (position_eci_km - self.orbit.R_eci)

    def world_velocity_from_eci(self, velocity_eci_m_s: np.ndarray) -> np.ndarray:
        """Convert absolute ECI velocity to MuJoCo chief-inertial world m/s."""
        velocity_eci_km_s = np.asarray(velocity_eci_m_s, dtype=float) * 1e-3
        return 1000.0 * (velocity_eci_km_s - self.orbit.V_eci)

    def eci_position_from_lvlh(self, position_lvlh_m: np.ndarray) -> np.ndarray:
        """Convert a chief-relative LVLH position to absolute ECI meters."""
        return self.eci_position_from_world(self.world_position_from_lvlh(position_lvlh_m))

    def eci_velocity_from_lvlh(
        self,
        position_lvlh_m: np.ndarray,
        velocity_lvlh_m_s: np.ndarray,
    ) -> np.ndarray:
        """Convert chief-relative LVLH velocity to absolute ECI m/s."""
        return self.eci_velocity_from_world(
            self.world_velocity_from_lvlh(position_lvlh_m, velocity_lvlh_m_s)
        )

    def lvlh_position_from_eci(self, position_eci_m: np.ndarray) -> np.ndarray:
        """Convert absolute ECI meters to chief-relative LVLH meters."""
        return self.lvlh_position_from_world(self.world_position_from_eci(position_eci_m))

    def lvlh_velocity_from_eci(
        self,
        position_eci_m: np.ndarray,
        velocity_eci_m_s: np.ndarray,
    ) -> np.ndarray:
        """Convert absolute ECI velocity to chief-relative LVLH m/s."""
        position_lvlh_km = self.lvlh_position_from_eci(position_eci_m) * 1e-3
        velocity_eci_km_s = np.asarray(velocity_eci_m_s, dtype=float) * 1e-3
        velocity_lvlh_km_s = (
            self.frame.C_LI @ (velocity_eci_km_s - self.orbit.V_eci)
            - np.cross(self.frame.omega_lvlh, position_lvlh_km)
        )
        return 1000.0 * velocity_lvlh_km_s

    def _initialize_freejoints_in_chief_inertial(self) -> None:
        """Keep free-joint XML positions as chief-centered inertial offsets.

        The orbit layer stores the chief reference state in km/km/s, while MuJoCo
        stores local multibody state in SI units. Root free-joint ``pos`` and
        ``qvel`` values are therefore interpreted directly as relative inertial
        offsets from the chief, not as absolute ECI coordinates.
        """
        return


__all__ = [
    "ControlMomentGyroMetadata",
    "MagneticMetadata",
    "MjoData",
    "MjoModel",
    "MagnetorquerMetadata",
    "ReactionWheelMetadata",
    "SurfaceMetadata",
    "ThrusterMetadata",
]
