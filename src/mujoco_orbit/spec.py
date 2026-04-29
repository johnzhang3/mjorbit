# pyright: reportAttributeAccessIssue=false, reportUnknownMemberType=false

"""Editable orbit overlay for MuJoCo-style model construction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from mujoco_orbit import _bindings
from mujoco_orbit.config import (
    ControlMomentGyroSpec,
    MagneticBodySpec,
    MagnetorquerSpec,
    ReactionWheelSpec,
    SurfaceSpec,
    ThrusterSpec,
)
from mujoco_orbit.constants import B0_EARTH, GM_EARTH, J2_EARTH, OMEGA_EARTH, R_EARTH


class CentralBodySpec:
    """Central-body and environment constants used by compiled orbit models."""

    __slots__ = ("_native",)

    def __init__(
        self,
        *,
        name: str = "earth",
        gm: float = GM_EARTH,
        radius: float = R_EARTH,
        j2: float = J2_EARTH,
        omega: Iterable[float] = (0.0, 0.0, OMEGA_EARTH),
        magnetic_b0: float = B0_EARTH,
        magnetic_axis: Iterable[float] = (0.0, 0.0, -1.0),
        atmosphere_h0: float = 400.0,
        atmosphere_rho0: float = 2.62e-13,
        atmosphere_scale_height: float = 58.2,
    ) -> None:
        self._native = _bindings.CentralBodySpec()
        self.name = name
        self.gm = gm
        self.radius = radius
        self.j2 = j2
        self.omega = omega
        self.magnetic_b0 = magnetic_b0
        self.magnetic_axis = magnetic_axis
        self.atmosphere_h0 = atmosphere_h0
        self.atmosphere_rho0 = atmosphere_rho0
        self.atmosphere_scale_height = atmosphere_scale_height

    @classmethod
    def _from_native(cls, native: Any) -> "CentralBodySpec":
        obj = cls.__new__(cls)
        obj._native = native
        return obj

    @property
    def name(self) -> str:
        return str(self._native.name)

    @name.setter
    def name(self, value: str) -> None:
        self._native.name = str(value)

    @property
    def gm(self) -> float:
        return float(self._native.gm)

    @gm.setter
    def gm(self, value: float) -> None:
        self._native.gm = float(value)

    @property
    def radius(self) -> float:
        return float(self._native.radius)

    @radius.setter
    def radius(self, value: float) -> None:
        self._native.radius = float(value)

    @property
    def j2(self) -> float:
        return float(self._native.j2)

    @j2.setter
    def j2(self, value: float) -> None:
        self._native.j2 = float(value)

    @property
    def omega(self) -> np.ndarray:
        return np.asarray(self._native.omega, dtype=np.float64)

    @omega.setter
    def omega(self, value: Iterable[float]) -> None:
        self._native.omega = _vec3(value, "omega")

    @property
    def magnetic_b0(self) -> float:
        return float(self._native.magnetic_b0)

    @magnetic_b0.setter
    def magnetic_b0(self, value: float) -> None:
        self._native.magnetic_b0 = float(value)

    @property
    def magnetic_axis(self) -> np.ndarray:
        return np.asarray(self._native.magnetic_axis, dtype=np.float64)

    @magnetic_axis.setter
    def magnetic_axis(self, value: Iterable[float]) -> None:
        self._native.magnetic_axis = _vec3(value, "magnetic_axis")

    @property
    def atmosphere_h0(self) -> float:
        return float(self._native.atmosphere_h0)

    @atmosphere_h0.setter
    def atmosphere_h0(self, value: float) -> None:
        self._native.atmosphere_h0 = float(value)

    @property
    def atmosphere_rho0(self) -> float:
        return float(self._native.atmosphere_rho0)

    @atmosphere_rho0.setter
    def atmosphere_rho0(self, value: float) -> None:
        self._native.atmosphere_rho0 = float(value)

    @property
    def atmosphere_scale_height(self) -> float:
        return float(self._native.atmosphere_scale_height)

    @atmosphere_scale_height.setter
    def atmosphere_scale_height(self, value: float) -> None:
        self._native.atmosphere_scale_height = float(value)

    def __repr__(self) -> str:
        return (
            "CentralBodySpec("
            f"name={self.name!r}, gm={self.gm!r}, radius={self.radius!r}, "
            f"j2={self.j2!r})"
        )


class MjoOrbitSpec:
    """Mutable orbit configuration overlay held by :class:`MjoSpec`."""

    __slots__ = ("_native",)

    def __init__(self, native: Any) -> None:
        self._native = native

    @property
    def plugin_body(self) -> str | None:
        return self._native.plugin_body

    @plugin_body.setter
    def plugin_body(self, value: str | None) -> None:
        self._native.plugin_body = None if value is None else str(value)

    @property
    def use_j2(self) -> bool:
        return bool(self._native.use_j2)

    @use_j2.setter
    def use_j2(self, value: bool) -> None:
        self._native.use_j2 = bool(value)

    @property
    def use_drag(self) -> bool:
        return bool(self._native.use_drag)

    @use_drag.setter
    def use_drag(self, value: bool) -> None:
        self._native.use_drag = bool(value)

    @property
    def use_srp(self) -> bool:
        return bool(self._native.use_srp)

    @use_srp.setter
    def use_srp(self, value: bool) -> None:
        self._native.use_srp = bool(value)

    @property
    def use_magnetic(self) -> bool:
        return bool(self._native.use_magnetic)

    @use_magnetic.setter
    def use_magnetic(self, value: bool) -> None:
        self._native.use_magnetic = bool(value)

    @property
    def use_gravity_gradient(self) -> bool:
        return bool(self._native.use_gravity_gradient)

    @use_gravity_gradient.setter
    def use_gravity_gradient(self, value: bool) -> None:
        self._native.use_gravity_gradient = bool(value)

    @property
    def orbit_dt(self) -> float | None:
        value = self._native.orbit_dt
        return None if value is None else float(value)

    @orbit_dt.setter
    def orbit_dt(self, value: float | None) -> None:
        self._native.orbit_dt = None if value is None else float(value)

    @property
    def central_body(self) -> CentralBodySpec:
        return CentralBodySpec._from_native(self._native.central_body)

    @central_body.setter
    def central_body(self, value: CentralBodySpec | Mapping[str, Any]) -> None:
        self._native.central_body = _central_body_native(value)

    @property
    def surfaces(self) -> list[SurfaceSpec]:
        return [_surface_from_native(spec) for spec in self._native.surfaces]

    @property
    def magnetic_bodies(self) -> list[MagneticBodySpec]:
        return [_magnetic_from_native(spec) for spec in self._native.magnetic_bodies]

    @property
    def reaction_wheels(self) -> list[ReactionWheelSpec]:
        return [_reaction_wheel_from_native(spec) for spec in self._native.reaction_wheels]

    @property
    def magnetorquers(self) -> list[MagnetorquerSpec]:
        return [_magnetorquer_from_native(spec) for spec in self._native.magnetorquers]

    @property
    def thrusters(self) -> list[ThrusterSpec]:
        return [_thruster_from_native(spec) for spec in self._native.thrusters]

    @property
    def cmgs(self) -> list[ControlMomentGyroSpec]:
        return [_cmg_from_native(spec) for spec in self._native.cmgs]

    def add_surface(self, spec: SurfaceSpec | None = None, **kwargs: Any) -> str:
        surface = _coerce(SurfaceSpec, spec, kwargs)
        return _native_call(self._native.add_surface, _surface_to_native(surface))

    def update_surface(
        self,
        element_name: str,
        spec: SurfaceSpec | None = None,
        **kwargs: Any,
    ) -> None:
        surface = _update_spec(element_name, self.surfaces, spec, kwargs)
        _native_call(self._native.update_surface, element_name, _surface_to_native(surface))

    def remove_surface(self, name: str) -> None:
        _native_call(self._native.remove_surface, name)

    def add_magnetic_body(self, spec: MagneticBodySpec | None = None, **kwargs: Any) -> str:
        magnetic = _coerce(MagneticBodySpec, spec, kwargs)
        return _native_call(self._native.add_magnetic_body, _magnetic_to_native(magnetic))

    def update_magnetic_body(
        self,
        element_name: str,
        spec: MagneticBodySpec | None = None,
        **kwargs: Any,
    ) -> None:
        magnetic = _update_spec(element_name, self.magnetic_bodies, spec, kwargs)
        _native_call(self._native.update_magnetic_body, element_name, _magnetic_to_native(magnetic))

    def remove_magnetic_body(self, name: str) -> None:
        _native_call(self._native.remove_magnetic_body, name)

    def add_reaction_wheel(self, spec: ReactionWheelSpec | None = None, **kwargs: Any) -> str:
        wheel = _coerce(ReactionWheelSpec, spec, kwargs)
        return _native_call(self._native.add_reaction_wheel, _reaction_wheel_to_native(wheel))

    def update_reaction_wheel(
        self,
        element_name: str,
        spec: ReactionWheelSpec | None = None,
        **kwargs: Any,
    ) -> None:
        wheel = _update_spec(element_name, self.reaction_wheels, spec, kwargs)
        _native_call(
            self._native.update_reaction_wheel,
            element_name,
            _reaction_wheel_to_native(wheel),
        )

    def remove_reaction_wheel(self, name: str) -> None:
        _native_call(self._native.remove_reaction_wheel, name)

    def add_magnetorquer(self, spec: MagnetorquerSpec | None = None, **kwargs: Any) -> str:
        magnetorquer = _coerce(MagnetorquerSpec, spec, kwargs)
        return _native_call(self._native.add_magnetorquer, _magnetorquer_to_native(magnetorquer))

    def update_magnetorquer(
        self,
        element_name: str,
        spec: MagnetorquerSpec | None = None,
        **kwargs: Any,
    ) -> None:
        magnetorquer = _update_spec(element_name, self.magnetorquers, spec, kwargs)
        _native_call(
            self._native.update_magnetorquer,
            element_name,
            _magnetorquer_to_native(magnetorquer),
        )

    def remove_magnetorquer(self, element_name: str) -> None:
        _native_call(self._native.remove_magnetorquer, element_name)

    def add_thruster(self, spec: ThrusterSpec | None = None, **kwargs: Any) -> str:
        thruster = _coerce(ThrusterSpec, spec, kwargs)
        return _native_call(self._native.add_thruster, _thruster_to_native(thruster))

    def update_thruster(
        self,
        element_name: str,
        spec: ThrusterSpec | None = None,
        **kwargs: Any,
    ) -> None:
        thruster = _update_spec(element_name, self.thrusters, spec, kwargs)
        _native_call(self._native.update_thruster, element_name, _thruster_to_native(thruster))

    def remove_thruster(self, name: str) -> None:
        _native_call(self._native.remove_thruster, name)

    def add_cmg(self, spec: ControlMomentGyroSpec | None = None, **kwargs: Any) -> str:
        cmg = _coerce(ControlMomentGyroSpec, spec, kwargs)
        return _native_call(self._native.add_cmg, _cmg_to_native(cmg))

    def update_cmg(
        self,
        element_name: str,
        spec: ControlMomentGyroSpec | None = None,
        **kwargs: Any,
    ) -> None:
        cmg = _update_spec(element_name, self.cmgs, spec, kwargs)
        _native_call(self._native.update_cmg, element_name, _cmg_to_native(cmg))

    def remove_cmg(self, name: str) -> None:
        _native_call(self._native.remove_cmg, name)


class MjoSpec:
    """Editable MuJoCo Orbit spec. Call ``compile()`` to create an immutable model."""

    __slots__ = ("_native", "mjorbit")

    def __init__(self, native: Any) -> None:
        self._native = native
        self.mjorbit = MjoOrbitSpec(native)

    @classmethod
    def from_xml_path(cls, path: str) -> "MjoSpec":
        return cls(_native_call(_bindings.MjoSpec.from_xml_path, str(path)))

    @classmethod
    def from_xml_string(
        cls,
        xml: str,
        assets: Mapping[str, bytes | bytearray | memoryview | Iterable[int]] | None = None,
    ) -> "MjoSpec":
        return cls(_native_call(_bindings.MjoSpec.from_xml_string, str(xml), _asset_map(assets)))

    @classmethod
    def from_mj_spec(
        cls,
        mj_spec: Any,
        assets: Mapping[str, bytes | bytearray | memoryview | Iterable[int]] | None = None,
    ) -> "MjoSpec":
        if not hasattr(mj_spec, "to_xml"):
            raise TypeError("mj_spec must provide a to_xml() method")
        xml = mj_spec.to_xml()
        if isinstance(xml, bytes):
            xml = xml.decode()
        return cls.from_xml_string(str(xml), assets=assets)

    def copy(self) -> "MjoSpec":
        return MjoSpec(_native_call(self._native.copy))

    def compile(self, *, mj_timestep: float | None = None):
        from mujoco_orbit.model import MjoModel

        native_model = _native_call(self._native.compile, mj_timestep)
        return MjoModel(native_model)

    def to_xml(self) -> str:
        return str(_native_call(self._native.to_xml))


def _vec3(value: Iterable[float], name: str) -> list[float]:
    arr = np.asarray(list(value), dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 values")
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _asset_map(
    assets: Mapping[str, bytes | bytearray | memoryview | Iterable[int]] | None,
) -> dict[str, list[int]]:
    if assets is None:
        return {}
    normalized: dict[str, list[int]] = {}
    for name, data in assets.items():
        if isinstance(data, bytes | bytearray | memoryview):
            normalized[str(name)] = list(bytes(data))
        else:
            normalized[str(name)] = [int(value) for value in data]
    return normalized


def _native_call(fn: Any, *args: Any) -> Any:
    try:
        return fn(*args)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc


def _central_body_native(value: CentralBodySpec | Mapping[str, Any]) -> Any:
    if isinstance(value, CentralBodySpec):
        return value._native
    if isinstance(value, Mapping):
        return CentralBodySpec(**dict(value))._native
    raise TypeError("central_body must be a CentralBodySpec or mapping")


def _coerce(cls: type[Any], spec: Any | None, kwargs: dict[str, Any]) -> Any:
    if spec is None:
        return cls(**kwargs)  # type: ignore[call-arg]
    current: Any = spec
    if kwargs:
        return replace(current, **kwargs)
    return current


def _update_spec(
    name: str,
    values: Iterable[Any],
    spec: Any | None,
    kwargs: dict[str, Any],
) -> Any:
    current: Any = _find_named(name, values) if spec is None else spec
    if kwargs:
        current = replace(current, **kwargs)
    if getattr(current, "name", None) is None:
        current = replace(current, name=name)
    return current


def _find_named(name: str, values: Iterable[Any]) -> Any:
    for value in values:
        if getattr(value, "name", None) == name:
            return value
    raise ValueError(f"<mjorbit> element {name!r} was not found")


def _surface_to_native(spec: SurfaceSpec) -> Any:
    native = _bindings.OrbitSurfaceSpec()
    native.name = spec.name or ""
    native.body_name = spec.body_name
    native.center_of_pressure_body = _vec3(spec.center_of_pressure_body, "center_of_pressure_body")
    native.normal_body = _vec3(spec.normal_body, "normal_body")
    native.area = float(spec.area)
    native.drag_coeff = float(spec.drag_coeff)
    native.srp_coeff = float(spec.srp_coeff)
    native.use_drag = bool(spec.use_drag)
    native.use_srp = bool(spec.use_srp)
    return native


def _surface_from_native(native: Any) -> SurfaceSpec:
    return SurfaceSpec(
        body_name=str(native.body_name),
        center_of_pressure_body=np.asarray(native.center_of_pressure_body, dtype=np.float64),
        normal_body=np.asarray(native.normal_body, dtype=np.float64),
        area=float(native.area),
        drag_coeff=float(native.drag_coeff),
        srp_coeff=float(native.srp_coeff),
        use_drag=bool(native.use_drag),
        use_srp=bool(native.use_srp),
        name=str(native.name),
    )


def _magnetic_to_native(spec: MagneticBodySpec) -> Any:
    native = _bindings.OrbitMagneticBodySpec()
    native.name = spec.name or ""
    native.body_name = spec.body_name
    native.dipole_body = _vec3(spec.dipole_body, "dipole_body")
    return native


def _magnetic_from_native(native: Any) -> MagneticBodySpec:
    return MagneticBodySpec(
        body_name=str(native.body_name),
        dipole_body=np.asarray(native.dipole_body, dtype=np.float64),
        name=str(native.name),
    )


def _reaction_wheel_to_native(spec: ReactionWheelSpec) -> Any:
    native = _bindings.OrbitReactionWheelSpec()
    native.name = spec.name or ""
    native.body_name = spec.body_name
    native.axis_body = _vec3(spec.axis_body, "axis_body")
    native.inertia = float(spec.inertia)
    native.speed_limit = spec.speed_limit
    native.torque_limit = spec.torque_limit
    return native


def _reaction_wheel_from_native(native: Any) -> ReactionWheelSpec:
    return ReactionWheelSpec(
        body_name=str(native.body_name),
        axis_body=np.asarray(native.axis_body, dtype=np.float64),
        inertia=float(native.inertia),
        speed_limit=None if native.speed_limit is None else float(native.speed_limit),
        torque_limit=None if native.torque_limit is None else float(native.torque_limit),
        name=str(native.name),
    )


def _magnetorquer_to_native(spec: MagnetorquerSpec) -> Any:
    native = _bindings.OrbitMagnetorquerSpec()
    native.name = spec.name or ""
    native.body_name = spec.body_name
    native.axis_body = _vec3(spec.axis_body, "axis_body")
    native.dipole_limit = float(spec.dipole_limit)
    return native


def _magnetorquer_from_native(native: Any) -> MagnetorquerSpec:
    return MagnetorquerSpec(
        body_name=str(native.body_name),
        axis_body=np.asarray(native.axis_body, dtype=np.float64),
        dipole_limit=float(native.dipole_limit),
        name=str(native.name),
    )


def _thruster_to_native(spec: ThrusterSpec) -> Any:
    native = _bindings.OrbitThrusterSpec()
    native.name = spec.name or ""
    native.body_name = spec.body_name
    native.position_body = _vec3(spec.position_body, "position_body")
    native.direction_body = _vec3(spec.direction_body, "direction_body")
    native.force_limit = float(spec.force_limit)
    return native


def _thruster_from_native(native: Any) -> ThrusterSpec:
    return ThrusterSpec(
        body_name=str(native.body_name),
        position_body=np.asarray(native.position_body, dtype=np.float64),
        direction_body=np.asarray(native.direction_body, dtype=np.float64),
        force_limit=float(native.force_limit),
        name=str(native.name),
    )


def _cmg_to_native(spec: ControlMomentGyroSpec) -> Any:
    native = _bindings.OrbitCmgSpec()
    native.name = spec.name or ""
    native.body_name = spec.body_name
    native.gimbal_axis_body = _vec3(spec.gimbal_axis_body, "gimbal_axis_body")
    native.spin_axis_body_0 = _vec3(spec.spin_axis_body_0, "spin_axis_body_0")
    native.rotor_momentum = float(spec.rotor_momentum)
    native.gimbal_rate_limit = spec.gimbal_rate_limit
    native.gimbal_angle_limit = spec.gimbal_angle_limit
    return native


def _cmg_from_native(native: Any) -> ControlMomentGyroSpec:
    return ControlMomentGyroSpec(
        body_name=str(native.body_name),
        gimbal_axis_body=np.asarray(native.gimbal_axis_body, dtype=np.float64),
        spin_axis_body_0=np.asarray(native.spin_axis_body_0, dtype=np.float64),
        rotor_momentum=float(native.rotor_momentum),
        gimbal_rate_limit=None
        if native.gimbal_rate_limit is None
        else float(native.gimbal_rate_limit),
        gimbal_angle_limit=None
        if native.gimbal_angle_limit is None
        else float(native.gimbal_angle_limit),
        name=str(native.name),
    )


__all__ = ["CentralBodySpec", "MjoOrbitSpec", "MjoSpec"]
