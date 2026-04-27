"""Public spec dataclasses for the MuJoCo-style mujoco_orbit API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class OrbitInit:
    """Initial chief orbit state in ECI (km, km/s)."""

    R_eci: np.ndarray  # shape (3,) km
    V_eci: np.ndarray  # shape (3,) km/s
    t: float = 0.0  # s

    def __post_init__(self) -> None:
        self.R_eci = np.asarray(self.R_eci, dtype=float)
        self.V_eci = np.asarray(self.V_eci, dtype=float)


@dataclass
class SurfaceSpec:
    """Metadata for one flat-plate aerodynamic / SRP surface."""

    body_name: str
    center_of_pressure_body: np.ndarray  # shape (3,) m
    normal_body: np.ndarray  # shape (3,) unit vector
    area: float  # m^2
    drag_coeff: float = 2.2
    srp_coeff: float = 1.8
    use_drag: bool = True
    use_srp: bool = True
    name: str | None = None

    def __post_init__(self) -> None:
        self.center_of_pressure_body = np.asarray(self.center_of_pressure_body, dtype=float)
        self.normal_body = np.asarray(self.normal_body, dtype=float)


@dataclass
class MagneticBodySpec:
    """Magnetic dipole source attached to a MuJoCo body."""

    body_name: str
    dipole_body: np.ndarray  # shape (3,) A·m^2, fixed residual dipole in body frame
    name: str | None = None

    def __post_init__(self) -> None:
        self.dipole_body = np.asarray(self.dipole_body, dtype=float)


@dataclass
class ReactionWheelSpec:
    """Configuration for one reaction wheel."""

    body_name: str  # MuJoCo body the wheel is mounted on
    axis_body: np.ndarray  # shape (3,) spin axis in body frame (unit vector)
    inertia: float  # kg·m^2
    speed_limit: Optional[float] = None  # rad/s, None = unlimited
    torque_limit: Optional[float] = None  # N·m
    name: str | None = None

    def __post_init__(self) -> None:
        self.axis_body = np.asarray(self.axis_body, dtype=float)


@dataclass
class MagnetorquerSpec:
    """Configuration for one magnetorquer."""

    body_name: str
    axis_body: np.ndarray  # shape (3,) axis in body frame (unit vector)
    dipole_limit: float  # A·m^2 max dipole magnitude
    name: str | None = None

    def __post_init__(self) -> None:
        self.axis_body = np.asarray(self.axis_body, dtype=float)


@dataclass
class ControlMomentGyroSpec:
    """Configuration for one single-gimbal control moment gyro (SGCMG).

    The rotor is idealized as spinning at constant angular momentum ``rotor_momentum``
    about the spin axis ``spin_axis_body_0`` (at gimbal angle ``θ = 0``). The gimbal
    axis ``gimbal_axis_body`` is body-fixed and must be orthogonal to
    ``spin_axis_body_0``. As the gimbal rotates by angle ``θ`` about ``gimbal_axis_body``,
    the spin axis rotates within the plane orthogonal to the gimbal axis, and the
    rotor momentum vector in the body frame becomes::

        ŝ(θ) = cos(θ) · spin_axis_body_0 + sin(θ) · (gimbal_axis_body × spin_axis_body_0)
        h(θ) = rotor_momentum · ŝ(θ)

    The controllable output torque on the spacecraft body from gimbaling at rate
    ``θ̇`` is ``-rotor_momentum · θ̇ · t̂(θ)`` where
    ``t̂(θ) = gimbal_axis_body × ŝ(θ)``.
    """

    body_name: str
    gimbal_axis_body: np.ndarray  # shape (3,) unit vector, body frame
    spin_axis_body_0: np.ndarray  # shape (3,) unit vector at θ=0, body frame, ⟂ gimbal_axis
    rotor_momentum: float  # kg·m^2/s, constant rotor angular momentum magnitude
    gimbal_rate_limit: Optional[float] = None  # rad/s, None = unlimited
    gimbal_angle_limit: Optional[float] = None  # rad, symmetric ±, None = unlimited
    name: str | None = None

    def __post_init__(self) -> None:
        self.gimbal_axis_body = np.asarray(self.gimbal_axis_body, dtype=float)
        self.spin_axis_body_0 = np.asarray(self.spin_axis_body_0, dtype=float)


@dataclass
class ThrusterSpec:
    """Configuration for one thruster."""

    body_name: str
    position_body: np.ndarray  # shape (3,) m, application point in body frame
    direction_body: np.ndarray  # shape (3,) unit thrust direction in body frame
    force_limit: float  # N max thrust
    name: str | None = None

    def __post_init__(self) -> None:
        self.position_body = np.asarray(self.position_body, dtype=float)
        self.direction_body = np.asarray(self.direction_body, dtype=float)


__all__ = [
    "ControlMomentGyroSpec",
    "MagneticBodySpec",
    "MagnetorquerSpec",
    "OrbitInit",
    "ReactionWheelSpec",
    "SurfaceSpec",
    "ThrusterSpec",
]
