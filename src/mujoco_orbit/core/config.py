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

    def __post_init__(self) -> None:
        self.center_of_pressure_body = np.asarray(self.center_of_pressure_body, dtype=float)
        self.normal_body = np.asarray(self.normal_body, dtype=float)


@dataclass
class MagneticBodySpec:
    """Magnetic dipole source attached to a MuJoCo body."""

    body_name: str
    dipole_body: np.ndarray  # shape (3,) A·m^2, fixed residual dipole in body frame

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

    def __post_init__(self) -> None:
        self.axis_body = np.asarray(self.axis_body, dtype=float)


@dataclass
class MagnetorquerSpec:
    """Configuration for one magnetorquer."""

    body_name: str
    axis_body: np.ndarray  # shape (3,) axis in body frame (unit vector)
    dipole_limit: float  # A·m^2 max dipole magnitude

    def __post_init__(self) -> None:
        self.axis_body = np.asarray(self.axis_body, dtype=float)


@dataclass
class ThrusterSpec:
    """Configuration for one thruster."""

    body_name: str
    position_body: np.ndarray  # shape (3,) m, application point in body frame
    direction_body: np.ndarray  # shape (3,) unit thrust direction in body frame
    force_limit: float  # N max thrust

    def __post_init__(self) -> None:
        self.position_body = np.asarray(self.position_body, dtype=float)
        self.direction_body = np.asarray(self.direction_body, dtype=float)


__all__ = [
    "MagneticBodySpec",
    "MagnetorquerSpec",
    "OrbitInit",
    "ReactionWheelSpec",
    "SurfaceSpec",
    "ThrusterSpec",
]
