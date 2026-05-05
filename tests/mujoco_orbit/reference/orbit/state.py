"""Core orbit and frame state dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OrbitState:
    """Chief spacecraft state in ECI frame."""

    R_eci: np.ndarray  # shape (3,) km
    V_eci: np.ndarray  # shape (3,) km/s
    t: float  # s

    def copy(self) -> "OrbitState":
        return OrbitState(self.R_eci.copy(), self.V_eci.copy(), self.t)


@dataclass
class FrameCache:
    """LVLH frame derived quantities, updated each step."""

    C_LI: np.ndarray  # shape (3,3)  LVLH-from-ECI rotation
    C_IL: np.ndarray  # shape (3,3)  ECI-from-LVLH rotation
    omega_lvlh: np.ndarray  # shape (3,) rad/s  angular velocity of LVLH in ECI, expressed in LVLH
    omega_dot_lvlh: np.ndarray  # shape (3,) rad/s^2

    @classmethod
    def identity(cls) -> "FrameCache":
        return cls(
            C_LI=np.eye(3),
            C_IL=np.eye(3),
            omega_lvlh=np.zeros(3),
            omega_dot_lvlh=np.zeros(3),
        )


@dataclass
class EnvironmentCache:
    """Environment quantities derived from orbit state and time."""

    sun_vector_eci: np.ndarray  # shape (3,) unit vector
    eclipse: float  # 0.0 = full shadow, 1.0 = full sun
    mag_field_eci: np.ndarray  # shape (3,) T
    atmosphere_omega_eci: np.ndarray  # shape (3,) rad/s  Earth rotation vector in ECI
    atm_density: float  # kg/m^3

    @classmethod
    def default(cls) -> "EnvironmentCache":
        return cls(
            sun_vector_eci=np.array([1.0, 0.0, 0.0]),
            eclipse=1.0,
            mag_field_eci=np.zeros(3),
            atmosphere_omega_eci=np.array([0.0, 0.0, 7.292115e-5]),
            atm_density=0.0,
        )
