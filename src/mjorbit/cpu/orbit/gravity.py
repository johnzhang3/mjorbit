"""Gravitational acceleration models.

Units throughout: km, s.
Ported from OrbitX dynamics/gravity.py (Vallado formulation).
"""

from __future__ import annotations

import numpy as np
from mjorbit.constants import GM_EARTH, R_EARTH, J2_EARTH


def point_mass_accel(r_eci: np.ndarray) -> np.ndarray:
    """Two-body gravitational acceleration.

    Args:
        r_eci: position in ECI, shape (3,) km

    Returns:
        acceleration, shape (3,) km/s^2
    """
    r = np.linalg.norm(r_eci)
    return -GM_EARTH / r**3 * r_eci


def j2_accel(r_eci: np.ndarray) -> np.ndarray:
    """J2 perturbation acceleration in ECI.

    Args:
        r_eci: position in ECI, shape (3,) km

    Returns:
        J2 acceleration, shape (3,) km/s^2
    """
    x, y, z = r_eci
    r = np.linalg.norm(r_eci)
    factor = (3.0 / 2.0) * J2_EARTH * GM_EARTH * R_EARTH**2 / r**5
    z_r2 = (z / r) ** 2
    ax = factor * x * (5.0 * z_r2 - 1.0)
    ay = factor * y * (5.0 * z_r2 - 1.0)
    az = factor * z * (5.0 * z_r2 - 3.0)
    return np.array([ax, ay, az])


def total_accel(r_eci: np.ndarray, use_j2: bool = True) -> np.ndarray:
    """Sum of point-mass and optional J2 acceleration."""
    a = point_mass_accel(r_eci)
    if use_j2:
        a = a + j2_accel(r_eci)
    return a
