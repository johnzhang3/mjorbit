"""Gravitational acceleration models.

Units throughout: km, s.
Ported from OrbitX dynamics/gravity.py (Vallado formulation).
"""

from __future__ import annotations

import numpy as np

from mujoco_orbit.constants import GM_EARTH, J2_EARTH, R_EARTH


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


def encke_point_mass_relative_accel(rho: np.ndarray, r_chief: np.ndarray) -> np.ndarray:
    """Encke's identity for the two-body differential ``g_pm(r_chief+rho) - g_pm(r_chief)``.

    Avoids the catastrophic subtraction of two near-equal large gravity
    vectors when ``||rho|| << ||r_chief||`` (paper eq. 'encke').

    Args:
        rho: chief-relative position, shape (3,) km.
        r_chief: chief absolute ECI position, shape (3,) km.

    Returns:
        Differential acceleration, shape (3,) km/s^2.
    """
    rho = np.asarray(rho, dtype=float)
    r_chief = np.asarray(r_chief, dtype=float)
    rc2 = float(r_chief @ r_chief)
    if rc2 == 0.0:
        return np.zeros(3)
    rc = np.sqrt(rc2)

    sigma = (2.0 * float(rho @ r_chief) + float(rho @ rho)) / rc2
    one_plus_sigma = 1.0 + sigma
    one_plus_sigma_3_2 = one_plus_sigma * np.sqrt(one_plus_sigma)
    f_sigma = (
        sigma
        * (3.0 + 3.0 * sigma + sigma * sigma)
        / ((1.0 + one_plus_sigma_3_2) * one_plus_sigma_3_2)
    )

    return -GM_EARTH / (rc2 * rc) * (rho - f_sigma * (r_chief + rho))


def relative_accel(
    rho: np.ndarray,
    r_chief: np.ndarray,
    use_j2: bool = True,
) -> np.ndarray:
    """Differential gravity ``g(r_chief+rho) - g(r_chief)``.

    Uses Encke's identity for the two-body term (well-conditioned at single
    precision) and direct subtraction for the much smaller J2 perturbation.
    """
    a = encke_point_mass_relative_accel(rho, r_chief)
    if use_j2:
        r_chief = np.asarray(r_chief, dtype=float)
        rho = np.asarray(rho, dtype=float)
        a = a + j2_accel(r_chief + rho) - j2_accel(r_chief)
    return a
