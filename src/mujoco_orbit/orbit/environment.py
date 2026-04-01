"""Environment models: sun/eclipse, magnetic field, atmosphere.

Units: km, s, T, kg/m^3.
Ported and adapted from OrbitX environment models.
"""

from __future__ import annotations

import numpy as np

from mujoco_orbit.constants import B0_EARTH, OMEGA_EARTH, R_EARTH
from mujoco_orbit.orbit.state import EnvironmentCache, FrameCache, OrbitState

# ---------------------------------------------------------------------------
# Sun vector (low-fidelity, mean ecliptic approximation)
# ---------------------------------------------------------------------------

_J2000_EPOCH = 946727941.816  # Unix time of J2000.0 (2000-01-01 12:00 TT) — approx


def sun_vector_eci(t: float) -> np.ndarray:
    """Unit vector from Earth to Sun in ECI, low-fidelity.

    Uses a simple mean ecliptic longitude model (good to ~1 deg).

    Args:
        t: seconds since J2000.0

    Returns:
        unit vector, shape (3,)
    """
    T_jc = t / (36525.0 * 86400.0)  # Julian centuries since J2000
    lambda_sun = np.deg2rad(280.460 + 36000.771 * T_jc)  # mean longitude
    M_sun = np.deg2rad(357.528 + 35999.050 * T_jc)  # mean anomaly
    lambda_ecl = lambda_sun + np.deg2rad(1.915) * np.sin(M_sun) + np.deg2rad(0.020) * np.sin(
        2.0 * M_sun
    )  # ecliptic longitude
    eps = np.deg2rad(23.439 - 0.013 * T_jc)  # obliquity
    return np.array([
        np.cos(lambda_ecl),
        np.sin(lambda_ecl) * np.cos(eps),
        np.sin(lambda_ecl) * np.sin(eps),
    ])


def eclipse_factor(R_eci: np.ndarray, sun_hat: np.ndarray) -> float:
    """Cylindrical shadow model.  Returns 1.0 in sun, 0.0 in shadow.

    Args:
        R_eci: spacecraft position, km
        sun_hat: unit vector to Sun

    Returns:
        eclipse factor in [0, 1]
    """
    # Project spacecraft position onto anti-sun direction
    proj = -np.dot(R_eci, sun_hat)  # positive = on shadow side
    if proj < 0.0:
        return 1.0  # sunward side
    # Perpendicular distance to sun-Earth axis
    d_perp = np.linalg.norm(R_eci - np.dot(R_eci, sun_hat) * sun_hat)
    if d_perp < R_EARTH:
        return 0.0  # umbra (cylindrical)
    return 1.0


# ---------------------------------------------------------------------------
# Centered-dipole magnetic field
# ---------------------------------------------------------------------------

def dipole_field_eci(R_eci: np.ndarray, t: float) -> np.ndarray:
    """Centered-dipole magnetic field in ECI.

    Tilted dipole approximation: dipole axis = geographic north pole (simple version).
    For a more accurate model, tilt the dipole axis.  Here we use aligned dipole for
    a first pass (same as OrbitX's centered-dipole model).

    Args:
        R_eci: spacecraft ECI position, km
        t: seconds since J2000.0 (unused for aligned dipole)

    Returns:
        magnetic field vector in ECI, T
    """
    r = np.linalg.norm(R_eci)
    r_hat = R_eci / r
    # Dipole axis along -ECI z (geographic south, matching OrbitX convention)
    # Earth's magnetic dipole moment points from geomagnetic north to south,
    # so at the geographic north pole, B points radially inward.
    m_hat = np.array([0.0, 0.0, -1.0])
    factor = B0_EARTH * (R_EARTH / r) ** 3
    B = factor * (3.0 * np.dot(m_hat, r_hat) * r_hat - m_hat)
    return B


# ---------------------------------------------------------------------------
# Atmosphere
# ---------------------------------------------------------------------------

# Exponential atmosphere: density = rho0 * exp(-(h - h0) / H)
# Simple two-zone model (LEO range)
_ATM_H0_KM = 400.0  # km reference altitude
_ATM_RHO0 = 2.62e-13  # kg/m^3 at 400 km
_ATM_H_SCALE = 58.2  # km scale height near 400 km


def atm_density(R_eci: np.ndarray) -> float:
    """Exponential atmosphere density at given ECI position.

    Args:
        R_eci: position, km

    Returns:
        density, kg/m^3
    """
    alt_km = np.linalg.norm(R_eci) - R_EARTH
    rho = _ATM_RHO0 * np.exp(-(alt_km - _ATM_H0_KM) / _ATM_H_SCALE)
    return float(np.maximum(rho, 0.0))


def atmosphere_relative_velocity_eci(
    V_sc_eci: np.ndarray, R_eci: np.ndarray
) -> np.ndarray:
    """Atmosphere-relative velocity of spacecraft in ECI.

    Atmosphere co-rotates with Earth at OMEGA_EARTH about ECI z-axis.

    Args:
        V_sc_eci: spacecraft ECI velocity, km/s
        R_eci: spacecraft ECI position, km

    Returns:
        v_rel in ECI, km/s
    """
    omega_earth = np.array([0.0, 0.0, OMEGA_EARTH])
    v_atm = np.cross(omega_earth, R_eci)  # km/s
    return V_sc_eci - v_atm


# ---------------------------------------------------------------------------
# Cache update
# ---------------------------------------------------------------------------

def update_environment_cache(orbit: OrbitState, frame_cache: FrameCache) -> EnvironmentCache:
    """Recompute all environment quantities from current chief orbit state."""
    sun_hat = sun_vector_eci(orbit.t)
    ecl = eclipse_factor(orbit.R_eci, sun_hat)
    B_eci = dipole_field_eci(orbit.R_eci, orbit.t)
    rho = atm_density(orbit.R_eci)
    omega_earth = np.array([0.0, 0.0, OMEGA_EARTH])
    return EnvironmentCache(
        sun_vector_eci=sun_hat,
        eclipse=ecl,
        mag_field_eci=B_eci,
        atmosphere_omega_eci=omega_earth,
        atm_density=rho,
    )
