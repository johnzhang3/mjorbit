"""Keplerian orbital element conversions.

Units: km, km/s, rad.
Ported from OrbitX elements/conversions.py (Vallado Algorithm 10).
"""

from __future__ import annotations

import numpy as np

from mjorbit.constants import GM_EARTH


def keplerian_to_cartesian(
    a: float,
    e: float,
    inc: float,
    raan: float,
    argp: float,
    nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Keplerian elements to ECI Cartesian state.

    Args:
        a: semi-major axis, km
        e: eccentricity
        inc: inclination, rad
        raan: right ascension of ascending node, rad
        argp: argument of perigee, rad
        nu: true anomaly, rad

    Returns:
        (R_eci, V_eci) each shape (3,) in km and km/s
    """
    p = a * (1.0 - e**2)
    r_pqw = p / (1.0 + e * np.cos(nu))

    # Position and velocity in perifocal frame
    cos_nu, sin_nu = np.cos(nu), np.sin(nu)
    R_pf = r_pqw * np.array([cos_nu, sin_nu, 0.0])
    V_pf = np.sqrt(GM_EARTH / p) * np.array([-sin_nu, e + cos_nu, 0.0])

    # Rotation matrices from perifocal to ECI
    # R3(-RAAN) @ R1(-inc) @ R3(-argp)
    R = _rot_pf_to_eci(raan, inc, argp)
    return R @ R_pf, R @ V_pf


def cartesian_to_keplerian(
    R_eci: np.ndarray, V_eci: np.ndarray
) -> dict[str, float]:
    """Convert ECI Cartesian state to Keplerian elements.

    Returns dict with keys: a, e, inc, raan, argp, nu (rad, km).
    """
    r = np.linalg.norm(R_eci)
    v = np.linalg.norm(V_eci)
    mu = GM_EARTH

    h = np.cross(R_eci, V_eci)
    h_mag = np.linalg.norm(h)

    n = np.cross(np.array([0.0, 0.0, 1.0]), h)
    n_mag = np.linalg.norm(n)

    e_vec = ((v**2 - mu / r) * R_eci - np.dot(R_eci, V_eci) * V_eci) / mu
    e = np.linalg.norm(e_vec)

    energy = v**2 / 2.0 - mu / r
    a = -mu / (2.0 * energy)

    inc = np.arccos(h[2] / h_mag)

    raan = np.arctan2(n[1], n[0]) if n_mag > 1e-12 else 0.0

    argp = np.arccos(np.clip(np.dot(n, e_vec) / (n_mag * e + 1e-30), -1.0, 1.0))
    if e_vec[2] < 0.0:
        argp = 2.0 * np.pi - argp

    nu = np.arccos(np.clip(np.dot(e_vec, R_eci) / (e * r + 1e-30), -1.0, 1.0))
    if np.dot(R_eci, V_eci) < 0.0:
        nu = 2.0 * np.pi - nu

    return dict(a=a, e=e, inc=inc, raan=raan, argp=argp, nu=nu)


def _rot_pf_to_eci(raan: float, inc: float, argp: float) -> np.ndarray:
    """3x3 rotation from perifocal to ECI frame."""
    c_O, s_O = np.cos(raan), np.sin(raan)
    c_i, s_i = np.cos(inc), np.sin(inc)
    c_w, s_w = np.cos(argp), np.sin(argp)

    R = np.array([
        [c_O * c_w - s_O * s_w * c_i,  -c_O * s_w - s_O * c_w * c_i,   s_O * s_i],
        [s_O * c_w + c_O * s_w * c_i,  -s_O * s_w + c_O * c_w * c_i,  -c_O * s_i],
        [s_w * s_i,                      c_w * s_i,                       c_i      ],
    ])
    return R
