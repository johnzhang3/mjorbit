"""LVLH frame construction and transforms.

LVLH convention (RSW / Hill frame):
  x̂ = radial (R̂ = R/|R|)
  ŷ = along-track (h × R̂ direction, approximately velocity for circular)
  ẑ = cross-track (h = R × V direction)

Wait — standard RSW:
  R̂ = radial outward
  Ŝ = in orbital plane, perpendicular to R̂, in direction of motion
  Ŵ = orbit normal (R × V)

We use RSW = LVLH:
  x̂_L = R̂
  ŷ_L = Ŝ (along-track in orbital plane)
  ẑ_L = Ŵ = (R × V) / |R × V|

Units: km, km/s, rad/s.
"""

from __future__ import annotations

import numpy as np
from mjorbit.cpu.orbit.state import OrbitState, FrameCache
from mjorbit.cpu.orbit.gravity import total_accel


def update_frame_cache(orbit: OrbitState, use_j2: bool = False) -> FrameCache:
    """Build LVLH frame from chief orbit state.

    Args:
        orbit: current chief orbit state
        use_j2: include J2 in omega_dot computation (uses finite differences)

    Returns:
        FrameCache with rotation matrices, omega, omega_dot
    """
    R, V = orbit.R_eci, orbit.V_eci
    r = np.linalg.norm(R)

    # Basis vectors
    x_hat = R / r  # radial
    h = np.cross(R, V)
    h_mag = np.linalg.norm(h)
    z_hat = h / h_mag  # cross-track (orbit normal)
    y_hat = np.cross(z_hat, x_hat)  # along-track

    # C_LI: rows are LVLH basis vectors expressed in ECI
    C_LI = np.stack([x_hat, y_hat, z_hat], axis=0)  # shape (3,3)
    C_IL = C_LI.T

    # Angular velocity of LVLH frame: omega = h / r^2 along orbit normal
    # In LVLH coordinates: omega = (0, 0, h/r^2) but let's express in ECI first
    omega_eci = h / r**2  # = (R x V) / r^2
    omega_lvlh = C_LI @ omega_eci  # express in LVLH

    # omega_dot: d/dt(omega) in LVLH frame
    # omega_eci = h / r^2, where h = R x V
    # d(omega)/dt = dh/dt / r^2 - 2 * h * rdot / r^3
    # For point-mass: dh/dt = R x a = R x (-mu/r^3 R) = 0
    # For J2: dh/dt = R x a_J2 != 0 in general
    a_eci = total_accel(R, use_j2=use_j2)
    dh_dt = np.cross(R, a_eci)
    dr_dt = np.dot(R, V) / r  # d|R|/dt
    domega_dt_eci = dh_dt / r**2 - 2.0 * h * dr_dt / r**3
    omega_dot_lvlh = C_LI @ domega_dt_eci

    return FrameCache(
        C_LI=C_LI,
        C_IL=C_IL,
        omega_lvlh=omega_lvlh,
        omega_dot_lvlh=omega_dot_lvlh,
    )


def eci_to_lvlh_pos(r_eci: np.ndarray, R_ref: np.ndarray, C_LI: np.ndarray) -> np.ndarray:
    """Convert ECI position to LVLH position relative to chief.

    Args:
        r_eci: absolute ECI position, km
        R_ref: chief ECI position, km
        C_LI: LVLH-from-ECI rotation

    Returns:
        relative position in LVLH, km
    """
    return C_LI @ (r_eci - R_ref)


def lvlh_to_eci_pos(r_lvlh: np.ndarray, R_ref: np.ndarray, C_IL: np.ndarray) -> np.ndarray:
    """Convert LVLH relative position to absolute ECI position."""
    return R_ref + C_IL @ r_lvlh


def eci_to_lvlh_vel(
    v_eci: np.ndarray,
    r_lvlh: np.ndarray,
    V_ref: np.ndarray,
    C_LI: np.ndarray,
    omega_lvlh: np.ndarray,
) -> np.ndarray:
    """Convert ECI velocity to LVLH relative velocity.

    v_lvlh = C_LI @ (v_eci - V_ref) - omega x r_lvlh
    """
    return C_LI @ (v_eci - V_ref) - np.cross(omega_lvlh, r_lvlh)
