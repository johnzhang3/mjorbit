"""Small orbit helpers for runnable examples."""

from __future__ import annotations

import numpy as np

from mujoco_orbit.constants import GM_EARTH


def circular_orbit_eci(radius_km: float, inclination_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ECI position/velocity for a circular orbit at true anomaly zero."""
    speed_km_s = np.sqrt(GM_EARTH / radius_km)
    return (
        np.array([radius_km, 0.0, 0.0]),
        np.array(
            [
                0.0,
                speed_km_s * np.cos(inclination_rad),
                speed_km_s * np.sin(inclination_rad),
            ]
        ),
    )
