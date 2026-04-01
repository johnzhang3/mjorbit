"""RK4 chief orbit propagator.

Units: km, km/s, s.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from mjorbit.cpu.orbit.state import OrbitState
from mjorbit.cpu.orbit.gravity import total_accel


def _derivatives(
    R: np.ndarray,
    V: np.ndarray,
    use_j2: bool,
    a_external: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """State derivatives: dR/dt = V, dV/dt = a(R) + a_external."""
    a = total_accel(R, use_j2=use_j2)
    if a_external is not None:
        a = a + a_external
    return V, a


def propagate_rk4(
    state: OrbitState,
    dt: float,
    use_j2: bool = True,
    a_external: Optional[np.ndarray] = None,
) -> OrbitState:
    """Propagate chief orbit by dt using RK4.

    Args:
        state: current chief orbit state
        dt: timestep in seconds
        use_j2: include J2 perturbation
        a_external: optional constant external acceleration in ECI, km/s^2
            (e.g. from thruster/drag orbit feedback)

    Returns:
        new OrbitState at t + dt
    """
    R, V = state.R_eci, state.V_eci

    k1R, k1V = _derivatives(R, V, use_j2, a_external)
    k2R, k2V = _derivatives(R + 0.5 * dt * k1R, V + 0.5 * dt * k1V, use_j2, a_external)
    k3R, k3V = _derivatives(R + 0.5 * dt * k2R, V + 0.5 * dt * k2V, use_j2, a_external)
    k4R, k4V = _derivatives(R + dt * k3R, V + dt * k3V, use_j2, a_external)

    R_new = R + (dt / 6.0) * (k1R + 2.0 * k2R + 2.0 * k3R + k4R)
    V_new = V + (dt / 6.0) * (k1V + 2.0 * k2V + 2.0 * k3V + k4V)

    return OrbitState(R_eci=R_new, V_eci=V_new, t=state.t + dt)
