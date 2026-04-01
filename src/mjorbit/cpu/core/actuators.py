"""External actuator state — reaction wheels, magnetorquers, thrusters.

These actuators are modeled outside MuJoCo rigid-body state on purpose:
- reaction wheels spin far faster than articulated-body timescales
- magnetorquers and thrusters have no meaningful internal degrees of freedom
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class ActuatorState:
    """Mutable external actuator state for one scenario."""

    # Reaction wheels
    rw_speed: np.ndarray  # shape (n_rw,) rad/s
    rw_momentum: np.ndarray  # shape (n_rw,) kg·m^2/s  (derived from speed * inertia)
    rw_inertia: np.ndarray  # shape (n_rw,) kg·m^2  (constant)

    # Magnetorquers — commanded dipole magnitude per unit (axes come from cfg)
    mtq_dipole: np.ndarray  # shape (n_mtq,) A·m^2

    # Thrusters — commanded force magnitude
    thr_force: np.ndarray  # shape (n_thr,) N

    @classmethod
    def zeros(cls, n_rw: int, rw_inertia: np.ndarray, n_mtq: int, n_thr: int) -> "ActuatorState":
        return cls(
            rw_speed=np.zeros(n_rw),
            rw_momentum=np.zeros(n_rw),
            rw_inertia=np.asarray(rw_inertia, dtype=float),
            mtq_dipole=np.zeros(n_mtq),
            thr_force=np.zeros(n_thr),
        )

    def update_rw_momentum(self) -> None:
        """Recompute wheel angular momentum from speed and inertia."""
        self.rw_momentum[:] = self.rw_speed * self.rw_inertia
