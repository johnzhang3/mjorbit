"""Runtime actuator state for the MuJoCo-style data API.

These actuators are modeled outside MuJoCo rigid-body state on purpose:
- reaction wheels spin far faster than articulated-body timescales
- magnetorquers and thrusters have no meaningful internal degrees of freedom
- control moment gyros carry a constant-magnitude rotor momentum whose direction
  is steered by gimbal angles, which live outside MuJoCo state
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ActuatorData:
    """Mutable external actuator state for one ``MjoData`` instance."""

    # Reaction wheels
    rw_speed: np.ndarray  # shape (n_rw,) rad/s
    rw_momentum: np.ndarray  # shape (n_rw,) kg·m^2/s  (derived from speed * inertia)
    rw_inertia: np.ndarray  # shape (n_rw,) kg·m^2
    rw_torque_cmd: np.ndarray  # shape (n_rw,) N·m

    # Magnetorquers — commanded dipole magnitude per unit (axes come from cfg)
    mtq_dipole_cmd: np.ndarray  # shape (n_mtq,) A·m^2

    # Thrusters — commanded force magnitude
    thr_force_cmd: np.ndarray  # shape (n_thr,) N

    # Control moment gyros
    cmg_gimbal_angle: np.ndarray  # shape (n_cmg,) rad
    cmg_gimbal_rate_cmd: np.ndarray  # shape (n_cmg,) rad/s
    cmg_rotor_momentum: np.ndarray  # shape (n_cmg,) kg·m^2/s (constant per CMG)

    @classmethod
    def zeros(
        cls,
        n_rw: int,
        rw_inertia: np.ndarray,
        n_mtq: int,
        n_thr: int,
        n_cmg: int,
        cmg_rotor_momentum: np.ndarray,
    ) -> "ActuatorData":
        inertia = np.asarray(rw_inertia, dtype=float)
        cmg_h = np.asarray(cmg_rotor_momentum, dtype=float)
        return cls(
            rw_speed=np.zeros(n_rw),
            rw_momentum=np.zeros(n_rw),
            rw_inertia=inertia.copy(),
            rw_torque_cmd=np.zeros(n_rw),
            mtq_dipole_cmd=np.zeros(n_mtq),
            thr_force_cmd=np.zeros(n_thr),
            cmg_gimbal_angle=np.zeros(n_cmg),
            cmg_gimbal_rate_cmd=np.zeros(n_cmg),
            cmg_rotor_momentum=cmg_h.copy(),
        )

    def update_rw_momentum(self, rw_inertia: np.ndarray | None = None) -> None:
        """Recompute wheel angular momentum from speed and inertia."""
        inertia = self.rw_inertia if rw_inertia is None else np.asarray(rw_inertia, dtype=float)
        self.rw_momentum[:] = self.rw_speed * inertia
        self.rw_inertia[:] = inertia
