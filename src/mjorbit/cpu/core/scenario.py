"""CPUScenario — owns all simulation state for the CPU reference path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import mujoco

from mjorbit.cpu.core.config import CPUScenarioCfg, SurfaceCfg, MagneticBodyCfg
from mjorbit.cpu.core.actuators import ActuatorState
from mjorbit.cpu.orbit.state import OrbitState, FrameCache, EnvironmentCache


@dataclass
class SurfaceMetadata:
    """Resolved surface metadata after compiling (body_id is resolved)."""

    body_id: int
    center_of_pressure_body: np.ndarray  # shape (3,) m  (MuJoCo SI)
    normal_body: np.ndarray  # shape (3,)
    area: float  # m^2
    drag_coeff: float
    srp_coeff: float
    use_drag: bool
    use_srp: bool


@dataclass
class MagneticMetadata:
    """Resolved magnetic body metadata after compiling."""

    body_id: int
    dipole_body: np.ndarray  # shape (3,) A·m^2


@dataclass
class CPUScenario:
    """Full simulation state for one CPU reference scenario."""

    mjm: mujoco.MjModel
    mjd: mujoco.MjData
    orbit: OrbitState
    frame_cache: FrameCache
    env_cache: EnvironmentCache
    surfaces: list[SurfaceMetadata]
    magnetic_bodies: list[MagneticMetadata]
    actuator_state: ActuatorState
    cfg: CPUScenarioCfg

    # Per-body wrench buffer [n_body, 6]: [fx, fy, fz, tx, ty, tz] in MuJoCo world frame
    # Units: N, N·m  (SI — MuJoCo boundary)
    _wrench_buffer: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._wrench_buffer = np.zeros((self.mjm.nbody, 6))

    def clear_wrench_buffer(self) -> None:
        self._wrench_buffer[:] = 0.0
        self.mjd.xfrc_applied[:] = 0.0

    # ------------------------------------------------------------------
    # Body state helpers (Phase 3)
    # ------------------------------------------------------------------

    def body_id(self, name: str) -> int:
        """Resolve MuJoCo body name to id. Raises ValueError if not found."""
        bid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' not found in model")
        return bid

    def body_com_pos(self, body_id: int) -> np.ndarray:
        """Body COM position in world (LVLH) frame, meters."""
        return self.mjd.xipos[body_id].copy()

    def body_com_quat(self, body_id: int) -> np.ndarray:
        """Body COM orientation as quaternion [w, x, y, z] in world frame."""
        return self.mjd.xquat[body_id].copy()

    def body_com_rotmat(self, body_id: int) -> np.ndarray:
        """Body COM orientation as 3x3 rotation matrix (world-from-body)."""
        return self.mjd.ximat[body_id].reshape(3, 3).copy()

    def body_com_vel(self, body_id: int) -> np.ndarray:
        """Body COM linear velocity in world frame, m/s."""
        return self.mjd.cvel[body_id, 3:].copy()

    def body_com_angvel(self, body_id: int) -> np.ndarray:
        """Body COM angular velocity in world frame, rad/s."""
        return self.mjd.cvel[body_id, :3].copy()

    def body_mass(self, body_id: int) -> float:
        """Body mass in kg."""
        return float(self.mjm.body_mass[body_id])
