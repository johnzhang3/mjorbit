"""Default MuJoCo + NumPy simulator for coupled orbital and multibody dynamics."""

from mujoco_orbit.core.compile import compile
from mujoco_orbit.core.step import step

__all__ = ["compile", "step"]
