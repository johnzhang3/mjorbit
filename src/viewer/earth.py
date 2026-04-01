"""Earth and body-trail rendering for viser.

Earth is rendered as an icosphere positioned at the correct LVLH offset
(radial direction, meters).  BodyTrail records a body's position history
and renders it as line segments.
"""

from __future__ import annotations

import numpy as np
import trimesh
import trimesh.visual
import trimesh.visual.material
import viser

from mujoco_orbit.constants import R_EARTH


def add_earth(
    server: viser.ViserServer,
    position: tuple[float, float, float],
) -> viser.SceneNodeHandle:
    """Add an Earth icosphere at *position* (meters, LVLH frame).

    In LVLH (RSW) the radial axis is +x, so Earth's centre sits at
    roughly ``(-R_orbit_m, 0, 0)``.
    """
    radius_m = R_EARTH * 1000.0  # km -> m
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=radius_m)
    color = np.array([30, 80, 200, 255], dtype=np.uint8)
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=color,
            metallicFactor=0.1,
            roughnessFactor=0.9,
        )
    )
    return server.scene.add_mesh_trimesh("/earth", mesh, position=position)


class BodyTrail:
    """Accumulates a body's position history and draws it as line segments."""

    def __init__(
        self,
        server: viser.ViserServer,
        name: str,
        max_points: int = 2000,
        color: tuple[int, int, int] = (255, 200, 50),
    ) -> None:
        self._server = server
        self._name = name
        self._max_points = max_points
        self._color = np.array(color, dtype=np.uint8)
        self._positions: list[np.ndarray] = []
        self._handle: viser.SceneNodeHandle | None = None

    def append(self, position: np.ndarray) -> None:
        self._positions.append(position.copy())
        if len(self._positions) > self._max_points:
            self._positions.pop(0)

    def render(self) -> None:
        if len(self._positions) < 2:
            return
        pts = np.array(self._positions)  # (N, 3)
        segments = np.stack([pts[:-1], pts[1:]], axis=1)  # (N-1, 2, 3)
        if self._handle is not None:
            self._handle.remove()
        self._handle = self._server.scene.add_line_segments(
            f"/trail/{self._name}",
            segments,
            colors=self._color,
            line_width=2.0,
        )
