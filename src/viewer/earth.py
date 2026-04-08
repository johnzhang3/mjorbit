# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Earth and body-trail rendering for viser.

Earth is rendered as an icosphere positioned at the correct LVLH offset
(radial direction, meters).  BodyTrail records a body's position history
and renders it as line segments.
"""

from __future__ import annotations

from typing import cast

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
        path_prefix: str = "/trail",
    ) -> None:
        self._server = server
        self._name = name
        self._max_points = max_points
        self._color = np.array(color, dtype=np.uint8)
        self._path_prefix = path_prefix.rstrip("/")
        self._scale = 1.0
        self._positions: list[np.ndarray] = []
        self._handle: viser.SplineCatmullRomHandle | None = None

    def append(self, position: np.ndarray) -> None:
        self._positions.append(position.copy())
        if len(self._positions) > self._max_points:
            self._positions.pop(0)

    def clear(self) -> None:
        self._positions.clear()
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def set_scale(self, scale: float) -> None:
        self._scale = float(scale)

    def _render_points(self, current_position: np.ndarray | None = None) -> np.ndarray | None:
        if not self._positions and current_position is None:
            return None

        pts = np.array(self._positions, dtype=float) if self._positions else np.empty((0, 3))
        if current_position is not None:
            current = np.asarray(current_position, dtype=float).reshape(1, 3)
            if pts.size == 0 or np.linalg.norm(current[0] - pts[-1]) > 1.0e-12:
                pts = np.concatenate([pts, current], axis=0)
        if len(pts) < 2:
            return None
        return self._scale * pts

    def render(
        self,
        *,
        current_position: np.ndarray | None = None,
        rotation: np.ndarray | None = None,
        translation: np.ndarray | None = None,
    ) -> None:
        pts = self._render_points(current_position=current_position)
        if pts is None:
            if self._handle is not None:
                self._handle.remove()
                self._handle = None
            return
        if rotation is not None:
            pts = pts @ np.asarray(rotation, dtype=float).T
        if translation is not None:
            pts = pts + np.asarray(translation, dtype=float)
        if self._handle is None:
            color = cast(tuple[int, int, int], tuple(int(c) for c in self._color))
            self._handle = self._server.scene.add_spline_catmull_rom(
                f"{self._path_prefix}/{self._name}",
                points=pts.astype(np.float32),
                line_width=2.0,
                color=color,
            )
        else:
            self._handle.points = pts.astype(np.float32)
