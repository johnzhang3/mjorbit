# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Shared scene construction + per-frame update for single-cluster recordings.

Reuses the viewer's own scene classes (``MuJoCoScene``, the Blue-Marble Earth
mesh, star field, framing helpers) so recorded frames match the interactive
viewer. Everything is rendered in the chief-centered "eci" floating-origin
frame: the spacecraft sits at the render origin and the Earth is placed at
``-R_eci``.

Why the world is uniformly scaled down (``world`` factor): viser's offscreen
``get_render`` (used for recording) silently drops geometry whose world
coordinates exceed a few thousand units — a float-precision limit the live
GPU viewer does not hit. The Earth lives at ~6.8e6 m, so every render
coordinate is multiplied by ``world`` (default 1e-4) to keep it in the safe
range. Uniform scaling leaves the rendered image unchanged. The spacecraft is
separately magnified by ``mag`` so a metre-scale craft reads against the globe.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import viser
import viser.transforms as vtf

# Make ``src/`` imports work when run as a plain script.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mjorbit.constants import OMEGA_EARTH, R_EARTH  # noqa: E402
from mjorbit.step import mjo_forward  # noqa: E402
from viewer.bodies import MuJoCoScene  # noqa: E402
from viewer.earth import (  # noqa: E402
    EARTH_TEXTURE_HQ_PATH,
    _create_atmosphere_mesh,
    add_star_field,
    create_earth_mesh,
)
from viewer.framing import (  # noqa: E402
    DEFAULT_CAMERA_FOV,
    default_camera_pose,
    distance_for_fill,
    lvlh_basis_eci,
    spacecraft_bounding_radius,
)


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class SingleScene:
    """A world-scaled viser scene for one spacecraft cluster (eci frame)."""

    def __init__(
        self,
        server: viser.ViserServer,
        model,
        data,
        *,
        mag: float = 5000.0,
        world: float = 1e-4,
        track_body_id: int = 1,
        star_radius_units: float = 1400.0,
        atmosphere: bool = True,
        earth_lat: int = 128,
        earth_lon: int = 256,
    ) -> None:
        self.server = server
        self.model = model
        self.data = data
        self.mag = float(mag)
        self.world = float(world)
        self.track_body_id = int(track_body_id)
        self._render_scale = self.mag * self.world

        server.scene.set_up_direction("+z")
        server.scene.set_background_image(np.zeros((4, 4, 3), dtype=np.uint8))
        # Coordinates are kept small by ``world``; a generous far plane is fine.
        server.initial_camera.far = 5.0e6

        self._scene = MuJoCoScene(server, model, root_path="/sc")
        self._scene.set_scale(self._render_scale)

        earth_radius = R_EARTH * 1000.0 * self.world
        self._earth = server.scene.add_mesh_trimesh(
            "/earth",
            create_earth_mesh(
                earth_radius,
                texture_path=EARTH_TEXTURE_HQ_PATH,
                lat_segments=earth_lat,
                lon_segments=earth_lon,
            ),
            cast_shadow=False,
            receive_shadow=False,
        )
        self._atmo = None
        if atmosphere:
            self._atmo = server.scene.add_mesh_trimesh(
                "/earth_atmosphere",
                _create_atmosphere_mesh(earth_radius * 1.015),
                cast_shadow=False,
                receive_shadow=False,
            )
        add_star_field(server, radius=float(star_radius_units))
        self.update()

    # -- per-frame sync ----------------------------------------------------

    def update(self) -> None:
        self._scene.update(self.data, rotation=None, translation=None, scale_origin=np.zeros(3))
        earth_pos = tuple(-1000.0 * np.asarray(self.data.orbit.R_eci, dtype=float) * self.world)
        spin = _rotation_z(OMEGA_EARTH * float(self.data.orbit.t))
        wxyz = tuple(vtf.SO3.from_matrix(spin).wxyz)
        with self.server.atomic():
            self._earth.position = earth_pos
            self._earth.wxyz = wxyz
            if self._atmo is not None:
                self._atmo.position = earth_pos

    # -- camera ------------------------------------------------------------

    def body_render_position(self, body_id: int | None = None) -> np.ndarray:
        bid = self.track_body_id if body_id is None else body_id
        return self._render_scale * np.asarray(self.data.xpos[bid], dtype=float)

    def cluster_center_render(self, body_ids) -> np.ndarray:
        pts = [np.asarray(self.data.xpos[b], dtype=float) for b in body_ids]
        return self._render_scale * np.mean(pts, axis=0)

    def auto_distance(self, *, fov: float = DEFAULT_CAMERA_FOV, fill: float = 0.6) -> float:
        radius = spacecraft_bounding_radius(self.model, self.data, self.track_body_id)
        return distance_for_fill(radius * self._render_scale, fov=fov, fill=fill)

    def camera_pose(
        self,
        *,
        distance: float,
        target: np.ndarray | None = None,
        offset_rsw: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Camera (position, look_at) in render units, framing *target*."""
        tgt = self.body_render_position() if target is None else np.asarray(target, dtype=float)
        basis = lvlh_basis_eci(self.data.orbit.R_eci, self.data.orbit.V_eci)
        if offset_rsw is None:
            return default_camera_pose(tgt, distance=distance, basis=basis)
        off = np.asarray(offset_rsw, dtype=float)
        off = off / np.linalg.norm(off)
        return tgt + float(distance) * (off @ basis), tgt


def forward_to(model, data, qpos: np.ndarray, *, R_eci=None, V_eci=None) -> None:
    """Set a stored kinematic state and run forward kinematics for rendering."""
    np.copyto(np.asarray(data.qpos), np.asarray(qpos, dtype=float))
    if R_eci is not None:
        np.copyto(np.asarray(data.orbit.R_eci), np.asarray(R_eci, dtype=float))
    if V_eci is not None:
        np.copyto(np.asarray(data.orbit.V_eci), np.asarray(V_eci, dtype=float))
    mjo_forward(model, data)
