# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Earth and body-trail rendering for viser.

The Earth is rendered as a UV-textured sphere using NASA's public-domain
"Blue Marble" composite (see assets/README.md). The texture wraps an
equirectangular lat/lon grid whose seam column is duplicated so the date
line interpolates cleanly. ``EarthVisual`` also supports an optional thin
atmosphere shell and spins the globe at the sidereal rate from the orbit
propagator clock.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import cast

import numpy as np
import trimesh
import trimesh.visual
import trimesh.visual.material
import viser
import viser.transforms as vtf

from mujoco_orbit.constants import OMEGA_EARTH, R_EARTH

EARTH_TEXTURE_PATH = Path(__file__).parent / "assets" / "earth_day_2k.jpg"
# Higher-resolution NASA Blue Marble (5400x2700) for banner-quality renders.
EARTH_TEXTURE_HQ_PATH = Path(__file__).parent / "assets" / "earth_day_5400.jpg"

_FALLBACK_RGBA = np.array([30, 80, 200, 255], dtype=np.uint8)


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def create_earth_mesh(
    radius: float,
    *,
    texture_path: str | Path | None = None,
    lat_segments: int = 64,
    lon_segments: int = 128,
) -> trimesh.Trimesh:
    """Textured UV sphere for an equirectangular Earth map.

    The +Z axis is the north pole and the texture's prime meridian crosses
    +X, so the mesh can be spun about +Z by an Earth rotation angle. Falls
    back to a flat-colored sphere when the texture is unavailable.
    """
    path = Path(texture_path) if texture_path is not None else EARTH_TEXTURE_PATH
    if not path.is_file():
        return create_fallback_earth_mesh(radius)
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(path.read_bytes())).convert("RGB")
    except Exception:
        return create_fallback_earth_mesh(radius)

    lat = np.linspace(np.pi / 2.0, -np.pi / 2.0, lat_segments + 1)
    lon = np.linspace(-np.pi, np.pi, lon_segments + 1)  # seam column duplicated
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

    cos_lat = np.cos(lat_grid)
    vertices = radius * np.stack(
        [
            cos_lat * np.cos(lon_grid),
            cos_lat * np.sin(lon_grid),
            np.sin(lat_grid),
        ],
        axis=-1,
    ).reshape(-1, 3)

    # OBJ-style UV origin (lower left); trimesh's GLB export flips V.
    uv = np.stack(
        [
            (lon_grid + np.pi) / (2.0 * np.pi),
            0.5 + lat_grid / np.pi,
        ],
        axis=-1,
    ).reshape(-1, 2)

    ncols = lon_segments + 1
    i, j = np.meshgrid(np.arange(lat_segments), np.arange(lon_segments), indexing="ij")
    i, j = i.ravel(), j.ravel()
    a = i * ncols + j  # north-west
    b = (i + 1) * ncols + j  # south-west
    c = (i + 1) * ncols + j + 1  # south-east
    d = i * ncols + j + 1  # north-east
    # Drop the degenerate triangle of each pole quad (the pole row collapses
    # to a single point).
    faces = np.concatenate(
        [
            np.stack([a, b, d], axis=-1)[i > 0],
            np.stack([b, c, d], axis=-1)[i < lat_segments - 1],
        ],
        axis=0,
    )

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=image,
        metallicFactor=0.0,
        roughnessFactor=1.0,
        # Faint self-illumination keeps the night side readable.
        emissiveTexture=image,
        emissiveFactor=[0.25, 0.25, 0.25],
    )
    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=trimesh.visual.TextureVisuals(uv=uv, material=material),
        process=False,  # keep the duplicated seam/pole vertices
    )


def create_fallback_earth_mesh(radius: float) -> trimesh.Trimesh:
    """Untextured blue sphere used when the texture asset is missing."""
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=_FALLBACK_RGBA,
            metallicFactor=0.1,
            roughnessFactor=0.9,
        )
    )
    return mesh


def _create_atmosphere_mesh(radius: float) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=np.array([135, 180, 235, 40], dtype=np.uint8),
            metallicFactor=0.0,
            roughnessFactor=1.0,
            alphaMode="BLEND",
        )
    )
    return mesh


class EarthVisual:
    """Textured, spinning Earth in the active render frame.

    Call :meth:`update` each frame with the Earth-center position (metres,
    render frame) and the orbit-propagator time so the globe spins at the
    sidereal rate. ``rotation`` carries an extra frame rotation (e.g. the
    inertial-to-LVLH matrix when rendering in the LVLH frame).
    """

    def __init__(
        self,
        server: viser.ViserServer,
        *,
        path: str = "/earth",
        textured: bool = True,
        atmosphere: bool = True,
        texture_path: str | Path | None = None,
        spin_epoch_angle: float = 0.0,
        lat_segments: int = 64,
        lon_segments: int = 128,
    ) -> None:
        self._server = server
        self._spin_epoch_angle = float(spin_epoch_angle)
        radius_m = R_EARTH * 1000.0  # km -> m

        mesh = (
            create_earth_mesh(
                radius_m,
                texture_path=texture_path,
                lat_segments=lat_segments,
                lon_segments=lon_segments,
            )
            if textured
            else create_fallback_earth_mesh(radius_m)
        )
        self.handle = server.scene.add_mesh_trimesh(
            path, mesh, cast_shadow=False, receive_shadow=False
        )
        self._atmo_handle: viser.SceneNodeHandle | None = None
        if textured and atmosphere:
            self._atmo_handle = server.scene.add_mesh_trimesh(
                f"{path}_atmosphere",
                _create_atmosphere_mesh(radius_m * 1.015),
                cast_shadow=False,
                receive_shadow=False,
            )

    def update(
        self,
        *,
        position: np.ndarray,
        sim_time: float = 0.0,
        rotation: np.ndarray | None = None,
    ) -> None:
        spin = _rotation_z(self._spin_epoch_angle + OMEGA_EARTH * float(sim_time))
        if rotation is not None:
            spin = np.asarray(rotation, dtype=float) @ spin
        wxyz = tuple(vtf.SO3.from_matrix(np.asarray(spin, dtype=np.float64)).wxyz)
        pos = tuple(np.asarray(position, dtype=float))
        with self._server.atomic():
            self.handle.position = pos
            self.handle.wxyz = wxyz
            if self._atmo_handle is not None:
                self._atmo_handle.position = pos

    def set_visible(self, visible: bool) -> None:
        self.handle.visible = visible
        if self._atmo_handle is not None:
            self._atmo_handle.visible = visible

    def remove(self) -> None:
        self.handle.remove()
        if self._atmo_handle is not None:
            self._atmo_handle.remove()
            self._atmo_handle = None


def add_earth(
    server: viser.ViserServer,
    position: tuple[float, float, float],
    *,
    textured: bool = True,
) -> viser.SceneNodeHandle:
    """Add an Earth sphere at *position* in the active viewer frame.

    Kept for backwards compatibility; new code should prefer
    :class:`EarthVisual`, which also handles per-frame position and spin.
    """
    visual = EarthVisual(server, textured=textured, atmosphere=False)
    visual.update(position=np.asarray(position, dtype=float))
    return visual.handle


def add_star_field(
    server: viser.ViserServer,
    *,
    radius: float,
    path: str = "/stars",
    count: int = 1500,
    seed: int = 0,
) -> viser.SceneNodeHandle:
    """Scatter faint points on a far sphere so orbit shots read as space."""
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(count, 3))
    points *= radius / np.linalg.norm(points, axis=1, keepdims=True)
    brightness = rng.integers(110, 255, size=(count, 1)).astype(np.uint8)
    colors = np.repeat(brightness, 3, axis=1)
    return server.scene.add_point_cloud(
        path,
        points=points.astype(np.float32),
        colors=colors,
        point_size=radius * 0.0015,
        point_shape="circle",
        precision="float32",  # star shell radii overflow float16
    )


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
