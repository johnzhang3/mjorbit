# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""MuJoCo body rendering for viser.

Extracts geoms from an MjoModel, creates trimesh primitives for each,
and places them under per-body frame nodes in the viser scene tree.
Call ``update()`` each frame to sync body transforms from MjoData.

Follows the approach from judo/visualizers/model.py:
- Frame node per body (transform updated from data.xpos / data.xquat)
- Mesh node per geom (static local offset from mjm.geom_pos / mjm.geom_quat)
"""

from __future__ import annotations

import mujoco
import numpy as np
import trimesh
import trimesh.visual
import trimesh.visual.material
import viser
import viser.transforms as vtf


def _rgba_to_uint8(rgba: np.ndarray) -> np.ndarray:
    return (np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8)


def _make_trimesh(geom_type: int, size: np.ndarray) -> trimesh.Trimesh | None:
    """Create a trimesh primitive matching the MuJoCo geom type."""
    size = np.asarray(size, dtype=float)
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2.0 * size)
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=3, radius=float(size[0]))
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(
            height=float(2.0 * size[1]), radius=float(size[0])
        )
    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(
            radius=float(size[0]), height=float(2.0 * size[1])
        )
    return None


def _apply_color(mesh: trimesh.Trimesh, rgba: np.ndarray) -> None:
    color = _rgba_to_uint8(rgba)
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=color,
            metallicFactor=0.3,
            roughnessFactor=0.7,
            alphaMode="BLEND" if color[3] < 255 else "OPAQUE",
        )
    )


class MuJoCoScene:
    """Renders MuJoCo model geometry in a viser scene.

    Creates a frame node for every non-world body and a mesh node for
    every geom attached to those bodies.  Geom meshes use the *model*
    local offset (``mjm.geom_pos``, ``mjm.geom_quat``); body frames
    are updated each tick from ``mjd.xpos`` / ``mjd.xquat``.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        mjm,
        *,
        root_path: str = "/spacecraft",
    ) -> None:
        self._server = server
        self._mjm = mjm
        self._root_path = root_path.rstrip("/")
        self._scale = 1.0
        self._root_frame = self._server.scene.add_frame(
            self._root_path,
            show_axes=False,
        )
        self._body_frames: list[viser.FrameHandle] = []
        self._geom_handles: list[viser.SceneNodeHandle] = []
        self._build()

    def _build(self) -> None:
        self._build_body_frames()
        self._build_geom_meshes()

    def _build_body_frames(self) -> None:
        mjm = self._mjm
        if self._body_frames:
            return
        for body_id in range(1, mjm.nbody):
            name = mjm.body_name(body_id)
            self._body_frames.append(
                self._server.scene.add_frame(
                    f"{self._root_path}/{name}",
                    show_axes=False,
                )
            )

    def _build_geom_meshes(self) -> None:
        mjm = self._mjm
        for geom_id in range(mjm.ngeom):
            body_id = int(mjm.geom_bodyid[geom_id])
            if body_id == 0:
                continue

            body_name = mjm.body_name(body_id)
            geom_name = mjm.geom_name(geom_id)

            mesh = _make_trimesh(
                mjm.geom_type[geom_id],
                self._scale * mjm.geom_size[geom_id],
            )
            if mesh is None:
                continue

            rgba = mjm.geom_rgba[geom_id].copy()
            if rgba[3] == 0.0:
                # Transparent fallback — use default grey
                rgba = np.array([0.5, 0.5, 0.5, 1.0])
            _apply_color(mesh, rgba)

            self._geom_handles.append(
                self._server.scene.add_mesh_trimesh(
                    f"{self._root_path}/{body_name}/{geom_name}",
                    mesh,
                    position=tuple(self._scale * mjm.geom_pos[geom_id]),
                    wxyz=tuple(mjm.geom_quat[geom_id]),
                )
            )

    def set_scale(self, scale: float) -> None:
        self._scale = float(scale)
        for handle in self._geom_handles:
            handle.remove()
        self._geom_handles.clear()
        self._build_geom_meshes()

    def remove(self) -> None:
        """Remove every scene node owned by this scene."""
        for handle in self._geom_handles:
            handle.remove()
        self._geom_handles.clear()
        for frame in self._body_frames:
            frame.remove()
        self._body_frames.clear()
        self._root_frame.remove()

    def update(
        self,
        mjd,
        *,
        rotation: np.ndarray | None = None,
        translation: np.ndarray | None = None,
        scale_origin: np.ndarray | None = None,
    ) -> None:
        """Sync body frame transforms from simulation state."""
        rot = None if rotation is None else np.asarray(rotation, dtype=float)
        trans = None if translation is None else np.asarray(translation, dtype=float)
        origin = None if scale_origin is None else np.asarray(scale_origin, dtype=float)
        with self._server.atomic():
            for i, frame in enumerate(self._body_frames):
                body_id = i + 1  # skip worldbody 0
                pos = mjd.xpos[body_id].copy()
                if rot is not None:
                    pos = rot @ pos
                if trans is not None:
                    pos = pos + trans
                if origin is None:
                    pos = self._scale * pos
                else:
                    pos = origin + self._scale * (pos - origin)
                frame.position = tuple(pos)

                if rot is None:
                    frame.wxyz = tuple(mjd.xquat[body_id])
                else:
                    body_rot = mjd.xmat[body_id].reshape(3, 3)
                    frame.wxyz = tuple(vtf.SO3.from_matrix(rot @ body_rot).wxyz)
