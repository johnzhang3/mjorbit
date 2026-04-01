# pyright: reportAttributeAccessIssue=false

"""MuJoCo body rendering for viser.

Extracts geoms from an MjModel, creates trimesh primitives for each,
and places them under per-body frame nodes in the viser scene tree.
Call ``update()`` each frame to sync body transforms from MjData.

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

    def __init__(self, server: viser.ViserServer, mjm: mujoco.MjModel) -> None:
        self._server = server
        self._mjm = mjm
        self._body_frames: list[viser.FrameHandle] = []
        self._build()

    def _build(self) -> None:
        mjm = self._mjm

        # One frame per non-world body
        for body_id in range(1, mjm.nbody):
            name = (
                mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, body_id)
                or f"body_{body_id}"
            )
            frame = self._server.scene.add_frame(
                f"/spacecraft/{name}",
                show_axes=False,
            )
            self._body_frames.append(frame)

        # One mesh per geom (skip world-body geoms)
        for geom_id in range(mjm.ngeom):
            body_id = mjm.geom_bodyid[geom_id]
            if body_id == 0:
                continue

            body_name = (
                mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, body_id)
                or f"body_{body_id}"
            )
            geom_name = (
                mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                or f"geom_{geom_id}"
            )

            mesh = _make_trimesh(mjm.geom_type[geom_id], mjm.geom_size[geom_id])
            if mesh is None:
                continue

            rgba = mjm.geom_rgba[geom_id].copy()
            if rgba[3] == 0.0:
                # Transparent fallback — use default grey
                rgba = np.array([0.5, 0.5, 0.5, 1.0])
            _apply_color(mesh, rgba)

            self._server.scene.add_mesh_trimesh(
                f"/spacecraft/{body_name}/{geom_name}",
                mesh,
                position=tuple(mjm.geom_pos[geom_id]),
                wxyz=tuple(mjm.geom_quat[geom_id]),
            )

    def update(self, mjd: mujoco.MjData) -> None:
        """Sync body frame transforms from simulation state."""
        with self._server.atomic():
            for i, frame in enumerate(self._body_frames):
                body_id = i + 1  # skip worldbody 0
                frame.position = tuple(mjd.xpos[body_id])
                frame.wxyz = tuple(mjd.xquat[body_id])
