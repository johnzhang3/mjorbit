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

import re
from pathlib import Path

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


def _read_mesh_assets(raw_xml: str, asset_dir: str | None) -> dict[str, bytes]:
    """Read mesh files referenced in ``raw_xml`` into a MuJoCo assets dict.

    Keys are the exact ``file`` strings used in the XML; values are the file
    bytes. Files are resolved relative to ``asset_dir`` (the model's source
    directory). Missing files are skipped so compilation can surface the error.
    """
    if asset_dir is None:
        return {}
    base = Path(asset_dir)
    assets: dict[str, bytes] = {}
    for file_ref in re.findall(r'<mesh\b[^>]*\bfile="([^"]+)"', raw_xml):
        candidate = Path(file_ref)
        path = candidate if candidate.is_absolute() else base / candidate
        if path.is_file():
            assets[file_ref] = path.read_bytes()
    return assets


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
        # Lazily-populated cache of geom_id -> (vertices, faces) for mesh geoms,
        # in the geom's local frame (mesh scale already baked in by MuJoCo).
        self._mesh_geometry: dict[int, tuple[np.ndarray, np.ndarray]] | None = None
        # Lazily-compiled standalone mujoco.MjModel rebuilt from the model's raw
        # XML, used to read geom attributes the native MjoModel bindings do not
        # expose (mesh vertices, geom_group, geom_matid, mat_rgba). ``False``
        # means "not attempted yet"; ``None`` means "attempted and unavailable".
        self._source_model: mujoco.MjModel | None | bool = False
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
        # geom_group / geom_matid / mat_rgba are not on the native bindings, so
        # read them from the rebuilt source model (same geom indexing) when it is
        # available; degrade gracefully when it is not.
        source = self._get_source_model()
        for geom_id in range(mjm.ngeom):
            body_id = int(mjm.geom_bodyid[geom_id])
            if body_id == 0:
                continue

            # Skip non-visual geoms (e.g. collision group 3); MuJoCo's default
            # visualizer shows geom groups 0-2, so we mirror that and avoid
            # drawing collision shells on top of the visual mesh.
            if source is not None and int(source.geom_group[geom_id]) > 2:
                continue

            body_name = mjm.body_name(body_id)
            geom_name = mjm.geom_name(geom_id)

            if int(mjm.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh = self._mesh_for_geom(geom_id)
            else:
                mesh = _make_trimesh(
                    mjm.geom_type[geom_id],
                    self._scale * mjm.geom_size[geom_id],
                )
            if mesh is None:
                continue

            # Color: a geom's rgba is only meaningful when it has no material;
            # if a material is assigned, MuJoCo renders with the material color
            # (it does not fold it into geom_rgba), so resolve it via the source
            # model. Fall back to the native geom_rgba when no source is available.
            if source is not None:
                matid = int(source.geom_matid[geom_id])
                if matid >= 0:
                    rgba = source.mat_rgba[matid].copy()
                else:
                    rgba = source.geom_rgba[geom_id].copy()
            else:
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

    def _mesh_for_geom(self, geom_id: int) -> trimesh.Trimesh | None:
        """Build a scaled trimesh for a MuJoCo mesh geom.

        The native ``MjoModel`` bindings do not expose mesh vertex/face data, so
        the geometry is recovered (once) by compiling a standalone
        ``mujoco.MjModel`` from the model's raw XML and reading its mesh arrays.
        Vertices are returned in the geom's local frame with the viewer's local
        scene scale applied.
        """
        if self._mesh_geometry is None:
            self._mesh_geometry = self._load_mesh_geometry()
        entry = self._mesh_geometry.get(geom_id)
        if entry is None:
            return None
        vertices, faces = entry
        return trimesh.Trimesh(
            vertices=self._scale * vertices,
            faces=faces,
            process=False,
        )

    def _get_source_model(self) -> mujoco.MjModel | None:
        """Compile (once) a standalone ``mujoco.MjModel`` from the raw XML.

        The native ``MjoModel`` bindings do not expose mesh vertex/face data,
        geom groups, or material colors, so the viewer recovers them from a
        vanilla MuJoCo model rebuilt from the model's source XML. Its geom
        indexing matches the ``MjoModel``'s. Returns ``None`` if the XML or its
        assets cannot be resolved.

        The XML is written back to a temp file in its own asset directory and
        compiled with ``from_xml_path`` so that *all* relative references —
        meshes plus ``<attach>`` / ``<include>`` sub-model files — resolve from
        disk. A flat ``from_xml_string`` asset dict cannot cover sub-model files.
        """
        if self._source_model is not False:
            return self._source_model  # type: ignore[return-value]
        self._source_model = None

        mjm = self._mjm
        raw_xml = getattr(mjm, "_raw_xml", None)
        if not raw_xml:
            return None
        asset_dir = getattr(mjm, "_asset_dir", None)
        try:
            if asset_dir is not None:
                import os
                import tempfile

                fd, tmp_path = tempfile.mkstemp(suffix=".xml", dir=asset_dir)
                try:
                    with os.fdopen(fd, "w") as handle:
                        handle.write(raw_xml)
                    self._source_model = mujoco.MjModel.from_xml_path(tmp_path)
                finally:
                    os.unlink(tmp_path)
            else:
                assets = _read_mesh_assets(raw_xml, asset_dir)
                self._source_model = mujoco.MjModel.from_xml_string(raw_xml, assets)
        except Exception:  # pragma: no cover - malformed/unresolvable assets
            self._source_model = None
        return self._source_model

    def _load_mesh_geometry(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Extract mesh vertices/faces from the standalone source model.

        Returns a mapping from geom_id (matching the MjoModel geom indexing) to
        ``(vertices, faces)`` arrays in the geom's local frame. Empty if the raw
        XML or asset files cannot be resolved.
        """
        source = self._get_source_model()
        if source is None:
            return {}

        geometry: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for geom_id in range(source.ngeom):
            if int(source.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            data_id = int(source.geom_dataid[geom_id])
            if data_id < 0:
                continue
            vadr = int(source.mesh_vertadr[data_id])
            vnum = int(source.mesh_vertnum[data_id])
            fadr = int(source.mesh_faceadr[data_id])
            fnum = int(source.mesh_facenum[data_id])
            vertices = source.mesh_vert[vadr : vadr + vnum].reshape(-1, 3).astype(float)
            faces = source.mesh_face[fadr : fadr + fnum].reshape(-1, 3).astype(np.int64)
            geometry[geom_id] = (vertices, faces)
        return geometry

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
