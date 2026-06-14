# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Instanced fleet rendering for viser.

``BatchedMuJoCoScene`` renders ``nworld`` copies of one MuJoCo model using
viser's batched-mesh API: one instanced draw call per geom, with per-instance
transforms supplied each frame. This keeps the scene-node count at
O(ngeom) instead of O(nworld * ngeom), which is what makes thousands of
simultaneously simulated robots (e.g. a ``mjorbit_warp`` ``nworld``
batch) renderable in a browser.

The class is backend-agnostic: it takes a CPU-side model for the static geom
metadata and per-frame ``(nworld, nbody, 3)`` / ``(nworld, nbody, 4)`` body
pose arrays, which the caller can pull from any simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import viser

from viewer.bodies import _make_trimesh, _rgba_to_uint8


def _quat_to_mats(quats: np.ndarray) -> np.ndarray:
    """Rotation matrices for an (..., 4) array of (w,x,y,z) quaternions."""
    w, x, y, z = quats[..., 0], quats[..., 1], quats[..., 2], quats[..., 3]
    mats = np.empty(quats.shape[:-1] + (3, 3), dtype=np.float64)
    mats[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    mats[..., 0, 1] = 2.0 * (x * y - w * z)
    mats[..., 0, 2] = 2.0 * (x * z + w * y)
    mats[..., 1, 0] = 2.0 * (x * y + w * z)
    mats[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    mats[..., 1, 2] = 2.0 * (y * z - w * x)
    mats[..., 2, 0] = 2.0 * (x * z - w * y)
    mats[..., 2, 1] = 2.0 * (y * z + w * x)
    mats[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return mats


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of (..., 4) by (4,) quaternions in (w,x,y,z) order."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


@dataclass
class _GeomEntry:
    body_id: int
    local_pos: np.ndarray  # (3,) scaled geom offset in the body frame
    local_quat: np.ndarray  # (4,) geom orientation in the body frame
    has_rotation: bool
    handle: viser.BatchedMeshHandle


class BatchedMuJoCoScene:
    """Renders ``nworld`` instances of one MuJoCo model with instanced meshes.

    Parameters
    ----------
    server:
        The viser server.
    mjm:
        CPU-side model exposing ``ngeom``, ``geom_bodyid``, ``geom_type``,
        ``geom_size``, ``geom_rgba``, ``geom_pos``, ``geom_quat``
        (``mjorbit.MjoModel`` works).
    nworld:
        Number of instances.
    scale:
        Visual scale applied to every spacecraft about its own anchor body
        (``anchor_body``), so real translations in the world frame render 1:1
        rather than scale-amplified.
    anchor_body:
        Body id whose position anchors the per-world scaling (default 1, the
        first non-world body).
    """

    def __init__(
        self,
        server: viser.ViserServer,
        mjm,
        nworld: int,
        *,
        root_path: str = "/fleet",
        scale: float = 1.0,
        anchor_body: int = 1,
        lod: str = "auto",
    ) -> None:
        self._server = server
        self._nworld = int(nworld)
        self._scale = float(scale)
        self._anchor_body = int(anchor_body)
        self._entries: list[_GeomEntry] = []

        identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
        zero_wxyz = np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (self._nworld, 1)
        )
        zero_pos = np.zeros((self._nworld, 3), dtype=np.float32)

        for geom_id in range(int(mjm.ngeom)):
            body_id = int(mjm.geom_bodyid[geom_id])
            if body_id == 0:
                continue
            mesh = _make_trimesh(
                int(mjm.geom_type[geom_id]),
                self._scale * np.asarray(mjm.geom_size[geom_id], dtype=float),
            )
            if mesh is None:
                continue
            rgba = np.asarray(mjm.geom_rgba[geom_id], dtype=float).copy()
            if rgba[3] == 0.0:
                rgba = np.array([0.5, 0.5, 0.5, 1.0])
            color = _rgba_to_uint8(rgba)

            local_quat = np.asarray(mjm.geom_quat[geom_id], dtype=np.float64).copy()
            handle = server.scene.add_batched_meshes_simple(
                f"{root_path.rstrip('/')}/geom_{geom_id}",
                vertices=np.asarray(mesh.vertices, dtype=np.float32),
                faces=np.asarray(mesh.faces, dtype=np.uint32),
                batched_wxyzs=zero_wxyz.copy(),
                batched_positions=zero_pos.copy(),
                batched_colors=(int(color[0]), int(color[1]), int(color[2])),
                opacity=float(rgba[3]) if rgba[3] < 1.0 else None,
                lod=lod,  # type: ignore[arg-type]
            )
            self._entries.append(
                _GeomEntry(
                    body_id=body_id,
                    local_pos=self._scale
                    * np.asarray(mjm.geom_pos[geom_id], dtype=np.float64),
                    local_quat=local_quat,
                    has_rotation=bool(np.abs(local_quat - identity_quat).max() > 1e-12),
                    handle=handle,
                )
            )

    @property
    def num_draw_calls(self) -> int:
        return len(self._entries)

    def update(
        self,
        xpos: np.ndarray,
        xquat: np.ndarray,
        *,
        offsets: np.ndarray | None = None,
    ) -> None:
        """Sync per-instance transforms from batched body poses.

        Parameters
        ----------
        xpos, xquat:
            Body world poses, shapes ``(nworld, nbody, 3)`` metres and
            ``(nworld, nbody, 4)`` (w,x,y,z). Each world's poses are scaled by
            ``scale`` about that world's anchor body, so the anchor's own
            world-frame motion renders unamplified.
        offsets:
            Optional ``(nworld, 3)`` per-instance render offsets in metres,
            applied after scaling (e.g. a banner-cloud placement).
        """
        xpos = np.asarray(xpos, dtype=np.float64)
        xquat = np.asarray(xquat, dtype=np.float64)
        if xpos.ndim != 3 or xpos.shape[0] != self._nworld:
            raise ValueError(f"xpos must have shape ({self._nworld}, nbody, 3)")

        anchor = xpos[:, self._anchor_body]
        mats = _quat_to_mats(xquat)  # (nworld, nbody, 3, 3)
        with self._server.atomic():
            for entry in self._entries:
                b = entry.body_id
                pos = anchor + self._scale * (xpos[:, b] - anchor)
                if np.any(entry.local_pos):
                    pos = pos + mats[:, b] @ entry.local_pos
                if offsets is not None:
                    pos = pos + offsets
                if entry.has_rotation:
                    quat = _quat_mul(xquat[:, b], entry.local_quat)
                else:
                    quat = xquat[:, b]
                entry.handle.batched_positions = pos.astype(np.float32)
                entry.handle.batched_wxyzs = quat.astype(np.float32)

    def remove(self) -> None:
        for entry in self._entries:
            entry.handle.remove()
        self._entries.clear()


__all__ = ["BatchedMuJoCoScene"]
