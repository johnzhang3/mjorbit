# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Contact-force visualization helpers for the browser viewer."""

from __future__ import annotations

import mujoco
import numpy as np
import viser

_DEFAULT_CONTACT_COLOR = np.array([255, 80, 80], dtype=np.uint8)


def contact_force_segments(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    *,
    force_scale: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return line segments for all active contact forces in world coordinates."""
    segments: list[np.ndarray] = []
    for contact_id in range(mj_data.ncon):
        contact = mj_data.contact[contact_id]
        force_contact = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(mj_model, mj_data, contact_id, force_contact)

        # MuJoCo reports force/torque in the contact frame; the contact-frame axes
        # are stored row-major in ``contact.frame``, so transpose to map into world.
        contact_frame = contact.frame.reshape(3, 3)
        force_world = contact_frame.T @ force_contact[:3]
        if np.linalg.norm(force_world) < 1e-9:
            continue

        start = np.asarray(contact.pos, dtype=float)
        end = start + force_world * force_scale
        segments.append(np.stack([start, end], axis=0))

    if not segments:
        return None

    segments_array = np.asarray(segments, dtype=np.float32)
    colors = np.broadcast_to(
        _DEFAULT_CONTACT_COLOR,
        (segments_array.shape[0], 2, _DEFAULT_CONTACT_COLOR.shape[0]),
    ).copy()
    return segments_array, colors


class ContactForceOverlay:
    """Render active MuJoCo contact forces as line segments."""

    def __init__(
        self,
        server: viser.ViserServer,
        *,
        path: str = "/contact_forces",
        force_scale: float = 5.0e-3,
        line_width: float = 6.0,
    ) -> None:
        self._server = server
        self._path = path
        self._force_scale = force_scale
        self._line_width = line_width
        self._scale = 1.0
        self._handle: viser.SceneNodeHandle | None = None

    def clear(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def set_scale(self, scale: float) -> None:
        self._scale = float(scale)

    def render(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData, *, visible: bool) -> None:
        if not visible:
            self.clear()
            return

        payload = contact_force_segments(
            mj_model,
            mj_data,
            force_scale=self._force_scale,
        )
        if payload is None:
            self.clear()
            return

        segments, colors = payload
        segments = self._scale * segments
        if self._handle is not None:
            self._handle.remove()
        self._handle = self._server.scene.add_line_segments(
            self._path,
            segments,
            colors=colors,
            line_width=self._line_width,
        )

    def render_transformed(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        *,
        visible: bool,
        rotation: np.ndarray | None = None,
        translation: np.ndarray | None = None,
    ) -> None:
        if not visible:
            self.clear()
            return

        payload = contact_force_segments(
            mj_model,
            mj_data,
            force_scale=self._force_scale,
        )
        if payload is None:
            self.clear()
            return

        segments, colors = payload
        segments = self._scale * segments
        if rotation is not None:
            segments = segments @ np.asarray(rotation, dtype=float).T
        if translation is not None:
            segments = segments + np.asarray(translation, dtype=float)

        if self._handle is not None:
            self._handle.remove()
        self._handle = self._server.scene.add_line_segments(
            self._path,
            segments,
            colors=colors,
            line_width=self._line_width,
        )
