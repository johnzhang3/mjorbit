# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Contact-force visualization helpers for the browser viewer."""

from __future__ import annotations

import numpy as np
import viser

from mjorbit.runtime import MjoData, MjoModel

_DEFAULT_CONTACT_COLOR = np.array([255, 80, 80], dtype=np.uint8)


def contact_force_segments(
    model: MjoModel,
    data: MjoData,
    *,
    force_scale: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return line segments for all active contact forces in world coordinates."""
    del model
    segments_array = data.contact_force_segments(force_scale=force_scale)
    if segments_array is None:
        return None

    segments_array = np.asarray(segments_array, dtype=np.float32)
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

    def render(self, model: MjoModel, data: MjoData, *, visible: bool) -> None:
        if not visible:
            self.clear()
            return

        payload = contact_force_segments(
            model,
            data,
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
        model: MjoModel,
        data: MjoData,
        *,
        visible: bool,
        rotation: np.ndarray | None = None,
        translation: np.ndarray | None = None,
    ) -> None:
        if not visible:
            self.clear()
            return

        payload = contact_force_segments(
            model,
            data,
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
