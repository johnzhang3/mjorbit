# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Predicted-rollout trace rendering (judo-style) for the viewer app.

Traces are predicted trajectories in chief-centered world coordinates
(metres). They are uploaded once per planner replan as a single merged
line-segments node parented to a frame; the per-render-frame motion
(LVLH rotation or ECI translation) only moves the parent frame, which is
cheap, instead of re-sending every vertex at 60 Hz.

The render-frame transform used by the app is
``render = origin + scale * (rot @ p + trans - origin)`` with
``trans == origin`` (ECI) or both zero (LVLH), which reduces to
``render = rot' @ (scale * p) + trans`` — i.e. scaled local points under a
rotated/translated parent frame.
"""

from __future__ import annotations

import numpy as np
import viser
import viser.transforms as vtf

from .tasks.base import PredictedTrace


class RolloutTraceScene:
    """Renders a task's predicted traces under a movable parent frame."""

    def __init__(self, server: viser.ViserServer, *, path: str = "/task/traces") -> None:
        self._server = server
        self._path = path
        self._frame = server.scene.add_frame(path, show_axes=False)
        self._handle: viser.SceneNodeHandle | None = None
        self._version: int | None = None
        self._scale = 1.0

    def set_scale(self, scale: float) -> None:
        """Mark the uploaded geometry stale so the next sync re-bakes scale."""
        if float(scale) != self._scale:
            self._scale = float(scale)
            self._version = None

    def clear(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._version = None

    def sync(self, traces: list[PredictedTrace], version: int) -> None:
        """Upload trace geometry when the task has published a new set."""
        if version == self._version:
            return
        self._version = version
        if not traces:
            self.clear()
            self._version = version
            return

        segments: list[np.ndarray] = []
        colors: list[np.ndarray] = []
        for trace in traces:
            pts = self._scale * np.asarray(trace.points, dtype=np.float32)
            if len(pts) < 2:
                continue
            seg = np.stack([pts[:-1], pts[1:]], axis=1)  # (T-1, 2, 3)
            segments.append(seg)
            color = np.asarray(trace.color, dtype=np.uint8)
            colors.append(np.broadcast_to(color, seg.shape).copy())
        if not segments:
            self.clear()
            self._version = version
            return

        points = np.concatenate(segments, axis=0)
        if self._handle is not None:
            self._handle.remove()
        self._handle = self._server.scene.add_line_segments(
            f"{self._path}/rollouts",
            points=points,
            colors=np.concatenate(colors, axis=0),
            line_width=3.0,
        )

    def update_transform(
        self, rotation: np.ndarray | None, translation: np.ndarray | None
    ) -> None:
        """Move the parent frame to the current render-frame transform."""
        wxyz = (1.0, 0.0, 0.0, 0.0)
        if rotation is not None:
            wxyz = tuple(
                vtf.SO3.from_matrix(np.asarray(rotation, dtype=np.float64)).wxyz
            )
        position = (0.0, 0.0, 0.0) if translation is None else tuple(translation)
        with self._server.atomic():
            self._frame.wxyz = wxyz
            self._frame.position = position
