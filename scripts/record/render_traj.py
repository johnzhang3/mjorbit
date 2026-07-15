# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Replay a saved (qpos, R_eci) trajectory through the world-scaled scene and
encode an mp4. Decouples slow controllers (MPPI) from rendering: run the
controller once to dump a trajectory, then render deterministically here.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import viser

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from recorder import HeadlessRecorder, VideoWriter, annotate, speedup_label  # noqa: E402
from scene import SingleScene, forward_to  # noqa: E402


@dataclass
class CameraConfig:
    track_body_ids: list[int]          # bodies whose render centroid the camera follows
    offset_rsw: np.ndarray             # view direction in (radial, along, cross)
    distance: float                    # camera distance in render units
    fov_deg: float = 48.0
    distance_end: float | None = None  # optional linear zoom over the clip
    offset_end: np.ndarray | None = None  # optional offset slew over the clip


def _frame_labels(idx: np.ndarray, fps: float, sim_dt) -> list[str] | None:
    """Per-rendered-frame ``"N× real-time"`` captions.

    Source frames are resampled uniformly onto ``idx`` for rendering, so each
    rendered frame advances ``density = (idx[-1] - idx[0]) / (len(idx) - 1)``
    source frames, i.e. ``density * fps`` source frames per real second. The
    shown speedup at a frame is therefore ``dt * density * fps`` for whichever
    ``dt`` applies at that source frame.

    ``sim_dt`` selects the caption:

    * ``None`` — no caption (returns ``None``);
    * scalar ``float`` — one constant speedup for the whole clip;
    * ``[[start_frame, dt], ...]`` — a piecewise schedule: source frames in
      ``[start_k, start_{k+1})`` advance ``dt_k`` sim-seconds each, so the
      caption steps at each phase boundary (e.g. a slow grasp shown at ~3×
      followed by a fast-forward coast shown at a few hundred ×).
    """
    if sim_dt is None:
        return None
    n_render = len(idx)
    density = (int(idx[-1]) - int(idx[0])) / max(1, n_render - 1)
    sched = np.atleast_2d(np.asarray(sim_dt, dtype=float))
    if sched.shape[1] == 2:  # [[start_frame, dt], ...] schedule
        order = np.argsort(sched[:, 0])
        starts, dts = sched[order, 0], sched[order, 1]
    else:  # scalar dt -> a single segment starting at frame 0
        starts, dts = np.array([0.0]), np.ravel(sched).astype(float)[:1]
    seg = np.clip(np.searchsorted(starts, idx, side="right") - 1, 0, None)
    return [speedup_label(float(dts[k]) * density * fps, 1.0) for k in seg]


def render_trajectory(
    *,
    model,
    data,
    qpos: np.ndarray,
    R_eci: np.ndarray,
    V_eci: np.ndarray,
    out_path: str,
    camera: CameraConfig,
    mag: float = 5000.0,
    world: float = 1e-4,
    video_seconds: float = 11.0,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    port: int = 8400,
    warmup: float = 9.0,
    sim_dt: float | np.ndarray | None = None,
) -> None:
    n_src = len(qpos)
    n_frames = max(2, int(round(video_seconds * fps)))
    idx = np.unique(np.linspace(0, n_src - 1, n_frames).astype(int))
    labels = _frame_labels(idx, fps, sim_dt)

    server = viser.ViserServer(host="127.0.0.1", port=port)
    scene = SingleScene(server, model, data, mag=mag, world=world,
                        track_body_id=camera.track_body_ids[0])
    rec = HeadlessRecorder(server, port=port, width=width, height=height, warmup=warmup)
    try:
        with VideoWriter(out_path, fps=fps, width=width, height=height) as vid:
            for f, step in enumerate(idx):
                u = f / (len(idx) - 1)
                forward_to(model, data, qpos[step], R_eci=R_eci[step], V_eci=V_eci)
                scene.update()
                target = scene.cluster_center_render(camera.track_body_ids)
                dist = camera.distance if camera.distance_end is None else (
                    (1 - u) * camera.distance + u * camera.distance_end
                )
                off = camera.offset_rsw if camera.offset_end is None else (
                    (1 - u) * camera.offset_rsw + u * camera.offset_end
                )
                pos, look = scene.camera_pose(distance=dist, target=target, offset_rsw=off)
                frame = rec.render(position=pos, look_at=look, fov_deg=camera.fov_deg)
                if labels is not None:
                    frame = annotate(frame, labels[f])
                vid.add(frame)
        print(f"wrote {out_path}  ({len(idx)} frames, {len(idx)/fps:.1f}s @ {fps}fps)")
    finally:
        rec.close()
        server.stop()
