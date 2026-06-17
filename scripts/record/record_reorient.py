# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Render the dual-arm 3-DOF reorientation clip from a saved trajectory.

The MPPI controller is expensive, so we record in two deterministic steps:
first dump a trajectory, then replay it through the world-scaled scene.

    pixi run python examples/reorient_mppi_3dof.py \
        --duration 60 --save-traj /tmp/reorient_traj.npz
    pixi run python scripts/record/record_reorient.py \
        --traj /tmp/reorient_traj.npz --out videos/reorient.mp4

The camera frames the whole bus+arms assembly (the bus is free-floating and does
not translate, so it stays centred) from a fixed oblique LVLH view, which keeps
the *attitude* slew unambiguous -- the bus visibly rotates while the camera holds
still. ``--seconds`` sets the clip length: 60 s of sim into a shorter clip plays
back sped up, and the burned-in caption reports the speed-up factor.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
for p in (str(_SRC), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from render_traj import CameraConfig, render_trajectory  # noqa: E402

from mjorbit import MjoModel, OrbitInit, mjo_forward  # noqa: E402
from viewer.framing import distance_for_fill, spacecraft_bounding_radius  # noqa: E402


def _frame_distance(model, data, qpos: np.ndarray, R_eci, V_eci, center_bid: int,
                    *, render_scale: float, fov_deg: float, fill: float) -> float:
    """Camera distance (render units) that keeps the assembly framed over the clip.

    The arms wave during the slew, so sample the bounding radius across the
    trajectory and frame the largest pose with a little margin (``fill``).
    """
    np.copyto(np.asarray(data.orbit.R_eci), np.asarray(R_eci[0], dtype=float))
    np.copyto(np.asarray(data.orbit.V_eci), np.asarray(V_eci, dtype=float))
    radius = 0.0
    for step in np.unique(np.linspace(0, len(qpos) - 1, 24).astype(int)):
        np.copyto(np.asarray(data.qpos), np.asarray(qpos[step], dtype=float))
        mjo_forward(model, data)
        radius = max(radius, spacecraft_bounding_radius(model, data, center_bid))
    return distance_for_fill(radius * render_scale, fov=np.deg2rad(fov_deg), fill=fill)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", default="/tmp/reorient_traj.npz")
    ap.add_argument("--out", default="videos/reorient.mp4")
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="clip length; 60 s of sim into 20 s plays back at 3x real-time")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mag", type=float, default=5000.0,
                    help="spacecraft magnification against the globe")
    ap.add_argument("--world", type=float, default=1e-4)
    ap.add_argument("--fov", type=float, default=46.0)
    ap.add_argument("--fill", type=float, default=0.5,
                    help="fraction of the view the assembly fills (smaller = more margin)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--port", type=int, default=8401)
    args = ap.parse_args()

    d = np.load(args.traj, allow_pickle=True)
    qpos, R_eci, V_eci = d["qpos"], d["R_eci"], d["V_eci"]
    xml_path = str(d["xml_path"])
    model = MjoModel.from_xml_path(xml_path)
    data = model.make_data(orbit=OrbitInit(R_eci=R_eci[0], V_eci=V_eci))

    bus = model.body_id("bus")
    render_scale = args.mag * args.world
    dist = _frame_distance(model, data, qpos, R_eci, V_eci, bus,
                           render_scale=render_scale, fov_deg=args.fov, fill=args.fill)

    camera = CameraConfig(
        # Oblique LVLH view (radial, along-track, cross-track) that shows all
        # three rotation axes; a very gentle arc keeps the shot alive without
        # masking the bus's own rotation as camera motion.
        track_body_ids=[bus],
        offset_rsw=np.array([0.45, 0.80, 0.40]),
        offset_end=np.array([0.55, 0.62, 0.56]),
        distance=dist,
        fov_deg=args.fov,
    )
    render_trajectory(
        model=model, data=data, qpos=qpos, R_eci=R_eci, V_eci=V_eci,
        out_path=args.out, camera=camera, mag=args.mag, world=args.world,
        video_seconds=args.seconds, fps=args.fps,
        width=args.width, height=args.height, port=args.port,
        sim_dt=float(d["dt"]),
    )


if __name__ == "__main__":
    main()
