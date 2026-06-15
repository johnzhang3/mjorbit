# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Render the grasping clip (paper example c) from a saved capture trajectory.

    pixi run python scripts/record/produce_grasp.py --out /tmp/grasp_traj.npz
    pixi run python scripts/record/record_grasp.py \
        --traj /tmp/grasp_traj.npz --out videos/grasping.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for p in (str(_ROOT / "src"), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from render_traj import CameraConfig, render_trajectory  # noqa: E402

from mjorbit import MjoModel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", default="/tmp/grasp_traj.npz")
    ap.add_argument("--out", default="videos/grasping.mp4")
    ap.add_argument("--seconds", type=float, default=11.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mag", type=float, default=40000.0)
    ap.add_argument("--distance", type=float, default=95.0)
    ap.add_argument("--fov", type=float, default=48.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--port", type=int, default=8410)
    args = ap.parse_args()

    d = np.load(args.traj, allow_pickle=True)
    qpos, R_eci, V_eci = d["qpos"], d["R_eci"], d["V_eci"]
    model = MjoModel.from_xml_path(str(d["xml_path"]))
    data = model.make_data()

    bus = model.body_id("bus")
    payload = model.body_id("payload")
    camera = CameraConfig(
        track_body_ids=[bus, payload],
        offset_rsw=np.array([0.75, -0.45, 0.48]),
        distance=args.distance,
        fov_deg=args.fov,
    )
    render_trajectory(
        model=model, data=data, qpos=qpos, R_eci=R_eci, V_eci=V_eci,
        out_path=args.out, camera=camera, mag=args.mag,
        video_seconds=args.seconds, fps=args.fps,
        width=args.width, height=args.height, port=args.port,
        sim_dt=float(d["dt"]),
    )


if __name__ == "__main__":
    main()
