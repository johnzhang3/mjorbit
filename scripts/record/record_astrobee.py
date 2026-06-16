# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Render the Astrobee detumble-and-grasp clip from a saved trajectory.

Produce the trajectory first with ``produce_astrobee.py``, then render here:

    pixi run -e rl python scripts/record/produce_astrobee.py \
        --checkpoint examples/ppo/logs/<run>/model_<it>.pt --out /tmp/astrobee_traj.npz
    pixi run -e rl python scripts/record/record_astrobee.py \
        --traj /tmp/astrobee_traj.npz --out videos/astrobee_grasp.mp4

The clip shows the Astrobee-style free-flyer detumbling from an initial tumble,
flying to a free-floating cargo module, and grasping its grapple bar.
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
    ap.add_argument("--traj", default="/tmp/astrobee_traj.npz")
    ap.add_argument("--out", default="videos/astrobee_grasp.mp4")
    ap.add_argument("--seconds", type=float, default=13.0)
    ap.add_argument("--clip-sim-seconds", type=float, default=26.0,
                    help="render only the first N seconds of sim (detumble + "
                         "approach + grasp + early hold)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mag", type=float, default=150000.0)
    ap.add_argument("--distance", type=float, default=80.0)
    ap.add_argument("--fov", type=float, default=48.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--port", type=int, default=8421)
    args = ap.parse_args()

    d = np.load(args.traj, allow_pickle=True)
    qpos, R_eci, V_eci = d["qpos"], d["R_eci"], d["V_eci"]
    dt = float(d["dt"])
    if args.clip_sim_seconds > 0:
        n = min(len(qpos), int(round(args.clip_sim_seconds / dt)))
        qpos, R_eci = qpos[:n], R_eci[:n]
    model = MjoModel.from_xml_path(str(d["xml_path"]))
    data = model.make_data()

    bus = model.body_id("bus")
    cargo = model.body_id("cargo")
    # Frame the robot + cargo cluster against the Earth, viewed mostly
    # cross-track so the approach + grasp reads clearly.
    camera = CameraConfig(
        track_body_ids=[bus, cargo],
        offset_rsw=np.array([0.35, 0.30, 0.88]),
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
