# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Render the RL truss earth-pointing clip (paper example d) from a saved
trajectory (replays it through the world-scaled scene to mp4).

Produce the trajectory first, then render here on CPU. For the full servicing
sequence (scripted fly-in + trained hug/stabilize) use ``produce_hug.py``:

    pixi run -e rl python scripts/record/produce_hug.py \
        --checkpoint examples/ppo/logs/<run>/model_<it>.pt --out /tmp/hug_traj.npz
    pixi run -e rl python scripts/record/record_truss.py \
        --traj /tmp/hug_traj.npz --out videos/truss_pointing.mp4 --clip-sim-seconds 0

(``examples/ppo/play.py --save-traj`` also dumps a single start-hugging rollout.)
The clip shows the bi-manual spacecraft hugging a large free-floating truss with
both arms and slewing it to point at Earth (nadir).
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
    ap.add_argument("--traj", default="/tmp/truss_traj.npz")
    ap.add_argument("--out", default="videos/truss_pointing.mp4")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--clip-sim-seconds", type=float, default=45.0,
                    help="render only the first N seconds of sim (the swing + early "
                         "tapping) so the reorientation is not compressed away")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mag", type=float, default=45000.0)
    ap.add_argument("--distance", type=float, default=100.0)
    ap.add_argument("--fov", type=float, default=48.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--port", type=int, default=8420)
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
    truss = model.body_id("truss")
    # Mostly cross-track (orbit-normal) view so the truss swing plane faces the
    # camera, with some radial offset so the Earth stays in frame below.
    camera = CameraConfig(
        track_body_ids=[bus, truss],
        offset_rsw=np.array([0.42, 0.28, 0.86]),
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
