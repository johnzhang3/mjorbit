# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Render the autonomous-docking clip (paper example b) from a saved trajectory.

First dump a trajectory:
    pixi run python examples/docking/main_mppi.py --duration 16 --save-traj /tmp/dock_traj.npz
Then:
    pixi run python scripts/record/record_docking.py \
        --traj /tmp/dock_traj.npz --out videos/docking.mp4
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
for p in (str(_SRC), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from render_traj import CameraConfig, render_trajectory  # noqa: E402

from mjorbit import MjoModel  # noqa: E402


def _compile_render_model(xml_path: str) -> MjoModel:
    xml = Path(xml_path).read_text().replace(
        "</mujoco>",
        '<mjorbit use_j2="false" use_drag="false" use_srp="false" use_magnetic="false"/></mujoco>',
    )
    near = Path(xml_path).parent
    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False, dir=near) as f:
        f.write(xml)
        tmp = Path(f.name)
    try:
        return MjoModel.from_xml_path(str(tmp), mj_timestep=0.01)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", default="/tmp/dock_traj.npz")
    ap.add_argument("--out", default="videos/docking.mp4")
    ap.add_argument("--seconds", type=float, default=11.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mag", type=float, default=5000.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--port", type=int, default=8400)
    args = ap.parse_args()

    d = np.load(args.traj, allow_pickle=True)
    qpos, R_eci, V_eci = d["qpos"], d["R_eci"], d["V_eci"]
    xml_path = str(d["xml_path"])
    model = _compile_render_model(xml_path)
    data = model.make_data()

    soy = model.body_id("soyuz")
    camera = CameraConfig(
        # Frame the Soyuz chaser as the subject (it only translates ~2 m — the
        # dock is a fine 6-DOF port alignment), with the ISS dock-port structure
        # as the target behind it and Earth beyond. Gentle arc + push-in for life.
        track_body_ids=[soy],
        offset_rsw=np.array([0.42, 0.86, 0.30]),     # behind the Soyuz, looking at the ISS
        offset_end=np.array([0.52, 0.52, 0.66]),     # swing to a 3/4 side view
        distance=21.0,
        distance_end=15.0,
        fov_deg=46.0,
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
