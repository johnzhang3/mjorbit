# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Render an RL capture-and-reorient rollout (examples/ppo/capture_env.py) to mp4.

The bi-manual free-flyer captures a free-floating client satellite by its
grapple bar and slews the stack until the client's antenna points at nadir.
Produce the trajectory with the evaluator (it picks a successful held-out
rollout and keeps a few seconds of hold after the success), then render here:

    pixi run -e rl python examples/ppo/capture_eval.py --run examples/ppo/logs/cap-sgs \
        --checkpoint model_2999.pt --save-traj videos/capture_traj.npz
    pixi run python scripts/record/record_capture_rl.py \
        --traj videos/capture_traj.npz --out videos/capture_reorient_rl.mp4

Captions: playback speed (bottom left) and the task state (top left): grip,
whether both hands hold the bar, and the antenna-to-nadir angle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import viser

_ROOT = Path(__file__).resolve().parents[2]
for p in (str(_ROOT / "src"), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from recorder import HeadlessRecorder, VideoWriter, annotate, speedup_label  # noqa: E402
from scene import SingleScene, forward_to  # noqa: E402

from mjorbit import MjoModel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--traj", default="videos/capture_traj.npz")
    ap.add_argument("--out", default="videos/capture_reorient_rl.mp4")
    ap.add_argument("--seconds", type=float, default=16.0, help="video length")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mag", type=float, default=45000.0)
    ap.add_argument("--distance", type=float, default=80.0)
    ap.add_argument(
        "--offset",
        type=float,
        nargs=3,
        default=(0.45, 0.35, 0.82),
        metavar=("RADIAL", "ALONG", "CROSS"),
        help="camera direction in LVLH",
    )
    ap.add_argument("--fov", type=float, default=48.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--port", type=int, default=8431)
    args = ap.parse_args()

    d = np.load(args.traj, allow_pickle=True)
    qpos, R_eci = d["qpos"], d["R_eci"]
    V_eci = (
        d["V_eci_per_frame"] if "V_eci_per_frame" in d else np.broadcast_to(d["V_eci"], R_eci.shape)
    )
    dt = float(d["dt"])
    point_err, held, grip = d["point_err_deg"], d["held"], d["grip_closed"]
    t_success = float(d["t_success_s"])

    model = MjoModel.from_xml_path(str(d["xml_path"]))
    data = model.make_data()
    track = [model.body_id("bus"), model.body_id("client")]

    n_frames = max(2, int(round(args.seconds * args.fps)))
    idx = np.unique(np.linspace(0, len(qpos) - 1, n_frames).astype(int))
    speed = speedup_label(len(qpos) * dt, len(idx) / args.fps)

    server = viser.ViserServer(host="127.0.0.1", port=args.port)
    scene = SingleScene(server, model, data, mag=args.mag, track_body_id=track[0])
    rec = HeadlessRecorder(server, port=args.port, width=args.width, height=args.height)
    try:
        with VideoWriter(args.out, fps=args.fps, width=args.width, height=args.height) as vid:
            for step in idx:
                forward_to(model, data, qpos[step], R_eci=R_eci[step], V_eci=V_eci[step])
                scene.update()
                pos, look = scene.camera_pose(
                    distance=args.distance,
                    target=scene.cluster_center_render(track),
                    offset_rsw=np.asarray(args.offset),
                )
                frame = rec.render(position=pos, look_at=look, fov_deg=args.fov)
                t = (step + 1) * dt
                state = (
                    "holding on nadir: success"
                    if t >= t_success
                    else (
                        "captured, slewing"
                        if held[step]
                        else ("closing grip" if grip[step] else "approach")
                    )
                )
                frame = annotate(
                    frame,
                    f"t = {t:4.1f} s   {state}   antenna-nadir {point_err[step]:5.1f}°",
                    corner="tl",
                )
                vid.add(annotate(frame, speed))
        print(f"wrote {args.out}  ({len(idx)} frames, {len(idx) / args.fps:.1f}s @ {args.fps}fps)")
    finally:
        rec.close()
        server.stop()


if __name__ == "__main__":
    main()
