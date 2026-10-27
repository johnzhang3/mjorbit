# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Render an RL capture-and-reorient rollout (examples/ppo/capture_env.py) to mp4.

The bi-manual free-flyer captures a free-floating client satellite by its
grapple bar and slews the stack until the client's antenna points at nadir.
Produce the trajectory with the evaluator (it picks a successful held-out
rollout), then render here; the clip ends on the success step:

    pixi run -e rl python examples/ppo/capture_eval.py --run examples/ppo/logs/cap-sgs \
        --checkpoint model_<it>.pt --save-traj videos/capture_traj.npz
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


def _state(t: float, t_success: float, held: bool, grip: bool, err_deg: float) -> str:
    if t >= t_success:
        return "SUCCESS: held within 10° of nadir for 3 s"
    if held:
        return "captured, holding on nadir" if err_deg < 10.0 else "captured, slewing to nadir"
    return "grip closed, securing the bar" if grip else "approaching"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--traj", default="videos/capture_traj.npz")
    ap.add_argument("--out", default="videos/capture_reorient_rl.mp4")
    ap.add_argument("--seconds", type=float, default=16.0, help="video length")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument(
        "--freeze-s",
        type=float,
        default=1.5,
        help="hold the success frame this long. The clip ends on the success step: episodes "
        "terminate on success, so the policy is not trained to keep holding after it",
    )
    ap.add_argument("--mag", type=float, default=45000.0)
    ap.add_argument("--distance", type=float, default=38.0)
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
    keep = min(len(qpos), int(round(t_success / dt)))
    qpos, R_eci, V_eci = qpos[:keep], R_eci[:keep], V_eci[:keep]

    xml = Path(str(d["xml_path"]))
    if not xml.exists():  # trajectory produced on another machine (e.g. the GPU host)
        xml = _ROOT / "examples" / "ppo" / xml.name
    model = MjoModel.from_xml_path(str(xml))
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
                state = _state(t, t_success, held[step], grip[step], point_err[step])
                frame = annotate(
                    frame,
                    f"t = {t:4.1f} s   {state}   antenna-nadir {point_err[step]:5.1f}°",
                    corner="tl",
                )
                frame = annotate(frame, speed)
                vid.add(frame)
            for _ in range(int(round(args.freeze_s * args.fps))):
                vid.add(frame)
        print(f"wrote {args.out}  ({len(idx)} frames, {len(idx) / args.fps:.1f}s @ {args.fps}fps)")
    finally:
        rec.close()
        server.stop()


if __name__ == "__main__":
    main()
