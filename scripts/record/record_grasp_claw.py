# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Render the 5-finger claw grasping clip from a saved trajectory.

    pixi run python scripts/record/produce_grasp_claw.py --out /tmp/grasp_claw_traj.npz
    pixi run python scripts/record/record_grasp_claw.py \
        --traj /tmp/grasp_claw_traj.npz --out videos/grasping_claw.mp4
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
from mjorbit.constants import GM_EARTH  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", default="/tmp/grasp_claw_traj.npz")
    ap.add_argument("--out", default="videos/grasping_claw.mp4")
    ap.add_argument("--seconds", type=float, default=11.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mag", type=float, default=10000.0)
    ap.add_argument("--distance", type=float, default=10.0,
                    help="camera distance (render units); smaller frames the claw tighter")
    ap.add_argument("--fov", type=float, default=48.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--port", type=int, default=8410)
    ap.add_argument("--plot", action="store_true",
                    help="show a position/quaternion sanity plot instead of rendering")
    args = ap.parse_args()

    d = np.load(args.traj, allow_pickle=True)
    qpos, R_eci, V_eci = d["qpos"], d["R_eci"], d["V_eci"]
    t = d["t"] if "t" in d else np.arange(len(qpos)) * float(d["dt"])
    # Per-phase [[start_frame, dt], ...] speedup schedule (newer trajectories);
    # fall back to the single constant dt for older files.
    sim_dt = d["sim_dt"] if "sim_dt" in d else float(d["dt"])
    t_latch = float(d["t_latch"]) if "t_latch" in d else None  # grasp instant, for annotation
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

    if args.plot:
        import matplotlib.pyplot as plt

        # orbit period (s) for marking each completed orbit after the grasp
        r_orbit = float(np.linalg.norm(R_eci[0]))
        period = 2.0 * np.pi * np.sqrt(r_orbit**3 / GM_EARTH)

        def add_markers(ax) -> None:
            if t_latch is not None and np.isfinite(t_latch):
                ax.axvline(x=t_latch, color="k", linestyle="--", label="grasp")
                k = 1
                while t_latch + k * period <= float(t[-1]):
                    ax.axvline(
                        x=t_latch + k * period, color="C7", linestyle=":",
                        label="orbit" if k == 1 else None,
                    )
                    k += 1

        fig, axs = plt.subplots(4, 1, sharex=True)
        axs[0].plot(t, qpos[:, :3])
        add_markers(axs[0])
        axs[0].set_title("bus position (km)")
        axs[0].legend(loc="upper right")
        for comp, lbl in zip(qpos[:, 3:7].T, ("w", "x", "y", "z")):
            axs[1].plot(t, comp, label=lbl)
        add_markers(axs[1])
        axs[1].set_title("bus quaternion (w, x, y, z)")
        axs[1].legend(loc="upper right", ncol=4)
        axs[2].plot(t, qpos[:, 7], label="boom (m)")
        axs[2].plot(t, qpos[:, 8:18:2], color="C1", alpha=0.5)
        add_markers(axs[2])
        axs[2].set_title("boom slide + proximal finger angles")
        axs[3].plot(t, qpos[:, 18:21])
        add_markers(axs[3])
        axs[3].set_title("payload position (km)")
        axs[3].set_xlabel("time (s)")
        plt.show()
        return

    render_trajectory(
        model=model, data=data, qpos=qpos, R_eci=R_eci, V_eci=V_eci,
        out_path=args.out, camera=camera, mag=args.mag,
        video_seconds=args.seconds, fps=args.fps,
        width=args.width, height=args.height, port=args.port,
        sim_dt=sim_dt,
    )


if __name__ == "__main__":
    main()
