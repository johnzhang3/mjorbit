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
import matplotlib.pyplot as plt

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
    t = d["t"] if "t" in d else np.arange(len(qpos)) * float(d["dt"])
    t_latch = float(d["t_latch"]) if "t_latch" in d else None  # optional, for annotation
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

    # orbit period (s) for marking each completed orbit after latch
    r_orbit = float(np.linalg.norm(R_eci[0]))
    period = 2.0 * np.pi * np.sqrt(r_orbit**3 / GM_EARTH)

    def add_markers(ax) -> None:
        if t_latch is not None:
            ax.axvline(x=t_latch, color="k", linestyle="--", label="t_latch")
            k = 1
            while t_latch + k * period <= float(t[-1]):
                ax.axvline(
                    x=t_latch + k * period, color="C7", linestyle=":",
                    label="orbit" if k == 1 else None,
                )
                k += 1

    # quickly plot the position and quaternion to sanity check
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(t, qpos[:, :3])
    add_markers(axs[0])
    axs[0].set_title("position (km)")
    axs[0].legend(loc="upper right")
    axs[1].plot(t, qpos[:, 3:7])
    add_markers(axs[1])
    axs[1].set_title("quaternion")
    axs[1].set_xlabel("time (s)")
    plt.show()


    # render_trajectory(
    #     model=model, data=data, qpos=qpos, R_eci=R_eci, V_eci=V_eci,
    #     out_path=args.out, camera=camera, mag=args.mag,
    #     video_seconds=args.seconds, fps=args.fps,
    #     width=args.width, height=args.height, port=args.port,
    #     sim_dt=float(d["dt"]),
    # )


if __name__ == "__main__":
    main()
