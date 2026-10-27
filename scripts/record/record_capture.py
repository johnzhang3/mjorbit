# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Render the MPPI fidelity study's capture-and-hold-nadir episode from a saved trajectory.

The servicer's two-link arm grasps the 400 kg payload, then the mjorbit planner swings the
stack to local vertical and holds it using only arm motion and the gravity gradient
(``examples/mppi/mppi_fidelity_capture.py``). The grasp and the long stabilization each get
their own share of the video, so the caption steps from one speed-up to the other.

    pixi run python examples/mppi/mppi_fidelity_capture.py --backends mjorbit \\
        --horizon-b 2400 --duration-b 10000 --save-traj /tmp/capture_traj.npz
    pixi run python scripts/record/record_capture.py \\
        --traj /tmp/capture_traj.npz --out videos/capture_nadir.mp4
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for p in (str(_ROOT / "src"), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from render_traj import CameraConfig, render_trajectory  # noqa: E402

from mjorbit import MjoModel  # noqa: E402
from mjorbit.testdata import SPACECRAFT_CAPTURE_XML  # noqa: E402

# The test model's bus is a bare box; give it solar panels for the video. They are massless
# and collision-free, so the replayed trajectory is unchanged.
BUS_GEOM = '<geom type="box" size="0.5 0.3 0.3" mass="800" rgba="0.3 0.5 0.9 1"/>'
PANELS = "".join(f'<geom type="box" pos="0 {y} 0" size="0.4 1.15 0.02" mass="0" contype="0" '
                 f'conaffinity="0" rgba="0.25 0.35 0.65 1"/>' for y in (1.5, -1.5))


def render_model() -> MjoModel:
    xml = Path(SPACECRAFT_CAPTURE_XML).read_text()
    assert xml.count(BUS_GEOM) == 1, "bus geom changed in the capture XML"
    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
        f.write(xml.replace(BUS_GEOM, BUS_GEOM + PANELS))
        path = Path(f.name)
    try:
        return MjoModel.from_xml_path(str(path))
    finally:
        path.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", default="/tmp/capture_traj.npz")
    ap.add_argument("--out", default="videos/capture_nadir.mp4")
    ap.add_argument("--capture-seconds", type=float, default=4.0, help="video time for the grasp")
    ap.add_argument("--stabilize-seconds", type=float, default=10.0,
                    help="video time for the stabilization")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mag", type=float, default=10000.0)
    ap.add_argument("--distance", type=float, default=15.0, help="camera distance (render units)")
    ap.add_argument("--offset", type=float, nargs=3, default=(0.3, 0.0, 1.0),
                    metavar=("RADIAL", "ALONG", "CROSS"),
                    help="camera direction in LVLH; mostly cross-track shows the swing in-plane, "
                         "with Earth below")
    ap.add_argument("--fov", type=float, default=48.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--port", type=int, default=8411)
    args = ap.parse_args()

    d = np.load(args.traj)
    t, n_a = d["t"], int(d["n_capture"])
    # Resample each phase onto its share of the video frames.
    f_a = max(2, int(round(args.capture_seconds * args.fps)))
    f_b = max(2, int(round(args.stabilize_seconds * args.fps)))
    idx = np.concatenate([np.linspace(0, n_a - 1, f_a), np.linspace(n_a, len(t) - 1, f_b)])
    idx = np.round(idx).astype(int)
    sim_dt = np.array([[0.0, (t[n_a - 1] - t[0]) / (f_a - 1)],
                       [float(f_a), (t[-1] - t[n_a]) / (f_b - 1)]])

    model = render_model()
    data = model.make_data()
    camera = CameraConfig(
        track_body_ids=[model.body_id("bus"), model.body_id("payload")],
        offset_rsw=np.asarray(args.offset, dtype=float),
        distance=args.distance,
        fov_deg=args.fov,
        up_radial=True,
    )
    render_trajectory(
        model=model, data=data, qpos=d["qpos"][idx], R_eci=d["R_eci"][idx], V_eci=d["V_eci"][idx],
        out_path=args.out, camera=camera, mag=args.mag,
        video_seconds=len(idx) / args.fps, fps=args.fps,
        width=args.width, height=args.height, port=args.port,
        sim_dt=sim_dt,
    )


if __name__ == "__main__":
    main()
