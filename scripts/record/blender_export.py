# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Export capture-and-stabilize trajectories to per-frame geom poses for ``blender_capture.py``.

Each ``--traj`` is one planner's run from ``examples/mppi/mppi_fidelity_capture.py
--save-traj`` (same seed, so the grasp is shared). The grasp and the stabilization get their
own share of the video, as in ``record_capture.py``. Poses are in the chief-centered inertial
frame (metres); the chief orbit comes along for the Earth, camera and lighting.

    pixi run python scripts/record/blender_export.py \\
        --traj /tmp/pair_mjorbit.npz /tmp/pair_zerog.npz --labels mjorbit zero-g \\
        --out /tmp/frames.npz
    blender -b -P scripts/record/blender_capture.py -- /tmp/frames.npz /tmp/frames_out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from mjorbit import MjoModel, mjo_forward  # noqa: E402
from mjorbit.testdata import SPACECRAFT_CAPTURE_XML  # noqa: E402

# Solar panels on the bus, for the video only (the test model's bus is a bare box).
PANELS = [dict(type=6, size=(0.4, 1.15, 0.02), pos=(0.0, y, 0.0), rgba=(0.25, 0.35, 0.65, 1.0))
          for y in (1.5, -1.5)]


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, w = q[1:], q[0]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--capture-seconds", type=float, default=4.0)
    ap.add_argument("--stabilize-seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    assert len(args.traj) == len(args.labels)

    model = MjoModel.from_xml_path(str(SPACECRAFT_CAPTURE_XML))
    data = model.make_data()
    bus = model.body_id("bus")
    gtype = list(np.asarray(model.geom_type)) + [p["type"] for p in PANELS]
    gsize = list(np.asarray(model.geom_size)) + [np.asarray(p["size"]) for p in PANELS]
    grgba = list(np.asarray(model.geom_rgba)) + [np.asarray(p["rgba"]) for p in PANELS]
    gbody = list(np.asarray(model.geom_bodyid)) + [bus] * len(PANELS)
    gpos = list(np.asarray(model.geom_pos)) + [np.asarray(p["pos"]) for p in PANELS]
    gquat = list(np.asarray(model.geom_quat)) + [np.array([1.0, 0, 0, 0])] * len(PANELS)

    runs = [np.load(p) for p in args.traj]
    t, n_a = runs[0]["t"], int(runs[0]["n_capture"])
    for r in runs[1:]:
        same = np.allclose(r["t"], t) and int(r["n_capture"]) == n_a
        assert same, "runs must share seed and durations"

    f_a = int(round(args.capture_seconds * args.fps))
    f_b = int(round(args.stabilize_seconds * args.fps))
    idx = np.round(np.concatenate([np.linspace(0, n_a - 1, f_a),
                                   np.linspace(n_a, len(t) - 1, f_b)])).astype(int)
    speed = np.concatenate([np.full(f_a, (t[n_a - 1] - t[0]) / (f_a - 1)),
                            np.full(f_b, (t[-1] - t[n_a]) / (f_b - 1))]) * args.fps

    G, F = len(gtype), len(idx)
    pos = np.zeros((len(runs), F, G, 3))
    quat = np.zeros((len(runs), F, G, 4))
    for k, r in enumerate(runs):
        for f, i in enumerate(idx):
            np.copyto(np.asarray(data.qpos), r["qpos"][i])
            np.copyto(np.asarray(data.orbit.R_eci), r["R_eci"][i])
            np.copyto(np.asarray(data.orbit.V_eci), r["V_eci"][i])
            mjo_forward(model, data)
            xpos, xquat = np.asarray(data.xpos), np.asarray(data.xquat)
            for g in range(G):
                b = gbody[g]
                pos[k, f, g] = xpos[b] + qrot(xquat[b], gpos[g])
                quat[k, f, g] = qmul(xquat[b], gquat[g])

    np.savez(args.out, labels=np.array(args.labels), geom_type=np.array(gtype),
             geom_size=np.array(gsize), geom_rgba=np.array(grgba), pos=pos, quat=quat,
             t=t[idx], speed=speed, R_km=runs[0]["R_eci"][idx], V_kms=runs[0]["V_eci"][idx],
             fps=args.fps, n_capture_frames=f_a)
    print(f"wrote {args.out}: {len(runs)} runs x {F} frames x {G} geoms")


if __name__ == "__main__":
    main()
