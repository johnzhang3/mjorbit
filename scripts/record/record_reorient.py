# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Render the dual-arm 3-DOF reorientation clip from a saved trajectory.

The MPPI controller is expensive, so we record in two deterministic steps:
first dump a trajectory, then replay it through the world-scaled scene.

    pixi run python examples/reorient_mppi_3dof.py \
        --duration 60 --save-traj /tmp/reorient_traj.npz
    pixi run python scripts/record/record_reorient.py \
        --traj /tmp/reorient_traj.npz --out videos/reorient.mp4

To instead render the 4:3 hero still used for the paper's Fig. 1(a)
(multibody attitude control), pass ``--still-out`` (sun-lit, bright Earth):

    pixi run python scripts/record/record_reorient.py --traj /tmp/reorient_traj.npz \
        --still-out .../mujoco_orbit_latex/figures/example_attitude.png

The camera frames the whole bus+arms assembly (the bus is free-floating and does
not translate, so it stays centred) from a fixed oblique LVLH view, which keeps
the *attitude* slew unambiguous -- the bus visibly rotates while the camera holds
still. ``--seconds`` sets the clip length: 60 s of sim into a shorter clip plays
back sped up, and the burned-in caption reports the speed-up factor.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
for p in (str(_SRC), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from render_traj import CameraConfig, render_trajectory  # noqa: E402

from mjorbit import MjoModel, OrbitInit, mjo_forward  # noqa: E402
from viewer.framing import distance_for_fill, spacecraft_bounding_radius  # noqa: E402


def _frame_distance(model, data, qpos: np.ndarray, R_eci, V_eci, center_bid: int,
                    *, render_scale: float, fov_deg: float, fill: float) -> float:
    """Camera distance (render units) that keeps the assembly framed over the clip.

    The arms wave during the slew, so sample the bounding radius across the
    trajectory and frame the largest pose with a little margin (``fill``).
    """
    np.copyto(np.asarray(data.orbit.R_eci), np.asarray(R_eci[0], dtype=float))
    np.copyto(np.asarray(data.orbit.V_eci), np.asarray(V_eci, dtype=float))
    radius = 0.0
    for step in np.unique(np.linspace(0, len(qpos) - 1, 24).astype(int)):
        np.copyto(np.asarray(data.qpos), np.asarray(qpos[step], dtype=float))
        mjo_forward(model, data)
        radius = max(radius, spacecraft_bounding_radius(model, data, center_bid))
    return distance_for_fill(radius * render_scale, fov=np.deg2rad(fov_deg), fill=fill)


def _look_rotation(direction) -> np.ndarray:
    """Rotation whose local -Z points along ``direction`` (a directional light
    then shines along ``direction``)."""
    f = np.asarray(direction, dtype=float)
    f = f / np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0]) if abs(f @ [0, 0, 1]) <= 0.98 else np.array([0.0, 1.0, 0.0])
    right = np.cross(up, -f)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(-f, right), -f], axis=1)


def render_still(model, data, qpos, R_eci, V_eci, *, out_path: str, frame_frac: float,
                 fill: float, fov_deg: float, mag: float, world: float,
                 width: int, height: int, port: int, offset_rsw) -> None:
    """Render one 4:3 hero still of the slew (sun-lit, bright Earth) for the paper.

    Unlike the video, this adds a directional "sun" and bumps the Earth's emissive
    factor so the limb reads bright (matching the other example panels), and frames
    a single posed attitude rather than the whole clip.
    """
    import viser  # noqa: PLC0415  (optional render path; keep the video path lean)
    import viser.transforms as vtf  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from recorder import HeadlessRecorder  # noqa: PLC0415
    from scene import SingleScene, forward_to  # noqa: PLC0415

    from mjorbit.constants import R_EARTH  # noqa: PLC0415
    from viewer.earth import EARTH_TEXTURE_HQ_PATH, create_earth_mesh  # noqa: PLC0415

    render_scale = mag * world
    bus = model.body_id("bus")
    dist = _frame_distance(model, data, qpos, R_eci, V_eci, bus,
                           render_scale=render_scale, fov_deg=fov_deg, fill=fill)
    server = viser.ViserServer(host="127.0.0.1", port=port)
    scene = SingleScene(server, model, data, mag=mag, world=world, track_body_id=bus)
    # The Earth mesh is emissive-driven, so re-add it with a boosted emissive factor
    # (and add a sun) to light the limb instead of rendering it dark.
    earth_r = R_EARTH * 1000.0 * world
    earth_mesh = create_earth_mesh(earth_r, texture_path=EARTH_TEXTURE_HQ_PATH,
                                   lat_segments=128, lon_segments=256)
    earth_mesh.visual.material.emissiveFactor = [0.7, 0.7, 0.7]
    earth = server.scene.add_mesh_trimesh("/earth", earth_mesh,
                                          cast_shadow=False, receive_shadow=False)
    server.scene.enable_default_lights(True)
    server.scene.add_light_directional(
        "/sun", color=(255, 250, 240), intensity=2.6,
        wxyz=tuple(vtf.SO3.from_matrix(_look_rotation([-1.0, -0.6, -0.5])).wxyz))
    rec = HeadlessRecorder(server, port=port, width=width, height=height, warmup=9.0)
    try:
        step = int(round(frame_frac * (len(qpos) - 1)))
        forward_to(model, data, qpos[step], R_eci=R_eci[step], V_eci=V_eci)
        scene.update()
        earth.position = tuple(-1000.0 * np.asarray(R_eci[step], dtype=float) * world)
        target = scene.cluster_center_render([bus])
        pos, look = scene.camera_pose(distance=dist, target=target,
                                      offset_rsw=np.asarray(offset_rsw, dtype=float))
        img = rec.render(position=pos, look_at=look, fov_deg=fov_deg)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(out_path)
        print(f"wrote {out_path}  ({width}x{height}, frame {step}/{len(qpos) - 1})")
    finally:
        rec.close()
        server.stop()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", default="/tmp/reorient_traj.npz")
    ap.add_argument("--out", default="videos/reorient.mp4")
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="clip length; 60 s of sim into 20 s plays back at 3x real-time")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mag", type=float, default=5000.0,
                    help="spacecraft magnification against the globe")
    ap.add_argument("--world", type=float, default=1e-4)
    ap.add_argument("--fov", type=float, default=46.0)
    ap.add_argument("--fill", type=float, default=1.6,
                    help="how much of the view the assembly fills (larger = bigger robot; "
                         ">1 lets the wide arm span exceed the frame height so the robot "
                         "takes up well over half the 16:9 frame)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--port", type=int, default=8401)
    # Still-figure mode: render one 4:3 hero frame (e.g. paper Fig. 1a) instead of
    # the video. Uses a sun + bright Earth and its own 4:3 size.
    ap.add_argument("--still-out", default=None,
                    help="render a single 4:3 PNG to this path (paper figure) and exit")
    ap.add_argument("--still-frac", type=float, default=0.45,
                    help="trajectory fraction to freeze for the still (0=start, 1=end)")
    ap.add_argument("--still-fill", type=float, default=1.3, help="framing for the still")
    ap.add_argument("--still-width", type=int, default=1280)
    ap.add_argument("--still-height", type=int, default=960)   # 4:3
    args = ap.parse_args()

    d = np.load(args.traj, allow_pickle=True)
    qpos, R_eci, V_eci = d["qpos"], d["R_eci"], d["V_eci"]
    xml_path = str(d["xml_path"])
    model = MjoModel.from_xml_path(xml_path)
    data = model.make_data(orbit=OrbitInit(R_eci=R_eci[0], V_eci=V_eci))

    bus = model.body_id("bus")
    render_scale = args.mag * args.world

    if args.still_out is not None:
        render_still(model, data, qpos, R_eci, V_eci, out_path=args.still_out,
                     frame_frac=args.still_frac, fill=args.still_fill, fov_deg=args.fov,
                     mag=args.mag, world=args.world, width=args.still_width,
                     height=args.still_height, port=args.port,
                     offset_rsw=np.array([0.45, 0.80, 0.40]))
        return

    dist = _frame_distance(model, data, qpos, R_eci, V_eci, bus,
                           render_scale=render_scale, fov_deg=args.fov, fill=args.fill)

    camera = CameraConfig(
        # Oblique LVLH view (radial, along-track, cross-track) that shows all
        # three rotation axes; a very gentle arc keeps the shot alive without
        # masking the bus's own rotation as camera motion.
        track_body_ids=[bus],
        offset_rsw=np.array([0.45, 0.80, 0.40]),
        offset_end=np.array([0.55, 0.62, 0.56]),
        distance=dist,
        fov_deg=args.fov,
    )
    render_trajectory(
        model=model, data=data, qpos=qpos, R_eci=R_eci, V_eci=V_eci,
        out_path=args.out, camera=camera, mag=args.mag, world=args.world,
        video_seconds=args.seconds, fps=args.fps,
        width=args.width, height=args.height, port=args.port,
        sim_dt=float(d["dt"]),
    )


if __name__ == "__main__":
    main()
