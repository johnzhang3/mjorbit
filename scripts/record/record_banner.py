# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Render the banner clip: a fleet of bimanual spacecraft robots in LEO.

CPU equivalent of ``examples/banner_viewer_gpu.py`` (which needs the warp/GPU
backend). The fleet motion in a banner is cosmetic — per-world random attitudes
and coordinated arm slews — so this animates each world kinematically (set the
attitude + arm pose, run forward kinematics) and renders all worlds with one
instanced ``BatchedMuJoCoScene``, world-scaled so the offscreen renderer keeps
the 6371 km Earth (see scripts/record/scene.py for the why).

    pixi run python scripts/record/record_banner.py --out videos/banner.mp4 --nworld 60
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import viser
import viser.transforms as vtf

_ROOT = Path(__file__).resolve().parents[2]
for p in (str(_ROOT / "src"), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from recorder import HeadlessRecorder, VideoWriter, annotate, speedup_label  # noqa: E402

from mjorbit import MjoModel, OrbitInit, mjo_forward  # noqa: E402
from mjorbit.constants import GM_EARTH, OMEGA_EARTH, R_EARTH  # noqa: E402
from mjorbit.testdata import SPACECRAFT_BIMANUAL_PANELS_XML  # noqa: E402
from viewer.batched import BatchedMuJoCoScene  # noqa: E402
from viewer.earth import (  # noqa: E402
    EARTH_TEXTURE_HQ_PATH,
    _create_atmosphere_mesh,
    add_star_field,
    create_earth_mesh,
)

# Coordinated bimanual stance + mirror-symmetric slew (from banner_viewer_gpu).
BIMANUAL_POSE = np.array([-1.1, -0.9, 1.1, 0.9])
BIMANUAL_SLEW = np.array([1.0, -0.6, -1.0, 0.6])


def _fleet_quats(n: int, spread_deg: float, rng: np.random.Generator) -> np.ndarray:
    if spread_deg >= 180.0:
        u1, u2, u3 = rng.uniform(0.0, 1.0, size=(3, n))
        r1, r2 = np.sqrt(1.0 - u1), np.sqrt(u1)
        return np.stack([r1 * np.sin(2 * np.pi * u2), r1 * np.cos(2 * np.pi * u2),
                         r2 * np.sin(2 * np.pi * u3), r2 * np.cos(2 * np.pi * u3)], axis=-1)
    axes = rng.standard_normal((n, 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    angles = np.deg2rad(spread_deg) * rng.uniform(0.0, 1.0, size=n)
    half = 0.5 * angles
    return np.concatenate([np.cos(half)[:, None], np.sin(half)[:, None] * axes], axis=-1)


def _grid_offsets_m(n, sy, sz, depth, rng, *, rows=0, aspect=2.5):
    if rows > 0:
        nrows = min(rows, n)
    else:
        nrows = max(1, min(round(math.sqrt(n * sy / (aspect * sz))), n))
    ncols = math.ceil(n / nrows)
    col = np.arange(n) % ncols
    row = np.arange(n) // ncols
    y = (col - 0.5 * (ncols - 1)) * sy
    z = (row - 0.5 * (nrows - 1)) * sz
    z = z + 0.18 * sz * (col % 2) * (1 if nrows > 1 else 0)
    x = rng.uniform(-depth, depth, size=n)
    y = y + rng.uniform(-0.08, 0.08, size=n) * sy
    z = z + rng.uniform(-0.08, 0.08, size=n) * sz
    return np.stack([x, y, z], axis=-1) * 1000.0


def _rotation_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="videos/banner.mp4")
    ap.add_argument("--nworld", type=int, default=60)
    ap.add_argument("--seconds", type=float, default=11.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--spacecraft-scale", type=float, default=440.0)
    ap.add_argument("--attitude-spread-deg", type=float, default=70.0)
    ap.add_argument("--slew-amplitude", type=float, default=0.5)
    ap.add_argument("--slew-period", type=float, default=18.0)
    ap.add_argument("--alt-km", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--world", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--port", type=int, default=8420)
    args = ap.parse_args()

    n = int(args.nworld)
    rng = np.random.default_rng(args.seed)
    W = float(args.world)

    model = MjoModel.from_xml_path(SPACECRAFT_BIMANUAL_PANELS_XML)
    nbody = int(model.nbody)
    r_orbit = R_EARTH + args.alt_km
    omega_orbit = float(np.sqrt(GM_EARTH / r_orbit**3))
    R0 = np.array([r_orbit, 0.0, 0.0])
    V0 = np.array([0.0, float(np.sqrt(GM_EARTH / r_orbit)), 0.0])
    data = model.make_data(orbit=OrbitInit(R_eci=R0, V_eci=V0))

    # Per-world attitudes + arm phases + render offsets.
    quats = _fleet_quats(n, args.attitude_spread_deg, rng)
    phases = 2.0 * np.pi * np.arange(n) / max(1, n)
    arm_base = BIMANUAL_POSE[None, :] + 0.12 * np.stack(
        [np.sin(phases), np.cos(1.7 * phases), -np.sin(phases + 0.8), -np.cos(1.7 * phases + 0.5)],
        axis=-1)
    offsets_m = _grid_offsets_m(n, 2.6, 2.2, 0.6, rng, aspect=2.5)
    offsets_render = offsets_m * W

    # Scene: instanced fleet + world-scaled Earth + stars.
    server = viser.ViserServer(host="127.0.0.1", port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.set_background_image(np.zeros((4, 4, 3), dtype=np.uint8))
    server.initial_camera.far = 5.0e6
    scene = BatchedMuJoCoScene(server, model, n, scale=args.spacecraft_scale * W)
    earth_r = R_EARTH * 1000.0 * W
    earth = server.scene.add_mesh_trimesh(
        "/earth", create_earth_mesh(earth_r, texture_path=EARTH_TEXTURE_HQ_PATH,
                                    lat_segments=128, lon_segments=256),
        cast_shadow=False, receive_shadow=False)
    atmo = server.scene.add_mesh_trimesh(
        "/earth_atmo", _create_atmosphere_mesh(earth_r * 1.015),
        cast_shadow=False, receive_shadow=False)
    add_star_field(server, radius=1400.0)

    span_y = float(np.ptp(offsets_m[:, 1])) + 2.0 * args.spacecraft_scale * 5.0
    span_z = float(np.ptp(offsets_m[:, 2])) + 2.0 * args.spacecraft_scale * 5.0
    cam_dist = max(0.85 * span_y, 2.1 * span_z, 8000.0) * W

    rec = HeadlessRecorder(server, port=args.port, width=args.width, height=args.height, warmup=9.0)

    n_frames = max(2, int(round(args.seconds * args.fps)))
    omega_slew = 2.0 * np.pi / args.slew_period
    # Cosmetic: advance the chief along a circular orbit so the Earth drifts/spins.
    sim_dt = 0.6  # s of sim-time per rendered frame
    label = speedup_label(n_frames * sim_dt, n_frames / args.fps)

    xpos = np.zeros((n, nbody, 3))
    xquat = np.zeros((n, nbody, 4))

    def fk_world(quat, arm):
        qp = np.asarray(data.qpos)
        qp[:] = 0.0
        qp[3:7] = quat
        qp[7:11] = arm
        mjo_forward(model, data)
        return np.asarray(data.xpos).copy(), np.asarray(data.xquat).copy()

    try:
        with VideoWriter(args.out, fps=args.fps, width=args.width, height=args.height) as vid:
            for f in range(n_frames):
                t = f * sim_dt
                slew = args.slew_amplitude * np.sin(omega_slew * t + phases)
                arms = arm_base + slew[:, None] * BIMANUAL_SLEW[None, :]
                for w in range(n):
                    xp, xq = fk_world(quats[w], arms[w])
                    xpos[w] = xp
                    xquat[w] = xq
                scene.update(xpos, xquat, offsets=offsets_render)
                # Earth: chief advances along its circular orbit; globe spins.
                ang = omega_orbit * t
                R_t = np.array([R0[0] * np.cos(ang), R0[0] * np.sin(ang), 0.0])
                earth_pos = tuple(-R_t * 1000.0 * W)
                spin = _rotation_z(OMEGA_EARTH * t)
                with server.atomic():
                    earth.position = earth_pos
                    earth.wxyz = tuple(vtf.SO3.from_matrix(spin).wxyz)
                    atmo.position = earth_pos
                # Slight 3/4 tilt for depth; Earth (at -X) stays behind the fleet.
                cam_pos = np.array([cam_dist, 0.28 * cam_dist, 0.20 * cam_dist])
                frame = rec.render(position=cam_pos, look_at=np.zeros(3), fov_deg=44.0)
                vid.add(annotate(frame, label))
        print(f"wrote {args.out}  ({n_frames} frames, {n_frames/args.fps:.1f}s, nworld={n})")
    finally:
        rec.close()
        server.stop()


if __name__ == "__main__":
    main()
