# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Render the banner clip (paper figure 1): a fleet of bimanual spacecraft
robots flying around in low Earth orbit, framed like ``figures/banner.png``.

CPU equivalent of ``examples/banner_viewer_gpu.py`` (which needs the warp/GPU
backend). Unlike the kinematic earlier version, every spacecraft is a real
coupled orbital + multibody simulation, fitted with three body-axis reaction
wheels and six (+/-x,y,z) RCS thrusters.

The motion is designed to stay *bounded* (a reaction-wheel craft commanded to
hold a tumble rate just spins its wheels up forever -- so don't): each robot
gets a small ballistic initial tumble (momentum-conserving), the wheels add
gentle random slews with a momentum-desaturation term and a hard speed limit,
and the thrusters apply a small damped random drift. The arms slew with the
position servos. All worlds render through one instanced
``BatchedMuJoCoScene``, world-scaled so the offscreen renderer keeps the
6371 km Earth (see scripts/record/scene.py for the why).

Defaults to real-time (``--speedup 1``) over 30 s; the timestep is chosen so
playback speed is exactly ``--speedup``. The camera matches paper figure 1: it
looks roughly along-track (+Y) so the nadir Earth's bright limb fills the left,
the cloud of robots recedes in depth, and black space fills the right. The
Earth is lit by a directional "sun" and a bumped emissive factor so the limb
reads bright like the figure.

    pixi run python scripts/record/record_banner.py --out videos/banner.mp4
"""

from __future__ import annotations

import argparse
import re
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

from mjorbit import mjo_forward, mjo_step  # noqa: E402
from mjorbit.config import OrbitInit, ReactionWheelSpec, ThrusterSpec  # noqa: E402
from mjorbit.constants import GM_EARTH, OMEGA_EARTH, R_EARTH  # noqa: E402
from mjorbit.spec import MjoSpec  # noqa: E402
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

# Empirical thruster gain for this model (measured by stepping a unit command):
# commanded thruster force -> body linear accel. The drift controller thinks in
# physical accel and converts to a force command through this.
THR_ACCEL_PER_FORCE = 2.9e-5  # m/s^2 per force-command unit


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


def _cloud_offsets_m(
    n: int,
    *,
    depth_km: float,
    width_km: float,
    height_km: float,
    x_bias: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Volumetric banner cloud in MuJoCo-world metres.

    Depth runs along +Y (along-track, the camera view axis -> perspective),
    width along +X (radial; -X is nadir/Earth, so the cloud sits on the
    Earth-facing side of the frame), height along +Z (cross-track). A jittered
    grid keeps spacing organic without rejection sampling.
    """
    # Choose a near-cubic cell grid matched to the box aspect ratio.
    extents = np.array([width_km, depth_km, height_km], dtype=float)
    ratios = extents / np.cbrt(np.prod(extents))
    counts = np.maximum(1, np.round(np.cbrt(n) * ratios).astype(int))
    while int(np.prod(counts)) < n:
        counts[np.argmax(extents / counts)] += 1
    cells = np.stack(
        np.meshgrid(*[np.arange(c) for c in counts], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    cells = cells[rng.permutation(len(cells))[:n]]
    cell_size = extents / counts
    pos_km = (cells + rng.uniform(0.1, 0.9, size=(n, 3))) * cell_size - 0.5 * extents
    # Bias depth so the cloud sits ahead of the camera (positive Y), and lift
    # it slightly toward nadir (-X) so robots overlap the bright limb.
    pos_km[:, 1] += 0.5 * depth_km
    pos_km[:, 0] += x_bias * width_km
    return pos_km * 1000.0


def _rotation_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """world-from-body rotation matrix from a (w,x,y,z) quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _build_model(*, timestep: float | None = None, wheel_speed_limit: float = 60.0):
    """Bimanual panels model fitted with 3 reaction wheels + 6 RCS thrusters.

    The wheels are speed-limited: a reaction wheel can only store so much
    momentum, and capping it (plus the desaturation term in the controller) is
    what stops the fleet from spinning up without bound.
    """
    txt = Path(SPACECRAFT_BIMANUAL_PANELS_XML).read_text()
    if timestep is not None:
        txt = re.sub(r'timestep="[^"]*"', f'timestep="{timestep:g}"', txt, count=1)
    spec = MjoSpec.from_xml_string(txt)
    for ax, nm in zip(np.eye(3), "xyz"):
        spec.mjorbit.add_reaction_wheel(
            ReactionWheelSpec(
                "bus", ax, 0.05, torque_limit=4.0,
                speed_limit=wheel_speed_limit, name=f"rw_{nm}",
            )
        )
    # Order matters: [+x, +y, +z, -x, -y, -z] so thr_force_cmd[0:3] push +axis
    # and [3:6] push -axis.
    for sgn, tag in ((+1.0, "p"), (-1.0, "n")):
        for ax, nm in zip(np.eye(3), "xyz"):
            spec.mjorbit.add_thruster(
                ThrusterSpec("bus", np.zeros(3), sgn * ax, 4000.0, name=f"thr_{nm}{tag}")
            )
    return spec.compile()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="videos/banner.mp4")
    ap.add_argument("--nworld", type=int, default=96)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speedup", type=float, default=1.0,
                    help="sim seconds per real second (1.0 = real-time)")
    ap.add_argument("--spacecraft-scale", type=float, default=440.0)
    ap.add_argument("--attitude-spread-deg", type=float, default=160.0)
    ap.add_argument("--slew-amplitude", type=float, default=0.5)
    ap.add_argument("--slew-period", type=float, default=18.0)
    ap.add_argument("--alt-km", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--world", type=float, default=1e-4)
    # Cloud extents (km, MuJoCo-world).
    ap.add_argument("--depth-km", type=float, default=26.0)
    ap.add_argument("--width-km", type=float, default=26.0)
    ap.add_argument("--height-km", type=float, default=18.0)
    ap.add_argument("--cloud-x-bias", type=float, default=-0.04,
                    help="shift cloud toward nadir/Earth (frac of width; -ve = over limb)")
    # Camera (render units): looks +Y; nadir Earth (-X) lands on the left. The
    # camera sits just behind the front of the cloud so robots fill the frame.
    ap.add_argument("--cam-back", type=float, default=0.14, help="camera dist behind cloud")
    ap.add_argument("--cam-right", type=float, default=0.08, help="camera +X offset (frac depth)")
    ap.add_argument("--cam-up", type=float, default=0.10, help="camera +Z offset (frac depth)")
    ap.add_argument("--look-ahead", type=float, default=0.42, help="look-at along depth (frac)")
    ap.add_argument("--fov", type=float, default=58.0)
    # Earth lighting.
    ap.add_argument("--earth-emissive", type=float, default=0.55)
    ap.add_argument("--sun-intensity", type=float, default=2.4)
    # Random-action motion. Each robot gets a small ballistic initial tumble
    # (momentum-conserving -> bounded), the wheels add gentle desaturated random
    # slews, and the thrusters do a small damped random drift.
    ap.add_argument("--spin-lo", type=float, default=0.025, help="min initial tumble rate (rad/s)")
    ap.add_argument("--spin-hi", type=float, default=0.055, help="max initial tumble rate (rad/s)")
    ap.add_argument("--wheel-torque", type=float, default=0.08, help="random wheel torque (N*m)")
    ap.add_argument("--desat", type=float, default=0.10, help="wheel desaturation gain")
    ap.add_argument("--wheel-speed-limit", type=float, default=60.0, help="wheel speed cap (rad/s)")
    ap.add_argument("--drift-accel", type=float, default=0.008, help="drift accel amp (m/s^2)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--port", type=int, default=8420)
    args = ap.parse_args()

    n = int(args.nworld)
    rng = np.random.default_rng(args.seed)
    W = float(args.world)
    fps = int(args.fps)

    # Pick an exact physics timestep so substeps * dt == sim-seconds-per-frame:
    # that makes the playback speed exactly --speedup (1.0 = real-time).
    sim_dt = float(args.speedup) / fps
    substeps = max(1, int(round(sim_dt / 0.01)))
    dt = sim_dt / substeps
    model = _build_model(timestep=dt, wheel_speed_limit=args.wheel_speed_limit)
    nbody = int(model.nbody)
    n_rw = len(model.reaction_wheels)
    n_thr = len(model.thrusters)

    r_orbit = R_EARTH + args.alt_km
    omega_orbit = float(np.sqrt(GM_EARTH / r_orbit**3))
    R0 = np.array([r_orbit, 0.0, 0.0])
    V0 = np.array([0.0, float(np.sqrt(GM_EARTH / r_orbit)), 0.0])

    # Per-world: random attitude (figure 1 has robots pointing every which way),
    # arm stance with a small per-robot variation, and an independent MjoData.
    quats = _fleet_quats(n, args.attitude_spread_deg, rng)
    phases = 2.0 * np.pi * np.arange(n) / max(1, n)
    arm_base = BIMANUAL_POSE[None, :] + 0.12 * np.stack(
        [np.sin(phases), np.cos(1.7 * phases), -np.sin(phases + 0.8), -np.cos(1.7 * phases + 0.5)],
        axis=-1)

    # Small ballistic initial tumble per robot: a random axis at a random rate.
    # With no external torque this is conserved, so the fleet keeps tumbling
    # gently at a *bounded* rate -- the wheels only perturb it.
    spin_axis = rng.standard_normal((n, 3))
    spin_axis /= np.linalg.norm(spin_axis, axis=1, keepdims=True)
    omega0 = spin_axis * rng.uniform(args.spin_lo, args.spin_hi, size=(n, 1))

    datas = []
    for w in range(n):
        d = model.make_data(orbit=OrbitInit(R_eci=R0, V_eci=V0))
        qp = np.asarray(d.qpos)
        qp[:] = 0.0
        qp[3:7] = quats[w]
        qp[7:11] = arm_base[w]
        np.asarray(d.ctrl)[:] = arm_base[w]
        mjo_forward(model, d)
        np.asarray(d.qvel)[3:6] = omega0[w]  # body-frame initial angular rate
        mjo_forward(model, d)
        datas.append(d)

    # Smooth bounded random references per world: a sum of sinusoids with random
    # frequency/phase/sign per axis (zero mean).
    def _modes(shape, t_lo, t_hi):
        omega = 2.0 * np.pi / rng.uniform(t_lo, t_hi, size=shape)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=shape)
        weight = rng.uniform(0.6, 1.0, size=shape) * rng.choice([-1.0, 1.0], size=shape)
        return omega, phase, weight

    n_modes = 2
    w_omega, w_phase, w_weight = _modes((n, 3, n_modes), 20.0, 55.0)  # wheel-torque ref
    t_omega, t_phase, t_weight = _modes((n, 3, n_modes), 38.0, 85.0)  # world-drift ref

    def wheel_ref(t):  # (n,3) open-loop random reaction-wheel torque (N*m)
        s = np.sin(w_omega * t + w_phase) * w_weight
        return args.wheel_torque * s.mean(axis=-1)

    def accel_ref(t):  # (n,3) target world linear accel (m/s^2)
        s = np.sin(t_omega * t + t_phase) * t_weight
        return args.drift_accel * s.mean(axis=-1)

    K_VEL = 0.06          # translation velocity damping (1/s)
    K_DESAT = float(args.desat)  # reaction-wheel momentum desaturation gain

    # ----------------------------------------------------------------
    # Scene: instanced fleet + world-scaled, sun-lit Earth + stars.
    # ----------------------------------------------------------------
    server = viser.ViserServer(host="127.0.0.1", port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.set_background_image(np.zeros((4, 4, 3), dtype=np.uint8))
    server.initial_camera.far = 5.0e6
    server.scene.enable_default_lights(True)
    # A "sun" from the camera/fleet side (+X +Y, slightly above) so the Earth's
    # camera-facing limb is lit instead of dark.
    server.scene.add_light_directional(
        "/sun", color=(255, 250, 240), intensity=float(args.sun_intensity),
        wxyz=tuple(vtf.SO3.from_matrix(_look_rotation(np.array([-1.0, -0.6, -0.5]))).wxyz),
    )

    scale = args.spacecraft_scale * W
    scene = BatchedMuJoCoScene(server, model, n, scale=scale)

    earth_r = R_EARTH * 1000.0 * W
    earth_mesh = create_earth_mesh(
        earth_r, texture_path=EARTH_TEXTURE_HQ_PATH, lat_segments=128, lon_segments=256)
    e = float(args.earth_emissive)
    earth_mesh.visual.material.emissiveFactor = [e, e, e]
    earth = server.scene.add_mesh_trimesh(
        "/earth", earth_mesh, cast_shadow=False, receive_shadow=False)
    atmo = server.scene.add_mesh_trimesh(
        "/earth_atmo", _create_atmosphere_mesh(earth_r * 1.015),
        cast_shadow=False, receive_shadow=False)
    add_star_field(server, radius=1400.0)

    offsets_m = _cloud_offsets_m(
        n, depth_km=args.depth_km, width_km=args.width_km,
        height_km=args.height_km, x_bias=args.cloud_x_bias, rng=rng)
    offsets_render = offsets_m * W

    # Camera: behind the cloud along -Y looking +Y (toward look-at), lifted and
    # shifted so the formation reads 3/4 with depth. Earth (-X) -> left of frame.
    depth_render = args.depth_km * 1000.0 * W
    cx = float(np.median(offsets_render[:, 0]))
    cz = float(np.median(offsets_render[:, 2]))
    cam_pos = np.array([
        cx + args.cam_right * depth_render,
        offsets_render[:, 1].min() - args.cam_back * depth_render,
        cz + args.cam_up * depth_render,
    ])
    look_at = np.array([cx, args.look_ahead * depth_render, cz])

    rec = HeadlessRecorder(server, port=args.port, width=args.width, height=args.height, warmup=9.0)

    n_frames = max(2, int(round(args.seconds * args.fps)))
    omega_slew = 2.0 * np.pi / args.slew_period
    label = speedup_label(n_frames * sim_dt, n_frames / fps)

    xpos = np.zeros((n, nbody, 3))
    xquat = np.zeros((n, nbody, 4))

    print("=" * 64)
    print(f"banner: {n} bimanual robots, real dynamics ({n_rw} wheels + {n_thr} thrusters each)")
    print(f"  dt={dt:g}s, {substeps} substeps/frame, sim_dt={sim_dt:g}s, {label}")
    print("=" * 64)

    try:
        with VideoWriter(args.out, fps=fps, width=args.width, height=args.height) as vid:
            sim_t = 0.0
            for f in range(n_frames):
                for _ in range(substeps):
                    tau_ref = wheel_ref(sim_t)   # (n,3) random wheel torque
                    aref = accel_ref(sim_t)      # (n,3) world-frame drift accel
                    arm_t = arm_base + (
                        args.slew_amplitude * np.sin(omega_slew * sim_t + phases)
                    )[:, None] * BIMANUAL_SLEW[None, :]
                    for w, d in enumerate(datas):
                        qvel = np.asarray(d.qvel)
                        quat = np.asarray(d.qpos)[3:7]
                        # Reaction wheels: gentle random slew torque, minus a
                        # desaturation term that bleeds stored wheel momentum so
                        # the fleet can't spin itself up without bound.
                        h_wheel = np.asarray(d.actuators.rw_momentum)
                        torque = np.clip(tau_ref[w] - K_DESAT * h_wheel, -4.0, 4.0)
                        np.asarray(d.actuators.rw_torque_cmd)[:] = torque
                        # Thrusters: track a small random world-frame drift, damped.
                        vel_world = qvel[0:3]
                        a_world = aref[w] - K_VEL * vel_world
                        Rwb = _quat_to_mat(quat)
                        a_body = Rwb.T @ a_world
                        f_body = a_body / THR_ACCEL_PER_FORCE
                        thr = np.zeros(n_thr)
                        thr[0:3] = np.clip(np.maximum(f_body, 0.0), 0.0, 4000.0)
                        thr[3:6] = np.clip(np.maximum(-f_body, 0.0), 0.0, 4000.0)
                        np.asarray(d.actuators.thr_force_cmd)[:] = thr
                        # Arm servos slew the bimanual stance.
                        np.asarray(d.ctrl)[:] = arm_t[w]
                        mjo_step(model, d)
                    sim_t += dt

                for w, d in enumerate(datas):
                    xpos[w] = np.asarray(d.xpos)
                    xquat[w] = np.asarray(d.xquat)

                # Render-scale fix: the scene leaves the anchor (bus) translation
                # un-amplified (raw metres), inconsistent with the W-scaled
                # offsets. Fold -(1-scale)*bus into the offset so the whole craft
                # -- bus drift included -- is uniformly magnified by `scale`.
                anchor = xpos[:, 1]
                offsets = offsets_render - (1.0 - scale) * anchor

                scene.update(xpos, xquat, offsets=offsets)

                ang = omega_orbit * sim_t
                R_t = np.array([R0[0] * np.cos(ang), R0[0] * np.sin(ang), 0.0])
                earth_pos = tuple(-R_t * 1000.0 * W)
                spin = _rotation_z(OMEGA_EARTH * sim_t)
                with server.atomic():
                    earth.position = earth_pos
                    earth.wxyz = tuple(vtf.SO3.from_matrix(spin).wxyz)
                    atmo.position = earth_pos

                frame = rec.render(position=cam_pos, look_at=look_at, fov_deg=args.fov)
                vid.add(annotate(frame, label))
        print(f"wrote {args.out}  ({n_frames} frames, {n_frames/args.fps:.1f}s, nworld={n})")
    finally:
        rec.close()
        server.stop()


def _look_rotation(direction: np.ndarray) -> np.ndarray:
    """A rotation matrix whose local -Z points along ``direction`` (for a
    directional light: the light shines along ``direction``)."""
    f = np.asarray(direction, dtype=float)
    f = f / np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0])
    if abs(f @ up) > 0.98:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, -f)
    right /= np.linalg.norm(right)
    true_up = np.cross(-f, right)
    return np.stack([right, true_up, -f], axis=1)


if __name__ == "__main__":
    main()
