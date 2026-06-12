"""GPU fleet banner: bimanual spacecraft simulated on the GPU, framed for a figure.

Steps an ``nworld`` batch of the bimanual dual-arm spacecraft with the
``mujoco_orbit_warp`` backend (one model, all worlds stepped together on the
GPU) and renders every world in the browser with instanced meshes — one draw
call per geom, not per spacecraft. The default composition is banner-friendly:
~sixty spacecraft in a staggered grid with uniformly random attitudes
(``--attitude-spread-deg`` dials the alignment), arms in a coordinated
bimanual stance slewing gently, a high-resolution NASA Blue Marble Earth
behind, and a black sky.

The same script scales to thousands of worlds (the simulation cost is the
same code path): pass ``--nworld 2048`` and the layout switches to a
banner-cloud automatically.

Cloud/grid placement is cosmetic (a per-world render offset, like
examples/arm_reach_viewer.py); the physics of every world is the standard
coupled orbital + multibody simulation at the shared chief orbit.

Usage:
    pixi install -e warp
    pixi run -e warp python examples/banner_viewer_gpu.py            # 24, grid
    pixi run -e warp python examples/banner_viewer_gpu.py --nworld 2048
"""

from __future__ import annotations

import argparse
import math
import time as wall_time

import numpy as np

try:
    import warp as wp

    import mujoco_orbit_warp as mjo_warp
except ImportError as exc:  # pragma: no cover - guidance for the common mistake
    raise SystemExit(
        "This example needs the warp environment: "
        "pixi install -e warp && pixi run -e warp python examples/banner_viewer_gpu.py"
    ) from exc

import viser

from mujoco_orbit import MjoModel as CpuMjoModel
from mujoco_orbit.constants import GM_EARTH, R_EARTH
from mujoco_orbit.testdata import SPACECRAFT_BIMANUAL_PANELS_XML
from viewer.batched import BatchedMuJoCoScene
from viewer.earth import EARTH_TEXTURE_HQ_PATH, EarthVisual, add_star_field

# Coordinated bimanual stance: both end effectors above the bus (+Z), facing
# each other over the shared workspace. Joint order: [shoulder_a, elbow_a,
# shoulder_b, elbow_b].
BIMANUAL_POSE = np.array([-1.1, -0.9, 1.1, 0.9])
# Slew direction that keeps the stance mirror-symmetric (end effectors move
# together/apart rather than waving independently).
BIMANUAL_SLEW = np.array([1.0, -0.6, -1.0, 0.6])


def _fleet_quats(n: int, spread_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Per-world attitudes in MuJoCo (w,x,y,z) order.

    ``spread_deg >= 180`` samples uniformly over SO(3) (Marsaglia); smaller
    values tilt the identity attitude by a random axis and an angle up to
    ``spread_deg``, keeping the fleet visually coherent (wings roughly
    parallel) while avoiding a sterile perfectly-aligned look.
    """
    if spread_deg >= 180.0:
        u1, u2, u3 = rng.uniform(0.0, 1.0, size=(3, n))
        r1, r2 = np.sqrt(1.0 - u1), np.sqrt(u1)
        return np.stack(
            [
                r1 * np.sin(2.0 * np.pi * u2),
                r1 * np.cos(2.0 * np.pi * u2),
                r2 * np.sin(2.0 * np.pi * u3),
                r2 * np.cos(2.0 * np.pi * u3),
            ],
            axis=-1,
        )
    axes = rng.standard_normal((n, 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    angles = np.deg2rad(spread_deg) * rng.uniform(0.0, 1.0, size=n)
    half = 0.5 * angles
    return np.concatenate(
        [np.cos(half)[:, None], np.sin(half)[:, None] * axes], axis=-1
    )


def _grid_offsets_m(
    n: int,
    spacing_y_km: float,
    spacing_z_km: float,
    depth_jitter_km: float,
    rng: np.random.Generator,
    *,
    rows: int = 0,
    aspect: float = 2.5,
) -> np.ndarray:
    """Staggered banner grid: columns along Y (horizontal), rows along Z.

    With ``rows=0`` the row count is chosen so the grid's width:height ratio
    is roughly ``aspect`` — a banner-shaped field rather than a thin line.
    """
    if rows > 0:
        nrows = min(rows, n)
    else:
        nrows = round(math.sqrt(n * spacing_y_km / (aspect * spacing_z_km)))
        nrows = max(1, min(nrows, n))
    ncols = math.ceil(n / nrows)
    col_idx = np.arange(n) % ncols
    row_idx = np.arange(n) // ncols
    y = (col_idx - 0.5 * (ncols - 1)) * spacing_y_km
    z = (row_idx - 0.5 * (nrows - 1)) * spacing_z_km
    # Stagger alternate columns vertically and jitter depth so the grid reads
    # as a formation rather than a lattice.
    z = z + 0.18 * spacing_z_km * (col_idx % 2) * (1 if nrows > 1 else 0)
    x = rng.uniform(-depth_jitter_km, depth_jitter_km, size=n)
    y = y + rng.uniform(-0.08, 0.08, size=n) * spacing_y_km
    z = z + rng.uniform(-0.08, 0.08, size=n) * spacing_z_km
    return np.stack([x, y, z], axis=-1) * 1000.0


def _cloud_offsets_m(
    n: int, extents_km: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Jittered-grid cloud for large fleets: organic spacing, no rejection."""
    ratios = extents_km / np.cbrt(np.prod(extents_km))
    counts = np.maximum(1, np.round(np.cbrt(n) * ratios).astype(int))
    while np.prod(counts) < n:
        counts[np.argmax(extents_km / counts)] += 1
    cells = np.stack(
        np.meshgrid(*[np.arange(c) for c in counts], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    cells = cells[rng.permutation(len(cells))[:n]]
    cell_size = 2.0 * extents_km / counts
    pos_km = (cells + rng.uniform(0.15, 0.85, size=(n, 3))) * cell_size - extents_km
    return pos_km * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nworld", type=int, default=60, help="GPU batch size")
    parser.add_argument(
        "--layout", choices=["auto", "grid", "cloud"], default="auto",
        help="banner grid for small fleets, jittered cloud for thousands",
    )
    parser.add_argument("--rows", type=int, default=0,
                        help="grid rows (0 = auto from --grid-aspect)")
    parser.add_argument("--grid-aspect", type=float, default=2.5,
                        help="target grid width:height ratio when --rows is 0")
    parser.add_argument("--alt-km", type=float, default=600.0, help="orbit altitude")
    parser.add_argument("--spacing-y-km", type=float, default=2.6,
                        help="grid horizontal spacing")
    parser.add_argument("--spacing-z-km", type=float, default=2.2,
                        help="grid row spacing")
    parser.add_argument(
        "--extent-x-km", type=float, default=8.0,
        help="cloud half-extent along ECI X (depth, toward the camera)",
    )
    parser.add_argument(
        "--extent-y-km", type=float, default=60.0,
        help="cloud half-extent along ECI Y (horizontal banner span)",
    )
    parser.add_argument(
        "--extent-z-km", type=float, default=14.0,
        help="cloud half-extent along ECI Z (vertical banner span)",
    )
    parser.add_argument(
        "--spacecraft-scale", type=float, default=300.0,
        help="visual scale per spacecraft (1.0 = truthful: 1 m bus, "
        "~5.4 m wingspan tip-to-tip)",
    )
    parser.add_argument(
        "--attitude-spread-deg", type=float, default=180.0,
        help="attitude randomization: 180 = fully random over SO(3); "
        "smaller values tilt the aligned attitude by at most this angle",
    )
    parser.add_argument("--slew-amplitude", type=float, default=0.25,
                        help="arm slew amplitude (rad); 0 freezes the pose")
    parser.add_argument("--slew-period", type=float, default=40.0,
                        help="arm slew period (s)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="sim speed multiplier vs realtime")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--earth-2k", action="store_true",
                        help="use the lighter 2k Earth texture")
    args = parser.parse_args()

    n = int(args.nworld)
    if n < 2:
        raise SystemExit("--nworld must be at least 2 (batched host mirrors)")
    layout = args.layout
    if layout == "auto":
        layout = "grid" if n <= 60 else "cloud"
    rng = np.random.default_rng(args.seed)

    # ----------------------------------------------------------------
    # Models: warp model for simulation, CPU model for render metadata
    # ----------------------------------------------------------------
    model = mjo_warp.MjoModel.from_xml_path(SPACECRAFT_BIMANUAL_PANELS_XML)
    cpu_model = CpuMjoModel.from_xml_path(SPACECRAFT_BIMANUAL_PANELS_XML)
    nu, nbody = int(model.nu), int(cpu_model.nbody)
    dt = float(cpu_model.opt.timestep)

    r_orbit = R_EARTH + args.alt_km
    omega_orbit = float(np.sqrt(GM_EARTH / r_orbit**3))
    R0 = np.array([r_orbit, 0.0, 0.0])
    V0 = np.array([0.0, np.sqrt(GM_EARTH / r_orbit), 0.0])
    data = model.make_data(
        orbit=mjo_warp.OrbitInit(R_eci=R0, V_eci=V0), nworld=n
    )

    # Per-world initial state: near-aligned attitude, bimanual arm stance with
    # a small per-robot variation.
    phases = 2.0 * np.pi * np.arange(n) / max(1, n)
    arm_base = BIMANUAL_POSE[None, :] + 0.12 * np.stack(
        [
            np.sin(phases),
            np.cos(1.7 * phases),
            -np.sin(phases + 0.8),
            -np.cos(1.7 * phases + 0.5),
        ],
        axis=-1,
    )
    data.qpos[:] = 0.0
    data.qpos[:, 3:7] = _fleet_quats(n, args.attitude_spread_deg, rng)
    data.qpos[:, 7:11] = arm_base
    data.ctrl[:] = arm_base.astype(np.float32)
    mjo_warp.mjo_upload(model, data, fields=("state", "inputs", "core"))
    mjo_warp.mjo_forward(model, data)

    # Warm up, then capture one step as a CUDA graph for the hot loop.
    for _ in range(4):
        mjo_warp.mjo_step(model, data)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        mjo_warp.mjo_step(model, data)
    graph = capture.graph
    wp.synchronize()

    # ----------------------------------------------------------------
    # Scene: instanced fleet + HQ Earth + stars, chief-centered frame
    # ----------------------------------------------------------------
    server = viser.ViserServer(port=args.port)
    server.scene.set_background_image(np.zeros((2, 2, 3), dtype=np.uint8))
    if layout == "grid":
        offsets_m = _grid_offsets_m(
            n, args.spacing_y_km, args.spacing_z_km, 0.6, rng,
            rows=args.rows, aspect=args.grid_aspect,
        )
    else:
        offsets_m = _cloud_offsets_m(
            n, np.array([args.extent_x_km, args.extent_y_km, args.extent_z_km]), rng
        )
    scene = BatchedMuJoCoScene(
        server, cpu_model, n, scale=args.spacecraft_scale
    )
    earth = EarthVisual(
        server,
        textured=True,
        atmosphere=True,
        texture_path=None if args.earth_2k else EARTH_TEXTURE_HQ_PATH,
        lat_segments=64 if args.earth_2k else 128,
        lon_segments=128 if args.earth_2k else 256,
    )
    add_star_field(server, radius=6.0e7)

    margin_m = 2.0 * args.spacecraft_scale * 5.0
    span_y_m = float(np.ptp(offsets_m[:, 1])) + margin_m
    span_z_m = float(np.ptp(offsets_m[:, 2])) + margin_m
    cam_dist = max(0.85 * span_y_m, 2.1 * span_z_m, 8000.0)
    server.initial_camera.position = (cam_dist, 0.0, 0.12 * cam_dist)
    server.initial_camera.look_at = (0.0, 0.0, 0.0)

    with server.gui.add_folder("Fleet"):
        pause_box = server.gui.add_checkbox("pause", initial_value=False)
        # Above ~0.6 rad the arms start sweeping through the bus/radiators
        # (contact is disabled in this model), so cap the slider there.
        slew_slider = server.gui.add_slider(
            "slew amplitude (rad)", min=0.0, max=0.6, step=0.05,
            initial_value=min(args.slew_amplitude, 0.6),
        )
        speed_slider = server.gui.add_slider(
            "speed", min=0.0, max=8.0, step=0.25, initial_value=args.speed
        )

    print("=" * 64)
    print(f"GPU fleet banner — {n} bimanual spacecraft on {wp.get_device()}")
    print(f"model: {nbody} bodies, {cpu_model.ngeom} geoms, nu={nu}, dt={dt:g} s")
    print(f"layout: {layout}, fleet rendered in {scene.num_draw_calls} instanced draw calls")
    print(f"browser: http://localhost:{args.port}  (Ctrl+C to stop)")
    print("=" * 64)

    # ----------------------------------------------------------------
    # Loop: budgeted GPU stepping, pull poses, update instanced meshes
    # ----------------------------------------------------------------
    sim_time = 0.0
    budget = 0.0
    last_wall = wall_time.perf_counter()
    last_report = last_wall
    steps_since_report = 0
    frames_since_report = 0
    omega_slew = 2.0 * np.pi / args.slew_period

    try:
        while True:
            now = wall_time.perf_counter()
            wall_dt, last_wall = now - last_wall, now
            if not pause_box.value:
                budget += wall_dt * float(speed_slider.value)
            budget = min(budget, 50.0 * dt)

            stepped = False
            if budget >= dt:
                amp = float(slew_slider.value)
                slew = amp * np.sin(omega_slew * sim_time + phases)
                ctrl = arm_base + slew[:, None] * BIMANUAL_SLEW[None, :]
                data.ctrl[:] = ctrl.astype(np.float32)
                mjo_warp.mjo_upload(model, data, fields=("ctrl",))
                while budget >= dt:
                    wp.capture_launch(graph)
                    budget -= dt
                    sim_time += dt
                    steps_since_report += n
                stepped = True

            if stepped or frames_since_report == 0:
                mjo_warp.mjo_pull(model, data, fields=("xpos", "xquat"))
                scene.update(
                    np.asarray(data.xpos), np.asarray(data.xquat), offsets=offsets_m
                )
                # Chief-centered frame: the Earth centre is -R(t), with the
                # circular chief orbit advanced analytically. This matches the
                # simulated chief because the XML disables J2/drag/SRP; the
                # residual is fp32 propagator rounding (~0.4 km per sim-hour,
                # invisible at this distance).
                ang = omega_orbit * sim_time
                c, s = np.cos(ang), np.sin(ang)
                R_t = np.array([R0[0] * c, R0[0] * s, 0.0])
                earth.update(position=-R_t * 1000.0, sim_time=sim_time)
            frames_since_report += 1

            if now - last_report >= 5.0:
                fps = frames_since_report / (now - last_report)
                sps = steps_since_report / (now - last_report)
                print(
                    f"  t={sim_time:7.1f} s  render {fps:5.1f} fps  "
                    f"sim {sps:.2e} world-steps/s  ({n} worlds)"
                )
                last_report = now
                steps_since_report = frames_since_report = 0

            wall_time.sleep(max(0.0, 1.0 / 30.0 - (wall_time.perf_counter() - now)))
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
