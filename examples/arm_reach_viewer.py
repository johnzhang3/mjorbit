"""Multi-spacecraft browser viewer for the dual-arm + solar-panels bus.

Spins up N independent simulations — each spacecraft has its own MjoModel
and MjoData (so its own orbit, qpos, qvel, controls, and integrator state) —
and renders them together in absolute ECI as a banner-style cloud: positions
sampled uniformly in a (wide-Y × short-Z × shallow-X) box around the chief
with a minimum-separation constraint, and a uniformly-random initial
attitude per spacecraft. The camera looks at the row from the +X (radial-
out) side so Earth sits behind the cloud.

The viewer runs indefinitely; stop with Ctrl+C.

Usage:
    pixi install
    pixi run python examples/arm_reach_viewer.py
    # tweak shape of the cloud:
    pixi run python examples/arm_reach_viewer.py \\
        --extent-y-km 60 --extent-z-km 10 --extent-x-km 6 --n 28
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from _orbit_reference import circular_orbit_eci

from mujoco_orbit import MjoData, MjoModel, OrbitInit, mjo_forward
from mujoco_orbit.constants import R_EARTH
from mujoco_orbit.testdata import SPACECRAFT_DUAL_ARM_PANELS_XML
from viewer import MjOrbitMultiViewer


def _compile_model(xml_path: str, *, mj_timestep: float) -> MjoModel:
    mjorbit = (
        '<mjorbit use_j2="false" use_drag="false" use_srp="false" '
        'use_magnetic="false">\n  </mjorbit>\n'
    )
    xml = Path(xml_path).read_text().replace("</mujoco>", f"  {mjorbit}</mujoco>")
    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as file:
        file.write(xml)
        configured_path = Path(file.name)
    try:
        return MjoModel.from_xml_path(str(configured_path), mj_timestep=mj_timestep)
    finally:
        configured_path.unlink(missing_ok=True)


def _sample_positions_km(
    n: int,
    extents_km: np.ndarray,
    min_sep_km: float,
    rng: np.random.Generator,
    max_attempts: int = 20000,
) -> np.ndarray:
    """Uniform sample of N points in ±extents_km with pairwise min_sep_km."""
    points: list[np.ndarray] = []
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-1.0, 1.0, size=3) * extents_km
        if min_sep_km <= 0.0 or all(
            np.linalg.norm(p - q) >= min_sep_km for q in points
        ):
            points.append(p)
    if len(points) < n:
        raise RuntimeError(
            f"Could not place {n} spacecraft with min-sep={min_sep_km} km in "
            f"a box ±{extents_km.tolist()} after {attempts} attempts. "
            "Reduce --min-sep-km or increase --extent-*-km."
        )
    return np.stack(points)


def _random_unit_quat(rng: np.random.Generator) -> np.ndarray:
    """Uniform random quaternion on S^3 (Marsaglia), MuJoCo (w,x,y,z) order."""
    u1, u2, u3 = rng.uniform(0.0, 1.0, size=3)
    r1, r2 = np.sqrt(1.0 - u1), np.sqrt(u1)
    return np.array(
        [
            r1 * np.sin(2.0 * np.pi * u2),  # w
            r1 * np.cos(2.0 * np.pi * u2),  # x
            r2 * np.sin(2.0 * np.pi * u3),  # y
            r2 * np.cos(2.0 * np.pi * u3),  # z
        ]
    )


def _build_entry(
    model: MjoModel,
    idx: int,
    n_total: int,
    chief_R_eci: np.ndarray,
    chief_V_eci: np.ndarray,
    offset_km: np.ndarray,
    rng: np.random.Generator,
    randomize_attitude: bool,
) -> tuple[MjoModel, MjoData]:
    """Construct an independent (model, data) entry for spacecraft *idx*.

    Position is the chief plus *offset_km*. Velocity matches the chief —
    long-term formation isn't physical, but this is for a screenshot.
    Attitude is uniformly random on SO(3) when *randomize_attitude* is True.
    """
    R_eci = chief_R_eci + offset_km
    V_eci = chief_V_eci.copy()
    data = model.make_data(orbit=OrbitInit(R_eci=R_eci, V_eci=V_eci))

    if randomize_attitude:
        data.qpos[3:7] = _random_unit_quat(rng)

    # Stagger initial arm targets across the cluster so each spacecraft
    # holds a slightly different pose. Position actuators converge in ~0.5 s.
    phase = 2.0 * np.pi * idx / n_total
    sa = 0.7 * np.sin(phase)
    ea = -0.9 * np.cos(phase)
    sb = 0.7 * np.sin(phase + np.pi / 3.0)
    eb = -0.9 * np.cos(phase + np.pi / 3.0)
    data.ctrl[:] = np.array([sa, ea, sb, eb])
    mjo_forward(model, data)
    return model, data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n", type=int, default=24, help="number of spacecraft to simulate"
    )
    parser.add_argument(
        "--alt-km", type=float, default=600.0,
        help="chief circular orbit altitude (km)",
    )
    parser.add_argument(
        "--inc-deg", type=float, default=51.6, help="chief orbit inclination (deg)"
    )
    parser.add_argument(
        "--extent-x-km", type=float, default=6.0,
        help="position spread ± along ECI X (depth, perpendicular to banner)",
    )
    parser.add_argument(
        "--extent-y-km", type=float, default=50.0,
        help="position spread ± along ECI Y (horizontal banner span)",
    )
    parser.add_argument(
        "--extent-z-km", type=float, default=10.0,
        help="position spread ± along ECI Z (vertical banner span)",
    )
    parser.add_argument(
        "--min-sep-km", type=float, default=4.0,
        help="minimum pairwise separation (km) — rejection sampling enforces this",
    )
    parser.add_argument(
        "--no-random-attitude", action="store_true",
        help="disable random initial attitudes (keep identity quaternion)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--port", type=int, default=8080, help="viser server port")
    parser.add_argument(
        "--spacecraft-scale", type=float, default=500.0,
        help=(
            "visual scale applied to each spacecraft. 1.0 = truthful (1 m bus); "
            "500x makes the bus read as ~500 m wide in the figure"
        ),
    )
    parser.add_argument(
        "--camera-distance-m", type=float, default=None,
        help=(
            "camera distance from cluster centre (m). Defaults to "
            "1.1 × the larger horizontal/vertical extent"
        ),
    )
    parser.add_argument(
        "--slew-amplitude", type=float, default=0.0,
        help="amplitude of the live arm slew (rad). 0 = static (screenshot mode)",
    )
    args = parser.parse_args()

    radius_km = R_EARTH + args.alt_km
    chief_R, chief_V = circular_orbit_eci(radius_km, np.deg2rad(args.inc_deg))
    rng = np.random.default_rng(args.seed)

    extents_km = np.array(
        [args.extent_x_km, args.extent_y_km, args.extent_z_km], dtype=float
    )
    offsets_km = _sample_positions_km(args.n, extents_km, args.min_sep_km, rng)

    # One compiled model — each entry gets its own MjoData with independent
    # orbit, qpos (incl. random attitude), qvel, ctrl, integrator state.
    model = _compile_model(SPACECRAFT_DUAL_ARM_PANELS_XML, mj_timestep=0.005)
    entries = [
        _build_entry(
            model, i, args.n, chief_R, chief_V,
            offsets_km[i], rng, not args.no_random_attitude,
        )
        for i in range(args.n)
    ]

    # Camera distance: fit the larger of the on-screen extents (Y horizontal,
    # Z vertical) with a little headroom.
    if args.camera_distance_m is not None:
        cam_dist_m = args.camera_distance_m
    else:
        on_screen_km = 2.0 * max(args.extent_y_km, args.extent_z_km)  # full diameter
        cam_dist_m = max(on_screen_km * 1.1 * 1000.0, 5000.0)

    viewer = MjOrbitMultiViewer(
        entries,
        port=args.port,
        show_earth=True,
        spacecraft_scale=args.spacecraft_scale,
        camera_distance=cam_dist_m,
    )

    # Frame the banner: camera at the +X (radial-out) side, slight Z
    # elevation, looking at cluster centre. Earth sits behind the cluster.
    cluster_center_m = 1000.0 * chief_R
    cam_pos = cluster_center_m + np.array(
        [cam_dist_m, 0.0, 0.15 * cam_dist_m]
    )
    viewer.server.initial_camera.position = tuple(cam_pos)
    viewer.server.initial_camera.look_at = tuple(cluster_center_m)

    # Action callback. Default amplitude=0 keeps the figure static for a
    # clean screenshot; pass --slew-amplitude > 0 for live motion.
    omega = 2.0 * np.pi / 30.0
    base_ctrls = np.stack([np.copy(d.ctrl) for _, d in entries])
    amplitude = args.slew_amplitude

    def action_fn(i, model, data, t):
        del model, data
        if amplitude == 0.0:
            return base_ctrls[i]
        phase = 2.0 * np.pi * i / args.n
        delta = amplitude * np.sin(omega * t + phase)
        ctrl = base_ctrls[i].copy()
        ctrl[0] += delta
        ctrl[1] -= 0.5 * delta
        ctrl[2] -= delta
        ctrl[3] += 0.5 * delta
        return ctrl

    print("=" * 60)
    print(f"Banner viewer  —  N = {args.n}, alt = {args.alt_km:g} km")
    print(
        f"Cloud ±(x,y,z) = ({args.extent_x_km:g}, {args.extent_y_km:g}, "
        f"{args.extent_z_km:g}) km, min-sep {args.min_sep_km:g} km, "
        f"scale {args.spacecraft_scale:g}x"
    )
    print(
        f"Random attitude = {not args.no_random_attitude}, "
        f"camera distance {cam_dist_m/1000:.1f} km"
    )
    print(f"Browser: http://localhost:{args.port}  (Ctrl+C to stop)")
    print("=" * 60)

    viewer.run(duration=None, action_fn=action_fn)


if __name__ == "__main__":
    main()
