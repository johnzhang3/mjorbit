"""MPPI docking demo: the Soyuz flies its port onto the ISS docking port.

The fully-actuated Soyuz from ``examples/docking/iss.xml`` (3 body-frame
thrusters + 3 body-frame "reaction wheels") is steered from its initial standoff
pose so that its docking port converges with the ISS docking port — both in
position and in orientation — using the spline-knot MPPI planner in
``mjorbit.planning``. Contact is disabled in the model, so this is a pure
guidance problem: bring the two port frames together, do not worry about the
collision/latch.

The Soyuz docking port is offset from the body CoM (the ``soyuz_dock_site`` sits
at ``(-1.7743, 0.033833, 0.23575)`` m in the body frame), so a body rotation
sweeps the port through an arc. The cost never works in body coordinates: it
reads the port frames straight from sensors, which already fold in that offset:
  * ``soyuz_port_rel`` — Soyuz port position **relative to the ISS port**,
    expressed in the ISS-port frame. Driving this to zero seats the ports.
  * ``soyuz_port_quat`` — Soyuz port orientation (world frame), driven to the
    target attitude.
Both thrusters (translation) and reaction wheels (rotation) are now sampled, so
MPPI commands a coupled 6-DOF approach. Rollouts are evaluated by
``mjorbit.rollout``.

Target orientation: by default the Soyuz port is aligned with the ISS port
attitude (read once at startup from ``iss_port_quat``). The Soyuz already starts
at that attitude, so the default run is a pure-translation dock: the planner
holds attitude with the reaction wheels while the thrusters fly the port in.
Pass ``--target-angle``/``--target-axis`` to instead command an explicit
world-frame target attitude; the planner then has to slew there as it closes
(see the authority note below).

Two modes, selected by ``--viewer``:
  * headless (default) — fast closed-loop run that logs port separation and
    attitude error and prints a PASS/WARN; use this to gather data and tune.
  * ``--viewer`` — drives the same controller while streaming to the browser
    viewer so you can watch (and capture) the approach for figures.

Every cost term is normalised by a tolerance (``--pos-tol``/``--att-tol``/
``--vel-tol``/``--rate-tol``), so the position, attitude, velocity, and
angular-rate penalties share one dimensionless scale and the ``--w-*`` weights
are plain priorities. This is what keeps the attitude locked through the
position-dominated approach: on the un-normalised cost the metre-scale port
separation dwarfs the radian-scale attitude error by ~1e3, the reaction-wheel
channel is left to be driven by sampling noise, and the injected angular
momentum runs the attitude away because the short horizon cannot brake it.

The translation authority is ~0.057 m/s^2 (400 N / 7000 kg) and the slew is
reaction-wheel limited (~0.005 rad/s^2 about the weak axes), so the default
aligned dock seats in ~10 s but a commanded slew takes far longer. A large
``--target-angle`` needs a longer ``--horizon`` (and more ``--num-nodes``) so
the planner can foresee the braking burn; otherwise the attitude overshoots and
limit-cycles short of the target. Raise ``--duration`` if the ports have not
seated. All weights, tolerances, and the target are CLI args.

models: https://science.nasa.gov/3d-resources/international-space-station-iss-a/

Usage:
    pixi run python examples/docking/main_mppi.py
    pixi run python examples/docking/main_mppi.py --duration 150
    # commanded slew (needs a longer horizon so the low-authority slew can brake):
    pixi run python examples/docking/main_mppi.py \\
        --target-angle -60 --target-axis "0 0 1" --horizon 8 --num-nodes 6
    pixi run python examples/docking/main_mppi.py --viewer
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
import time as wall_time
from pathlib import Path

import numpy as np
from util import DockingCollision, quat_mult

from mjorbit import MjoModel, OrbitInit, mjo_forward, mjo_get_state, mjo_set_state, mjo_step
from mjorbit.constants import GM_EARTH, R_EARTH
from mjorbit.planning import MppiConfig, MppiPlanner
from mjorbit.rollout import mjo_control_size

# Actuator saturations, matching the ctrlrange in examples/docking/iss.xml.
THRUST_FORCE_MAX = 400.0  # body-frame translation thrusters, N
WHEEL_TORQUE_MAX = 150.0  # body-frame reaction wheels, N*m


def circular_orbit_eci(radius_km: float, inclination_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Circular orbit at ascending node: R along +X, V in the Y-Z plane."""
    speed = np.sqrt(GM_EARTH / radius_km)
    R_eci = np.array([radius_km, 0.0, 0.0])
    V_eci = speed * np.array([0.0, np.cos(inclination_rad), np.sin(inclination_rad)])
    return R_eci, V_eci


def sensor_slice(model: MjoModel, name: str) -> slice:
    descriptor = model.sensor(name)
    return slice(int(descriptor.adr), int(descriptor.adr) + int(descriptor.dim))


def _compile_xml_text(xml: str, *, near_dir: Path, mj_timestep: float) -> MjoModel:
    """Compile an XML string from a temp file next to ``near_dir`` (asset resolution)."""
    with tempfile.NamedTemporaryFile(
        suffix=".xml", mode="w", delete=False, dir=near_dir
    ) as file:
        file.write(xml)
        configured_path = Path(file.name)
    try:
        return MjoModel.from_xml_path(str(configured_path), mj_timestep=mj_timestep)
    finally:
        configured_path.unlink(missing_ok=True)


def _with_mjorbit(xml: str) -> str:
    """Inject a perturbation-free <mjorbit> overlay (iss.xml has no <mjorbit> block)."""
    mjorbit = (
        '<mjorbit use_j2="false" use_drag="false" use_srp="false" '
        'use_magnetic="false">\n  </mjorbit>\n'
    )
    return xml.replace("</mujoco>", f"  {mjorbit}</mujoco>")


def _compile_model(xml_path: str, *, mj_timestep: float) -> MjoModel:
    """Compile iss.xml with the <mjorbit> overlay (environment perturbations off)."""
    xml = _with_mjorbit(Path(xml_path).read_text())
    return _compile_xml_text(xml, near_dir=Path(xml_path).parent, mj_timestep=mj_timestep)


def _set_body_pose(xml: str, body: str, pos: np.ndarray, quat: np.ndarray) -> str:
    """Rewrite the pos/quat of a named body's opening tag (sets the free-joint qpos0)."""
    def repl(match: "re.Match[str]") -> str:
        tag = re.sub(r'pos="[^"]*"', f'pos="{pos[0]} {pos[1]} {pos[2]}"', match.group(0))
        return re.sub(r'quat="[^"]*"', f'quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}"', tag)

    return re.sub(rf'<body name="{body}"[^>]*?>', repl, xml, count=1)


def _compile_latched(xml_path: str, *, qpos: np.ndarray, mj_timestep: float) -> MjoModel:
    """Compile the latched twin: dock_latch weld on, frozen at the current pose.

    The CPU backend cannot toggle an equality at runtime, so capture swaps to
    this model and transfers the packed state (as in the capture demo). MuJoCo
    derives a weld's relative pose from the reference configuration (it ignores
    the XML ``relpose``), so the current ISS/Soyuz poses are baked into the body
    qpos0 — this makes the weld hold the ports exactly where they are at latch,
    instead of snapping back to the standoff.
    """
    xml = Path(xml_path).read_text().replace('active="false"', 'active="true"')
    xml = _set_body_pose(xml, "iss_main", qpos[0:3], qpos[3:7])
    xml = _set_body_pose(xml, "soyuz", qpos[7:10], qpos[10:14])
    return _compile_xml_text(
        _with_mjorbit(xml), near_dir=Path(xml_path).parent, mj_timestep=mj_timestep
    )


def quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Unit quaternion [w, x, y, z] (MuJoCo order) for a rotation about ``axis``."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        raise ValueError("target axis must be non-zero")
    axis = axis / norm
    half = 0.5 * angle_rad
    return np.concatenate([[np.cos(half)], np.sin(half) * axis])


def attitude_error_deg(quat: np.ndarray, target_quat: np.ndarray) -> float:
    """Geodesic angle between two unit quaternions, in degrees (double-cover safe)."""
    dot = float(np.clip(abs(np.dot(quat, target_quat)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def main() -> None:
    ps = argparse.ArgumentParser(description=__doc__)
    ps.add_argument("--duration", type=float, default=120.0, help="closed-loop sim time (s)")
    ps.add_argument("--horizon", type=float, default=3.5, help="MPPI planning horizon (s)")
    ps.add_argument("--num-rollouts", type=int, default=128, help="MPPI samples per replan")
    ps.add_argument("--num-nodes", type=int, default=4, help="spline knots over the horizon")
    ps.add_argument("--replan-period", type=float, default=0.1, help="time between replans (s)")

    ps.add_argument("--sigma-thrust", type=float, default=150.0, help="thruster-force noise (N)")
    ps.add_argument("--sigma-wheel", type=float, default=20.0, help="wheel-torque noise std (N*m)")
    ps.add_argument("--temperature", type=float, default=0.03, help="MPPI temperature")
    ps.add_argument("--seed", type=int, default=0, help="RNG seed")
    ps.add_argument(
        "--nthread", type=int, default=max(1, os.cpu_count() or 1),
        help="rollout worker threads",
    )
    # Cost weights. Every term is normalised by a tolerance (see the --*-tol
    # args below), so these are dimensionless priorities rather than
    # unit-dependent gains, and position/attitude/velocity/rate are directly
    # comparable.
    # Position (docking-port separation).
    ps.add_argument("--w-pos", type=float, default=1.0, help="running port-separation weight")
    ps.add_argument("--w-term-pos", type=float, default=60.0, help="terminal port separation")
    ps.add_argument("--w-vel", type=float, default=1.0, help="running linear-rate (damping)")
    ps.add_argument("--w-term-vel", type=float, default=40.0, help="terminal linear-rate (braking)")
    # Orientation. The attitude weights are large because in the default aligned
    # dock the attitude error is ~0, so they cost nothing there; they earn their
    # keep on a commanded slew, pulling the last few degrees in against the
    # rate-damping term (which otherwise leaves a steady-state attitude error).
    ps.add_argument("--w-run", type=float, default=8.0, help="running orientation weight")
    ps.add_argument("--w-term", type=float, default=250.0, help="terminal orientation weight")
    ps.add_argument("--w-rate", type=float, default=6.0, help="running angular-rate (damping)")
    ps.add_argument("--w-term-rate", type=float, default=40.0, help="terminal angular-rate brake")
    # Control effort (normalised by actuator saturation).
    ps.add_argument("--w-ctrl", type=float, default=0.05, help="control-effort weight")
    # Error tolerances that set the normalisation scale of each cost term.
    ps.add_argument("--pos-tol", type=float, default=0.1, help="port-separation scale (m)")
    ps.add_argument("--vel-tol", type=float, default=0.05, help="linear-rate scale (m/s)")
    ps.add_argument("--att-tol", type=float, default=2.0, help="attitude-error scale (deg)")
    ps.add_argument("--rate-tol", type=float, default=0.005, help="angular-rate scale (rad/s)")
    # Collision avoidance: keep the Soyuz structure from plowing through the ISS.
    # Checked at the terminal rollout state only (the docking port sits 1 m off the
    # ISS hull, so a seated dock has clearance and never reads as a collision).
    ps.add_argument("--w-collision", type=float, default=1e5, help="structure-collision weight")
    ps.add_argument(
        "--collision-margin", type=float, default=+0.05,
        help="allowed structure overlap (m); interpenetration deeper than this is "
        "penalised. A small negative tolerance for grazing convex-hull contact.",
    )

    # Capture/latch: once the ports are close, aligned, and slow, activate the
    # dock_latch weld so the two craft are rigidly mated (data.eq_active[0] = 1).
    ps.add_argument("--latch", action=argparse.BooleanOptionalAction, default=True,
                    help="auto-activate the docking weld once the seat criteria are met")
    ps.add_argument("--latch-sep", type=float, default=0.15, help="latch: max port separation (m)")
    ps.add_argument("--latch-att", type=float, default=10.0, help="latch: max attitude error (deg)")
    ps.add_argument("--latch-vel", type=float, default=0.2, help="latch: max linear rate (m/s)")
    ps.add_argument("--latch-rate", type=float, default=0.1, help="latch: max angular rate (rad/s)")

    ps.add_argument(
        "--target-angle", type=float, default=None,
        help="explicit world-frame target attitude (deg about --target-axis); "
        "omit to align the Soyuz port with the ISS port",
    )
    ps.add_argument(
        "--target-axis", type=str, default="0 0 1",
        help="slew axis in the world frame for an explicit --target-angle, e.g. '0 0 1'",
    )
    ps.add_argument("--viewer", action="store_true", help="stream to the browser viewer")
    ps.add_argument("--port", type=int, default=8080, help="viewer port")
    ps.add_argument(
        "--camera-distance", type=float, default=50000.0, help="viewer camera distance (m)"
    )
    args = ps.parse_args()

    alt_km = 400.0
    R_eci, V_eci = circular_orbit_eci(R_EARTH + alt_km, np.deg2rad(51.6))

    xml_path = Path(__file__).with_name("iss_explicit.xml")
    model = _compile_model(str(xml_path), mj_timestep=0.01)
    data = model.make_data(orbit=OrbitInit(R_eci=R_eci, V_eci=V_eci))
    mjo_forward(model, data)

    # Real-geometry collision proxy: a contact-enabled twin of the model whose
    # convex mesh hulls report when the Soyuz structure overlaps the ISS. The ISS
    # is unactuated, so its pose is frozen once from the initial state.
    collision = DockingCollision.from_xml(str(xml_path), margin=args.collision_margin)
    collision.set_reference(np.asarray(data.qpos))

    nq = int(model.nq)
    quat_sl = sensor_slice(model, "soyuz_port_quat")
    rel_sl = sensor_slice(model, "soyuz_port_rel")
    iss_quat_sl = sensor_slice(model, "iss_port_quat")
    sensordata = np.asarray(data.sensordata)

    # Target port orientation: by default align with the ISS port (face-to-face
    # dock), which is read once here and held fixed (the ISS is unactuated and at
    # rest). With --target-angle, command an explicit world-frame slew instead.
    if args.target_angle is None:
        target_quat = sensordata[iss_quat_sl].copy()
        target_desc = "ISS port attitude (face-to-face dock)"
    else:
        axis = np.array([float(v) for v in args.target_axis.split()], dtype=np.float64)
        if axis.shape != (3,):
            ps.error("--target-axis must be three numbers, e.g. '0 0 1'")
        target_quat = quat_from_axis_angle(axis, np.deg2rad(args.target_angle))
        target_desc = f"{args.target_angle:g} deg about [{args.target_axis}] (world frame)"

    # Packed rollout state layout: [time, qpos(nq), qvel(nv), ...]. The Soyuz free
    # joint is the second free joint, owning qvel[6:12]: linear velocity is
    # qvel[6:9], body-frame angular rate is qvel[9:12].
    soyuz_vlin_sl = slice(1 + nq + 6, 1 + nq + 9)
    soyuz_rate_sl = slice(1 + nq + 9, 1 + nq + 12)

    ncontrol = mjo_control_size(model)
    if ncontrol != int(model.nu):
        raise RuntimeError("this demo expects a model without orbital actuators")

    # Sample all six actuators: thrusters (indices 0:3) for translation and
    # reaction wheels (indices 3:6) for rotation, so the approach is full 6-DOF.
    sigma = np.empty(ncontrol)
    sigma[0:3] = args.sigma_thrust
    sigma[3:6] = args.sigma_wheel
    ctrl_low = np.empty(ncontrol)
    ctrl_high = np.empty(ncontrol)
    ctrl_low[0:3], ctrl_high[0:3] = -THRUST_FORCE_MAX, THRUST_FORCE_MAX
    ctrl_low[3:6], ctrl_high[3:6] = -WHEEL_TORQUE_MAX, WHEEL_TORQUE_MAX

    # Cost-term tolerances. Each error is normalised by the precision we care
    # about, so the position, attitude, velocity, and angular-rate terms land on
    # one common (dimensionless) scale and the CLI weights act as plain
    # priorities. Without this the metre-scale port separation dwarfs the
    # radian-scale attitude error by ~1e3: the planner spends essentially all of
    # its selection pressure on translation, and the under-weighted
    # reaction-wheel channel is left to be optimised by sampling noise. That
    # injects angular momentum the short horizon and low (~0.005 rad/s^2) slew
    # authority cannot brake out, so the attitude drifts away from a lock it
    # started at (the "oscillation"/run-away on the un-normalised cost).
    # Normalising lets the rate-damping term bite as the ports close (sep -> 0),
    # holding attitude instead of letting the wheels spin the craft up.
    w_pos, w_term_pos, w_vel, w_term_vel = args.w_pos, args.w_term_pos, args.w_vel, args.w_term_vel
    w_run, w_term = args.w_run, args.w_term
    w_rate, w_term_rate, w_ctrl = args.w_rate, args.w_term_rate, args.w_ctrl
    w_collision = args.w_collision
    inv_pos2 = 1.0 / args.pos_tol**2
    inv_vel2 = 1.0 / args.vel_tol**2
    inv_att2 = 1.0 / np.deg2rad(args.att_tol) ** 2
    inv_rate2 = 1.0 / args.rate_tol**2
    target_conj = np.array(
        [target_quat[0], -target_quat[1], -target_quat[2], -target_quat[3]]
    )

    def cost_fn(states: np.ndarray, sensors: np.ndarray, controls: np.ndarray) -> np.ndarray:
        # Port separation, normalised by the seat tolerance.
        n_pos = np.sum(sensors[:, :, rel_sl] ** 2, axis=-1) * inv_pos2
        # Port orientation error: the geodesic angle between the Soyuz port and
        # the target attitude. q_error = q_port (x) conj(q_target); its scalar
        # part is cos(theta/2) (made double-cover safe with abs), so theta is the
        # true angle between the two attitudes. Using the angle (not 1 - w**2)
        # keeps the term on the same quadratic footing as the others.
        q_error = quat_mult(sensors[:, :, quat_sl], target_conj)
        cos_half = np.clip(np.abs(q_error[:, :, 0]), 0.0, 1.0)
        ang = 2.0 * np.arccos(cos_half)
        n_att = ang**2 * inv_att2
        # Linear/angular rates, normalised — the damping that brings the ports in
        # seated rather than passing through, and that keeps the reaction wheels
        # from spinning the craft up during the position-dominated approach.
        n_vel = np.sum(states[:, :, soyuz_vlin_sl] ** 2, axis=-1) * inv_vel2
        n_rate = np.sum(states[:, :, soyuz_rate_sl] ** 2, axis=-1) * inv_rate2
        # Control effort, normalised by each actuator's saturation.
        n_ctrl = np.sum((controls / ctrl_high) ** 2, axis=-1)

        # Structure-collision keep-out: flag rollouts whose Soyuz hull touches the
        # ISS (1.0 in collision, 0.0 clear), read from real convex-mesh contacts.
        # Evaluated at the terminal rollout state only (one query per rollout).
        collision_pen = collision.collision_from_states(states[:, -1, :])
        return (
            w_pos * np.mean(n_pos, axis=1)
            + w_term_pos * n_pos[:, -1]
            + w_vel * np.mean(n_vel, axis=1)
            + w_term_vel * n_vel[:, -1]
            + w_run * np.mean(n_att, axis=1)
            + w_term * n_att[:, -1]
            + w_rate * np.mean(n_rate, axis=1)
            + w_term_rate * n_rate[:, -1]
            + w_ctrl * np.mean(n_ctrl, axis=1)
            + w_collision * collision_pen
        )

    config = MppiConfig(
        horizon=args.horizon,
        num_rollouts=args.num_rollouts,
        num_nodes=args.num_nodes,
        spline_order="linear",
        sigma=sigma,
        temperature=args.temperature,
        use_noise_ramp=True,
        noise_ramp=2.5,
        nthread=args.nthread,
        seed=args.seed,
    )
    planner = MppiPlanner(model, config, cost_fn, ctrl_low=ctrl_low, ctrl_high=ctrl_high)
    planner.reset(data)

    def current_separation_m() -> float:
        return float(np.linalg.norm(np.asarray(data.sensordata)[rel_sl]))

    def current_error_deg() -> float:
        return attitude_error_deg(np.asarray(data.sensordata)[quat_sl], target_quat)

    def current_collision() -> float:
        return collision.in_collision(np.asarray(data.qpos))

    def seat_ready(data_arg) -> bool:
        """True when the ports are close, aligned, and slow enough to latch."""
        sd = np.asarray(data_arg.sensordata)
        sep = float(np.linalg.norm(sd[rel_sl]))
        att = attitude_error_deg(sd[quat_sl], target_quat)
        qvel = np.asarray(data_arg.qvel)
        vlin = float(np.linalg.norm(qvel[6:9]))
        rate = float(np.linalg.norm(qvel[9:12]))
        return (
            sep <= args.latch_sep and att <= args.latch_att
            and vlin <= args.latch_vel and rate <= args.latch_rate
        )

    # Shared controller: replan on cadence, otherwise hold the current plan. Used
    # both by the headless loop and by the viewer's per-step action callback.
    last_replan = [-np.inf]

    def controller(data_arg, _sim_t: float) -> np.ndarray:
        if data_arg.time - last_replan[0] >= args.replan_period - 1e-9:
            planner.update_action(data_arg)
            last_replan[0] = data_arg.time
        return planner.action(float(data_arg.time))

    print("=" * 60)
    print("MPPI Docking — Soyuz thrusters + reaction wheels")
    print("=" * 60)
    print(
        f"rollouts={config.num_rollouts}  nodes={config.num_nodes}  "
        f"horizon={config.horizon:g}s  replan={args.replan_period:g}s  "
        f"sigma=({args.sigma_thrust:g}N, {args.sigma_wheel:g}N*m)  threads={config.nthread}"
    )
    print(f"target orientation: {target_desc}")
    print(f"initial port separation: {current_separation_m():.3f} m")
    print(f"initial attitude error:  {current_error_deg():.2f} deg")
    print()

    if args.viewer:
        from viewer import MjOrbitViewer

        viewer = MjOrbitViewer(
            model,
            data,
            port=args.port,
            show_earth=True,
            show_axes=True,
            track_body="iss_main",
            camera_distance=args.camera_distance,
            render_frame="eci",
        )
        # Scene magnification
        viewer.set_local_scene_scale(5000.0)
        print(f"Browser: http://localhost:{args.port}  — watch the Soyuz dock (Ctrl+C to stop)")
        print("=" * 60)
        viewer.run(duration=args.duration, action_fn=controller)
        print(f"\nfinal port separation: {current_separation_m():.3f} m")
        print(f"final attitude error:  {current_error_deg():.2f} deg")
        return

    dt = float(model.opt.timestep)
    n_steps = int(round(args.duration / dt))
    log_every = int(round(1.0 / dt))

    separations = np.empty(n_steps)
    errors = np.empty(n_steps)
    collisions = np.empty(n_steps)
    latched = False
    t_start = wall_time.perf_counter()
    for step in range(n_steps):
        if latched:
            # Docked: thrusters off, the weld holds the mate.
            np.copyto(data.ctrl, np.zeros(int(model.nu)))
        else:
            np.copyto(data.ctrl, controller(data, float(data.time)))
        mjo_step(model, data)

        # Capture: once the ports are seated, compile the welded twin (frozen at
        # the current pose), carry the packed state across, and swap to it so the
        # dock_latch weld rigidly mates the craft. Rebinding model/data here
        # propagates to the metric closures above.
        if not latched and args.latch and seat_ready(data):
            latch_model = _compile_latched(
                str(xml_path), qpos=np.asarray(data.qpos), mj_timestep=0.01
            )
            state = mjo_get_state(model, data)
            new_data = latch_model.make_data(orbit=OrbitInit(R_eci=R_eci, V_eci=V_eci))
            mjo_set_state(latch_model, new_data, state)
            mjo_forward(latch_model, new_data)
            model, data = latch_model, new_data
            latched = True
            rel_vel = float(np.linalg.norm(np.asarray(data.qvel)[6:9]))
            print(
                f"\n*** DOCK LATCHED at t={data.time:.1f} s  "
                f"(sep={current_separation_m():.3f} m, att={current_error_deg():.2f} deg, "
                f"rel vel={rel_vel:.4f} m/s) ***\n"
            )

        separations[step] = current_separation_m()
        errors[step] = current_error_deg()
        collisions[step] = current_collision()
        if (step + 1) % log_every == 0:
            assert planner.last_costs is not None
            vlin = float(np.linalg.norm(np.asarray(data.qvel)[6:9]))
            rate = float(np.linalg.norm(np.asarray(data.qvel)[9:12]))
            col_str = "COLLIDING" if collisions[step] else "clear"
            print(
                f"  t={data.time:6.1f} s   sep={separations[step]:6.3f} m   "
                f"att={errors[step]:6.2f} deg   v={vlin:6.4f} m/s   "
                f"w={rate:7.4f} rad/s   {col_str:>9}   latched: {latched}   "
                f"best cost={planner.last_costs.min():9.4f}"
            )
    elapsed = wall_time.perf_counter() - t_start

    settle_window = max(1, int(round(2.0 / dt)))
    settle_sep = float(np.mean(separations[-settle_window:]))
    settle_error = float(np.mean(errors[-settle_window:]))
    final_vlin = float(np.linalg.norm(np.asarray(data.qvel)[6:9]))
    final_rate = float(np.linalg.norm(np.asarray(data.qvel)[9:12]))
    print()
    print(f"final port separation: {separations[-1]:.3f} m  (mean last 2 s: {settle_sep:.3f} m)")
    print(f"final attitude error:  {errors[-1]:.2f} deg  (mean last 2 s: {settle_error:.2f} deg)")
    print(f"residual linear rate:  {final_vlin:.5f} m/s")
    print(f"residual angular rate: {final_rate:.5f} rad/s")
    # Structure collisions over the run (1 whenever the Soyuz hull came within the
    # keep-out margin of the ISS).
    n_collisions = int(np.count_nonzero(collisions))
    if n_collisions == 0:
        print("structure collisions:  none")
    else:
        print(
            f"structure collisions:  {n_collisions} of {n_steps} steps "
            f"(margin {args.collision_margin:+.2f} m)"
        )
    print(f"wall time: {elapsed:.1f} s ({n_steps * dt / elapsed:.1f}x realtime)")

    all_finite = bool(
        np.all(np.isfinite(np.asarray(data.qpos))) and np.all(np.isfinite(np.asarray(data.qvel)))
    )
    seated = settle_sep < 0.2 and settle_error < 5.0
    slow = final_vlin < 0.02 and final_rate < 5.0e-3
    if all_finite and seated and slow:
        print("PASS: Soyuz port seated on the ISS port at the target attitude.")
    else:
        print("WARN: ports did not seat within tolerance — see log above (try raising --duration).")


if __name__ == "__main__":
    main()
