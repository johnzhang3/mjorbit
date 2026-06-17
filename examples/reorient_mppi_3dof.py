"""MPPI free-floating 3-DOF reorientation: slew the bus with two multi-DOF arms.

No bus actuators. Two arms (each a 3-DOF shoulder about x/y/z plus an elbow) sit
on opposite (+x / -x) faces of the bus. Because the shoulder axes span all three
body axes, coordinated arm motion can exchange angular momentum about ANY axis,
so the bus can be driven to an arbitrary target attitude.

The default target (60 deg about (1,1,1)) is a large slew well outside the
small-angle regime, where a linearized attitude model would be invalid -- MPPI
plans through the full nonlinear coupled dynamics by sampling, so it handles it
directly. The rate penalty (w_arm_rate) is tuned heavy and the running-attitude
term dropped, so the slew is calm: low joint and bus rates, trading speed for
gentleness (and there is deliberately no cost on reaching the target fast).

MPPI plans the 8 arm setpoints to drive the bus to a target quaternion and arrive
slowly. Two hold strategies (``--hold-mode``):
  active (default): keep replanning so the arms do momentum management --
     continuous small reactive strokes that reject the disturbances and hold
     attitude. The arms load up toward a non-neutral pose; the oscillatory
     gravity-gradient torque is largely returned each orbit so they don't
     saturate quickly, while the small secular drag/magnetic part slowly walks
     them toward their limits. This matches the viewer.
  freeze: park the arms once on target. Open-loop hold -- fine briefly, but with
     the environmental torques on it drifts badly over an orbit.

Run headless; prints convergence and saves plots to examples/reorient_3dof_plots.png.
Pass ``--save-traj PATH`` to dump a (qpos, R_eci, V_eci) trajectory for the
deterministic video recorder (scripts/record/record_reorient.py).

Usage:
    pixi run example-reorient
    pixi run python examples/reorient_mppi_3dof.py --target-deg 35 --axis 1 0 1
    # long hold-dominated run, cheap planner (~70x faster, holds ~5 deg):
    pixi run python examples/reorient_mppi_3dof.py --duration 1000 --fast
    # dump a 60 s trajectory for the recorder:
    pixi run python examples/reorient_mppi_3dof.py --duration 60 --save-traj /tmp/reorient_traj.npz
    # plots need matplotlib (report env): pixi run -e report python examples/reorient_mppi_3dof.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from mjorbit import MjoModel, OrbitInit, mjo_forward, mjo_step
from mjorbit.constants import GM_EARTH, R_EARTH
from mjorbit.planning import MppiConfig, MppiPlanner

REORIENT_XML = Path(__file__).parent / "reorient_dualarm.xml"

# Packed state layout (nq=15, nv=14): [time, qpos(15), qvel(14)]
# qpos: bus pos 1:4, bus quat 4:8, arm joints 8:16
#       arm order = [A_sx, A_sy, A_sz, A_elbow, B_sx, B_sy, B_sz, B_elbow]
# qvel: bus linvel 16:19, bus angvel 19:22, arm joint rates 22:30
BUS_QUAT = slice(4, 8)
BUS_W = slice(19, 22)      # bus body-frame angular velocity (x,y,z)
ARM_Q = slice(8, 16)
ARM_QVEL = slice(22, 30)

# --- Geometry for self-collision avoidance (contact is off) -----------------
SHOULDER_X, L1, L2 = 0.4, 1.0, 1.0   # mount offset + link lengths (XML)
BUS_HALF = 0.4                        # bus box half-extent (symmetric cube)
KEEPOUT = 0.18                        # m clearance margin (capsule radius + buffer)
_TS = np.linspace(0.2, 1.0, 5)        # sample fractions along each link
_XHAT = np.array([1.0, 0.0, 0.0])


def _rot(axis: str, a: np.ndarray) -> np.ndarray:
    """Batched rotation matrix about a principal axis; a is (...,) -> (...,3,3)."""
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    if axis == "x":
        rows = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == "y":
        rows = [[c, z, s], [z, o, z], [-s, z, c]]
    else:  # z
        rows = [[c, -s, z], [s, c, z], [z, z, o]]
    return np.stack([np.stack(r, axis=-1) for r in rows], axis=-2)


# keep track of points on arm to collision avoidance with the spacecraft body
def _arm_points(sx, sy, sz, elbow, mount_sign: float) -> np.ndarray:
    """Sampled points along one arm, in the bus frame; shape (..., 2*len(_TS), 3).

    Validated against MuJoCo xpos: link1 dir = Rx(sx)Ry(sy)Rz(sz) @ (sign*xhat),
    link2 adds an elbow rotation about y. mount_sign = +1 for arm A, -1 for B.
    """
    r_sh = _rot("x", sx) @ _rot("y", sy) @ _rot("z", sz)
    base = mount_sign * _XHAT
    dir1 = (r_sh @ base[..., None])[..., 0]                       # (...,3)
    elb = (r_sh @ _rot("y", elbow) @ base[..., None])[..., 0]      # (...,3)
    mount = mount_sign * SHOULDER_X * _XHAT                        # (3,)
    elbow_pt = mount + L1 * dir1                                   # (...,3)
    pts = [mount + (t * L1) * dir1 for t in _TS]
    pts += [elbow_pt + (t * L2) * elb for t in _TS]
    return np.stack(pts, axis=-2)


def _clearance_penalty(states: np.ndarray) -> np.ndarray:
    """Barrier penalty for either arm penetrating the bus box. Reduces over time."""
    q = states[..., ARM_Q]
    pts = np.concatenate(
        [_arm_points(q[..., 0], q[..., 1], q[..., 2], q[..., 3], +1.0),
         _arm_points(q[..., 4], q[..., 5], q[..., 6], q[..., 7], -1.0)],
        axis=-2,
    )                                                              # (...,P,3)
    outside = np.maximum(np.abs(pts) - BUS_HALF, 0.0)
    dist = np.linalg.norm(outside, axis=-1)                        # (...,P)
    barrier = np.maximum(KEEPOUT - dist, 0.0) ** 2
    return np.mean(np.sum(barrier, axis=-1), axis=-1)              # sum P, mean T


def att_err_deg(q: np.ndarray, qt: np.ndarray) -> np.ndarray:
    """Geodesic attitude error (deg) between quaternions (w,x,y,z)."""
    dot = np.clip(np.abs(np.sum(q * qt, axis=-1)), 0.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dot))


def make_reorient_cost(num_timesteps: int, q_target: np.ndarray,
                       w_arm_rate: float = 8.0, w_clear: float = 25.0,
                       w_att_term: float = 40.0, w_att_run: float = 0.0,
                       w_stop: float = 8.0, w_bus_run: float = 0.0):
    """Slew-phase cost: reach the target with the gentlest possible arm motion.

    There is deliberately NO reward for reaching the target *fast*: ``w_att_run``
    (the running, "sooner is better" attitude term) defaults to 0. The only
    attitude reward is the TERMINAL one (``w_att_term``, the last 20%% of each
    rollout); with receding-horizon replanning that terminal carrot still pulls
    the bus to the target, just without any time pressure.

    ``w_arm_rate`` is the dominant term and the knob to make the motion gentle:
    it penalizes the 8 arm-joint rates over the whole rollout, so the planner
    uses slow strokes. Raising it makes the arms calmer but eventually parks the
    bus a few degrees short of the target (the closing strokes stop being worth
    their rate cost) -- that is the gentleness/accuracy trade-off.

    ``w_stop`` is a terminal bus-rate penalty for a slow arrival. ``w_bus_run``
    (off) penalizes bus rotation over the whole slew -- it is NOT a joint rate and
    it fights the reorientation itself (the bus must rotate to reslew), so prefer
    ``w_arm_rate`` for gentleness and leave this at 0.
    """
    term = slice(int(0.8 * num_timesteps), None)
    qt = np.asarray(q_target)

    def cost_fn(states, sensors, controls):
        del sensors, controls
        q = states[:, :, BUS_QUAT]
        att_err = 1.0 - np.sum(q * qt, axis=-1) ** 2          # 0 aligned, in [0,1]
        bus_rate_sq = np.sum(states[:, :, BUS_W] ** 2, axis=-1)
        arm_rate_sq = np.sum(states[:, :, ARM_QVEL] ** 2, axis=-1)
        return (
            w_att_term * np.mean(att_err[:, term], axis=1)        # reach target
            + w_att_run * np.mean(att_err, axis=1)                # sooner is better
            + w_stop * np.mean(bus_rate_sq[:, term], axis=1)      # arrive slowly
            + w_bus_run * np.mean(bus_rate_sq, axis=1)            # gentle bus slew
            + w_arm_rate * np.mean(arm_rate_sq, axis=1)           # keep motors slow
            + w_clear * _clearance_penalty(states)                # avoid hitting bus
        )

    return cost_fn


def run(target_deg: float, axis: np.ndarray, duration: float, horizon: float,
        rollouts: int, replan: float, seed: int, w_arm_rate: float, w_clear: float,
        w_bus_run: float = 0.0, hold_mode: str = "active"):
    model = MjoModel.from_xml_path(str(REORIENT_XML))
    dt = float(model.opt.timestep)

    r_orbit = R_EARTH + 400.0
    v = float(np.sqrt(GM_EARTH / r_orbit))
    data = model.make_data(orbit=OrbitInit(R_eci=[r_orbit, 0, 0], V_eci=[0, v, 0]))

    # Start at rest, arms slightly bent outward, bus at identity attitude.
    data.qpos[7:15] = [0.0, 0.3, 0.0, -0.6, 0.0, 0.3, 0.0, -0.6]
    np.copyto(data.ctrl, data.qpos[7:15])
    mjo_forward(model, data)

    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    psi = np.deg2rad(target_deg)
    q_target = np.array([np.cos(psi / 2), *(np.sin(psi / 2) * axis)])

    cfg = MppiConfig(
        horizon=horizon, num_rollouts=rollouts, num_nodes=5, spline_order="linear",
        sigma=0.18, temperature=0.05, use_noise_ramp=True, noise_ramp=1.5,
        nthread=max(1, os.cpu_count() or 1), seed=seed,
    )
    # Per-joint bounds: keep the shoulder-y hinges (indices 1, 5) clear of the
    # x/y/z gimbal-lock singularity at sy=+-90 deg; the other joints get the full
    # range. Order = [A_sx, A_sy, A_sz, A_elbow, B_sx, B_sy, B_sz, B_elbow].
    ctrl_hi = np.full(model.nu, 2.6)
    ctrl_hi[[1, 5]] = 1.2
    planner = MppiPlanner(
        model, cfg,
        make_reorient_cost(int(np.ceil(horizon / dt)), q_target, w_arm_rate, w_clear,
                            w_bus_run=w_bus_run),
        ctrl_low=-ctrl_hi, ctrl_high=ctrl_hi,
    )
    planner.reset(data, nominal_knots=np.tile(np.asarray(data.ctrl), (5, 1)))

    n = int(round(duration / dt))
    replan_every = max(1, int(round(replan / dt)))
    log = {k: np.empty(n) for k in ("t", "att", "rate", "armrate", "clear")}
    log["joints"] = np.empty((n, 8))
    # Full kinematic trajectory for the deterministic video recorder.
    traj_qpos = np.empty((n, model.nq))
    traj_R_eci = np.empty((n, 3))
    V_eci0 = np.asarray(data.orbit.V_eci, dtype=float).copy()

    print("=" * 60)
    print("MPPI Free-Floating 3-DOF Reorientation (dual arm, no bus actuator)")
    print("=" * 60)
    print(f"target {target_deg:g} deg about axis {np.round(axis, 3)} | "
          f"horizon {horizon:g}s | rollouts {rollouts} | duration {duration:g}s")
    hold_desc = ("freeze arms once on target" if hold_mode == "freeze"
                 else "arms keep working (active momentum management)")
    print(f"hold mode: {hold_mode} ({hold_desc})")
    log_every = int(round(2.0 / dt))
    freeze = hold_mode == "freeze"
    held = False
    hold_t = None
    sat_margin_deg = 180.0  # smallest joint clearance to a control bound seen during hold
    for step in range(n):
        # In freeze mode the arms park once on target; in active mode the planner
        # keeps running so the arms make reactive strokes to hold attitude.
        if not (freeze and held):
            if step % replan_every == 0:
                planner.update_action(data)
            np.copyto(data.ctrl, planner.action(float(data.time)))
        mjo_step(model, data)
        q = np.asarray(data.qpos[3:7])
        w = np.asarray(data.qvel[3:6])
        log["t"][step] = data.time
        log["att"][step] = att_err_deg(q, q_target)
        log["rate"][step] = np.rad2deg(np.linalg.norm(w))
        log["joints"][step] = np.rad2deg(np.asarray(data.qpos[7:15]))
        log["armrate"][step] = np.rad2deg(np.max(np.abs(np.asarray(data.qvel[6:14]))))
        traj_qpos[step] = np.asarray(data.qpos, dtype=float)
        traj_R_eci[step] = np.asarray(data.orbit.R_eci, dtype=float)
        qarm = np.asarray(data.qpos[7:15])
        pts = np.concatenate([_arm_points(*qarm[:4], +1.0),
                              _arm_points(*qarm[4:], -1.0)], axis=0)
        log["clear"][step] = float(
            np.linalg.norm(np.maximum(np.abs(pts) - BUS_HALF, 0.0), axis=-1).min()
        )
        if (not held and data.time > 4.0
                and log["att"][step] < 3.0 and log["rate"][step] < 2.5):
            held = True
            hold_t = float(data.time)
            print(f"  TARGET reached at t={hold_t:.1f}s (att err={log['att'][step]:.1f} deg, "
                  f"{'freezing arms' if freeze else 'holding actively'})")
        if held and not freeze:  # track how close the arms get to saturating
            margin = float(np.min(ctrl_hi - np.abs(np.asarray(data.qpos[7:15]))))
            sat_margin_deg = min(sat_margin_deg, np.rad2deg(margin))
        if step % log_every == 0:
            print(f"  t={data.time:5.1f}s  att err={log['att'][step]:6.1f} deg  "
                  f"|rate|={log['rate'][step]:6.2f} deg/s")

    settle = max(1, int(round(2.0 / dt)))
    final_att = float(np.mean(log["att"][-settle:]))
    final_rate = float(np.mean(log["rate"][-settle:]))
    print()
    print(f"final attitude error (last 2s mean): {final_att:.1f} deg")
    print(f"final |rate| (last 2s mean): {final_rate:.2f} deg/s")
    print(f"peak arm joint rate: {log['armrate'].max():.1f} deg/s")
    if hold_mode == "active" and hold_t is not None:
        print(f"active hold: smallest joint margin to a control bound: {sat_margin_deg:.0f} deg "
              f"({'SATURATED' if sat_margin_deg < 5.0 else 'no saturation'})")
    # Active hold keeps the arms working, so the bus jitters a bit (higher rate);
    # judge it on attitude error held, not on a near-zero rate.
    rate_ok = final_rate < (3.0 if hold_mode == "active" else 2.0)
    print("PASS" if final_att < 5.0 and rate_ok else "PARTIAL",
          f"— attitude error {final_att:.1f} deg")
    log["qpos"] = traj_qpos
    log["R_eci"] = traj_R_eci
    log["V_eci"] = V_eci0
    log["dt"] = dt
    return log, target_deg, axis


def save_trajectory(log, path: Path) -> None:
    """Dump (qpos, R_eci, V_eci, dt, xml_path) for scripts/record/record_reorient.py."""
    np.savez(
        path,
        qpos=log["qpos"], R_eci=log["R_eci"], V_eci=log["V_eci"],
        dt=log["dt"], xml_path=str(REORIENT_XML),
    )
    print(f"saved trajectory -> {path}  ({len(log['qpos'])} steps)")


def make_plots(log, target_deg: float, axis: np.ndarray, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(f"Dual-arm 3-DOF reorientation by reaction "
                 f"({target_deg:g} deg about {np.round(axis, 2)}, no bus actuator)")

    ax[0, 0].plot(log["t"], log["att"], color="tab:blue")
    ax[0, 0].axhline(0, ls="--", color="tab:red")
    ax[0, 0].set(title="Attitude error to target", xlabel="t [s]", ylabel="deg")
    ax[0, 0].grid(alpha=0.3)

    ax[0, 1].plot(log["t"], log["rate"], color="tab:green")
    ax[0, 1].set(title="Bus angular rate (→0 = stopped)", xlabel="t [s]", ylabel="deg/s")
    ax[0, 1].grid(alpha=0.3)

    labels = ["A_sx", "A_sy", "A_sz", "A_el", "B_sx", "B_sy", "B_sz", "B_el"]
    for j in range(8):
        ax[1, 0].plot(log["t"], log["joints"][:, j], label=labels[j], lw=1)
    ax[1, 0].set(title="Arm joint angles (the control)", xlabel="t [s]", ylabel="deg")
    ax[1, 0].legend(ncol=2, fontsize=7)
    ax[1, 0].grid(alpha=0.3)

    ax[1, 1].plot(log["t"], log["armrate"], color="tab:orange")
    ax[1, 1].set(title="Peak arm joint rate", xlabel="t [s]", ylabel="deg/s")
    ax[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"saved plots -> {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-deg", type=float, default=60.0,
                   help="default 60 deg: outside the small-angle regime but "
                        "reachable with slow arms")
    p.add_argument("--axis", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                   help="rotation axis (need not be normalized)")
    p.add_argument("--duration", type=float, default=100.0)
    p.add_argument("--horizon", type=float, default=14.0,
                   help="long enough to see a multi-stroke pumping maneuver")
    p.add_argument("--rollouts", type=int, default=128)
    p.add_argument("--replan", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--w-arm-rate", type=float, default=8.0,
                   help="arm joint-rate penalty -- the gentleness knob; raise for calmer arms")
    p.add_argument("--w-bus-run", type=float, default=0.0,
                   help="bus angular-rate penalty (fights the slew; leave 0)")
    p.add_argument("--w-clear", type=float, default=25.0, help="bus-collision penalty")
    p.add_argument("--hold-mode", choices=["freeze", "active"], default="active",
                   help="active (default): keep replanning so the arms do momentum management "
                        "and reject disturbances (matches the viewer). freeze: park the arms "
                        "once on target (open-loop hold, drifts under disturbances over long "
                        "horizons).")
    p.add_argument("--fast", action="store_true",
                   help="cheap planner preset for long, hold-dominated runs (e.g. --duration "
                        "1000): rollouts=24, horizon=6, replan=2.0 -- ~70x faster than the "
                        "default planner, slews a bit slower and holds ~5 deg instead of ~2 deg. "
                        "Overrides --rollouts/--horizon/--replan.")
    p.add_argument("--save-traj", type=Path, default=None,
                   help="dump a (qpos, R_eci, V_eci) trajectory .npz for the video recorder "
                        "(scripts/record/record_reorient.py)")
    p.add_argument("--save-plots", type=Path,
                   default=REORIENT_XML.parent / "reorient_3dof_plots.png",
                   help="path for the diagnostic plot PNG (needs matplotlib / the report env)")
    args = p.parse_args()
    if args.fast:
        args.rollouts, args.horizon, args.replan = 24, 6.0, 2.0

    log, target, axis = run(args.target_deg, args.axis, args.duration, args.horizon,
                            args.rollouts, args.replan, args.seed,
                            args.w_arm_rate, args.w_clear, args.w_bus_run, args.hold_mode)
    if args.save_traj is not None:
        save_trajectory(log, args.save_traj)
    try:
        make_plots(log, target, axis, args.save_plots)
    except ModuleNotFoundError:
        print("(matplotlib not available; skipping plots — run with `pixi run -e report`)")


if __name__ == "__main__":
    main()
