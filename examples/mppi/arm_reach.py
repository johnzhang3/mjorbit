"""MPPI arm-reach demo: a free-floating spacecraft arm reaches for a target.

A 2-link arm on a free-floating bus (no attitude control) must bring its end
effector to a target object co-moving on the chief orbit. Arm motion torques
the unactuated base, so the planner has to account for the coupled
base-arm dynamics — a fixed-base IK pose would miss.

The planner is the spline-knot MPPI in mjorbit.planning (judo-style
sampling, variance ramp across the horizon), with rollouts evaluated by
mjorbit.rollout. The cost reads end-effector and target world positions
from framepos sensors, so no Python-side kinematics is needed.

Usage:
    pixi run python examples/mppi/arm_reach.py
    pixi run python examples/mppi/arm_reach.py --duration 8 --num-rollouts 128
"""

from __future__ import annotations

import argparse
import os
import time as wall_time

import numpy as np

from mjorbit import MjoModel, OrbitInit, mjo_forward, mjo_step
from mjorbit.constants import GM_EARTH, R_EARTH
from mjorbit.planning import MppiConfig, MppiPlanner
from mjorbit.rollout import mjo_control_size
from mjorbit.testdata import SPACECRAFT_ARM_REACH_XML as ARM_REACH_XML


def circular_orbit_eci(radius_km: float, inclination_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Circular orbit at ascending node: R along +X, V in the Y-Z plane."""
    speed = np.sqrt(GM_EARTH / radius_km)
    R_eci = np.array([radius_km, 0.0, 0.0])
    V_eci = speed * np.array([0.0, np.cos(inclination_rad), np.sin(inclination_rad)])
    return R_eci, V_eci


def sensor_slice(model: MjoModel, name: str) -> slice:
    descriptor = model.sensor(name)
    return slice(int(descriptor.adr), int(descriptor.adr) + int(descriptor.dim))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0, help="closed-loop sim time (s)")
    parser.add_argument("--horizon", type=float, default=1.5, help="MPPI planning horizon (s)")
    parser.add_argument("--num-rollouts", type=int, default=64, help="MPPI samples per replan")
    parser.add_argument("--num-nodes", type=int, default=4, help="spline knots over the horizon")
    parser.add_argument("--replan-period", type=float, default=0.1, help="time between replans (s)")
    parser.add_argument("--sigma", type=float, default=0.2, help="knot noise std (rad)")
    parser.add_argument("--temperature", type=float, default=0.05, help="MPPI temperature")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--nthread", type=int, default=max(1, os.cpu_count() or 1),
        help="rollout worker threads",
    )
    args = parser.parse_args()

    alt_km = 400.0
    R_eci, V_eci = circular_orbit_eci(R_EARTH + alt_km, np.deg2rad(51.6))

    model = MjoModel.from_xml_path(str(ARM_REACH_XML))
    data = model.make_data(orbit=OrbitInit(R_eci=R_eci, V_eci=V_eci))
    mjo_forward(model, data)

    nq, nv = int(model.nq), int(model.nv)
    ee_sl = sensor_slice(model, "ee_pos")
    target_sl = sensor_slice(model, "target_pos")
    # Packed rollout state layout: [time, qpos(nq), qvel(nv), ...]. The free
    # joint owns the first six DOFs (angular rates in qvel[3:6]); the arm
    # hinges are the last two.
    base_rate_sl = slice(1 + nq + 3, 1 + nq + 6)
    arm_qvel_sl = slice(1 + nq + nv - 2, 1 + nq + nv)

    w_running, w_terminal, w_qvel, w_base = 1.0, 10.0, 1.0e-2, 0.5

    def cost_fn(states: np.ndarray, sensors: np.ndarray, controls: np.ndarray) -> np.ndarray:
        del controls
        dist_sq = np.sum((sensors[:, :, ee_sl] - sensors[:, :, target_sl]) ** 2, axis=-1)
        arm_rate_sq = np.sum(states[:, :, arm_qvel_sl] ** 2, axis=-1)
        base_rate_sq = np.sum(states[:, :, base_rate_sl] ** 2, axis=-1)
        return (
            w_running * np.mean(dist_sq, axis=1)
            + w_terminal * dist_sq[:, -1]
            + w_qvel * np.mean(arm_rate_sq, axis=1)
            + w_base * np.mean(base_rate_sq, axis=1)
        )

    config = MppiConfig(
        horizon=args.horizon,
        num_rollouts=args.num_rollouts,
        num_nodes=args.num_nodes,
        spline_order="linear",
        sigma=args.sigma,
        temperature=args.temperature,
        use_noise_ramp=True,
        noise_ramp=2.5,
        nthread=args.nthread,
        seed=args.seed,
    )
    ncontrol = mjo_control_size(model)
    if ncontrol != int(model.nu):
        raise RuntimeError("this demo expects a model without orbital actuators")
    planner = MppiPlanner(
        model,
        config,
        cost_fn,
        ctrl_low=np.full(ncontrol, -3.14),
        ctrl_high=np.full(ncontrol, 3.14),
    )
    planner.reset(data)

    def ee_target_distance() -> float:
        sensordata = np.asarray(data.sensordata)
        return float(np.linalg.norm(sensordata[ee_sl] - sensordata[target_sl]))

    dt = float(model.opt.timestep)
    n_steps = int(round(args.duration / dt))
    replan_every = max(1, int(round(args.replan_period / dt)))
    log_every = int(round(1.0 / dt))

    print("=" * 60)
    print("MPPI Arm Reach — free-floating spacecraft arm")
    print("=" * 60)
    print(
        f"rollouts={config.num_rollouts}  nodes={config.num_nodes}  "
        f"horizon={config.horizon:g}s  replan={args.replan_period:g}s  "
        f"sigma={args.sigma:g}  threads={config.nthread}"
    )
    print(f"initial EE-target distance: {ee_target_distance():.3f} m")
    print()

    distances = np.empty(n_steps)
    t_start = wall_time.perf_counter()
    for step in range(n_steps):
        if step % replan_every == 0:
            planner.update_action(data)
        np.copyto(data.ctrl, planner.action(float(data.time)))
        mjo_step(model, data)
        distances[step] = ee_target_distance()
        if (step + 1) % log_every == 0:
            assert planner.last_costs is not None
            print(
                f"  t={data.time:5.1f} s   distance={distances[step]:6.3f} m   "
                f"best rollout cost={planner.last_costs.min():8.4f}"
            )
    elapsed = wall_time.perf_counter() - t_start

    settle_window = max(1, int(round(2.0 / dt)))
    settle_distance = float(np.mean(distances[-settle_window:]))
    base_rate = float(np.linalg.norm(np.asarray(data.qvel)[3:6]))
    print()
    print(f"final EE-target distance: {distances[-1]:.3f} m")
    print(f"mean distance over last 2 s: {settle_distance:.3f} m")
    print(f"residual base angular rate: {base_rate:.4f} rad/s")
    print(f"wall time: {elapsed:.1f} s ({n_steps * dt / elapsed:.1f}x realtime)")

    all_finite = bool(
        np.all(np.isfinite(np.asarray(data.qpos))) and np.all(np.isfinite(np.asarray(data.qvel)))
    )
    if all_finite and settle_distance < 0.05:
        print("PASS: end effector reached the target and held position.")
    else:
        print("WARN: end effector did not settle within 5 cm — see log above.")


if __name__ == "__main__":
    main()
