"""MPPI sim-fidelity check: does the *rollout model* matter for long-horizon
gravity-gradient stabilization?

We capture a free-flyer payload once, then run the long-horizon phase-B
stabilization (damp libration about local vertical using only the slow arm +
gravity-gradient torque) from the SAME latched state under two planners that
differ ONLY in the dynamics model used for their internal MPPI rollouts:

  * ``mjorbit`` -- full orbital coupling (gravity gradient + differential gravity
    + orbit propagation). The correct model.
  * ``zerog``   -- the "disable gravity in an off-the-shelf engine" baseline: the
    same compiled model with the central body's GM set to 0, so the rollouts see
    no gravity gradient, no differential gravity, no orbital reference motion
    (plain free-floating MuJoCo). This is the SmallSatSim-style approximation.
  * ``zerog_fair`` -- keeps GM (so the rollout still propagates the orbit and sees
    the rotating nadir) but disables the gravity-gradient *torque*. Isolates the
    attitude dynamics gap from the reference-motion gap.

BOTH planners are EXECUTED on the full-mjorbit plant (the common "reality"). A
zero-g planner predicts the captured stack stays put, so it plans almost no arm
motion; on the true plant the gravity gradient librates the stack and it drifts /
tumbles. The mjorbit planner predicts the libration and damps it.

This is the cheap, no-training precursor to the RL sim-fidelity study: if the
planner's dynamics model doesn't matter here, it won't matter for RL either.

    pixi run python examples/mppi/mppi_fidelity_capture.py --backends mjorbit zerog
    # show horizon matters: a short rollout can't see the libration
    pixi run python examples/mppi/mppi_fidelity_capture.py --backends mjorbit zerog --horizon-b 300
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time as wall_time
from pathlib import Path

import numpy as np

from mjorbit import MjoData, MjoModel, OrbitInit, mjo_forward, mjo_step
from mjorbit.constants import GM_EARTH, R_EARTH
from mjorbit.planning import MppiConfig, MppiPlanner
from mjorbit.rollout import mjo_get_state, mjo_set_state
from mjorbit.spec import MjoSpec
from mjorbit.testdata import SPACECRAFT_CAPTURE_XML

CAPTURE_XML = Path(SPACECRAFT_CAPTURE_XML)

# Packed state layout (see capture_stabilize.py): bus pos 1:4, bus quat 4:8,
# arm 8:10, payload pos 10:13, ..., bus angvel 20:23, arm vel 23:25, R_eci 31:34.
BUS_POS = slice(1, 4)
PAYLOAD_POS = slice(10, 13)
BUS_WZ = 22
ARM_QVEL = slice(23, 25)
R_ECI = slice(31, 34)

EE_POS = slice(0, 3)
EE_QUAT = slice(3, 7)
EE_LINVEL = slice(7, 10)
SENS_PAYLOAD_POS = slice(10, 13)
PAYLOAD_LINVEL = slice(13, 16)
GRASP_OFFSET = 0.6

# The compiler requires GM > 0, so the "disable gravity" baseline uses a
# negligible GM (g0 = GM/r^2 ~ 1e-14 km/s^2 at LEO): no gravity gradient, no
# differential gravity, the reference free-drifts -- i.e. plain zero-g MuJoCo.
EPS_GM = 1e-6


def quat_to_xaxis(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)], axis=-1
    )


def _compile(active: bool, *, gm: float, use_gg: bool, use_j2: bool) -> MjoModel:
    """Compile the capture XML with weld on/off and a configurable gravity model."""
    text = CAPTURE_XML.read_text().replace('active="false"', f'active="{str(active).lower()}"')
    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
        f.write(text)
        path = Path(f.name)
    try:
        spec = MjoSpec.from_xml_path(str(path))
        spec.mjorbit.central_body.gm = gm
        spec.mjorbit.use_gravity_gradient = use_gg
        spec.mjorbit.use_j2 = use_j2
        return spec.compile()
    finally:
        path.unlink(missing_ok=True)


# Planner dynamics models (weld active = the captured stack). The plant is always
# the full-fidelity "mjorbit" model.
def stack_model(backend: str) -> MjoModel:
    if backend == "mjorbit":
        return _compile(True, gm=GM_EARTH, use_gg=True, use_j2=True)
    if backend == "zerog":
        return _compile(True, gm=EPS_GM, use_gg=False, use_j2=False)
    if backend == "zerog_fair":
        return _compile(True, gm=GM_EARTH, use_gg=False, use_j2=False)
    raise ValueError(f"unknown backend {backend!r}")


def grasp_error(sensors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grasp_pt = sensors[..., EE_POS] + GRASP_OFFSET * quat_to_xaxis(sensors[..., EE_QUAT])
    dist = np.linalg.norm(sensors[..., SENS_PAYLOAD_POS] - grasp_pt, axis=-1)
    relspeed = np.linalg.norm(sensors[..., PAYLOAD_LINVEL] - sensors[..., EE_LINVEL], axis=-1)
    return dist, relspeed


def stack_pitch(states: np.ndarray) -> np.ndarray:
    axis = states[..., PAYLOAD_POS] - states[..., BUS_POS]
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    rhat = states[..., R_ECI]
    rhat = rhat / np.linalg.norm(rhat, axis=-1, keepdims=True)
    sin = rhat[..., 0] * axis[..., 1] - rhat[..., 1] * axis[..., 0]
    cos = np.sum(rhat * axis, axis=-1)
    return np.arctan2(sin, cos)


def make_capture_cost(num_timesteps: int):
    term = slice(int(0.8 * num_timesteps), None)

    def cost_fn(states, sensors, controls):
        del controls
        dist, relspeed = grasp_error(sensors)
        arm_rate_sq = np.sum(states[:, :, ARM_QVEL] ** 2, axis=-1)
        return (0.3 * np.mean(dist**2, axis=1) + 3.0 * np.mean(dist[:, term] ** 2, axis=1)
                + 100.0 * np.mean(relspeed[:, term] ** 2, axis=1)
                + 20.0 * np.mean(arm_rate_sq, axis=1))

    return cost_fn


def make_stabilize_cost(num_timesteps: int, omega_orbit: float):
    term = slice(int(0.6 * num_timesteps), None)

    def cost_fn(states, sensors, controls):
        del sensors, controls
        pitch = stack_pitch(states)
        align = np.sin(pitch) ** 2
        libr = ((states[:, :, BUS_WZ] - omega_orbit) / omega_orbit) ** 2
        arm_rate_sq = np.sum((states[:, :, ARM_QVEL] / 0.01) ** 2, axis=-1)
        return (2.0 * np.mean(align[:, term], axis=1) + 0.5 * np.mean(libr[:, term], axis=1)
                + 0.3 * np.mean(align, axis=1) + 0.005 * np.mean(arm_rate_sq, axis=1))

    return cost_fn


def plant_pitch(data: MjoData) -> float:
    bus = np.asarray(data.qpos[0:3])
    payload = np.asarray(data.qpos[9:12])
    axis = payload - bus
    axis /= np.linalg.norm(axis)
    rhat = data.orbit.R_eci / np.linalg.norm(data.orbit.R_eci)
    return float(np.arctan2(rhat[0] * axis[1] - rhat[1] * axis[0], rhat @ axis))


def capture(model_cap, R0, V0, omega, dt, args):
    """Phase A: grasp with the full-fidelity planner.

    Returns ``(latched_state, t_hist, pitch_hist)`` -- the latched state plus the
    grasp-phase trajectory (so the figure can show the full grasp+stabilize
    timeline). This phase is shared by all conditions: the grasp is identical;
    only the subsequent stabilization differs by planner model.
    """
    data = model_cap.make_data(orbit=OrbitInit(R_eci=R0, V_eci=V0))
    data.qpos[3:7] = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    data.qpos[7:9] = [1.0, -2.0]
    data.qvel[3:6] = [0.0, 0.0, omega]
    payload_pos = np.array([0.5, 6.3, 0.0])
    data.qpos[9:12] = payload_pos
    data.qvel[8:11] = np.cross([0.0, 0.0, omega], payload_pos)
    data.qvel[11:14] = [0.0, 0.0, 0.002]
    np.copyto(data.ctrl, data.qpos[7:9])
    mjo_forward(model_cap, data)

    cfg = MppiConfig(horizon=args.horizon_a, num_rollouts=args.rollouts, num_nodes=5,
                     spline_order="linear", sigma=0.3, temperature=0.05,
                     use_noise_ramp=True, noise_ramp=2.5, nthread=args.nthread, seed=args.seed)
    planner = MppiPlanner(model_cap, cfg, make_capture_cost(int(np.ceil(args.horizon_a / dt))),
                          ctrl_low=np.full(2, -2.8), ctrl_high=np.full(2, 2.8))
    planner.reset(data, nominal_knots=np.tile(np.asarray(data.ctrl), (5, 1)))

    replan_every = max(1, int(round(args.replan_a / dt)))
    t_hist, pitch_hist = [], []
    for step in range(int(round(args.max_capture_time / dt))):
        if step % replan_every == 0:
            planner.update_action(data)
        np.copyto(data.ctrl, planner.action(float(data.time)))
        mjo_step(model_cap, data)
        t_hist.append(float(data.time))
        pitch_hist.append(plant_pitch(data))
        dist, relspeed = grasp_error(np.asarray(data.sensordata))
        if dist < 0.30 and relspeed < 0.04:
            print(f"  grasped at t={data.time:.0f}s (dist={float(dist):.3f} m, "
                  f"rel speed={float(relspeed)*100:.2f} cm/s)")
            return mjo_get_state(model_cap, data), np.asarray(t_hist), np.asarray(pitch_hist)
    raise RuntimeError("capture failed within max-capture-time")


def stabilize(backend: str, latched_state, model_plant, R0, V0, omega, dt, args) -> dict:
    """Phase B from the latched state. Plant = full mjorbit; planner = `backend`."""
    model_plan = model_plant if backend == "mjorbit" else stack_model(backend)

    data = model_plant.make_data(orbit=OrbitInit(R_eci=R0, V_eci=V0))
    mjo_set_state(model_plant, data, latched_state)
    np.copyto(data.ctrl, np.asarray(data.qpos[7:9]))
    mjo_forward(model_plant, data)

    # Separate planner workspace (consistent with the planner model), synced to
    # the plant state at every replan.
    data_plan = model_plan.make_data(orbit=OrbitInit(R_eci=R0, V_eci=V0))

    n_steps_b = int(np.ceil(args.horizon_b / dt))
    cfg = MppiConfig(horizon=args.horizon_b, num_rollouts=args.rollouts, num_nodes=6,
                     spline_order="linear", sigma=0.25, temperature=0.02,
                     use_noise_ramp=True, noise_ramp=2.5, nthread=args.nthread, seed=args.seed + 1)
    planner = MppiPlanner(model_plan, cfg, make_stabilize_cost(n_steps_b, omega),
                          ctrl_low=np.full(2, -2.8), ctrl_high=np.full(2, 2.8))
    planner.reset(data, nominal_knots=np.tile(np.asarray(data.qpos[7:9]), (6, 1)))

    replan_every = max(1, int(round(args.replan_b / dt)))
    log_every = int(round(500.0 / dt))
    n_b = int(round(args.duration_b / dt))
    pitches = np.empty(n_b)
    times = np.empty(n_b)
    for step in range(n_b):
        if step % replan_every == 0:
            # sync planner workspace to the plant state, then replan
            mjo_set_state(model_plan, data_plan, mjo_get_state(model_plant, data))
            np.copyto(data_plan.ctrl, np.asarray(data.ctrl))
            mjo_forward(model_plan, data_plan)
            planner.update_action(data_plan)
        np.copyto(data.ctrl, planner.action(float(data.time)))
        mjo_step(model_plant, data)
        pitches[step] = plant_pitch(data)
        times[step] = float(data.time)
        if step % log_every == 0:
            pitch_deg = np.rad2deg(pitches[step])
            print(f"    [{backend:11s}] t={data.time:7.0f}s  pitch={pitch_deg:+7.1f} deg")

    abs_deg = np.rad2deg(np.abs(pitches))
    err_deg = np.minimum(abs_deg, 180.0 - abs_deg)  # angle to nearest vertical
    libration_period = 2.0 * np.pi / (omega * np.sqrt(3.0))
    window = min(int(round(libration_period / dt)), n_b // 2)
    winding = float(np.abs(np.unwrap(pitches)[-1] - np.unwrap(pitches)[0]))
    finite = bool(np.all(np.isfinite(np.asarray(data.qpos))) and
                  np.all(np.isfinite(np.asarray(data.qvel))))
    summary = {
        "backend": backend,
        "first_window_max_deg": float(err_deg[:window].max()),
        "last_window_max_deg": float(err_deg[-window:].max()),
        "last_window_mean_deg": float(err_deg[-window:].mean()),
        "final_pitch_deg": float(np.rad2deg(pitches[-1])),
        "winding_deg": float(np.rad2deg(winding)),
        "tumbled": bool(winding > np.deg2rad(540.0)),
        "finite": finite,
        "stabilized": bool(finite and winding <= np.deg2rad(540.0)
                           and err_deg[-window:].max() < 35.0
                           and err_deg[-window:].mean() < 18.0),
    }
    return summary, times, pitches


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backends", nargs="+", default=["mjorbit", "zerog"],
                   choices=["mjorbit", "zerog", "zerog_fair"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon-a", type=float, default=300.0)
    p.add_argument("--horizon-b", type=float, default=2400.0)
    p.add_argument("--rollouts", type=int, default=64)
    p.add_argument("--replan-a", type=float, default=10.0)
    p.add_argument("--replan-b", type=float, default=60.0)
    p.add_argument("--max-capture-time", type=float, default=900.0)
    p.add_argument("--duration-b", type=float, default=14000.0,
                   help="phase-B duration (s); long enough (~4 libration periods) for "
                        "the orbit-coupled controller to settle")
    p.add_argument("--nthread", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument("--out", default="/tmp/mppi_fidelity.json")
    args = p.parse_args()

    alt_km = 400.0
    r_orbit = R_EARTH + alt_km
    omega = float(np.sqrt(GM_EARTH / r_orbit**3))
    v_orbit = float(np.sqrt(GM_EARTH / r_orbit))
    R0 = np.array([r_orbit, 0.0, 0.0])
    V0 = np.array([0.0, v_orbit, 0.0])

    model_cap = _compile(False, gm=GM_EARTH, use_gg=True, use_j2=True)   # weld off (capture)
    model_stk = _compile(True, gm=GM_EARTH, use_gg=True, use_j2=True)    # weld on  (plant)
    dt = float(model_cap.opt.timestep)
    lib = 2.0 * np.pi / (omega * np.sqrt(3.0))

    print("=" * 70)
    print("MPPI sim-fidelity: planner rollout model = {mjorbit | zerog}, plant = mjorbit")
    print("=" * 70)
    print(f"orbit {alt_km:.0f} km, dt={dt:g}s, phase-B horizon {args.horizon_b:g}s "
          f"({args.horizon_b/lib:.2f} libration periods), duration {args.duration_b:g}s")
    print("--- phase A: grasp (full-fidelity planner, shared by all conditions) ---")
    t0 = wall_time.perf_counter()
    latched, cap_t, cap_pitch = capture(model_cap, R0, V0, omega, dt, args)
    # transfer to the weld-on plant model
    data_b = model_stk.make_data(orbit=OrbitInit(R_eci=R0, V_eci=V0))
    mjo_set_state(model_stk, data_b, latched)
    mjo_forward(model_stk, data_b)
    print(f"  stack pitch at grasp: {np.rad2deg(plant_pitch(data_b)):+.1f} deg\n")
    latched_stk = mjo_get_state(model_stk, data_b)
    cap_err = np.minimum(np.rad2deg(np.abs(cap_pitch)), 180.0 - np.rad2deg(np.abs(cap_pitch)))

    def pack(t_full, err_full):
        stride = max(1, len(t_full) // 1500)
        return {"t_s": t_full[::stride].tolist(), "err_deg": err_full[::stride].tolist()}

    results = {}
    for backend in args.backends:
        print(f"--- phase B: stabilize with planner='{backend}' (executed on mjorbit) ---")
        summary, stab_t, stab_pitch = stabilize(backend, latched_stk, model_stk,
                                                 R0, V0, omega, dt, args)
        stab_abs_deg = np.rad2deg(np.abs(stab_pitch))
        stab_err = np.minimum(stab_abs_deg, 180.0 - stab_abs_deg)
        # full grasp+stabilize timeline (the grasp prefix is the same for all conditions)
        t_full = np.concatenate([cap_t, stab_t])
        err_full = np.concatenate([cap_err, stab_err])
        results[backend] = {"summary": summary, **pack(t_full, err_full)}
        print(f"  => {backend}: last-window max {summary['last_window_max_deg']:.1f} / "
              f"mean {summary['last_window_mean_deg']:.1f} deg, "
              f"winding {summary['winding_deg']:.0f} deg, "
              f"{'STABILIZED' if summary['stabilized'] else 'FAILED'}\n")

    out = {"meta": {"alt_km": alt_km, "dt": dt, "horizon_b": args.horizon_b,
                    "duration_b": args.duration_b, "libration_period_s": lib,
                    "rollouts": args.rollouts, "seed": args.seed},
           "results": results}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wall time {wall_time.perf_counter()-t0:.0f}s; wrote {args.out}")
    print("\nSUMMARY (eval on full mjorbit plant):")
    for b in args.backends:
        s = results[b]["summary"]
        print(f"  {b:11s}  last-window mean {s['last_window_mean_deg']:6.1f} deg  "
              f"max {s['last_window_max_deg']:6.1f} deg  "
              f"{'STABILIZED' if s['stabilized'] else 'FAILED'}")


if __name__ == "__main__":
    main()
