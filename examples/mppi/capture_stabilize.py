"""Long-horizon MPPI demo: capture a free-flyer, then let gravity gradient work.

A free-floating bus with a long, slow 2-link arm captures a 400 kg payload
drifting ~6 m away on the same orbit, then stabilizes the combined stack about
the local vertical. The stack is captured roughly along-track, i.e. nearly 90
degrees from the gravity-gradient equilibrium, so the planner must manage the
swing through local vertical and bleed off libration energy using nothing but
slow arm motions — the only torques on the free base are gravity-gradient
torques and arm reaction torques.

This exercises the simulator's long-horizon environment modelling: rollouts
span ~40 minutes of coupled orbital + multibody dynamics per replan (J2 and
per-body differential gravity on, libration period ~56 min), and the phase-B
cost is evaluated on the terminal window of each rollout, so the planner only
"sees" that an arm strategy works through the passive dynamics the simulator
predicts — there is no attitude actuator to force the outcome.

Phases:
  A (capture): reach the grasp pose gently (low relative velocity).
  latch: the weld equality activates — modelled by recompiling the same XML
     with active="true" and transferring the packed mjo state across models.
  B (stabilize): long-horizon MPPI damps libration via the arm.

Usage:
    pixi run python examples/mppi/capture_stabilize.py
    pixi run python examples/mppi/capture_stabilize.py --quick
    # show that a short horizon cannot see the passive stabilization:
    pixi run python examples/mppi/capture_stabilize.py --horizon-b 300
"""

from __future__ import annotations

import argparse
import os
import time as wall_time
from pathlib import Path

import numpy as np

from mujoco_orbit import MjoData, MjoModel, OrbitInit, mjo_forward, mjo_step
from mujoco_orbit.constants import GM_EARTH, R_EARTH
from mujoco_orbit.planning import MppiConfig, MppiPlanner
from mujoco_orbit.rollout import mjo_get_state, mjo_set_state
from mujoco_orbit.testdata import SPACECRAFT_CAPTURE_XML

CAPTURE_XML = Path(SPACECRAFT_CAPTURE_XML)

# Packed rollout state layout for this model (nstate = 1 + nq + nv + 7):
# [time, bus pos 1:4, bus quat 4:8, shoulder 8, elbow 9, payload pos 10:13,
#  payload quat 13:17, bus linvel 17:20, bus angvel 20:23, shoulder vel 23,
#  elbow vel 24, payload linvel 25:28, payload angvel 28:31,
#  R_eci 31:34, V_eci 34:37, orbit t 37]
BUS_POS = slice(1, 4)
ARM_Q = slice(8, 10)
PAYLOAD_POS = slice(10, 13)
BUS_WZ = 22
ARM_QVEL = slice(23, 25)
R_ECI = slice(31, 34)

# Sensor layout: ee_pos 0:3, ee_quat 3:7, ee_linvel 7:10, payload_pos 10:13,
# payload_linvel 13:16
EE_POS = slice(0, 3)
EE_QUAT = slice(3, 7)
EE_LINVEL = slice(7, 10)
SENS_PAYLOAD_POS = slice(10, 13)
PAYLOAD_LINVEL = slice(13, 16)

GRASP_OFFSET = 0.6  # m, weld relpose offset along the end-effector x-axis


def quat_to_xaxis(q: np.ndarray) -> np.ndarray:
    """Body x-axis in world frame for quaternions (..., 4) in (w,x,y,z) order."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)],
        axis=-1,
    )


def compile_models(dt_override: float | None = None) -> tuple[MjoModel, MjoModel]:
    """Compile the capture (weld off) and stack (weld on) variants of one XML."""
    import tempfile

    xml = CAPTURE_XML.read_text()
    models = []
    for active in ("false", "true"):
        text = xml.replace('active="false"', f'active="{active}"')
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(text)
            path = Path(f.name)
        try:
            models.append(MjoModel.from_xml_path(str(path), mj_timestep=dt_override))
        finally:
            path.unlink(missing_ok=True)
    return models[0], models[1]


def grasp_error(sensors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance and relative speed between payload and the grasp point.

    Works on any (..., nsensordata) array, so it serves both the batched
    rollout cost and the plant-side latch check.
    """
    grasp_pt = sensors[..., EE_POS] + GRASP_OFFSET * quat_to_xaxis(sensors[..., EE_QUAT])
    dist = np.linalg.norm(sensors[..., SENS_PAYLOAD_POS] - grasp_pt, axis=-1)
    relspeed = np.linalg.norm(
        sensors[..., PAYLOAD_LINVEL] - sensors[..., EE_LINVEL], axis=-1
    )
    return dist, relspeed


def stack_pitch(states: np.ndarray) -> np.ndarray:
    """Signed in-plane angle between the bus->payload axis and local vertical."""
    axis = states[..., PAYLOAD_POS] - states[..., BUS_POS]
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    rhat = states[..., R_ECI]
    rhat = rhat / np.linalg.norm(rhat, axis=-1, keepdims=True)
    sin = rhat[..., 0] * axis[..., 1] - rhat[..., 1] * axis[..., 0]
    cos = np.sum(rhat * axis, axis=-1)
    return np.arctan2(sin, cos)


def make_capture_cost(num_timesteps: int):
    term = slice(int(0.8 * num_timesteps), None)

    def cost_fn(states: np.ndarray, sensors: np.ndarray, controls: np.ndarray) -> np.ndarray:
        del controls
        dist, relspeed = grasp_error(sensors)
        arm_rate_sq = np.sum(states[:, :, ARM_QVEL] ** 2, axis=-1)
        return (
            0.3 * np.mean(dist**2, axis=1)
            + 3.0 * np.mean(dist[:, term] ** 2, axis=1)
            + 100.0 * np.mean(relspeed[:, term] ** 2, axis=1)
            + 20.0 * np.mean(arm_rate_sq, axis=1)
        )

    return cost_fn


def make_stabilize_cost(num_timesteps: int, omega_orbit: float):
    term = slice(int(0.6 * num_timesteps), None)

    def cost_fn(states: np.ndarray, sensors: np.ndarray, controls: np.ndarray) -> np.ndarray:
        del sensors, controls
        pitch = stack_pitch(states)
        align = np.sin(pitch) ** 2  # 0 at either vertical, 1 at horizontal
        libr = ((states[:, :, BUS_WZ] - omega_orbit) / omega_orbit) ** 2
        arm_rate_sq = np.sum((states[:, :, ARM_QVEL] / 0.01) ** 2, axis=-1)
        return (
            2.0 * np.mean(align[:, term], axis=1)
            + 0.5 * np.mean(libr[:, term], axis=1)
            + 0.3 * np.mean(align, axis=1)
            + 0.005 * np.mean(arm_rate_sq, axis=1)
        )

    return cost_fn


def plant_pitch(data: MjoData) -> float:
    bus = np.asarray(data.qpos[0:3])
    payload = np.asarray(data.qpos[9:12])
    axis = payload - bus
    axis /= np.linalg.norm(axis)
    rhat = data.orbit.R_eci / np.linalg.norm(data.orbit.R_eci)
    return float(np.arctan2(rhat[0] * axis[1] - rhat[1] * axis[0], rhat @ axis))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="half-length phase B")
    parser.add_argument("--horizon-a", type=float, default=300.0, help="capture horizon (s)")
    parser.add_argument(
        "--horizon-b", type=float, default=2400.0,
        help="stabilization horizon (s); must cover a good fraction of the "
        "~3400 s libration period for the planner to see the passive dynamics",
    )
    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--replan-a", type=float, default=10.0)
    parser.add_argument("--replan-b", type=float, default=60.0)
    parser.add_argument("--max-capture-time", type=float, default=900.0)
    parser.add_argument("--duration-b", type=float, default=7000.0)
    parser.add_argument("--nthread", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.quick:
        args.duration_b = 3500.0

    alt_km = 400.0
    r_orbit = R_EARTH + alt_km
    omega = float(np.sqrt(GM_EARTH / r_orbit**3))
    v_orbit = float(np.sqrt(GM_EARTH / r_orbit))
    R0 = np.array([r_orbit, 0.0, 0.0])
    V0 = np.array([0.0, v_orbit, 0.0])  # equatorial: orbit plane = world XY

    model_cap, model_stk = compile_models()
    dt = float(model_cap.opt.timestep)
    libration_period = 2.0 * np.pi / (omega * np.sqrt(3.0))

    data = model_cap.make_data(orbit=OrbitInit(R_eci=R0, V_eci=V0))
    # Bus arm points along-track (+Y); attitude tracks the rotating LVLH frame.
    data.qpos[3:7] = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    data.qpos[7:9] = [1.0, -2.0]  # folded arm: a real reach is needed to capture
    data.qvel[3:6] = [0.0, 0.0, omega]
    # Payload ~6.3 m ahead on (almost) the same orbit, with the co-orbital
    # velocity in chief-centered inertial axes and a slow residual tumble.
    payload_pos = np.array([0.5, 6.3, 0.0])
    data.qpos[9:12] = payload_pos
    data.qvel[8:11] = np.cross([0.0, 0.0, omega], payload_pos)
    data.qvel[11:14] = [0.0, 0.0, 0.002]
    np.copyto(data.ctrl, data.qpos[7:9])
    mjo_forward(model_cap, data)

    cfg_a = MppiConfig(
        horizon=args.horizon_a, num_rollouts=args.rollouts, num_nodes=5,
        spline_order="linear", sigma=0.3, temperature=0.05,
        use_noise_ramp=True, noise_ramp=2.5, nthread=args.nthread, seed=args.seed,
    )
    planner_a = MppiPlanner(
        model_cap, cfg_a, make_capture_cost(int(np.ceil(args.horizon_a / dt))),
        ctrl_low=np.full(2, -2.8), ctrl_high=np.full(2, 2.8),
    )
    planner_a.reset(data, nominal_knots=np.tile(np.asarray(data.ctrl), (5, 1)))

    print("=" * 64)
    print("MPPI Capture & Gravity-Gradient Stabilize")
    print("=" * 64)
    print(
        f"orbit: {alt_km:.0f} km equatorial, rate {omega:.3e} rad/s, "
        f"libration period ~{libration_period:.0f} s"
    )
    print(
        f"phase A horizon {args.horizon_a:g} s | phase B horizon {args.horizon_b:g} s "
        f"({args.horizon_b / libration_period:.2f} libration periods)"
    )
    dist0, _ = grasp_error(np.asarray(data.sensordata))
    print(f"initial grasp distance {float(dist0):.2f} m, payload along-track +{payload_pos[1]:g} m")
    print()

    # ----------------------------------------------------------------
    # Phase A: capture
    # ----------------------------------------------------------------
    print("--- phase A: approach and capture ---")
    t_start = wall_time.perf_counter()
    replan_every = max(1, int(round(args.replan_a / dt)))
    log_every = int(round(60.0 / dt))
    latched = False
    latch_dist = latch_speed = float("nan")
    n_max = int(round(args.max_capture_time / dt))
    for step in range(n_max):
        if step % replan_every == 0:
            planner_a.update_action(data)
        np.copyto(data.ctrl, planner_a.action(float(data.time)))
        mjo_step(model_cap, data)
        dist, relspeed = grasp_error(np.asarray(data.sensordata))
        if step % log_every == 0:
            print(
                f"  t={data.time:6.0f} s  grasp dist={float(dist):6.3f} m  "
                f"rel speed={float(relspeed)*100:5.2f} cm/s"
            )
        if dist < 0.30 and relspeed < 0.04:
            latched = True
            latch_dist, latch_speed = float(dist), float(relspeed)
            break

    if not latched:
        print("WARN: no capture within the time limit — aborting.")
        return
    t_latch = float(data.time)
    print(
        f"  LATCH at t={t_latch:.0f} s: dist={latch_dist:.3f} m, "
        f"rel speed={latch_speed*100:.2f} cm/s — weld engaged"
    )

    # ----------------------------------------------------------------
    # Latch: transfer the packed state into the weld-active model
    # ----------------------------------------------------------------
    state = mjo_get_state(model_cap, data)
    data_b = model_stk.make_data(orbit=OrbitInit(R_eci=R0, V_eci=V0))
    mjo_set_state(model_stk, data_b, state)
    np.copyto(data_b.ctrl, np.asarray(data_b.qpos[7:9]))  # hold joints
    mjo_forward(model_stk, data_b)
    pitch0 = np.rad2deg(plant_pitch(data_b))
    print(f"  stack pitch from local vertical at capture: {pitch0:+.1f} deg")
    print()

    # ----------------------------------------------------------------
    # Phase B: long-horizon stabilization
    # ----------------------------------------------------------------
    print("--- phase B: gravity-gradient stabilization ---")
    n_steps_b = int(np.ceil(args.horizon_b / dt))
    cfg_b = MppiConfig(
        horizon=args.horizon_b, num_rollouts=args.rollouts, num_nodes=6,
        spline_order="linear", sigma=0.25, temperature=0.02,
        use_noise_ramp=True, noise_ramp=2.5, nthread=args.nthread, seed=args.seed + 1,
    )
    planner_b = MppiPlanner(
        model_stk, cfg_b, make_stabilize_cost(n_steps_b, omega),
        ctrl_low=np.full(2, -2.8), ctrl_high=np.full(2, 2.8),
    )
    planner_b.reset(data_b, nominal_knots=np.tile(np.asarray(data_b.ctrl), (6, 1)))

    replan_every = max(1, int(round(args.replan_b / dt)))
    log_every = int(round(250.0 / dt))
    n_b = int(round(args.duration_b / dt))
    pitches = np.empty(n_b)
    times_b = np.empty(n_b)
    for step in range(n_b):
        if step % replan_every == 0:
            planner_b.update_action(data_b)
        np.copyto(data_b.ctrl, planner_b.action(float(data_b.time)))
        mjo_step(model_stk, data_b)
        pitches[step] = plant_pitch(data_b)
        times_b[step] = float(data_b.time)
        if step % log_every == 0:
            wz = float(np.asarray(data_b.qvel)[5])
            print(
                f"  t={data_b.time:7.0f} s  pitch={np.rad2deg(pitches[step]):+7.1f} deg  "
                f"(wz-w)/w={wz/omega - 1.0:+6.2f}  joints="
                f"[{np.rad2deg(data_b.qpos[7]):+6.1f}, {np.rad2deg(data_b.qpos[8]):+6.1f}] deg"
            )
    elapsed = wall_time.perf_counter() - t_start

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    # Either vertical is a gravity-gradient equilibrium (payload toward or
    # away from Earth); measure the angle to the nearest one.
    abs_deg = np.rad2deg(np.abs(pitches))
    err_deg = np.minimum(abs_deg, 180.0 - abs_deg)
    window = min(int(round(libration_period / dt)), n_b // 2)
    first_max = float(err_deg[:window].max())
    last_max = float(err_deg[-window:].max())
    last_mean = float(err_deg[-window:].mean())
    # True tumbling = the stack keeps winding instead of librating.
    winding = float(np.abs(np.unwrap(pitches)[-1] - np.unwrap(pitches)[0]))
    tumbled = winding > np.deg2rad(540.0)
    equilibrium = "payload-zenith (0 deg)" if abs_deg[-window:].mean() < 90.0 else \
        "payload-nadir (180 deg)"
    finite = bool(
        np.all(np.isfinite(np.asarray(data_b.qpos)))
        and np.all(np.isfinite(np.asarray(data_b.qvel)))
    )

    print()
    print(f"captured at t={t_latch:.0f} s ({latch_speed*100:.2f} cm/s contact speed)")
    print(f"angle to nearest vertical: first window max {first_max:.1f} deg -> "
          f"last window max {last_max:.1f} / mean {last_mean:.1f} deg")
    print(f"settled toward {equilibrium}, final pitch {np.rad2deg(pitches[-1]):+.1f} deg")
    print(f"wall time {elapsed:.0f} s "
          f"({(t_latch + args.duration_b) / elapsed:.0f}x realtime)")
    if finite and not tumbled and last_max < 35.0 and last_mean < 18.0:
        print("PASS: payload captured and stack stabilized about local vertical "
              "by gravity-gradient torques + slow arm motion.")
    else:
        print("WARN: stabilization criteria not met "
              f"(tumbled={tumbled}, finite={finite}, "
              f"last max={last_max:.1f}, last mean={last_mean:.1f}).")


if __name__ == "__main__":
    main()
