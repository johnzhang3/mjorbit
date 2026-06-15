# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Run the MPPI capture (phase A) + a short post-latch hold and dump a
trajectory for offline rendering (paper example c: grasping under gravity
gradient). Reuses the capture cost + initial state from
``examples/mppi/capture_stabilize.py`` verbatim; skips the multi-minute
phase-B stabilization (too long for a ~10 s clip) and instead holds the mated
stack so the capture moment is the focus.

    pixi run python scripts/record/produce_grasp.py --out /tmp/grasp_traj.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for p in (str(_ROOT / "src"), str(_ROOT / "examples" / "mppi"), str(_ROOT / "scripts" / "record")):
    if p not in sys.path:
        sys.path.insert(0, p)

import capture_stabilize as cs  # noqa: E402  (examples/mppi)

from mjorbit import OrbitInit, mjo_forward, mjo_step  # noqa: E402
from mjorbit.constants import GM_EARTH, R_EARTH  # noqa: E402
from mjorbit.planning import MppiConfig, MppiPlanner  # noqa: E402
from mjorbit.rollout import mjo_get_state, mjo_set_state  # noqa: E402
from mjorbit.testdata import SPACECRAFT_CAPTURE_XML  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/grasp_traj.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollouts", type=int, default=64)
    ap.add_argument("--max-capture-time", type=float, default=900.0)
    ap.add_argument("--hold-seconds", type=float, default=120.0,
                    help="post-latch passive hold to show the captured stack")
    args = ap.parse_args()

    alt_km = 400.0
    r_orbit = R_EARTH + alt_km
    omega = float(np.sqrt(GM_EARTH / r_orbit**3))
    v_orbit = float(np.sqrt(GM_EARTH / r_orbit))
    R0 = np.array([r_orbit, 0.0, 0.0])
    V0 = np.array([0.0, v_orbit, 0.0])

    model_cap, model_stk = cs.compile_models()
    dt = float(model_cap.opt.timestep)

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

    cfg_a = MppiConfig(
        horizon=300.0, num_rollouts=args.rollouts, num_nodes=5,
        spline_order="linear", sigma=0.3, temperature=0.05,
        use_noise_ramp=True, noise_ramp=2.5,
        nthread=max(1, os.cpu_count() or 1), seed=args.seed,
    )
    planner_a = MppiPlanner(
        model_cap, cfg_a, cs.make_capture_cost(int(np.ceil(300.0 / dt))),
        ctrl_low=np.full(2, -2.8), ctrl_high=np.full(2, 2.8),
    )
    planner_a.reset(data, nominal_knots=np.tile(np.asarray(data.ctrl), (5, 1)))

    qpos_hist: list[np.ndarray] = []
    R_hist: list[np.ndarray] = []
    replan_every = max(1, int(round(10.0 / dt)))
    n_max = int(round(args.max_capture_time / dt))
    latched = False
    for step in range(n_max):
        if step % replan_every == 0:
            planner_a.update_action(data)
        np.copyto(data.ctrl, planner_a.action(float(data.time)))
        mjo_step(model_cap, data)
        qpos_hist.append(np.asarray(data.qpos).copy())
        R_hist.append(np.asarray(data.orbit.R_eci).copy())
        dist, relspeed = cs.grasp_error(np.asarray(data.sensordata))
        if dist < 0.30 and relspeed < 0.04:
            latched = True
            break
    n_capture = len(qpos_hist)
    t_latch = float(data.time)
    print(f"phase A: {'LATCHED' if latched else 'no latch'} at t={t_latch:.0f}s "
          f"(dist={float(dist):.3f} m), {n_capture} steps")
    if not latched:
        print("WARN: did not latch; rendering the approach only")

    # Post-latch passive hold (no phase-B planning): mate the stack and hold joints.
    if latched and args.hold_seconds > 0:
        state = mjo_get_state(model_cap, data)
        data_b = model_stk.make_data(orbit=OrbitInit(R_eci=R0, V_eci=V0))
        mjo_set_state(model_stk, data_b, state)
        np.copyto(data_b.ctrl, np.asarray(data_b.qpos[7:9]))
        mjo_forward(model_stk, data_b)
        for _ in range(int(round(args.hold_seconds / dt))):
            mjo_step(model_stk, data_b)
            qpos_hist.append(np.asarray(data_b.qpos).copy())
            R_hist.append(np.asarray(data_b.orbit.R_eci).copy())

    np.savez(
        args.out,
        qpos=np.asarray(qpos_hist),
        R_eci=np.asarray(R_hist),
        V_eci=V0,
        dt=dt,
        xml_path=str(SPACECRAFT_CAPTURE_XML),
        n_capture=n_capture,
    )
    print(f"saved {len(qpos_hist)}-step trajectory to {args.out} (capture={n_capture})")


if __name__ == "__main__":
    main()
