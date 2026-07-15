"""Live viewer for the dual-arm 3-DOF reorientation.

Runs the same MPPI as reorient_mppi_3dof.py, replanning inside the viewer's
per-step callback, so you watch the bus slew to an arbitrary target attitude
purely by the reaction of two multi-DOF arms -- no bus actuators. The planner
is trimmed (fewer rollouts / shorter horizon than the headless demo) so it stays
responsive inside the live render loop.

The target (60 deg about the (1,1,1) axis) is a large slew that mixes roll,
pitch and yaw, well outside the small-angle regime, so both arms must cooperate.
This viewer uses ACTIVE momentum management: the planner keeps running after the
bus reaches the target, so the arms make continuous small reactive strokes to
hold attitude against the (enabled) environmental torques, rather than freezing.
You can watch the arms load up to a non-neutral pose and keep working to hold.

To record a deterministic clip instead of watching live, dump a trajectory with
reorient_mppi_3dof.py --save-traj and render it with scripts/record/record_reorient.py.

Usage:
    pixi run example-reorient-viewer
    # then open http://localhost:8093 in the VSCode Simple Browser
"""

from __future__ import annotations

import os

import numpy as np
from reorient_mppi_3dof import (
    REORIENT_XML,
    att_err_deg,
    make_plots,
    make_reorient_cost,
)

from mjorbit import MjoModel, OrbitInit, mjo_forward
from mjorbit.constants import GM_EARTH, R_EARTH
from mjorbit.planning import MppiConfig, MppiPlanner
from viewer import MjOrbitViewer

PORT = 8093
TARGET_DEG = 60.0
AXIS = np.array([1.0, 1.0, 1.0])
# Trimmed for live replanning (the choppiness in a live MPPI viewer is the
# per-replan rollout cost blocking the render thread, so we keep rollouts*horizon
# small). This 48/10 config is verified to reach the 60 deg target and then hold
# it actively; each replan is ~120 ms. The headless demo uses 128 rollouts / 14 s
# for a tighter solution. Active hold replans the WHOLE time, so the small replan
# hitch persists during the hold (not just the slew).
HORIZON = 10.0
ROLLOUTS = 48
REPLAN = 0.5            # s between replans


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-deg", type=float, default=TARGET_DEG)
    p.add_argument("--axis", type=float, nargs=3, default=list(AXIS),
                   help="rotation axis (need not be normalized)")
    args = p.parse_args()
    target_deg = args.target_deg

    model = MjoModel.from_xml_path(str(REORIENT_XML))
    dt = float(model.opt.timestep)

    r_orbit = R_EARTH + 400.0
    v = float(np.sqrt(GM_EARTH / r_orbit))
    data = model.make_data(orbit=OrbitInit(R_eci=[r_orbit, 0, 0], V_eci=[0, v, 0]))
    data.qpos[7:15] = [0.0, 0.3, 0.0, -0.6, 0.0, 0.3, 0.0, -0.6]
    np.copyto(data.ctrl, data.qpos[7:15])
    mjo_forward(model, data)

    axis = np.asarray(args.axis, float) / np.linalg.norm(args.axis)
    psi = np.deg2rad(target_deg)
    q_target = np.array([np.cos(psi / 2), *(np.sin(psi / 2) * axis)])

    cfg = MppiConfig(
        horizon=HORIZON, num_rollouts=ROLLOUTS, num_nodes=5, spline_order="linear",
        sigma=0.18, temperature=0.05, use_noise_ramp=True, noise_ramp=1.5,
        nthread=max(1, os.cpu_count() or 1), seed=0,
    )
    # Per-joint bounds: keep the shoulder-y hinges (indices 1, 5) clear of the
    # x/y/z gimbal-lock singularity at sy=+-90 deg.
    ctrl_hi = np.full(model.nu, 2.6)
    ctrl_hi[[1, 5]] = 1.2
    planner = MppiPlanner(
        model, cfg, make_reorient_cost(int(np.ceil(HORIZON / dt)), q_target),
        ctrl_low=-ctrl_hi, ctrl_high=ctrl_hi,
    )
    planner.reset(data, nominal_knots=np.tile(np.asarray(data.ctrl), (5, 1)))

    replan_every = max(1, int(round(REPLAN / dt)))
    state = {"step": 0, "reached": False}
    # Log the live run so the same reorient_3dof_plots.png the headless demo makes
    # gets refreshed on exit (Ctrl+C / browser close).
    log = {k: [] for k in ("t", "att", "rate", "armrate", "joints")}
    # Last sim time seen by control(); kept outside `state` so the Reset-detection
    # below survives the state re-init it triggers. Start at -inf so the first call
    # never falsely trips the rewind branch.
    prev_time = [float("-inf")]

    def control(d, _t):
        # The viewer's Reset button rewinds d.time to 0 (reset_simulation ->
        # mjo_set_state restores the t=0 snapshot). Detect that rewind and re-seed
        # the planner/state/log exactly as main() does, so a reset run does not
        # replay a stale plan or keep the old run's `reached` flag and plot log.
        if d.time < prev_time[0]:
            planner.reset(d, nominal_knots=np.tile(np.asarray(d.ctrl), (5, 1)))
            state["step"] = 0
            state["reached"] = False
            for series in log.values():
                series.clear()
        prev_time[0] = float(d.time)
        # Active momentum management: MPPI keeps replanning the whole time (never
        # freezes), so after the slew the arms make continuous reactive strokes to
        # hold attitude against the environmental torques.
        att = float(att_err_deg(np.asarray(d.qpos[3:7]), q_target))
        rate = np.rad2deg(np.linalg.norm(np.asarray(d.qvel[3:6])))
        log["t"].append(float(d.time))
        log["att"].append(att)
        log["rate"].append(rate)
        log["joints"].append(np.rad2deg(np.asarray(d.qpos[7:15]).copy()))
        log["armrate"].append(np.rad2deg(np.max(np.abs(np.asarray(d.qvel[6:14])))))
        if state["step"] % replan_every == 0:
            planner.update_action(d)
        state["step"] += 1
        if not state["reached"] and d.time > 4.0 and att < 3.0:
            state["reached"] = True
            print(f"  target reached at t={d.time:.1f}s (att err={att:.1f} deg) "
                  f"-- arms now holding actively")
        return planner.action(float(d.time))

    viewer = MjOrbitViewer(
        model, data,
        port=PORT,
        show_earth=False,
        show_axes=True,                  # LVLH R/S/W axes for reference
        track_bodies=["A_ee", "B_ee"],   # trail both end-effector reaction paths
        camera_distance=6.0,
        render_frame="lvlh",
    )
    viewer.set_local_scene_scale(1.0)

    print("=" * 60)
    print("  Dual-Arm 3-DOF Reorientation — live viewer")
    print("=" * 60)
    print(f"  Watch the bus slew {target_deg:g} deg about {np.round(axis, 2)} using ONLY "
          f"the two arms, then hold it actively (arms keep working).")
    print(f"  Open: http://localhost:{PORT}")
    print("  Plots are saved to reorient_3dof_plots.png on exit (Ctrl+C / close).")
    print()

    try:
        viewer.run(duration=None, action_fn=control)
    finally:
        if len(log["t"]) > 1:
            arr = {k: np.asarray(v) for k, v in log.items()}
            try:
                make_plots(arr, target_deg, axis,
                           REORIENT_XML.parent / "reorient_3dof_plots.png")
            except ModuleNotFoundError:
                print("(matplotlib not available; run with `pixi run -e report` "
                      "to refresh reorient_3dof_plots.png)")


if __name__ == "__main__":
    main()
