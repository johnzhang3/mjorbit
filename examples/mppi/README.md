# MPPI examples

Sampling-based MPC on top of the `mujoco_orbit` CPU backend.

- `planner.py` — generic spline-knot MPPI planner (judo-style: control knots
  interpolated over the horizon, Gaussian sampling with an optional variance
  ramp across the horizon, exponentially-weighted nominal update). Rollouts go
  through `mujoco_orbit.rollout`, so the optimized control vector is
  `[data.ctrl | rw_torque | mtq_dipole | thr_force | cmg_rate]` and works for
  models with orbital actuators too.
- `arm_reach.py` — simplest demo: a 2-link arm on a free-floating bus reaches
  for a target object co-moving on the chief orbit. Costs read end-effector
  and target world positions from `framepos` sensors in
  `spacecraft_arm_reach.xml`.
- `capture_stabilize.py` — long-horizon demo: a bus with a slow 6.5 m arm
  captures a 400 kg free-flyer drifting ~6 m away, then stabilizes the stack
  about the local vertical using only gravity-gradient torques and slow arm
  motion (the base has no attitude actuators). The grasp is a weld equality
  activated at latch by recompiling `spacecraft_capture.xml` with
  `active="true"` and transferring the packed mjo state across models. Each
  phase-B replan rolls out ~40 min of coupled orbital + multibody dynamics
  (J2 + per-body differential gravity + per-body gravity-gradient torque),
  and the cost is evaluated on the terminal window of each rollout, so the
  planner only succeeds by predicting the passive dynamics. Horizon scaling
  measured across seeds: 300 s horizons fail outright (~50 deg residual
  libration), 600–1200 s are hit-or-miss, and the default 2400 s
  (~0.7 libration periods) passes all tested seeds — final pitch within ~4 deg
  of either gravity-gradient equilibrium, with 1–14 deg mean residual
  libration over the last window. Runs at ~300x realtime on 20 CPU threads.

Run:

```bash
pixi run example-mppi-arm-reach
pixi run example-mppi-capture
# or with knobs:
pixi run python examples/mppi/arm_reach.py --num-rollouts 128 --horizon 2.0
# show that a short horizon cannot see the passive stabilization:
pixi run python examples/mppi/capture_stabilize.py --horizon-b 300
```

Tuning lives in `MppiConfig`: `horizon`, `num_rollouts`, `num_nodes`,
`spline_order` (`zero`/`linear`/`cubic`), `sigma` (scalar or per-control),
`temperature`, `use_noise_ramp`/`noise_ramp`, `seed`, and `nthread` for the
threaded CPU rollout.
