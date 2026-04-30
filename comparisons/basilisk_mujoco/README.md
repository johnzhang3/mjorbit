# Basilisk-MuJoCo Comparison Harness

This folder contains opt-in comparisons between `mujoco_orbit` and
Basilisk-MuJoCo-style scenarios. The goal is to separate frame/coupling
questions from larger GN&C examples before reproducing the full paper cases.

## Frame Mapping

Basilisk-MuJoCo reports body and site states in an inertial simulation frame
that is J2000/ECI aligned. When Earth is the central body, the body free-joint
position can be interpreted as an absolute Earth-centered inertial state.

`mujoco_orbit` keeps MuJoCo `world` chief-centered but ECI aligned:

```text
absolute ECI body state = chief OrbitInit/data.orbit state + MuJoCo world offset
```

LVLH is only a derived rotating frame for conversions and displays. Comparisons
in this harness therefore use converted physical states, not raw `qpos`.

## Cases

- `single_body_orbit.xml`: one free rigid body in a 400 km circular orbit for
  one orbital period, with J2, drag, SRP, magnetic effects, and gravity-gradient
  torque disabled.
- `hinged_satellite.xml`: one free hub with two hinged panels/links and
  deterministic hinge position targets. This is intentionally simpler than the
  six-panel and thruster-arm paper examples.
- `two_arm_free_drift.xml`: one free hub with two unlimited passive hinged
  arms, no actuators, and one-orbit drift at matched `0.1 s`
  MuJoCo/orbit/Basilisk timesteps. The arms start aligned by default; the
  direct Basilisk leg is run with RKF45 by default.

## Running

From the repository root:

```bash
pixi run compare-basilisk-single
pixi run compare-basilisk-hinges
pixi run compare-basilisk
pixi run compare-basilisk-integrators
pixi run compare-basilisk-matched-dt
pixi run compare-basilisk-two-arm
pixi run diagnose-basilisk-two-arm
pixi run compare-two-arm-eci-frame
pixi run benchmark-basilisk-rollouts
```

The scripts always run the `mujoco_orbit` leg and write summaries/samples to
`comparisons/basilisk_mujoco/out/`. That directory is ignored by git.

Basilisk is not a default dependency. If `Basilisk.simulation.mujoco` is not
installed, the direct Basilisk leg is reported as skipped in the JSON summary.
When Basilisk with MuJoCo support is installed, the scripts also attempt direct
`MJScene` + `NBodyGravity` runs using the same MJCF assets and add the sampled
Basilisk inertial trajectories and joint states to the `.npz` outputs.

The direct Basilisk runners accept `--basilisk-integrator` with `euler`, `rk2`,
`rk4`, `rkf45`, or `rkf78`. `compare-basilisk-integrators` sweeps the
single-body case over Euler and RKF45 by default. `benchmark-basilisk-rollouts`
compares `mujoco_orbit.rollout(..., nthread=...)` against independent Basilisk
`MJScene` simulations launched through a Python process pool; that is not a
shared-model batch rollout API, so the summary JSON records this caveat.

`compare-basilisk-matched-dt` is the fairer ECI-frame integrator stress test:
`mujoco_orbit` uses `mj_timestep=0.1 s` and `orbit_dt=0.1 s`, while Basilisk
uses a `0.1 s` task period. It runs Basilisk Euler and RKF45 and reports both
ECI position error and attitude error for a torque-free spinning body.

`compare-basilisk-two-arm` extends that matched-step setup to a passive
multibody satellite. It records hub attitude, hub/body-origin ECI positions,
hinge angles/rates, and system COM, then compares the same `mujoco_orbit`
trajectory against Basilisk RKF45. `--mj-integrator RK4` can be used to test
whether MuJoCo's internal integrator is the mismatch source.

`diagnose-basilisk-two-arm` runs shorter isolating experiments for the passive
joint mismatch: plain MuJoCo versus Basilisk without gravity/orbit coupling,
`mujoco_orbit` Euler versus RK4 against Basilisk RKF45, and a snapshot check of
the analytic tidal wrench applied by `mujoco_orbit`.

`compare-two-arm-eci-frame` removes Basilisk entirely. It runs the two-arm
passive drift once in raw MuJoCo absolute ECI coordinates and once through
`mujoco_orbit`'s chief-centered ECI-aligned world. The default raw ECI leg uses
a MuJoCo passive callback so point-mass gravity is refreshed inside the MuJoCo
dynamics evaluation; `--gravity-application xfrc` is also available to mirror
the original `experiments/frame_study` applied-force loop exactly. Matching
trajectories in callback mode is the frame-conversion falsification test, while
the `xfrc` mode is useful for exposing stale-force integration error.

## Next Milestones

1. Tighten the articulated actuator path so Basilisk and direct MuJoCo use the
   same hinge-control semantics. 
2. Reproduce the paper's reaction-wheel attitude case with explicit wheel bodies.
3. Extend the hinged satellite into staged branching panel deployment with locks.
4. Add the thruster-arm control case once the simpler articulated parity cases
   agree.
 