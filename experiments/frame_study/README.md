# Frame Study

Minimal raw-MuJoCo experiment for comparing orbital frame choices.

This experiment intentionally does not import `mujoco_orbit`. It defines a tiny
two-body propagator and a one-body MuJoCo XML string locally, then compares:

- `eci`: absolute Earth-centered inertial MuJoCo coordinates.
- `chief_inertial`: chief-centered coordinates with inertially fixed axes.
- `lvlh`: chief-centered rotating LVLH coordinates.

The default sweep runs four scenarios:

- Circular equatorial LEO.
- Circular inclined LEO.
- Elliptical equatorial orbit.
- Elliptical inclined orbit.

Each scenario starts from the same small LVLH relative offset with torque-free
attitude and runs for three chief orbits.

The orbit/reference propagation and MuJoCo integration are intentionally
decoupled:

- `ORBIT_TIMESTEP` is the coarse RK4 chief/reference orbit step.
- `MUJOCO_TIMESTEP` is the MuJoCo free-body/contact step.

The default uses a `0.5 s` orbit step and a `0.25 s` MuJoCo step, so each orbit
step contains two MuJoCo substeps. Lowering `MUJOCO_TIMESTEP` increases the
local dynamics frequency without changing the orbit propagation step.

Run:

```bash
pixi run python experiments/frame_study/run.py
```

The table reports ECI position error against an independent RK4 two-body
reference, relative-motion radius, orbital invariant drift, and attitude
invariant drift.

## RK4 Force Evaluation

The experiment intentionally uses only MuJoCo RK4 for the free body. All frame
accelerations are applied through MuJoCo's passive callback, so the force is
re-evaluated inside RK4's dynamics evaluations instead of being held constant
for an entire step.

To run a smaller reproduction:

```bash
pixi run python experiments/frame_study/run.py --scenario circular_equatorial
```

To generate the one-orbit ECI/local-chief integrator comparison:

```bash
pixi run python experiments/frame_study/run.py --integrator-study
```

This uses one shared timestep for MuJoCo, chief propagation, and the RK4 ECI
reference. It writes position-error and energy-error plots plus raw `.npz`
samples under `experiments/frame_study/out/` for:

- `ECI + Euler`
- `ECI + RK4`
- `ECI + implicit`
- `ECI + implicitfast`
- `local chief + Euler`
- `local chief + RK4`
- `local chief + implicit`
- `local chief + implicitfast`

ECI curves are plotted with solid lines and local-chief curves with dashed
lines. Use `--study-precision float32` to stress state-storage conditioning by
rounding MuJoCo state and callback force inputs to float32; MuJoCo itself still
uses the precision compiled into the installed wheel.

Use `--study-orbits` to extend the same comparison, for example:

```bash
pixi run python experiments/frame_study/run.py --integrator-study --study-orbits 3
```

Use `--study-rel-vel-lvlh VX VY VZ` to add a small initial LVLH relative
velocity in m/s, for example a 1 cm/s radial perturbation:

```bash
pixi run python experiments/frame_study/run.py \
  --integrator-study \
  --study-orbits 3 \
  --study-rel-vel-lvlh 0.01 0 0
```

Use `--study-length-unit-m` to rescale the coordinate length unit used by the
study propagator and MuJoCo free joint while keeping plotted errors in meters.
For example, `--study-length-unit-m 1000` propagates in km and
`--study-length-unit-m 100000` propagates in 100 km units.

The chief-centered frames avoid integrating the large two-body central motion
inside MuJoCo. They integrate only meter-scale relative motion and small tidal
accelerations, while the chief/reference orbit is advanced by the separate RK4
propagator.
