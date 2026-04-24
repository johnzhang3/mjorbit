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
reference, relative-motion radius, and attitude invariant drift.
