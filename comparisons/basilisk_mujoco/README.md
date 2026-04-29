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

## Running

From the repository root:

```bash
pixi run compare-basilisk-single
pixi run compare-basilisk-hinges
pixi run compare-basilisk
```

The scripts always run the `mujoco_orbit` leg and write summaries/samples to
`comparisons/basilisk_mujoco/out/`. That directory is ignored by git.

Basilisk is not a default dependency. If `Basilisk.simulation.mujoco` is not
installed, the direct Basilisk leg is reported as skipped in the JSON summary.
When Basilisk with MuJoCo support is installed, the scripts also attempt direct
`MJScene` + `NBodyGravity` runs using the same MJCF assets and add the sampled
Basilisk inertial trajectories and joint states to the `.npz` outputs.

## Next Milestones

1. Tighten the articulated actuator path so Basilisk and direct MuJoCo use the
   same hinge-control semantics.
2. Reproduce the paper's reaction-wheel attitude case with explicit wheel bodies.
3. Extend the hinged satellite into staged branching panel deployment with locks.
4. Add the thruster-arm control case once the simpler articulated parity cases
   agree.
