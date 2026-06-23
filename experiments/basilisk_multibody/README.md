# Basilisk Multibody Experiment

This experiment is a paper-facing validation and throughput study for the bimanual
spacecraft model used in the example figure. It writes generated XML, JSON summaries, and
optional plots under `experiments/basilisk_multibody/out/`.

Run from the repository root:

```bash
pixi run basilisk-multibody-experiment
```

The script always runs the `mjorbit` CPU leg. It runs the Basilisk comparison only when
`Basilisk.simulation.mujoco` is installed, and it runs the GPU leg only when the Warp
environment is available. The `report` environment includes matplotlib for plot generation,
while the `warp` environment enables the GPU leg:

```bash
pixi run -e report python experiments/basilisk_multibody/run.py --no-warp
pixi run -e warp python experiments/basilisk_multibody/run.py --no-figure
```

The generated XML removes joint limits from the bimanual model and disables the explicit
rigid-body gravity-gradient torque in `mjorbit`, matching Basilisk's point-mass
`NBodyGravity` applied to each articulated body.
