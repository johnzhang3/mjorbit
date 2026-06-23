# Basilisk Multibody Experiment

This experiment is a paper-facing validation and throughput study for the bimanual
spacecraft model used in the example figure. It writes generated XML, JSON summaries, and
optional plots under `experiments/basilisk_multibody/out/`.

Run from the repository root:

```bash
pixi run basilisk-multibody-experiment
```

## Local Basilisk MuJoCo install

The Linux `bsk` wheel may not include `Basilisk.simulation.mujoco`, so build Basilisk
from source into the Pixi environment that will run the comparison:

```bash
git clone --depth 1 --branch v2.10.2 https://github.com/AVSLab/basilisk.git /tmp/basilisk-src
CONAN_ARGS='--mujoco True --opNav False --vizInterface False --examples False' \
  uv pip install --python .pixi/envs/default/bin/python --force-reinstall /tmp/basilisk-src
```

Repeat the same install with a fresh source checkout for the Warp environment:

```bash
git clone --depth 1 --branch v2.10.2 https://github.com/AVSLab/basilisk.git /tmp/basilisk-src-warp
CONAN_ARGS='--mujoco True --opNav False --vizInterface False --examples False' \
  uv pip install --python .pixi/envs/warp/bin/python --force-reinstall /tmp/basilisk-src-warp
```

When launching the experiment locally, expose the Pixi C++ runtime libraries:

```bash
pixi run -e warp bash -lc 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"; python experiments/basilisk_multibody/run.py --no-figure'
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
