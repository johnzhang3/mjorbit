# Paper reproduction

This directory contains validation studies and figure generators. It stays in
the source repository and source distribution so results remain reproducible;
it is not installed in the runtime wheel. For introductory simulations, start
with [examples](../examples/README.md).

Run commands from the repository root. `pixi install -e report` adds plotting
dependencies. Use the committed lockfile and record the Git revision, command,
seed, hardware, and optional external simulator versions with reproduced results.

| Study | Entry point | Requirements and outputs |
| --- | --- | --- |
| Frame precision and integrators | [frame_study](frame_study/README.md) | Standalone Newton–Euler reference; `report` for plots; `frame_study/out/` |
| Basilisk multibody comparison | [basilisk_multibody](basilisk_multibody/README.md) | CPU leg runs locally; Basilisk MuJoCo source build and GPU leg are optional; `basilisk_multibody/out/` |
| Cross-track collision | [cross_track_collision/run.py](cross_track_collision/run.py) | CPU orbit/contact analysis; `report` for plots; `cross_track_collision/out/` |
| MPPI model fidelity | [mppi_fidelity](mppi_fidelity/README.md) | Multiple CPU controller runs; JSON summaries and TikZ figure in `mppi_fidelity/out/` |

## Entry commands

```bash
# Frame study: a smaller single-scenario run.
pixi run -e report python experiments/frame_study/run.py --scenario circular_equatorial

# Frame paper figure (see --help for sweep parameters).
pixi run -e report python experiments/frame_study/make_paper_figure.py --help

# CPU comparison without requiring optional Basilisk or Warp installations.
pixi run -e report python experiments/basilisk_multibody/run.py --no-warp

# Collision analysis and figure options.
pixi run -e report python experiments/cross_track_collision/run.py --help
pixi run -e report python experiments/cross_track_collision/make_paper_figure.py --help
```

The MPPI fidelity guide provides the multi-seed sequence and figure command.
Generated `out/` directories are ignored. Existing tracked figures under
`cross_track_collision/figures/` are retained as reference artifacts; update
them only when intentionally changing the reference result.

PPO training and checkpoint playback are documented in
[examples/ppo](../examples/ppo/README.md). Trajectory recording and rendering
are documented in [scripts/record](../scripts/record/README.md). These workflows
need additional compute, checkpoints, or rendering tools; they are separate
from the minimal installation smoke test.
