# Example catalog

Run from the repository root after `pixi install`. Begin with the CPU examples;
they need no trained policy or NVIDIA GPU. See the
[quick-start tutorial](../docs/quickstart.md) for the first model and simulation.

## Small examples

| File | Command | What it demonstrates |
| --- | --- | --- |
| [minimal.py](minimal.py) | `pixi run example-minimal` | Complete model/data/step lifecycle; one second of free drift |
| [free_drift.py](free_drift.py) | `pixi run example-free-drift` | 60-second comparison against Clohessy–Wiltshire relative motion |
| [arm_reach.py](arm_reach.py) | `pixi run python examples/arm_reach.py` | Joint motion and floating-base reaction with orbital propagation |
| [collision_viewer.py](collision_viewer.py) | `pixi run python examples/collision_viewer.py` | Interactive contact between two free bodies |
| [batched.py](batched.py) | `pixi run -e warp example-batched` | 256 worlds with explicit device upload/pull; Linux/NVIDIA GPU |

The task app is the easiest interactive entry point:

```bash
pixi run viewer                 # free_drift by default
pixi run viewer --list-tasks
pixi run viewer --task arm_reach_mppi
pixi run viewer --task capture_stabilize_mppi
```

Open the printed local URL. Use Pause, Reset, and the task dropdown to explore.

## Paper scenarios and controllers

| Scenario | Command | Notes |
| --- | --- | --- |
| MPPI arm reach | `pixi run example-mppi-arm-reach --nthread 2` | Shortest controller example; [MPPI guide](mppi/README.md) |
| Dual-arm reorientation | `pixi run example-reorient` | 3-DOF attitude control using internal arm motion |
| Reorientation viewer | `pixi run example-reorient-viewer` | Dedicated browser demo |
| Docking | `pixi run example-docking` | Uses the source-tree ISS and Soyuz meshes; [provenance](docking/assets/README.md) |
| Capture and stabilization | `pixi run example-mppi-capture` | Long-horizon MPPI; allow substantially more time than arm reach |
| PPO truss pointing | `pixi run -e rl ppo-truss-train` | Linux/NVIDIA GPU; [training and playback](ppo/README.md) |
| Batched banner viewer | `pixi run -e warp banner-viewer-gpu` | GPU fleet simulation with a browser view |

Each controller script accepts `--help` for duration, seed, sampling, and output
options. The PPO guide also covers Astrobee grasping. Checkpoints are produced
by training and are not bundled with the project.

The specialized docking scripts and parameter sweep, MPPI fidelity comparison,
and PPO environment definitions remain alongside their scenarios. They support
the paper workflows; use the commands above to get started. Figure generators
are indexed under [experiments](../experiments/README.md), and video production
under [scripts/record](../scripts/record/README.md).
