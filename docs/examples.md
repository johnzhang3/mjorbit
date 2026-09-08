# Examples and paper reproduction

Run commands from the repository root after `pixi install`. The examples are
source files; they are included in the source distribution but are not Python
runtime packages in the wheel.

| Goal | Command | Requirements |
| --- | --- | --- |
| First simulation | `pixi run example-minimal` | CPU; no browser |
| Analytical free drift comparison | `pixi run example-free-drift` | CPU; no browser |
| Interactive free drift | `pixi run viewer` | CPU and WebGL browser |
| MPPI arm reach | `pixi run example-mppi-arm-reach --nthread 2` | CPU |
| Dual-arm reorientation | `pixi run example-reorient` | CPU; longer run |
| ISS–Soyuz docking | `pixi run example-docking` | CPU and source mesh assets |
| Capture and stabilize | `pixi run example-mppi-capture` | CPU; long planning horizons |
| Batched simulation | `pixi run -e warp example-batched` | Linux/NVIDIA GPU |
| PPO training | `pixi run -e rl ppo-truss-train` | Linux/NVIDIA GPU; training time |

Run a scenario with `--help` to inspect its arguments. The
[source example catalog](https://github.com/johnzhang3/mjorbit/blob/main/examples/README.md)
links the dedicated MPPI and PPO guides.

## Paper artifacts

`experiments/` contains the frame-precision study, Basilisk comparison,
cross-track collision analysis, and controller-fidelity figure generators.
Use the `report` environment for plotting. Basilisk and RL comparisons have
additional external dependencies or checkpoints described in their guides.

The [reproduction index](https://github.com/johnzhang3/mjorbit/blob/main/experiments/README.md)
lists commands, output locations, and requirements. Generated trajectories,
training logs, and local output directories are ignored by Git; deliberately
maintained reference figures remain tracked.
