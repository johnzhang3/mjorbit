# Batched GPU simulation

The optional `mjorbit_warp` backend uses MuJoCo Warp. The supported Pixi GPU
workflow targets Linux x86-64 with an NVIDIA GPU. The CPU package remains
available in the same environment; backend selection is explicit by import.

```bash
pixi install -e warp
pixi run -e warp example-batched
```

## Complete example

```{literalinclude} ../examples/batched.py
:language: python
```

This runs 256 copies of the bundled free-body model with different initial
offsets. Initial kernel compilation can take longer than later runs.

## Shapes and synchronization

Use `model.make_data(orbit=..., nworld=N)`. With `N > 1`, host state arrays
have a leading world dimension: `qpos` is `(N, nq)`, `qvel` is `(N, nv)`, and
`ctrl` is `(N, nu)`. With `nworld=1`, host arrays use the unbatched shape.

The device state is authoritative. The public NumPy arrays are host mirrors:

1. After editing host `qpos`, `qvel`, `ctrl`, or orbit, call `mjo_upload` with
   the fields to transfer.
2. Call `mjo_forward` after changing initial conditions, then `mjo_step` in
   the simulation loop.
3. Call `mjo_pull` before reading current state on the host.

Spacecraft command buffers (`rw_torque_cmd`, `mtq_dipole_cmd`, and
`thr_force_cmd`) are uploaded automatically by ordinary stepping. CUDA graph
capture uses device-side control; do not assume a host edit will be captured.
For simple host-driven code, `mjo_step(model, data, sync=True)` uploads and
pulls the full supported state each step, at an additional transfer cost.

## Precision and current scope

MuJoCo Warp's multibody state, device chief position/velocity, and reference
orbit propagation use single precision. The absolute epoch anchor, solar and
magnetic angle accumulation, and the cancellation-sensitive J2 difference use
double precision. The chief-centered formulation keeps robot coordinates near
the origin. CPU/GPU results should be compared with precision-appropriate
tolerances, especially around contacts and eclipse boundaries.

The gravity kernels currently use fixed Earth constants. Use the CPU backend
for custom central-body gravity; supplying different `gm`, `j2`, or gravity
radius values does not change the GPU gravity kernels. The environment kernels
do consume the configured atmosphere, magnetic, and eclipse parameters. See
[forces and disturbances](forces-disturbances.md#inspecting-loads-and-backend-scope)
for model equations and the shared articulated-body drag limitation.

Reaction wheels, thrusters, and magnetorquers are supported. CMGs raise
`NotImplementedError` during GPU model construction. Native sensor data is
available; the CPU noisy sensor namespace is not implemented on GPU. The
standard task viewer and `mjorbit.planning` planner use the CPU backend.

GPU/RL demonstrations have their own entry points:

```bash
pixi run -e warp banner-viewer-gpu
pixi run -e rl ppo-truss-train --help
```

See the source [PPO guide](https://github.com/johnzhang3/mjorbit/tree/main/examples/ppo)
for training and checkpoint playback. A trained policy is not bundled.
