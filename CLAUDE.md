# mjorbit

MuJoCo-style simulator for coupled orbital dynamics and MuJoCo multibody dynamics.
`mjorbit` is the stable CPU reference backend, and `mjorbit_warp` is the
GPU-targeting MJWarp backend.

## Public API

The supported public surface is built around `MjoModel` and `MjoData`. Backend choice is
explicit by import path, and `model.make_data(...)` is the preferred way to construct runtime
state.

```python
from mjorbit import MjoModel, OrbitInit, mjo_forward, mjo_step

model = MjoModel.from_xml_path("model.xml")
data = model.make_data(
    orbit=OrbitInit(R_eci=[7000.0, 0.0, 0.0], V_eci=[0.0, 7.5, 0.0]),
)

mjo_forward(model, data)  # call after direct state edits
mjo_step(model, data)     # advance one step
```

The warp backend uses the same API names, plus `nworld` on `make_data(...)` for batched GPU
simulation:

```python
from mjorbit_warp import MjoModel, OrbitInit, mjo_step

model = MjoModel.from_xml_path("model.xml")
data = model.make_data(
    orbit=OrbitInit(R_eci=[7000.0, 0.0, 0.0], V_eci=[0.0, 7.5, 0.0]),
    nworld=256,
)

data.actuators.mtq_dipole_cmd[:, 0] = 5.0  # actuator commands auto-sync each step
mjo_step(model, data)
```

On the warp backend the **device state is authoritative**. `mjo_step`/`mjo_forward`
auto-sync only the cheap actuator *command* inputs (`rw_torque_cmd`, `mtq_dipole_cmd`,
`thr_force_cmd`) from the public buffers, so `set cmd; mjo_step` produces torque exactly
as on the CPU backend. Edits to larger or device-integrated buffers (`qpos`, `qvel`,
`ctrl`, `orbit`, `rw_speed`) are **not** pushed by default — call `mjo_upload(model, data,
fields=...)` (or `mjo_step(model, data, sync=True)`) after editing those, and `mjo_pull`
before reading public buffers or host `MjData` back from the device.

`MjoModel` owns compiled/static state. `MjoData` owns runtime state, orbit state, actuator
commands, caches, and sensor runtime state. On the CPU backend,
`MjoData(model, orbit=...)` still works for compatibility, but prefer `model.make_data(...)`
for new code.

`OrbitInit` and `data.orbit` store the chief/reference orbit in absolute ECI coordinates
(`R_eci`, `V_eci`). MuJoCo `world` coordinates are different: they are chief-centered local
inertial offsets in SI units, with axes parallel to ECI. Root free-joint `qpos`/`qvel`,
`xpos`, `xmat`, `cvel`, and `xfrc_applied` should be interpreted in that local world frame,
not as absolute ECI state and not as LVLH state.

## Project Layout

- `src/mjorbit/` — CPU reference backend
- `src/mjorbit/config.py` — public specs
- `src/mjorbit/model.py`, `data.py`, `step.py`, `rollout.py` — public runtime API
- `src/mjorbit/planning/` — spline-knot MPPI planner on the batched rollout
- `src/cpp/` — native MuJoCo plugin and C++ orbit implementation
- `tests/mjorbit/reference/orbit/` — Python reference/analysis orbit helpers
- `tests/mjorbit/reference/coupling/` — Python reference/analysis coupling helpers
- `tests/mjorbit/reference/sensors.py` — Python reference sensor helpers
- `src/mjorbit/testdata/` — bundled XML assets
- `src/mjorbit_warp/` — optional MJWarp backend, host/device sync, and batched runtime API
- `src/viewer/` — browser viewer integration (`mjo-viewer` task app, framing, textured Earth)
- `src/viewer/tasks/` — viewer task registry and built-in tasks
- `examples/` — lightweight demos of the public API
- `ISS/` — separate homework/report analysis workspace and artifacts
- `tests/mjorbit/` — unit and integration tests
- `tests/mjorbit_warp/` — guarded MJWarp tests

## Development

```bash
pixi install
pixi install -e report
pixi install -e warp
pixi run test
pixi run test-warp
pixi run lint
pixi run typecheck
pixi run cpp-test
```

Useful entrypoints:

```bash
pixi run viewer            # interactive task viewer (judo-style), or: mjo-viewer
pixi run example-free-drift
pixi run example-mppi-arm-reach
pixi run example-mppi-capture
pixi run example-reorient            # dual-arm 3-DOF attitude slew by reaction (headless)
pixi run example-reorient-viewer     # same, live in the browser viewer
pixi run python examples/arm_reach.py
pixi run iss-hw2
```

## Units

All physics quantities use km, s, kg, rad, T unless stated otherwise.

- distances: km
- velocities: km/s
- accelerations: km/s²
- forces: kg·km/s² (= kN)
- torques: kg·km²/s²
- magnetic field: T

MuJoCo uses SI (m, s, kg) internally. Conversions happen at the MuJoCo boundary.

## MuJoCo Frame Conventions

These conventions are critical for correctness. Getting them wrong causes silent
energy/momentum non-conservation.

- **MuJoCo `world`** is the chief-centered local inertial frame. The origin follows the
  chief/reference orbit, and the axes are parallel to ECI. Do not add `data.orbit.R_eci` or
  `data.orbit.V_eci` into MuJoCo `qpos`/`qvel` or XML free-joint initial conditions.
- **Absolute ECI** lives in `data.orbit`, environment caches, and helper methods whose names
  explicitly include `eci`. Use `world_*` helpers for MuJoCo state and `eci_*` helpers only
  when an absolute inertial quantity is intended.
- **LVLH** is a derived rotating frame in `data.frame`. Convert through the runtime helpers
  (`world_position_from_lvlh`, `lvlh_position_from_world`, etc.) rather than treating
  MuJoCo `world` axes as LVLH axes.
- **`qvel[3:6]`** for a free joint is angular velocity in the **body frame** (child frame),
  not the world frame.
- **`xmat`** is the body-frame orientation matrix (world-from-body). Use this for body-frame
  and world-frame transforms.
- **`ximat`** is the **inertia-frame** orientation matrix (world-from-principal-axes). This
  equals `xmat` only when `body_iquat` is identity. Do not use `ximat` as a body rotation
  matrix.
- **`xfrc_applied`** torques are in the **world frame**. To apply a body-frame torque, use
  `xfrc[bid, 3:] = xmat @ tau_body`.
- **`body_iquat`** is the rotation from body frame to inertia frame.
- **`fullinertia`** should be set in XML for non-diagonal inertia. Do not patch
  `body_inertia` or `body_iquat` post-compilation and expect dynamics to change correctly.

## Key Design Rules

- The public interface is `MjoModel` / `MjoData`; do not rebuild the old scenario wrapper.
- Choose the backend explicitly by import path:
  - CPU: `mjorbit`
  - MJWarp: `mjorbit_warp`
- Static metadata belongs on `MjoModel`; per-run state belongs on `MjoData`.
- Prefer `model.make_data(...)` for runtime construction. Warp users specify batched parallel
  simulation count with `nworld`.
- After directly mutating `data.qpos`, `data.qvel`, `data.orbit`, or actuator commands, call
  `mjo_forward(model, data)`.
- Advance the simulation with `mjo_step(model, data)`. MuJoCo controls come from `data.ctrl`,
  and orbital actuators come from `data.actuators.*_cmd`.
- The default MuJoCo-side integrator is `implicitfast`, not MuJoCo's stock semi-implicit
  Euler: model compilation upgrades an unspecified (or explicitly Euler) `opt.integrator` to
  `implicitfast`, because Euler does not conserve angular momentum for tumbling/asymmetric
  bodies (issue #12). For the orbit-following frame's position-only forces `implicitfast`
  matches semi-implicit Euler, so translational behavior is unchanged. Explicit `implicit` /
  `implicitfast` / `RK4` choices are preserved; set `model.opt.integrator` to override.
- On the warp backend the device is authoritative: `mjo_step`/`mjo_forward` auto-sync the
  actuator command inputs (`*_cmd`) but not `qpos`/`qvel`/`ctrl`/`orbit`/`rw_speed`. After
  editing those, call `mjo_upload(...)` (or pass `sync=True`); call `mjo_pull(...)` before
  reading public buffers back. The CPU backend has no host/device split, so all edits take
  effect directly.
- `data.sensordata` is the canonical forward/step-updated sensor buffer. Stochastic sampling
  lives behind `data.sensors`.
- Keep warp-specific implementation under `src/mjorbit_warp/`. Preserve MuJoCo frame
  conventions exactly across both backends. The warp device core is float32, including the
  orbit clock (`orbit_t`): per-step rounding grows past ~1 h of sim time, so time-keyed
  environment models (sun vector, eclipse, magnetic field) lose timing accuracy on very long
  device-resident runs (~tens of seconds per sim-hour, worst case).
- Keep lightweight demos in `examples/`. Keep ISS homework/report analysis and generated
  artifacts in `ISS/`. The `src/viewer/` module itself stays backend-agnostic (no warp
  imports); warp-driven visualization lives in examples (e.g.
  `examples/banner_viewer_gpu.py` feeds the instanced fleet renderer from
  `mjorbit_warp`). The `ISS/` workspace remains CPU-only.
