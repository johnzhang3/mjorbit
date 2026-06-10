# mujoco_orbit

MuJoCo-style simulator for coupled orbital dynamics and MuJoCo multibody dynamics.
`mujoco_orbit` is the stable CPU reference backend, and `mujoco_orbit_warp` is the
GPU-targeting MJWarp backend.

## Public API

The supported public surface is built around `MjoModel` and `MjoData`. Backend choice is
explicit by import path, and `model.make_data(...)` is the preferred way to construct runtime
state.

```python
from mujoco_orbit import MjoModel, OrbitInit, mjo_forward, mjo_step

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
from mujoco_orbit_warp import MjoModel, OrbitInit, mjo_step

model = MjoModel.from_xml_path("model.xml")
data = model.make_data(
    orbit=OrbitInit(R_eci=[7000.0, 0.0, 0.0], V_eci=[0.0, 7.5, 0.0]),
    nworld=256,
)

mjo_step(model, data)
```

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

- `src/mujoco_orbit/` — CPU reference backend
- `src/mujoco_orbit/config.py` — public specs
- `src/mujoco_orbit/model.py`, `data.py`, `step.py`, `rollout.py` — public runtime API
- `src/cpp/` — native MuJoCo plugin and C++ orbit implementation
- `tests/mujoco_orbit/reference/orbit/` — Python reference/analysis orbit helpers
- `tests/mujoco_orbit/reference/coupling/` — Python reference/analysis coupling helpers
- `tests/mujoco_orbit/reference/sensors.py` — Python reference sensor helpers
- `src/mujoco_orbit/testdata/` — bundled XML assets
- `src/mujoco_orbit_warp/` — optional MJWarp backend, host/device sync, and batched runtime API
- `src/viewer/` — browser viewer integration
- `examples/` — lightweight demos of the public API
- `ISS/` — separate homework/report analysis workspace and artifacts
- `tests/mujoco_orbit/` — unit and integration tests
- `tests/mujoco_orbit_warp/` — guarded MJWarp tests

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
pixi run example-free-drift
pixi run example-mppi-arm-reach
pixi run example-mppi-capture
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
  - CPU: `mujoco_orbit`
  - MJWarp: `mujoco_orbit_warp`
- Static metadata belongs on `MjoModel`; per-run state belongs on `MjoData`.
- Prefer `model.make_data(...)` for runtime construction. Warp users specify batched parallel
  simulation count with `nworld`.
- After directly mutating `data.qpos`, `data.qvel`, `data.orbit`, or actuator commands, call
  `mjo_forward(model, data)`.
- Advance the simulation with `mjo_step(model, data)`. MuJoCo controls come from `data.ctrl`,
  and orbital actuators come from `data.actuators.*_cmd`.
- `data.sensordata` is the canonical forward/step-updated sensor buffer. Stochastic sampling
  lives behind `data.sensors`.
- Keep warp-specific implementation under `src/mujoco_orbit_warp/`. Preserve MuJoCo frame
  conventions exactly across both backends.
- Keep lightweight demos in `examples/`. Keep ISS homework/report analysis and generated
  artifacts in `ISS/`. Viewer integration and the `ISS/` workspace remain CPU-only for now.
