# mujoco_orbit

MuJoCo-style simulator for coupled orbital dynamics and MuJoCo multibody dynamics.

## Public API

The supported public surface is built around `MjoModel` and `MjoData`.

```python
from mujoco_orbit import MjoData, MjoModel, OrbitInit, mjo_forward, mjo_step

model = MjoModel.from_xml_path("model.xml")
data = MjoData(
    model,
    orbit=OrbitInit(R_eci=[7000.0, 0.0, 0.0], V_eci=[0.0, 7.5, 0.0]),
)

mjo_forward(model, data)  # call after direct state edits
mjo_step(model, data)     # advance one step
```

`MjoModel` owns compiled/static state. `MjoData` owns runtime state, orbit state, actuator
commands, caches, and sensor runtime state.

## Project Layout

- `src/mujoco_orbit/` — main package
- `src/mujoco_orbit/core/` — public specs, model/data wrappers, stepping
- `src/mujoco_orbit/orbit/` — orbital propagation, gravity, LVLH, environment
- `src/mujoco_orbit/coupling/` — external wrench assembly and actuator/environment coupling
- `src/mujoco_orbit/sensors.py` — sensor catalogs, callback plumbing, measurement helpers
- `src/mujoco_orbit/testdata/` — bundled XML assets
- `src/viewer/` — optional viewer integration
- `examples/` — lightweight demos of the public API
- `ISS/` — separate homework/report analysis workspace and artifacts
- `tests/mujoco_orbit/` — unit and integration tests

## Development

```bash
uv sync --dev
uv sync --dev --extra report
uv sync --dev --extra viewer
uv run pytest -q
uv run ruff check .
uv run pyright
```

Useful entrypoints:

```bash
uv run python examples/free_drift.py
uv run python examples/arm_reach.py
uv run python ISS/hw2/spacecraft_dynamics.py
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
- Static metadata belongs on `MjoModel`; per-run state belongs on `MjoData`.
- After directly mutating `data.qpos`, `data.qvel`, `data.orbit`, or actuator commands, call
  `mjo_forward(model, data)`.
- Advance the simulation with `mjo_step(model, data)`. MuJoCo controls come from `data.ctrl`,
  and orbital actuators come from `data.actuators.*_cmd`.
- `data.sensordata` is the canonical forward/step-updated sensor buffer. Stochastic sampling
  lives behind `data.sensors`.
- Keep lightweight demos in `examples/`. Keep ISS homework/report analysis and generated
  artifacts in `ISS/`.
