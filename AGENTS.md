# Repository Guidelines

## Project Structure & Module Organization
`src/mujoco_orbit/` is the main package and now exposes a MuJoCo-style model/data API.

- `src/mujoco_orbit/core/config.py` defines public `OrbitInit` and `*Spec` dataclasses.
- `src/mujoco_orbit/core/runtime.py` defines `MjoModel` and `MjoData`.
- `src/mujoco_orbit/core/step.py` defines `mjo_forward` and `mjo_step`.
- `src/mujoco_orbit/core/rollout.py` defines state and rollout helpers.
- `src/cpp/` contains the native MuJoCo plugin and C++ orbit implementation.
- `tests/mujoco_orbit/reference/orbit/` contains Python reference/analysis orbit helpers.
- `tests/mujoco_orbit/reference/coupling/` contains Python reference/analysis coupling helpers.
- `tests/mujoco_orbit/reference/sensors.py` contains Python reference sensor helpers.
- `src/mujoco_orbit/testdata/` contains XML fixtures used by tests and examples.
- `src/viewer/` contains the browser viewer integration.
- `examples/` contains small runnable demos of the public API.
- `ISS/` is a separate top-level workspace for homework analysis, plots, and report material.
- `tests/mujoco_orbit/` mirrors the production package; shared setup lives in `_helpers.py`.

## Build, Test, and Development Commands
Use `pixi` for environment management and command execution.

- `pixi install`: install the default Python 3.12 dev environment and editable package.
- `pixi install -e py311`: install the Python 3.11 dev environment.
- `pixi install -e report`: add report/analysis dependencies for `ISS/`.
- `pixi run test`: run the full test suite.
- `pixi run test-api`: run the public API tests.
- `pixi run lint`: run linting and import-order checks.
- `pixi run typecheck`: run static type checks.
- `pixi run cpp-test`: configure, build, and test the native C++ plugin.
- `pixi run example-free-drift`: run a minimal API example.
- `pixi run iss-hw2`: run an ISS analysis script.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, explicit type hints, and small focused
modules. Ruff enforces a 100-character line length and import sorting.

Use `snake_case` for functions and modules, `PascalCase` for classes/dataclasses, `*Spec`
for static compile-time metadata, and `OrbitInit` for orbital initial conditions.

Do not reintroduce the removed compatibility layer. New public code should target
`MjoModel`, `MjoData`, `mjo_forward`, and `mjo_step` directly. Avoid adding new
`Scenario`, `compile`, `step`, or `*Cfg` surfaces unless the user explicitly asks for a new
compatibility wrapper.

## Testing Guidelines
Add tests in `tests/mujoco_orbit/test_*.py` beside the subsystem you change. Prefer
deterministic numeric assertions with `numpy.testing` or `pytest.approx`, and cover both
nominal behavior and validation errors.

If you touch the runtime API, stepping, sensors, or coupling code, update
`tests/mujoco_orbit/test_api_model_data.py` and the relevant subsystem tests. Run
`pixi run test` before submitting. If you touch `ISS/` analysis scripts, run the affected
script directly and keep generated plots or PDFs intentional.

## Commit & Pull Request Guidelines
Use short imperative commit subjects such as `Add reaction wheel saturation test`. Keep
commits scoped to one change. PRs should include a concise summary, the commands you ran,
and any remaining issues.

Link the related issue when available. Include screenshots or short recordings for viewer
changes, and include plot diffs or artifact notes when the change intentionally updates
material under `ISS/`.

## Environment Notes
Target Python `>=3.11,<3.13` as defined in `pyproject.toml`. MuJoCo and viewer support are
required runtime dependencies. Report tooling remains an optional extra.

Do not commit generated caches such as `__pycache__/`, `.pytest_cache/`, or `.ruff_cache/`.
Treat `ISS/` plots, PDFs, and saved data as intentional analysis artifacts rather than
incidental byproducts.

## MuJoCo / Orbit Frame Conventions

These conventions are critical for correctness. Getting them wrong causes silent
energy/momentum non-conservation.

- **`OrbitInit` / `data.orbit`** store the chief/reference orbit in absolute ECI
  coordinates (`R_eci`, `V_eci`) using km and km/s.
- **MuJoCo `world`** is the chief-centered local inertial frame in SI units, with origin at
  the chief and axes parallel to ECI. It is not absolute ECI and not LVLH.
- **Root free-joint `qpos`/`qvel`** are local inertial offsets from the chief. Do not add
  `data.orbit.R_eci` or `data.orbit.V_eci` to XML free-joint initial conditions or MuJoCo
  state unless you are explicitly converting to absolute ECI via helper methods.
- **LVLH** is a derived rotating frame in `data.frame`. Use `MjoData` conversion helpers for
  LVLH/world/ECI transforms instead of assuming MuJoCo `world` axes are LVLH axes.
- **`qvel[3:6]`** for a free joint is angular velocity in the **body frame** (child frame),
  not the world frame.
- **`xmat`** is the body orientation matrix (world-from-body). Use this for body-frame and
  world-frame transforms.
- **`ximat`** is the **inertia-frame** orientation matrix (world-from-principal-axes). This
  equals `xmat` only when `body_iquat` is identity. Do not use `ximat` as a body rotation
  matrix.
- **`xfrc_applied`** torques are in the **world frame**. To apply a body-frame torque, use
  `xfrc[bid, 3:] = xmat @ tau_body`.
- **`body_iquat`** is the rotation from body frame to inertia (principal axes) frame. When
  inertia is diagonal in body frame, this is identity.
- **`fullinertia`** in XML is the correct way to specify non-diagonal inertia. MuJoCo
  internally decomposes it into `body_inertia` plus `body_iquat`. Do not patch
  `body_inertia` or `body_iquat` after compilation and expect dynamics to update correctly.
