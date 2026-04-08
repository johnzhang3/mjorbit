# Repository Guidelines

## Project Structure & Module Organization
`src/mujoco_orbit/` is the main package and now exposes a MuJoCo-style model/data API.

- `src/mujoco_orbit/core/config.py` defines public `OrbitInit` and `*Spec` dataclasses.
- `src/mujoco_orbit/core/runtime.py` defines `MjoModel` and `MjoData`.
- `src/mujoco_orbit/core/step.py` defines `mjo_forward` and `mjo_step`.
- `src/mujoco_orbit/core/actuators.py` holds runtime actuator-state containers.
- `src/mujoco_orbit/orbit/` contains propagation, gravity, LVLH, and environment models.
- `src/mujoco_orbit/coupling/` assembles external wrenches from environment and actuators.
- `src/mujoco_orbit/sensors.py` contains sensor catalogs, callbacks, and measurement helpers.
- `src/mujoco_orbit/testdata/` contains XML fixtures used by tests and examples.
- `src/viewer/` contains the browser viewer integration.
- `examples/` contains small runnable demos of the public API.
- `ISS/` is a separate top-level workspace for homework analysis, plots, and report material.
- `tests/mujoco_orbit/` mirrors the production package; shared setup lives in `_helpers.py`.

## Build, Test, and Development Commands
Use `uv` for environment management and command execution.

- `uv sync --dev`: install the package plus test, lint, and type-check tools.
- `uv sync --dev --extra report`: add report/analysis dependencies for `ISS/`.
- `uv run pytest -q`: run the full test suite.
- `uv run pytest tests/mujoco_orbit/test_api_model_data.py -q`: run the public API tests.
- `uv run ruff check .`: run linting and import-order checks.
- `uv run pyright`: run static type checks.
- `uv run python examples/free_drift.py`: run a minimal API example.
- `uv run python ISS/hw2/spacecraft_dynamics.py`: run an ISS analysis script.

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
`uv run pytest -q` before submitting. If you touch `ISS/` analysis scripts, run the affected
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

## MuJoCo Frame Conventions

These conventions are critical for correctness. Getting them wrong causes silent
energy/momentum non-conservation.

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
