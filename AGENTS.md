# Repository Guidelines

## Project Structure & Module Organization
`src/mujoco_orbit` contains the default simulation package. Core code lives under `src/mujoco_orbit/`: `core/` for scenario setup and stepping, `orbit/` for orbital math and environment models, `coupling/` for forces and actuators, and `testdata/` for bundled XML assets used by examples and tests. Optional browser visualization lives in `src/viewer/`. Keep runnable demos in `examples/` and mirror production modules with tests in `tests/mujoco_orbit/`.

## Build, Test, and Development Commands
Use `uv` for environment and command execution.

- `uv sync --dev`: install the package plus test, lint, and type-check tools.
- `uv sync --dev --extra viewer`: add optional viewer dependencies (`viser`, `trimesh`).
- `uv run pytest -q`: run the full test suite.
- `uv run pytest tests/mujoco_orbit/test_compile.py -q`: run one focused test file while iterating.
- `uv run ruff check .`: run linting and import-order checks.
- `uv run pyright`: run static type checks.
- `uv run python examples/free_drift.py`: execute a focused example.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, explicit type hints, and small focused modules. Ruff enforces a 100-character line length and import sorting; fix import order before opening a PR. Use `snake_case` for functions and modules, `PascalCase` for dataclasses/classes, and descriptive config suffixes such as `OrbitCfg` or `ScenarioCfg`. Test helpers are typically private functions prefixed with `_`.

## Testing Guidelines
Add tests in `tests/mujoco_orbit/test_*.py` beside the subsystem you change. Prefer deterministic numeric assertions with `numpy.testing` or `pytest.approx`, and cover both nominal paths and validation errors. Run `uv run pytest -q` before submitting; if you touch viewer code, also run the relevant example script manually.

## Commit & Pull Request Guidelines
The Git history is still sparse, so use short imperative commit subjects such as `Add surface load regression test`. Keep commits scoped to one change. PRs should include a concise summary, the commands you ran, and any remaining issues. Link the related issue when available, and include screenshots or a short recording for changes under `src/viewer/` or `examples/*viewer.py`.

## Environment Notes
Target Python `>=3.11,<3.13` as defined in `pyproject.toml`. MuJoCo is a required runtime dependency; viewer support is optional. Do not commit generated caches such as `__pycache__/`, `.pytest_cache/`, or `.ruff_cache/`.

## MuJoCo frame conventions

These conventions are critical for correctness — getting them wrong causes silent energy/momentum non-conservation.

- **`qvel[3:6]`** for a free joint is angular velocity in the **body frame** (child frame), NOT the world frame.
- **`xmat`** is the body-frame orientation matrix (world-from-body). Use this for body-frame ↔ world-frame transforms.
- **`ximat`** is the **inertia-frame** orientation matrix (world-from-principal-axes). This equals `xmat` only when `body_iquat` is identity (i.e., diagonal inertia aligned with body axes). **Do NOT use `ximat` as a body rotation matrix.**
- **`xfrc_applied`** torques are in the **world frame**. To apply a body-frame torque: `xfrc[bid, 3:] = xmat @ tau_body`.
- **`body_iquat`**: rotation from body frame to inertia (principal axes) frame. When inertia is diagonal in body frame, this is identity.
- **`fullinertia`** XML attribute: MuJoCo internally eigendecomposes into `body_inertia` (diagonal, possibly reordered) + `body_iquat` (non-trivial rotation). The full tensor is preserved. To set non-diagonal inertia, use `fullinertia` in the XML — do NOT modify `body_inertia`/`body_iquat` post-compilation (changes may not take effect on the dynamics).