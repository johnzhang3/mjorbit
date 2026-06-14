# MJWarp Backend Phase 1 (`mjorbit_warp`)

## Summary
- Add a new optional backend package at `src/mjorbit_warp` alongside the existing CPU package `src/mjorbit/`.
- Users select the backend explicitly by import path:
  - CPU: `mjorbit`
  - GPU/MJWarp: `mjorbit_warp`
- Standardize runtime creation around `model.make_data(...)` for both backends.
- For the warp backend, `model.make_data(..., nworld=N)` is the public way to request parallel GPU simulations.
- Keep MJWarp transfer primitives internal but central to the implementation:
  - `mjw.put_model(...)`
  - `mjw.make_data(...)`
  - `mjw.put_data(...)`

## Public API Direction
- Keep the core API names consistent across backends:
  - `MjoModel`
  - `MjoData`
  - `mjo_forward`
  - `mjo_step`
  - shared config types like `OrbitInit`, `SurfaceSpec`, `ReactionWheelSpec`, etc.
- Add `MjoModel.make_data(...)` and make it the recommended creation path for both backends.
- CPU backend:
  - keep `MjoData(model, orbit=...)` working for backward compatibility
  - implement `model.make_data(...)` as a thin convenience wrapper over the existing constructor
- Warp backend:
  - use `model.make_data(..., nworld=N)` as the primary runtime construction API
  - allow `nworld=1` by default
  - expose `model.backend == "warp"` and `data.backend == "warp"` for clarity
  - expose `data.nworld` so users can tell when they are running batched GPU simulations

## Backend Internals
- `MjoModel.from_xml_path(...)` in `mjorbit_warp`:
  - compile a host `mujoco.MjModel`
  - resolve ids and static metadata from the host model
  - upload it with `mjw.put_model(host_model)`
  - store both `host_model` and `warp_model` on the wrapper
- `MjoModel.make_data(...)` in `mjorbit_warp`:
  - call `mjw.make_data(host_model, nworld=nworld)` for fresh device allocations
  - create the wrapper `MjoData` object around that device state
- `mjw.put_data(...)` role:
  - internal tool for state seeding, host-to-device restore, and future CPU↔GPU handoff
  - not the primary user-facing creation path
- Wrapper-owned state:
  - orbit propagation state
  - frame/environment caches
  - actuator bookkeeping
  - sensor noise/bias state
  - host-visible numpy buffers that mirror the public API
- Synchronization model:
  - direct edits happen on wrapper-managed host-facing buffers
  - `mjo_forward` / `mjo_step` push those edits to device state
  - after execution, the wrapper refreshes public observables back from device state

## Compatibility Findings To Design Around
- MJWarp supports the main MuJoCo engine features we currently rely on:
  - free joints
  - hinge joints
  - MuJoCo position actuators
  - contact dynamics
  - `xfrc_applied`
  - `xmat`, `ximat`, `site_xmat`
  - `body_iquat`, `fullinertia`
  - built-in gyro / accelerometer / magnetometer sensors
  - `framequat`
- Wrapper-owned compatibility work is still required for:
  - named metadata access, because we should keep the host model for name/id lookup and sensor metadata
  - `sensor_user` / `nuser_sensor`, which our CPU sensor catalog currently depends on
  - custom orbit sensors, which should move from global `mjcb_sensor` to MJWarp’s per-model `callback.sensor`
- Treat custom orbit-sensor semantics and stochastic measurement behavior as wrapper responsibilities, not as raw MJWarp behavior.
- Expect backend-specific tolerances because MJWarp is float32-based and documented to differ numerically from MuJoCo, especially for contact-rich scenes.

## Scope For Phase 1
- Implement a real core warp backend for:
  - free-body and articulated multibody stepping
  - orbital propagation and inertial/environment wrench coupling
  - reaction wheels, magnetorquers, and thrusters
  - built-in sensor truth and custom orbit sensor callbacks
  - single-world and batched-world runtime creation through `model.make_data(...)`
- Defer:
  - viewer migration
  - `ISS/` migration
  - full CPU/warp behavioral parity guarantees for every test
  - public backend auto-detection
  - plugin-based sensor/actuator support

## Tests
- Add `tests/mjorbit_warp/`, guarded with `pytest.importorskip("mujoco_warp")`.
- Cover:
  - `MjoModel.from_xml_path(...)` uploads with `put_model`
  - `model.make_data(..., nworld=1)` and `model.make_data(..., nworld>1)`
  - public `data.nworld` / `data.backend` / `model.backend`
  - free-body and articulated-arm construction
  - `mjo_forward` after direct host-side edits
  - external wrench application through `xfrc_applied`
  - reaction wheel, magnetorquer, and thruster behavior
  - built-in sensor truth plus custom orbit sensor callbacks
  - contact smoke tests and contact-force access
  - batched stepping shape/invariant checks for `nworld>1`
- CPU backend tests:
  - add coverage for `model.make_data(...)` so both backends share the same recommended creation pattern
- Local validation on this machine:
  - package structure and skip behavior only
  - no required runtime MJWarp execution yet
- Deferred GPU validation:
  - `uv sync --dev --extra warp`
  - run the warp suite on an NVIDIA-enabled machine
  - compare CPU vs warp invariants with looser tolerances

## Documentation Updates
- Update `AGENTS.md` and `CLAUDE.md` to state:
  - CPU backend lives in `src/mjorbit`
  - MJWarp backend lives in `src/mjorbit_warp`
  - backend selection is by import path
  - `model.make_data(...)` is the preferred runtime construction API
  - warp users specify parallel simulation count with `nworld`
  - warp support is optional-dependency gated via `uv sync --dev --extra warp`
  - viewer and `ISS/` remain CPU-only for now
- Add short examples for:
  - CPU single-world usage
  - warp single-world usage
  - warp batched `nworld=N` usage

## Assumptions And Defaults
- Use `src/mjorbit_warp`, not `src/mujoco_orbot_warp`.
- Keep `mujoco-warp` as an optional extra, not a core dependency.
- Backend choice is explicit by import path, not hidden behind auto-detection.
- `model.make_data(...)` becomes the preferred construction path across both backends.
- `MjoData(model, orbit=...)` remains supported on CPU for compatibility, but is not the preferred new pattern.
- Warp batching is requested through `model.make_data(..., nworld=N)`, not through a separate batch type.
- `put_model` / `make_data` / `put_data` remain implementation mechanisms behind the wrapper API.
