# mjorbit: Orbital Dynamics + MuJoCo WARP Coupled Simulator

## Context

**Problem:** On-orbit robotic manipulation requires simulating both orbital mechanics (km-scale, ECI frame) and multi-body contact dynamics (mm-scale, local frame). No existing GPU simulator couples these two regimes at the fidelity needed for RL training.

**Approach:** Two-loop architecture — an outer orbit/environment loop (Warp kernels, ECI frame) and an inner multi-body loop (MuJoCo WARP, LVLH frame) — coupled via per-body wrench injection. Both are GPU-parallel across `nworld` environments.

**Frame mismatch resolution:** Define MuJoCo's world frame as the LVLH (Local Vertical Local Horizontal) / Hill's frame centered on the reference spacecraft. Disable MuJoCo's built-in gravity. At every MuJoCo substep, assemble external wrenches in LVLH and write them into `d.xfrc_applied` per body. These wrenches include exact non-inertial frame terms from the chief orbit, per-body gravity/J2, and configuration-dependent environment loads such as drag, SRP, magnetic torques, and actuator/environment torques. This keeps all MuJoCo body positions at meter-scale (avoiding float32 precision issues at 6000+ km), while MuJoCo handles contacts, joints, and actuators in the local frame.

**Existing assets:**
- OrbitX (`/home/johnzhang/Documents/OrbitX/`) — JAX-native orbital sim with gravity, element conversions, frame transforms. Algorithms to port to Warp.
- MuJoCo WARP prototypes (`/home/johnzhang/Documents/prototypes/mjwarp/`) — basic `put_model`/`put_data`/`step` patterns.

---

## Architecture

```
                    ┌──────────────────────────────┐
                    │   Outer Orbit/Env Loop        │
                    │  Warp: RK4 chief propagation  │
                    │  ECI frame, km units          │
                    │  (R, V, t) per world          │
                    │  + frame/environment cache    │
                    └──────────┬───────────────────┘
                               │ chief state, LVLH frame,
                               │ sun/B-field/atmosphere cache
                    ┌──────────▼───────────────────┐
                    │     Coupling Layer            │
                    │  Warp kernels: assemble       │
                    │  per-body / per-surface       │
                    │  external wrenches            │
                    │  Read: d.xipos, d.ximat,      │
                    │        d.cvel, metadata       │
                    │  Write: d.xfrc_applied        │
                    └──────────┬───────────────────┘
                               │ xfrc_applied
                    ┌──────────▼───────────────────┐
                    │   Inner MuJoCo Loop           │
                    │   MuJoCo WARP: step(m, d)     │
                    │   LVLH frame, meter units     │
                    │   Contacts, joints, actuators │
                    └──────────────────────────────┘
```

All data stays on GPU. No host transfers in the hot path.

**Layer interaction contract:**
- The outer layer owns the chief orbital state in ECI, LVLH frame kinematics (`C_LI`, `ω`, `ω̇`), and environment caches ported from OrbitX: `J2`, sun vector, eclipse, atmospheric co-rotation / relative velocity, and magnetic field.
- The MuJoCo layer owns the articulated configuration and velocity of each rigid body: COM pose/velocity, body orientation, joint states, contacts, and actuator states.
- The coupling layer combines them into per-body wrenches. Frame/inertial terms are evaluated at each body COM. Configuration-dependent loads such as drag and SRP are evaluated at user-defined surfaces or force application points attached to MuJoCo bodies, then scattered into that body's force/torque slot in `d.xfrc_applied`.
- This means the force model is not "apply one net force at the spacecraft COM". Instead, each body or surface contributes its own wrench, and the articulated system responds through MuJoCo.

---

## Project Structure

```
mjorbit/
  pyproject.toml              # uv, hatchling, deps: mujoco, mujoco-warp, warp-lang
  CLAUDE.md
  .gitignore

  src/mjorbit/
    __init__.py               # Public API: compile, step, ScenarioCfg
    constants.py              # GM_EARTH, R_EARTH, J2 (km units, from OrbitX)

    core/
      config.py               # ScenarioCfg, OrbitCfg, MuJoCoCfg dataclasses
      compile.py              # compile(cfg) -> Scenario: load MJCF, init orbits, put to GPU
      scenario.py             # Scenario: holds warp Model/Data + OrbitState + LVLHParams
      step.py                 # step(scenario, ctrl): coupled outer+inner step

    orbit/
      state.py                # OrbitState: Warp arrays (nworld, 3) for R, V
      propagator.py           # @wp.kernel rk4_orbit_step: RK4 on (R,V) with gravity/env
      gravity.py              # @wp.func point_mass_accel, j2_accel (port from OrbitX)
      elements.py             # @wp.kernel keplerian_to_cartesian (port from OrbitX)
      lvlh.py                 # @wp.func compute_lvlh_params: n, omega, omega_dot from (R,V)
      environment.py          # @wp.func sun/eclipse, atmospheric rel vel, dipole magnetic field

    coupling/
      inertial.py             # @wp.kernel exact LVLH non-inertial + gravity/J2 body forces
      surfaces.py             # @wp.kernel drag/SRP surface loads -> body wrenches
      magnetic.py             # @wp.kernel magnetic torques / field projection
      sync.py                 # coupled_step(): orbit/env update + N mujoco substeps

    mjcf/
      assets/                 # Static MJCF XML files
        free_body.xml         # Single free body (for validation)
        spacecraft_arm.xml    # Spacecraft with robotic arm
      builders.py             # Programmatic MJCF + surface/load metadata helpers

  tests/
    test_orbit_propagation.py # Kepler period, energy conservation, J2 drift
    test_lvlh_forces.py       # CW free-drift vs analytical solution
    test_coupling.py          # Multi-rate sync, force correctness
    test_compile.py           # Config -> Scenario round-trip
    test_e2e.py               # End-to-end coupled step

  examples/
    free_drift.py             # CW free drift validation + plotting
    arm_reach.py              # Arm reaching in microgravity
```

---

## Implementation Phases

### Phase 0: Project Scaffolding
**Files:** `pyproject.toml`, `.gitignore`, `CLAUDE.md`, all `__init__.py`, `constants.py`

- Set up uv project with deps: `mujoco>=3.3`, `mujoco-warp>=3.6`, `warp-lang>=1.6`, `numpy>=1.26,<2`
- Dev deps: `pytest`, `ruff`, `pyright`
- Hatchling build backend, `src/mjorbit` package layout
- Port constants from OrbitX (`GM_EARTH=398600.4418 km³/s²`, `R_EARTH=6378.137 km`, `J2=1.08263e-3`)
- `uv sync` must succeed

### Phase 1: Orbit and Environment Kernels in Warp
**Files:** `orbit/state.py`, `orbit/gravity.py`, `orbit/propagator.py`, `orbit/elements.py`, `orbit/lvlh.py`, `orbit/environment.py`

- `OrbitState` dataclass holding `wp.array` for R, V, t with shape `(nworld, 3)`
- Port J2 gravity from OrbitX `dynamics/gravity.py` as `@wp.func`
- RK4 orbit propagation as `@wp.kernel` with `dim=(nworld,)`
- Port Keplerian-to-Cartesian from OrbitX `elements/conversions.py` as `@wp.kernel`
- LVLH parameter computation: mean motion `n = sqrt(mu/r³)`, angular velocity `ω = h/r²` along orbit normal
- Port OrbitX environment cache helpers needed by force models: eclipse / sun vector, dipole magnetic field, atmosphere-relative velocity, and frame cache updates
- **Test:** Circular orbit period matches `T=2π√(a³/μ)` to <0.1%; energy conservation <1e-8 relative; environment cache values match OrbitX for a single rigid spacecraft state

### Phase 2: MuJoCo WARP Integration
**Files:** `core/config.py`, `core/scenario.py`, `core/compile.py`, `mjcf/assets/*.xml`

- Config dataclasses: `OrbitCfg` (Keplerian elements), `MuJoCoCfg` (mjcf_path, timestep), `ScenarioCfg` (nworld, orbit_dt, force flags)
- Define load metadata attached to MuJoCo bodies: flat-plate surfaces / center-of-pressure points with drag and SRP coefficients, optional magnetic dipoles / magnetorquer axes, and per-body area/mass properties needed by the coupling kernels
- `compile(cfg)` loads MJCF, overrides `mjm.opt.gravity = [0,0,0]`, calls `mjwarp.put_model`/`put_data(nworld=N)`, initializes `OrbitState` from Keplerian elements, and uploads surface/body load metadata to Warp arrays
- Create minimal MJCF: `free_body.xml` (single free body, 1 kg, for validation)
- **Test:** `compile()` produces valid Scenario; gravity is zero; nworld matches; surface/body metadata round-trips correctly

### Phase 3: Coupling Layer (core innovation)
**Files:** `coupling/inertial.py`, `coupling/surfaces.py`, `coupling/magnetic.py`, `coupling/sync.py`

**Force model split:**

**A. `compute_inertial_body_wrenches` Warp kernel** (`dim=(nworld, nbody)`):
- Reads each body COM position from `d.xipos[worldid, bodyid]` (meters, LVLH frame)
- Reads each body COM velocity from `d.cvel[worldid, bodyid]` (spatial_vector: angular, linear)
- Reads body mass from `m.body_mass[worldid % m.body_mass.shape[0], bodyid]`
- Reads chief orbit state and LVLH frame kinematics (`R_ref`, `V_ref`, `C_LI`, `ω`, `ω̇`) from the outer loop
- Computes exact LVLH apparent acceleration at each body COM:
  ```
  a_body =
      C_IL * (g_eci(r_body_eci) - g_eci(r_ref_eci))
      - 2 * ω × v_body_lvlh
      - ω̇ × r_body_lvlh
      - ω × (ω × r_body_lvlh)
  ```
  where `g_eci` includes point-mass gravity + J2
- Writes `mass * a_body` into the force part of `d.xfrc_applied`
- Skips body 0 (worldbody)
- This bodywise formulation naturally produces gravity-gradient torque on extended / articulated spacecraft without needing a separate rigid-body-only approximation

**B. `compute_surface_environment_wrenches` Warp kernel** (`dim=(nworld, nsurface)`):
- Each surface is attached to a MuJoCo body and has local metadata: center of pressure, normal, area, drag coefficient, SRP coefficient, and enable flags
- Reads body pose (`d.xipos`, `d.ximat`) and spatial velocity (`d.cvel`) to recover world-space surface point position, normal, and velocity
- Computes drag using OrbitX's atmosphere model and atmosphere-relative velocity at the surface point
- Computes SRP using OrbitX's sun/eclipse model and the current surface orientation
- Converts each surface force into a body wrench:
  - `F_surface` applied at center of pressure
  - `τ_body = (r_cp - r_com) × F_surface`
- Scatter-adds the wrench into that body's slot in `d.xfrc_applied`

**C. `compute_magnetic_wrenches` Warp kernel** (`dim=(nworld, nbody_or_component)`):
- Reads magnetic field from the outer environment cache using OrbitX's centered-dipole model
- Applies magnetic torques such as `τ = m × B` for magnetorquers or fixed dipoles
- Writes torques into the torque part of `d.xfrc_applied`
- Magnetic field primarily affects attitude / sensors, not translation; no translational magnetic force model is required for the MVP unless a specific payload needs it

**`coupled_step(scenario)`**:
1. Propagate orbit by `orbit_dt` (Warp kernel)
2. Update LVLH frame params and environment cache from new orbital state
3. For each MuJoCo substep (`n_substeps = orbit_dt / mj_timestep`):
   a. Zero `d.xfrc_applied`
   b. Launch `compute_inertial_body_wrenches`
   c. Launch `compute_surface_environment_wrenches`
   d. Launch `compute_magnetic_wrenches`
   e. Call `mjwarp.step(m, d)`

**Coupling rule of thumb:**
- Chief-orbit effects that depend only on orbital state live in the outer layer and are cached once per outer update.
- Loads that depend on articulated configuration, body pose, surface normal, or local point velocity are evaluated in the coupling layer from MuJoCo state every inner substep.
- This is how drag/SRP can depend on spacecraft configuration and attitude while still being applied to individual components rather than to the overall COM.

**Key details verified:**
- `xfrc_applied` is `wp.spatial_vector` = (torque_xyz, force_xyz) — confirmed in types.py:1758
- `cvel` is `wp.spatial_vector` = (angular, linear) COM velocity — confirmed in types.py:1804
- `xipos` is `(nworld, nbody, 3)` COM position — confirmed in types.py:1671
- `body_mass` is `(*, nbody)` float — confirmed in types.py:1253
- `ximat` / world orientation is needed to rotate surface normals and body-fixed dipoles into LVLH / ECI
- Forces persist between steps; must overwrite each substep — confirmed from io.py

**Tests (gold standard):**
- Place a 1 kg free body at (100, 0, 0) m in LVLH, zero velocity, ISS circular orbit (a=6778 km). Simulate one orbit period. Compare trajectory against CW analytical solution in the circular, point-mass limit:
```
x(t) = (4-3cos(nt))*x₀
y(t) = 6(sin(nt)-nt)*x₀
z(t) = 0
```
- Expected accuracy: <1 cm over one orbit (~5400 s) in the CW limit
- Compare per-body J2, drag, SRP, and magnetic cache outputs against OrbitX for an equivalent single rigid-body spacecraft state
- Validate gravity-gradient torque by comparing a rigid spacecraft's attitude torque against OrbitX in a no-contact scenario
- Validate surface-force aggregation by confirming that two equal and opposite panel loads produce zero net force and the correct net torque

### Phase 4: End-to-End API
**Files:** `core/step.py`, `__init__.py`

- `step(scenario, ctrl=None)` — public API for advancing simulation
- Optional `ctrl` array writes to `d.ctrl` before stepping
- Warm-up JIT compilation in `compile()` by running one dummy step
- **Test:** Full scenario with spacecraft_arm.xml, step 1000 times, verify finite state

### Phase 5: Examples and MJCF Models
**Files:** `mjcf/assets/spacecraft_arm.xml`, `mjcf/builders.py`, `examples/free_drift.py`, `examples/arm_reach.py`

- `spacecraft_arm.xml`: Free-floating base (100 kg) + 3-link arm with hinge joints + target object
- `free_drift.py`: Run CW validation, plot trajectory vs analytical
- `arm_reach.py`: Arm reaching for a target in microgravity, demonstrate coupled sim

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Orbit propagation framework | Warp (not JAX) | Same framework as MuJoCo WARP; no data transfer overhead; all GPU |
| MuJoCo world frame | LVLH (Hill's frame) | Meter-scale positions avoid float32 precision issues; natural for proximity ops |
| Gravity in MuJoCo | Disabled (`gravity="0 0 0"`) | External orbital/environment wrenches injected via `xfrc_applied` instead |
| Force injection | `xfrc_applied` (Cartesian) | Assemble per-body force/torque contributions directly on GPU |
| Coupling granularity | Bodywise + surfacewise | COM-level force is insufficient for drag, SRP, magnetic torques, and articulated gravity-gradient effects |
| Multi-rate scheme | Orbit/env cache at `orbit_dt`, MuJoCo at `mj_timestep` | Slow orbital/environment state cached outside; configuration-dependent loads recomputed every MuJoCo substep |
| Units | Orbit: km,km/s; MuJoCo: m,m/s | Standard conventions; mean motion `n` is in rad/s (unit-agnostic) |
| LVLH axes | x=radial out, y=along-track, z=orbit normal | Standard Hill frame convention |
| Relative dynamics model | Exact LVLH + J2 first | Match OrbitX fidelity where available; use CW only as a special-case validation oracle |

---

## Verification

1. **Unit tests (Phase 1-2):** Orbit propagation period, energy, J2 drift; environment cache values match OrbitX
2. **CW limit validation (Phase 3):** Free-drift trajectory matches closed-form CW solution to <1 cm when J2/SRP/drag are disabled
3. **OrbitX parity checks:** J2 gravity, eclipse/SRP, atmosphere-relative velocity, and dipole magnetic field match OrbitX for identical states
4. **Force sanity check:** ISS orbit, 100 kg body at 100 m radial: frame tidal force is on the order of `3*n²*x*m = 0.038 N`; drag/SRP magnitudes remain in expected LEO ranges
5. **Batch consistency:** nworld=1 vs nworld=512 produce identical per-world results
6. **Multi-body validation (Phase 4):** Two free bodies drift independently in the no-contact limit; articulated arm motion plus external panel loads conserve momentum / produce expected net wrench
7. **Run all tests:** `uv run pytest tests/ -v`

---

## Dependencies

```toml
[project]
dependencies = [
    "mujoco>=3.3",
    "mujoco-warp>=3.6",
    "warp-lang>=1.6",
    "numpy>=1.26,<2",
]

[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.8.0", "pyright>=1.1.0"]
```

---

## Risks

| Risk | Mitigation |
|------|------------|
| Float32 at orbital scale | LVLH frame keeps MuJoCo positions at meter-scale; orbit prop can use float64 if needed |
| Multi-rate environment/cache drift | Reduce `orbit_dt` or interpolate chief state / frame cache within inner loop |
| Warp JIT compilation latency | Warmup step during `compile()` |
| Mapping surface loads to body wrenches | Store explicit center-of-pressure / normal metadata and test scatter-add logic on simple rigid-body fixtures |
| `cvel` interpretation in rotating frame | MuJoCo treats world as inertial; coupling kernels must convert LVLH point velocities to absolute ECI / atmosphere-relative velocities carefully |
