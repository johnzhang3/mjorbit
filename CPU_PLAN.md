# mjorbit CPU-First Plan: NumPy + Standard MuJoCo Reference Simulator

## Goal

Implement a CPU reference simulator for coupled orbital dynamics and multi-body contact dynamics before adding GPU/Warp complexity.

This reference path should:
- use standard MuJoCo on CPU for rigid-body dynamics, contacts, joints, and actuators
- use NumPy for orbital propagation, frame transforms, and environment models
- implement the same coupling contract planned for the GPU version
- use bidirectional coupling at the net external wrench level
- prioritize correctness, inspectability, and testability over speed
- become the oracle for later GPU parity tests

This is not intended to be the final high-throughput RL simulator. It is the correctness baseline.

---

## Why CPU First

The highest-risk part of this project is not raw simulation throughput. It is the coupling logic:
- frame definitions and transforms
- how chief-orbit dynamics couple into the local LVLH frame
- which multibody effects should feed back into the orbit dynamics
- how per-body gravity/J2 loads are computed
- how drag and SRP depend on articulated configuration and attitude
- how per-surface loads are converted into body wrenches
- how those wrenches are applied into MuJoCo correctly
- how spacecraft actuators that do not belong as explicit MuJoCo rigid bodies should be represented

All of these are easier to debug on CPU with:
- scalar code
- explicit arrays
- step-by-step logging
- direct comparison to analytical solutions and OrbitX

The CPU simulator should therefore be treated as the reference implementation for model semantics.

---

## Scope

### In Scope

- Single-world simulation (`nworld=1`)
- Chief orbit propagation in ECI
- LVLH frame definition and exact rotating-frame coupling
- Standard MuJoCo CPU stepping
- Per-body wrench injection through `xfrc_applied`
- Bidirectional coupling between orbit and multibody layers through net external force/torque bookkeeping
- OrbitX-level environment fidelity:
  - point-mass gravity
  - J2
  - gravity-gradient effects through bodywise gravity
  - atmosphere-relative velocity
  - drag via flat-plate surfaces
  - SRP via flat-plate surfaces + eclipse
- magnetic field via OrbitX's centered-dipole model
- magnetic torques such as magnetorquers / fixed dipoles
- custom spacecraft actuators modeled outside MuJoCo rigid-body state:
  - reaction wheels
  - magnetorquers
  - thrusters
- Validation against analytical CW limits and OrbitX parity checks

### Out of Scope

- GPU execution
- Warp kernels
- batched `nworld > 1`
- performance optimization
- differentiability
- advanced atmosphere beyond OrbitX's current fidelity
- higher-order gravity beyond J2
- complex geom-derived aerodynamic modeling

---

## Architecture

```
                  ┌──────────────────────────────┐
                  │   Orbit/Env Layer (NumPy)     │
                  │  Chief orbit in ECI           │
                  │  LVLH frame cache             │
                  │  Sun / eclipse / B-field      │
                  │  Atmosphere-relative velocity │
                  └──────────┬───────────────────┘
                             │ chief state, env cache
                  ┌──────────▼───────────────────┐
                  │    Coupling Layer (NumPy)     │
                  │  Read MuJoCo state            │
                  │  Compute per-body wrenches    │
                  │  Compute per-surface loads    │
                  │  Write d.xfrc_applied         │
                  └──────────┬───────────────────┘
                             │ xfrc_applied
                  ┌──────────▼───────────────────┐
                  │   MuJoCo CPU Layer            │
                  │  mujoco.mj_step              │
                  │  contacts, joints, actuators │
                  └──────────────────────────────┘
```

---

## Layer Interaction Contract

- The orbit/environment layer owns:
  - chief state in ECI: `R_ref`, `V_ref`, `t`
  - frame cache: LVLH rotation, `omega`, `omega_dot`
  - environment cache: sun vector, eclipse, magnetic field, atmosphere-relative velocity

- The MuJoCo layer owns:
  - articulated body states
  - body COM positions/velocities
  - body orientations
  - contacts, joints, actuator state

- The coupling layer reads from both and produces:
  - per-body inertial/gravity/J2 forces
  - per-surface drag and SRP forces
  - per-body actuator and magnetic torques
  - final body wrenches written into `MjData.xfrc_applied`
  - net external force/torque on the coupled system used to update the chief orbit

Bidirectional coupling policy:
- The orbit layer drives the LVLH frame seen by MuJoCo.
- The multibody layer feeds back only through net external wrench on the system.
- Internal joint forces, constraint forces, and contact forces within the coupled spacecraft/manipulator stack do not directly change the orbit propagator.
- External loads do feed back into orbit propagation:
  - thrusters
  - drag
  - SRP
  - net gravity/J2 over the distributed mass model
  - contact impulses with truly external objects if they are modeled outside the current coupled system boundary
- Reaction wheels and magnetorquers affect attitude but should not produce translational orbit feedback in the nominal model.

This means the simulator does not collapse external effects into a single net wrench at the spacecraft COM. Forces are applied where they physically belong.

---

## Project Structure

```
mjorbit/
  CPU_PLAN.md
  PLAN.md

  src/mjorbit/
    __init__.py
    constants.py

    cpu/
      __init__.py

      core/
        config.py            # CPUScenarioCfg, OrbitCfg, MuJoCoCfg, surface metadata cfg
        scenario.py          # CPUSenario: mjModel, mjData, chief orbit state, caches
        compile.py           # compile_cpu(cfg) -> CPUScenario
        step.py              # step_cpu(scenario, ctrl=None)
        actuators.py         # external actuator states and updates

      orbit/
        state.py             # OrbitState dataclass, numpy arrays
        gravity.py           # point_mass_accel, j2_accel
        propagator.py        # RK4 chief propagation
        elements.py          # Keplerian <-> Cartesian helpers
        lvlh.py              # LVLH rotation, omega, omega_dot, frame transforms
        environment.py       # sun/eclipse, dipole field, v_rel, atmosphere helpers

      coupling/
        inertial.py          # per-body exact LVLH + J2 forces
        surfaces.py          # drag/SRP loads on attached surfaces
        actuators.py         # reaction wheel, thruster, magnetorquer wrench assembly
        magnetic.py          # tau = m x B helpers / field projection
        feedback.py          # reduce body/surface loads to system net external wrench
        apply.py             # write assembled wrenches into MjData.xfrc_applied

      mjcf/
        assets/
          free_body.xml
          spacecraft_arm.xml
        builders.py          # optional model + metadata helpers

  tests/
    cpu/
      test_orbit.py
      test_environment.py
      test_inertial_wrenches.py
      test_surface_loads.py
      test_magnetic.py
      test_compile_cpu.py
      test_coupled_cpu.py
      test_orbitx_parity_cpu.py

  examples/
    cpu_free_drift.py
    cpu_arm_reach.py
```

---

## Data Model

### `OrbitState`

NumPy dataclass holding:
- `R_eci: np.ndarray shape (3,)` in km
- `V_eci: np.ndarray shape (3,)` in km/s
- `t: float` in s

### `FrameCache`

- `C_LI: np.ndarray shape (3, 3)` LVLH-from-ECI rotation
- `C_IL: np.ndarray shape (3, 3)` inverse rotation
- `omega_lvlh: np.ndarray shape (3,)` in rad/s
- `omega_dot_lvlh: np.ndarray shape (3,)` in rad/s^2

### `EnvironmentCache`

- `sun_vector_eci: np.ndarray shape (3,)`
- `eclipse: float`
- `mag_field_eci: np.ndarray shape (3,)`
- `atmosphere_omega_eci: np.ndarray shape (3,)`

### `ActuatorState`

External actuator state that is not represented as explicit MuJoCo rigid bodies:
- `rw_speed: np.ndarray shape (n_rw,)`
- `rw_momentum: np.ndarray shape (n_rw,)` or derived from speed and inertia
- optional thruster state such as valve lag / fuel bookkeeping
- optional magnetorquer state if dynamics beyond direct command are needed

Reaction wheels are modeled externally on purpose:
- do not add wheel rigid bodies or wheel joints to MuJoCo
- wheel spin rates can be much faster than the articulated-body timescales we care about
- representing them in MuJoCo would add stiff fast states and unnecessary integration burden
- instead, track wheel momentum externally and inject equal-and-opposite torques on the host body

### `SurfaceMetadata`

Each surface entry contains:
- `body_id`
- `center_of_pressure_body`
- `normal_body`
- `area`
- `drag_coeff`
- `srp_coeff`
- `use_drag`
- `use_srp`

This mirrors OrbitX's flat-plate representation instead of trying to infer loads from MuJoCo geoms.

### `CPUScenario`

Owns:
- `mjm: mujoco.MjModel`
- `mjd: mujoco.MjData`
- `orbit: OrbitState`
- `frame_cache: FrameCache`
- `env_cache: EnvironmentCache`
- `surface_metadata`
- `magnetic_metadata`
- `actuator_state`
- `cfg`

---

## Implementation Phases

### Phase 0: Scaffolding

**Files:** `pyproject.toml`, `.gitignore`, `CLAUDE.md`, all `__init__.py`, `constants.py`

- Set up uv project with deps: `mujoco>=3.3`, `numpy>=1.26,<2`
- Dev deps: `pytest`, `ruff`, `pyright`
- Hatchling build backend, `src/mjorbit` package layout
- Port constants from OrbitX (`GM_EARTH=398600.4418 km³/s²`, `R_EARTH=6378.137 km`, `J2=1.08263e-3`)

**Files:** `src/mjorbit/cpu/**`, `constants.py`, `tests/cpu/**`
- Create CPU package layout under `src/mjorbit/cpu/`
- Reuse Earth constants from OrbitX
- Add NumPy + MuJoCo CPU dependencies if missing
- Define core config dataclasses
- Define external actuator config/state structures for reaction wheels, magnetorquers, and thrusters
- Keep interfaces parallel to the eventual GPU API where possible:
  - `compile_cpu(cfg)`
  - `step_cpu(scenario, ctrl=None)`

**Exit criteria:**
- imports succeed
- a trivial `compile_cpu()` can load a MuJoCo model and step once
- `uv sync` must succeed

### Phase 1: Orbit and Frame Utilities

**Files:** `cpu/orbit/state.py`, `gravity.py`, `propagator.py`, `elements.py`, `lvlh.py`

- Port point-mass gravity and J2 from OrbitX into NumPy
- Implement RK4 chief propagation
- Implement Keplerian-to-Cartesian helper
- Implement LVLH frame construction from chief `R,V`
- Implement `omega` and `omega_dot`
- Implement ECI <-> LVLH position and velocity transforms

**Validation:**
- circular orbit period
- energy conservation
- J2 drift sanity check
- frame orthonormality and roundtrip transforms

### Phase 2: Environment Cache

**Files:** `cpu/orbit/environment.py`

- Port OrbitX-equivalent environment helpers:
  - sun vector
  - eclipse
  - centered-dipole magnetic field
  - atmosphere-relative velocity
  - exponential atmosphere density
- Build a cache update function:
  - `update_environment_cache(orbit, frame_cache, attitude_if_needed, ...)`

**Validation:**
- parity against OrbitX for identical rigid-body states

### Phase 3: MuJoCo CPU Integration

**Files:** `cpu/core/config.py`, `scenario.py`, `compile.py`

- Load MuJoCo MJCF with standard `mujoco.MjModel.from_xml_path`
- Set `mjm.opt.gravity[:] = 0`
- Build `MjData`
- Initialize chief orbit from config
- Upload surface and magnetic metadata into scenario-owned NumPy structures
- Initialize external actuator state outside MuJoCo:
  - reaction wheel inertias / speeds / momentum
  - thruster locations and directions
  - magnetorquer axes and dipole limits
- Provide helper accessors for body COM pose, orientation, and spatial velocity

**Validation:**
- `compile_cpu()` returns a valid scenario
- one `mujoco.mj_step` runs without NaNs
- metadata matches configured bodies/surfaces
- actuator state initializes correctly and is independent of MuJoCo generalized coordinates

### Phase 4: Inertial / Gravity Coupling

**Files:** `cpu/coupling/inertial.py`

- Read each MuJoCo body COM in LVLH coordinates
- Convert body COM state into ECI position where needed:
  - `r_body_eci = R_ref + C_IL @ r_body_lvlh_km`
- Compute bodywise apparent acceleration:
  ```
  a_body_lvlh =
      C_LI @ (g_eci(r_body_eci) - g_eci(R_ref))
      - 2 * omega x v_body_lvlh
      - omega_dot x r_body_lvlh
      - omega x (omega x r_body_lvlh)
  ```
- Multiply by body mass
- Write force contribution into a body-wrench buffer

Important:
- use bodywise gravity/J2, not a single rigid-body gravity-gradient torque formula
- this lets gravity-gradient effects emerge naturally from the force distribution

**Validation:**
- free-body CW limit when J2 is disabled
- exact-vs-CW agreement in the small-offset circular limit
- gravity-gradient torque sanity checks on a dumbbell-like rigid model

### Phase 5: Surface Loads

**Files:** `cpu/coupling/surfaces.py`

- For each configured surface:
  - recover body world pose from MuJoCo
  - rotate local center of pressure and surface normal into LVLH/world frame
  - compute local point velocity from body COM linear velocity + angular contribution
  - compute atmosphere-relative velocity at the surface point
  - compute drag force from OrbitX's atmosphere model
  - compute SRP force from OrbitX's sun/eclipse model
  - convert force-at-point into body torque
- Accumulate into per-body wrench buffer

Important modeling choice:
- use explicit user-defined flat-plate surfaces
- do not infer aerodynamic/SRP properties from MuJoCo geoms in the first version

**Validation:**
- single panel drag magnitude sanity check
- eclipse disables SRP
- symmetric panels can cancel net force while producing net torque

### Phase 6: Magnetic Torques

**Files:** `cpu/coupling/magnetic.py`

- Compute `B_eci` from environment cache
- Rotate into body frame or LVLH as needed
- Apply:
  - magnetorquer torque `tau = m x B`
  - optional fixed residual dipole torques
- Accumulate torque into body wrench buffer

**Validation:**
- parity with OrbitX dipole model
- sign / axis sanity checks for simple dipole orientations

### Phase 7: External Actuators

**Files:** `cpu/core/actuators.py`, `cpu/coupling/actuators.py`

Implement spacecraft actuators outside MuJoCo rigid-body state:

- Reaction wheels
  - maintain wheel momentum / speed as external state
  - integrate wheel speed from commanded wheel torque
  - apply equal-and-opposite torque to the host body
  - include saturation and optional speed limits
  - no translational force contribution

- Magnetorquers
  - command magnetic dipole moment with axis and amplitude limits
  - compute `tau = m x B`
  - no translational force contribution in the nominal model

- Thrusters
  - command force magnitude at a body-fixed application point
  - convert to body force + torque
  - contributes to orbit feedback through net external force
  - optional fuel bookkeeping can live in external actuator state

Reason for external modeling:
- these actuators are spacecraft-domain devices, not contact-domain rigid bodies
- reaction wheels in particular can spin at very high rates and should not be introduced as fast explicit MuJoCo joints unless there is a very specific need

**Validation:**
- reaction wheel command changes wheel momentum and applies equal/opposite body torque
- magnetorquer torque matches `m x B`
- thruster force-at-point produces correct net force and moment

### Phase 8: Wrench Application, Orbit Feedback, and Coupled Step

**Files:** `cpu/coupling/apply.py`, `cpu/core/step.py`

Implement:
1. update chief orbit by `orbit_dt`
2. update frame cache
3. update environment cache
4. zero `mjd.xfrc_applied`
5. assemble inertial/gravity body wrenches
6. assemble drag/SRP surface wrenches
7. assemble actuator and magnetic torques/wrenches
8. reduce all external contributions to a net system external wrench
9. feed the net external force back into chief orbit propagation
10. write final per-body wrenches into `mjd.xfrc_applied`
11. write controls if provided
12. call `mujoco.mj_step(mjm, mjd)`

Design choice:
- start with one orbit/cache update per MuJoCo step for simplicity
- once correctness is established, optionally introduce a slower outer loop if needed
- bidirectional feedback is limited to net external wrench, not raw internal MuJoCo forces

**Validation:**
- full step remains finite
- external wrench bookkeeping is inspectable per body
- no-contact articulated motions behave sensibly
- internal manipulator motion alone does not change COM orbit
- thruster and drag forces do change COM orbit in the expected direction

### Phase 9: Examples and Golden Tests

**Files:** `examples/cpu_free_drift.py`, `examples/cpu_arm_reach.py`, `tests/cpu/*`

- `cpu_free_drift.py`
  - chief in circular LEO
  - one free body with no drag/SRP/magnetic
  - compare to CW analytical solution in the linearized limit

- `cpu_arm_reach.py`
  - articulated spacecraft arm
  - no contact first
  - then add contact target

- `test_orbitx_parity_cpu.py`
  - compare CPU environment/gravity outputs to OrbitX on shared states

**Exit criteria:**
- CPU reference simulator passes tests
- coupling logic is stable enough to serve as GPU oracle

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Multibody engine | Standard MuJoCo CPU | Mature, stable, easy to inspect |
| Orbit/env implementation | NumPy | Simple, explicit, debuggable |
| Coupling direction | Bidirectional via net external wrench | Realistic without feeding back internal MuJoCo forces into orbit dynamics |
| Coupling granularity | Per-body + per-surface | Needed for articulated systems and OrbitX-level environment effects |
| Gravity model | Point mass + J2 | Matches current OrbitX fidelity |
| Drag/SRP representation | Explicit flat-plate surfaces | Matches OrbitX and avoids ambiguous geom inference |
| Magnetic model | Centered dipole + torque coupling | Matches OrbitX fidelity |
| Actuator representation | External spacecraft actuator state | Best fit for reaction wheels, magnetorquers, and thrusters |
| Gravity-gradient modeling | Emergent from bodywise gravity | More general than a single rigid-body torque formula |
| Time stepping | Simple and inspectable first | Correctness before multi-rate optimization |

---

## Verification Strategy

1. Orbit-only checks:
   - Kepler period
   - energy conservation
   - J2 drift sanity

2. Frame checks:
   - LVLH basis orthonormality
   - ECI/LVLH roundtrip consistency

3. Environment parity checks:
   - eclipse parity with OrbitX
   - magnetic field parity with OrbitX
   - atmosphere-relative velocity parity with OrbitX

4. Coupling checks:
   - CW limit for a free body
   - bodywise J2 force differs correctly with radial offset
   - drag/SRP act at configured centers of pressure
   - magnetic torques match `m x B`
   - reaction wheels change attitude but not orbit
   - manipulator self-motion does not change orbit
   - thrusters change orbit through net external force feedback

5. End-to-end checks:
   - free bodies drift correctly with no contact
   - articulated model remains finite for long horizons
   - contact scenario remains stable with external loads

6. GPU handoff criterion:
   - every future Warp kernel must be validated against the CPU reference on matched initial conditions

---

## Risks

| Risk | Mitigation |
|------|------------|
| CPU reference grows into a second product | Keep it single-world, correctness-only, and stop after it becomes a reliable oracle |
| Coupling is still hard to debug | Log per-body/per-surface wrench contributions explicitly |
| Orbit feedback double-counts forces | Define a single reducer that maps external body/surface loads to system net external wrench |
| Surface models are underspecified | Require explicit metadata in config rather than inferring from MuJoCo geoms |
| Frame/sign mistakes | Build small analytical tests before large end-to-end tests |
| MuJoCo state interpretation confusion | Add helper functions for COM pose, velocity, and orientation extraction with focused tests |
| Reaction wheel timescales are too fast for MuJoCo | Keep wheel states external and inject only the resulting body torques |

---

## Handoff to GPU Plan

Do not begin Warp/MJWarp implementation until the CPU path can do all of the following:
- compile and step a free-body case
- reproduce the CW limit when high-fidelity effects are disabled
- match OrbitX environment outputs on rigid-body parity tests
- apply drag/SRP/magnetic effects as per-body/per-surface wrenches
- run an articulated spacecraft model stably on CPU

Once these are true, the GPU plan becomes a translation and batching effort rather than a first-principles dynamics debugging effort.
