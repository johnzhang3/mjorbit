# Frame Conventions and Equations of Motion

This note documents the dynamics actually integrated by `mjorbit`, the
frame in which they live, and how the orbit-side propagator and MuJoCo's
multibody solver are stitched together by the `mjorbit.orbit` plugin.

## Frames

Three frames are in play.

- **ECI** — Earth-centered inertial. Standard non-rotating geocentric frame.
  Used by everything in `data.orbit` and the environment cache.
- **MuJoCo `world`** — chief-centered, ECI-axis-parallel. Origin follows the
  chief reference orbit; axes are ***not*** rotating, they are parallel to ECI
  axes. This is the frame in which `qpos`, `qvel`, `xpos`, `xmat`, `xfrc`
  live. It is **non-inertial** (translating with the chief) but **non-rotating**
  (zero angular velocity).
- **LVLH** — chief-centered rotating frame (radial / along-track / cross-track).
  This is a *derived* frame; conversions go through `data.frame.C_LI` /
  `C_IL` and `data.frame.omega_lvlh`. Nothing in MuJoCo itself uses LVLH.

The frame configuration is critical: because `world` is **not** rotating,
the only fictitious force that appears in it is the translational pseudo-force
`-m·a_chief`. Coriolis, centrifugal, and Euler terms are all zero.

## Equation of Motion: Inertial → Chief-Centered

Let

- `R_c(t)` = chief ECI position,
- `r_i(t)` = body *i* position in MuJoCo `world` (chief-relative, ECI-parallel axes),
- absolute ECI position of body *i* is `R_i = R_c + r_i`.

Newton in ECI:

```
m_i  d²R_i/dt²  =  m_i · g(R_i)  +  F_i^non-grav
```

Substituting `R_i = R_c + r_i` and rearranging:

```
m_i  d²r_i/dt²  =  m_i · [g(R_c + r_i) − a_chief]  +  F_i^non-grav
```

where `a_chief ≡ d²R_c/dt²` is whatever the chief reference is actually doing.

The last expression is **what MuJoCo integrates**. Specifically, MuJoCo sees:

```
m_i  d²r_i/dt²  =  F_i^applied  −  C_i + Γ_i^contact
```

where `F_i^applied` is the sum of forces we hand it via `qfrc_passive` and
`C_i` is the Coriolis/centripetal terms generated internally by MuJoCo for
multibody coupling. We need `F_i^applied` to equal

```
F_i^applied  =  m_i · g(R_c + r_i)  −  m_i · a_chief  +  F_i^non-grav  +  τ_GG^i
```

where `τ_GG^i` is the rotational gravity-gradient torque that the COM-level
formulation misses for a rigid body of finite extent (see below).

The orbit plugin assembles exactly this `F_i^applied` per body.

## What the Chief Reference Does

The chief is a *reference trajectory*; nothing physical depends on its
specific path. The propagator (`src/cpp/src/propagator.cc`) integrates

```
dR_c/dt  =  V_c
dV_c/dt  =  g(R_c)  +  a_feedback
```

with RK4 at `orbit_dt` (defaults to `mj_timestep`). `g(R_c)` is point-mass +
optional J2 gravity. `a_feedback` is a swarm-averaged non-gravity acceleration
described below; it exists so the chief tracks the centroid of the swarm
rather than drifting away when drag or thrust are active.

So `a_chief = g(R_c) + a_feedback`. This is what must be cancelled as a
fictitious force.

## Forces Applied Per Body

All physical forces on each body are evaluated at **that body's** absolute ECI
position and velocity, not at the chief's. Specifically, in
`src/cpp/src/coupling_passive.cc`:

| Effect                | Function                          | Per-body input                                |
|-----------------------|-----------------------------------|-----------------------------------------------|
| Differential gravity  | `apply_inertial_wrenches`         | `R_i = R_c + xipos_i` (per body)             |
| Drag / SRP            | `apply_surface_wrenches`          | `R_i, V_i` at each surface center-of-pressure |
| Residual magnetic     | `apply_magnetic_wrenches`         | body-frame dipole crossed with `B_eci`       |
| Gravity-gradient torque | `apply_gravity_gradient_torques`| `R_i = R_c + xipos_i` (per body)             |
| Reaction wheels / CMGs| `apply_*_wrenches`                | per-body angular velocity                    |
| Magnetorquers         | `apply_magnetorquer_wrenches`     | per-body                                     |
| Thrusters             | `apply_thruster_wrenches`         | per-body                                     |

So drag *is* computed at the altitude/velocity of each body part; gravity *is*
evaluated at each body's position. The translational tidal effect (a.k.a. the
"gravity gradient" force on a swarm of bodies) falls out automatically from
the per-body `g(R_i) − g(R_c)` evaluation — there is no separate tidal-force
correction. Empirically this reproduces Clohessy–Wiltshire to ~1 % (see
`tests/mjorbit/test_inertial_wrenches.py`).

### Why `apply_gravity_gradient_torques` is still needed

The translational equation evaluates gravity at the body's COM, which gives
the right net force for that body but no net torque. For an extended rigid
body, gravity falls off across the body and produces a torque

```
τ_GG^i  =  (3 μ / |R_i|³)  R̂_i × (J_i · R̂_i)
```

with `R̂_i = R_i / |R_i|` and `J_i` the body's inertia tensor in world axes
(via `ximat`). This is the analytic limit of integrating the differential
gravity over the body's mass distribution, in the limit where the body's
extent is small compared to `|R_i|`. The COM-only translational scheme misses
exactly this torque, and `apply_gravity_gradient_torques` adds it back per
body. It is *not* a separate "gravity gradient" model — it is the rotational
piece of the same physics that the differential force already captures
translationally.

For multi-body articulated systems, each body contributes its own `τ_GG`
about its own COM; the inter-body translational tidal coupling is already in
the per-body differential-gravity force, so there is no double-counting.

## Where the Swarm-Mean Enters

The swarm mean is used **only** for one thing: deciding how the chief
reference moves so the swarm stays close to the origin.

`add_feedback_force` (called inside drag/SRP and thruster routines)
accumulates the world-frame non-gravity force `F^non-grav` summed across all
bodies. Then

```
a_feedback  =  ( Σ_i F_i^non-grav )  /  ( Σ_i m_i )      [in km/s²]
```

That single shared `a_feedback` is

1. fed to the chief propagator (so `a_chief = g(R_c) + a_feedback`), which
   keeps the chief tracking the swarm's centroid acceleration; and
2. subtracted from **each body** in `apply_origin_acceleration_wrenches` as
   `-m_i · a_feedback`, completing the fictitious-force compensation
   `-m_i · a_chief`.

Step (2) is required because the chief is moving with `g(R_c) + a_feedback`,
so the pseudo-force must cancel both pieces.

The swarm mean is **not** applied to any body as a physical force. Each body
still feels its own per-body drag, SRP, thrust, gravity, etc. The only
*net* observable effect of the feedback is on the chief reference; relative
motion `r_i(t)` is identical for any choice of chief tracking, by
construction (it just changes whether the bodies appear to drift in the
chief frame or whether the origin chases them).

## Total Wrench on Each Body

Putting it all together, the world-frame wrench applied to body *i* per step
is the sum of the rows below:

```
F_i  =  m_i [ g(R_c + r_i) − g(R_c) ]            differential gravity (incl. J2)
       + F_i^drag(R_i, V_i)                       per-body drag
       + F_i^SRP(R_i, n̂_i)                       per-body SRP
       + F_i^thrust                                per-body thrust
       − m_i · a_feedback                          chief non-grav cancellation

τ_i  =  τ_GG^i                                    rigid-body GG torque
       + τ_i^drag/SRP                              drag/SRP torque about COM
       + τ_i^magnetic                              residual dipole × B
       + τ_i^magnetorquer                          commanded dipole × B
       + τ_i^RW + τ_i^CMG                          actuator gyro + command
       + τ_i^thruster                              thrust offset × force
```

`τ_GG` is added for translation+rotation completeness; the rest are direct
physics.

## Frame Approximations Worth Knowing

These are deliberate small-extent approximations, not bugs:

- **Atmospheric density** is evaluated at the *chief* ECI position only and
  shared across bodies (`environment.cc:atm_density`). For LEO swarms with
  ≤O(km) separation this is fine; for widely separated bodies the per-body
  altitude difference would matter.
- **Magnetic field `B_eci`** is evaluated at the chief position only.
- **Sun vector / eclipse** is evaluated at the chief position only.

Per-body inputs that *are* per-body: gravity, body-velocity for drag,
body-attitude for surface normals, surface position for torque arms.

## How Orbit and MuJoCo Are Stitched: the Plugin

The integration is glued together by `mjorbit.orbit`, a MuJoCo plugin
declaring two capability hooks (`src/cpp/plugin/orbit_plugin.cc`):

- `mjPLUGIN_PASSIVE` → `Compute(...)` callback,
- per-step `Advance(...)` callback.

`MjoModel.from_xml_path` injects `<extension><plugin plugin="mjorbit.orbit"/></extension>`
into the user's XML and attaches one plugin instance to a host body. It also
sets `mj_model.opt.gravity[:] = 0.0` so that MuJoCo's built-in uniform
gravity does *not* duplicate the gravity model.

Per-step lifecycle inside `mj_step`:

1. **Forces phase (`Compute`, `mjPLUGIN_PASSIVE`).** MuJoCo calls
   `Compute(...)` during the passive-force stage. The native code:
   1. `refresh_orbit_caches(inst)` — recomputes `C_LI`, `omega_lvlh`,
      `omega_dot_lvlh`, sun, B-field, eclipse, atm density at current `R_c`.
   2. `apply_passive_wrenches(...)` — evaluates per-body force/torque and
      writes them into MuJoCo's `qfrc_passive` via `mj_applyFT`. The same
      values are tee'd into `inst->wrench_buffer` for Python-side
      observability.
2. **Integration.** MuJoCo integrates multibody dynamics in chief-frame
   coordinates with `qfrc_passive` plus internal Coriolis terms.
3. **Reference-orbit phase (`Advance`).** After the multibody step,
   `Advance(...)`:
   1. `advance_actuators(...)` — Euler-integrates RW speeds and CMG gimbal
      angles using the just-applied torque commands (these live outside
      MuJoCo's qpos so they need their own integrator).
   2. `propagate_rk4(...)` — advances `R_c, V_c` by `orbit_dt` under
      `g(R_c) + a_feedback`.
   3. `refresh_orbit_caches(inst)` — re-syncs cached frame/environment
      quantities at the new `R_c`.

Python-side glue (`src/mjorbit/core/step.py`) is intentionally thin:

```python
def mjo_step(model, data):
    data.wrench_buffer[:] = 0.0
    data.xfrc_applied[:] = 0.0
    mujoco.mj_step(model.mj_model, data.mj_data)
    data.xfrc_applied[:] = data.wrench_buffer   # for inspection only
```

`xfrc_applied` is *not* the integration channel; it is set after the step so
Python users can read what was applied. The actual force flow is
plugin → `qfrc_passive` → MuJoCo integrator.

## What MuJoCo Computes vs. What the Plugin Computes

| Quantity                                | Computed by                          |
|-----------------------------------------|--------------------------------------|
| Multibody mass matrix, constraints      | MuJoCo                               |
| Joint, contact, limit forces            | MuJoCo                               |
| Internal Coriolis/centripetal (multibody)| MuJoCo                              |
| Time integration of `qpos`, `qvel`      | MuJoCo                               |
| Built-in uniform gravity                | **Disabled** (`opt.gravity = 0`)     |
| Per-body gravity (incl. J2)             | Plugin (`apply_inertial_wrenches`)   |
| Translational fictitious-force cancellation | Plugin (`apply_inertial_wrenches` subtracts `g_chief`; `apply_origin_acceleration_wrenches` subtracts `a_feedback`) |
| Gravity-gradient torque                 | Plugin (`apply_gravity_gradient_torques`) |
| Atmospheric drag                        | Plugin (`apply_surface_wrenches`)    |
| Solar radiation pressure                | Plugin (`apply_surface_wrenches`)    |
| Magnetic / magnetorquer                 | Plugin                               |
| RW / CMG / thruster                     | Plugin                               |
| Chief orbit propagation                 | Plugin (`propagate_rk4`)             |
| RW speed, CMG gimbal angle integration  | Plugin (Euler in `advance_actuators`)|
| LVLH frame cache (`C_LI`, `ω_lvlh`)     | Plugin (`refresh_orbit_caches`)      |
| Environment cache (sun, B, ρ_atm, eclipse) | Plugin (`refresh_orbit_caches`)   |
| Sensor measurements                     | Plugin sensor hook + Python noise model |

## Validation

`tests/mjorbit/test_inertial_wrenches.py::TestCWLimit` integrates the
full coupled simulator from a CW initial condition and compares against the
closed-form Clohessy–Wiltshire trajectory:

- `test_radial_offset_drift`     — radial offset, 10 s
- `test_along_track_velocity`    — along-track velocity kick
- `test_cross_track_oscillation` — cross-track oscillation
- `test_combined_motion`         — mixed radial/along/cross over 20 s

All match CW to 1 % in position and velocity. CW is the analytic linearised
two-body relative-motion solution in the chief frame, so this verifies the
frame physics and the plugin's force assembly end-to-end.

`test_gravity_gradient.py` verifies the rotational `τ_GG` against the
analytic formula on a dumbbell, and `test_orbitx_parity.py` cross-checks
against an independent Python reference implementation.
