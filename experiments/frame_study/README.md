# Frame Study

Self-contained Newton–Euler experiment for comparing orbital frame choices. The
local simulator is a 6-DOF rigid-body propagator implemented in `run.py`; this
experiment does **not** depend on MuJoCo or `mjorbit`.

Three frame choices are compared:

- `eci`: absolute Earth-centered inertial coordinates.
- `chief_inertial`: chief-centered coordinates with inertially fixed axes.
- `lvlh`: chief-centered rotating LVLH coordinates.

The default sweep runs four scenarios:

- Circular equatorial LEO.
- Circular inclined LEO.
- Elliptical equatorial orbit.
- Elliptical inclined orbit.

Each scenario starts from the same small LVLH relative offset with torque-free
attitude and runs for three chief orbits.

## Asymmetric units

The body propagator and the orbit propagator carry their own length units, the
same way an off-the-shelf robotic simulator (in meters by XML convention) would
be coupled to an external orbit propagator (in km or 100 km):

- **Body propagator**: always meters. Carries `r, v, q, omega_body`.
- **Chief and truth-body**: in `--orbit-length-unit-m` units (default 1000, i.e. km).
  The chief is converted back to meters at the body's RHS interface.

This matches what a MuJoCo-based pipeline looks like and decouples the
"where MuJoCo lives" choice from the "where Kepler lives" choice.

## Cancellation-free differential gravity (Encke)

In the chief-centered frames the body's RHS needs

    g_diff = g(chief + r) - g(chief)

with `chief ~ 6.78e6 m` and `r ~ 5 m`. Computing this as the literal subtraction
of two ~9.8 m/s² gravity vectors that differ by ~1e-5 m/s² loses ~5 digits to
catastrophic cancellation in single precision (~5 m position floor over 3 orbits).

`encke_diff_gravity` rewrites the difference algebraically:

    g_diff = -(mu / |chief|^3) * (r - f(q) * (chief + r))

with `q = r . (2 chief + r) / |chief|^2` and
`f(q) = 1 - 1/(1+q)^{3/2}` evaluated cancellation-free as

    f(q) = [q (3 + 3q + q^2) / (1 + (1+q)^{3/2})] / (1+q)^{3/2}.

This is exact (no Taylor truncation) and never forms the literal difference of
two near-equal large vectors. The chief-related dot products are computed in
float64 since the chief reference orbit is always float64; the result is then
cast to the body propagator's dtype. The body's `r` is the only float32-carried
quantity that enters the formula.

In the **LVLH frame** the same `encke_diff_gravity` is used, with the body's
relative position rotated into ECI (`c_il @ r_local`), then the result rotated
back into LVLH (`c_li @ ...`). The Coriolis, Euler, and centrifugal terms
remain explicit and are unchanged.

ECI mode keeps the literal `g(r_body)` formulation: the body's `qpos` *is* the
absolute ECI position, so the float32 ~0.8 m floor at 6.78e6 m scale is
unavoidable without rescaling MuJoCo itself. ECI is left as a strawman to
illustrate that floor.

## Time stepping

- `ORBIT_TIMESTEP = 0.5 s` is the coarse RK4 chief/reference orbit step (default sweep).
- `SIM_TIMESTEP = 0.1 s` is the body Newton–Euler step.
- `TRUTH_SUBSTEPS = 10`: the truth-body reference uses RK4 with 10 substeps per
  body step, so its truncation error is far below the body integrator's.

The `Euler` integrator is **semi-implicit (symplectic) Euler** (matches MuJoCo's
`mjINT_EULER`): velocity is updated with the current acceleration; position and
orientation are then updated with the *new* velocity. On Hamiltonian systems this
preserves energy on average and produces bounded periodic error rather than the
secular blow-up of forward Euler. `RK4` is standard explicit RK4.

The `implicit` integrator is **linearly-implicit (implicit-in-velocity) Euler**
(matches MuJoCo's `mjINT_IMPLICIT` velocity update): the velocity DOFs are
advanced via `(I - dt J) dvel = dt accel` using the analytic Jacobian
`J = d(accel)/d(vel)`, then positions/orientation follow with the new velocities.
For position-only accelerations (ECI, chief-inertial) `J = 0`, so the
translational update reduces *exactly* to semi-implicit Euler; in the LVLH frame
the velocity-dependent Coriolis term is treated implicitly. Note this is a
generic linearly-implicit scheme: it reproduces MuJoCo's translational behavior
but, unlike MuJoCo's `mjINT_IMPLICIT`, it does not preserve the rotation-group
structure, so its torque-free attitude `|H|` is only first-order accurate. The
paper figure reports translation only, so this does not affect it.

## Run

```bash
pixi run python experiments/frame_study/run.py
```

The table reports ECI position error against the substepped RK4 two-body
reference, relative-motion radius, orbital invariant drift, and attitude
invariant drift.

To run a smaller reproduction:

```bash
pixi run python experiments/frame_study/run.py --scenario circular_equatorial
```

To generate the one-orbit ECI/local-chief integrator comparison:

```bash
pixi run python experiments/frame_study/run.py --integrator-study
```

This uses one shared timestep for the body, chief propagation, and the
substepped RK4 ECI reference. It writes position-error and energy-error plots
plus raw `.npz` samples under `experiments/frame_study/out/` for:

- `ECI + Euler`
- `ECI + RK4`
- `local chief + Euler`
- `local chief + RK4`

ECI curves are plotted with solid lines and local-chief curves with dashed
lines. Use `--study-precision float32` to run the body propagator end-to-end in
single precision; the chief and truth-body references stay in float64.

Use `--study-orbits` to extend the same comparison, for example:

```bash
pixi run python experiments/frame_study/run.py --integrator-study --study-orbits 3
```

Use `--study-rel-vel-lvlh VX VY VZ` to add a small initial LVLH relative
velocity in m/s, for example a 1 cm/s radial perturbation:

```bash
pixi run python experiments/frame_study/run.py \
  --integrator-study \
  --study-orbits 3 \
  --study-rel-vel-lvlh 0.01 0 0
```

Use `--orbit-length-unit-m` to choose the orbit propagator's length unit while
the body propagator stays in meters. For example,
`--orbit-length-unit-m 1000` runs the chief in km and
`--orbit-length-unit-m 100000` in 100 km units. The default is 1000 (km), the
standard astrodynamics convention. The make-paper-figure script also defaults
to km.

The chief-centered frames, paired with the Encke differential-gravity
formulation, integrate only meter-scale relative motion plus small tidal
accelerations and avoid the float32 cancellation that would otherwise cap the
local-chief floor at meters per orbit. The chief and reference orbits are
advanced by the separate RK4 propagator regardless of the body precision.
