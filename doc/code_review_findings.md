# Code Review Findings

Full-codebase review of the coupled orbital + multibody simulator covering
physics correctness (CPU C++ core and Warp GPU kernels), CPU↔Warp feature
parity, concurrency/thread-safety, the Python runtime/MPPI planner, the viewer,
and branding consistency.

Each finding lists the location, the defect, the failure scenario, and the
status in this change set (Fixed / Documented).

**Verification scope.** The CPU backend (Python + C++) is built and tested on
the development host (`pixi run test`: full suite green). The Warp backend is
`linux-64`-only (`pixi.toml`), so Warp code changes here are verified by static
review and `py_compile` only, **not** by running the Warp test suite — that must
be done on Linux (`pixi run -e warp test-warp`).

## Physics correctness

### P1 — CPU drag & SRP use the chief's cached scalar for every surface — **Fixed**
`src/cpp/src/coupling_passive.cc` (`apply_surface_wrenches`).
Drag scaled by `inst->atm_density` and SRP gated/scaled by `inst->eclipse`
(both chief-centered scalars) for *all* surfaces. The function already computes
each surface's `r_point_eci_km` but discarded it. The Warp kernel and the Python
reference (`tests/.../reference/coupling/surfaces.py`) both evaluate density and
the eclipse factor per surface.
**Failure:** a formation spanning altitude or straddling the shadow terminator
gets the wrong differential drag and wrong SRP, and the CPU and Warp backends
silently disagree.
**Fix:** evaluate `atm_density(r_point_eci_km)` and `eclipse_factor(r_point_eci_km,
sun_hat)` per surface on the CPU path, matching the Warp kernel and reference.

### P2 — Warp drops user `xfrc_applied` — **Documented (fix recommended, not landed)**
`src/mjorbit_warp/device/kernels/coupling.py` (`_assemble_wrenches`) and
`src/mjorbit_warp/device/shared.py` (`_apply_origin_compensation`).
The assemble kernel zeros `xfrc_applied` at the top of every pass and then
overwrites it with the internal wrench buffer, so any user-supplied external
wrench is dropped. The CPU backend routes orbit forces through `qfrc_passive`
(`mj_applyFT`) and leaves `xfrc_applied` for the user.
**Failure:** code that sets `data.xfrc_applied` for an external disturbance works
on CPU and is silently ignored on Warp.
**Why not landed here:** the obvious "add instead of overwrite" is not safe on
its own — `xfrc_applied` is in `_MIRRORED_ARRAY_FIELDS` (`field_registry.py`),
so it is pulled back to the host each step and re-uploaded the next step;
accumulating the orbit wrench into it would double-count across steps. The
correct fix routes the assembled orbit wrench through a dedicated device wrench
channel that the integrator reads *in addition* to the user's `xfrc_applied`
(or drops `xfrc_applied` from the pull set and adds the uploaded user value
inside the kernel). That is a non-local change to the GPU step pipeline, and the
Warp backend is `linux-64`-only (`pixi.toml`) so it cannot be built or
runtime-tested on the macOS host this change set was prepared on. Recommended as
a follow-up landed on Linux with the Warp test suite. (The upstream MJWarp
`xfrc_accumulate` helper is re-exported by `mjorbit_warp` via its `__getattr__`
re-export shim and may be a useful primitive for the dedicated channel.)

### P3 — Warp J2 differential is a catastrophic-cancellation float32 subtraction — **Fixed**
`src/mjorbit_warp/device/kernels/gravity.py` (`_relative_accel`).
The point-mass differential uses the cancellation-safe Encke form, but the J2
differential was a literal `_j2_accel(R+rho) - _j2_accel(R)`. Cancellation
severity depends on `||rho|| / ||R_chief||` (~1e-7 for metre-scale bodies at
7000 km), which is at float32 epsilon — not on J2's magnitude as the old comment
claimed.
**Failure:** differential J2 force on coupled bodies becomes numerical noise on
the float32 GPU path (CPU is float64, unaffected).
**Fix:** compute the J2 differential from a first-order tidal expansion
(directional-derivative form) that avoids subtracting two near-equal vectors.

### P4 — Warp forward `total_mass` includes the worldbody — **Fixed**
`src/mjorbit_warp/device/kernels/assembly.py` (`_assemble_forward_kernel`).
The forward pass recomputes `total_mass` over `range(nbody)` (includes body 0),
while the step kernel and the CPU `total_body_mass` start at body 1.
**Failure:** a forward/step inconsistency in the origin-acceleration
compensation; benign only because the MuJoCo worldbody mass is conventionally 0.
**Fix:** sum over `range(1, nbody)`.

### P5 — Warp `_atm_density` lacks the CPU zero-guard — **Fixed**
`src/mjorbit_warp/device/kernels/environment.py` (`_atm_density`).
CPU returns 0 when `rho0 <= 0 || scale_height <= 0`; the Warp version divides by
the scale height unconditionally.
**Failure:** `scale_height = 0` produces inf/NaN density that poisons the whole
world's wrench buffer; CPU returns 0 safely.
**Fix:** add the same guard.

### P6 — CMG command torque vanishes when `dt <= 0` — **Documented**
`src/cpp/src/coupling_passive.cc` (`effective_cmg_rate`).
`return dt > 0.0 ? (theta_new - gimbal_angle)/dt : 0.0` drops commanded CMG
torque whenever a forward/finite-difference path runs with `timestep == 0`.
Low severity (forward passes normally use the configured timestep); left as-is to
avoid changing the rate-limited gimbal integration semantics.

### P7 — Sun-vector epoch assumption — **Documented**
`src/cpp/src/environment.cc` (`sun_vector_eci`).
Treats the orbit clock as seconds-since-J2000; if the sim epoch is not J2000 the
eclipse/SRP direction is offset. The Python reference defines an explicit J2000
offset. Documented as a known modelling assumption; a configurable epoch is
future work.

## CPU ↔ Warp feature parity

These are documented as known gaps (tracked for follow-up); they are large
features rather than localized bugs.

### F1 — Control Moment Gyros missing on Warp — **Documented**
CPU has `ControlMomentGyroSpec` + `apply_cmg_wrenches`; the Warp backend uploads
no CMG arrays and has no CMG term in the coupling kernel. A `<mjorbit>` model
with `<cmg>` loads on Warp and produces zero torque with no error.

### F2 — `CentralBodySpec` ignored on Warp — **Documented**
Warp kernels hardcode Earth constants (`GM_EARTH`, `R_EARTH`, `J2_EARTH`,
`B0_EARTH`, aligned dipole, `OMEGA_EARTH`); only atmosphere params are uploaded.
A non-Earth body, custom J2/B0/omega, or tilted dipole changes CPU results but is
silently ignored on GPU.

### F3 — Sensors namespace missing on Warp — **Documented**
CPU exposes `data.sensors` (measure/measure_all/bias, noisy sampling) and
orbit-plugin sun/magnetometer truth in `sensordata`; Warp `data.py` has a
`# TODO(sensors)` stub. `data.sensors.measure(...)` raises `AttributeError` on
Warp.

### F4 — Rollout/state API, frame helpers, `reset`, `make_data` signature diverge — **Documented**
Warp exports neither `rollout`, `mjo_get_state`, nor `mjo_set_state`; Warp
`MjoData` lacks `world_position_from_lvlh` / `lvlh_position_from_world` /
`eci_position_from_world` / `reset()`; Warp `make_data` requires `orbit` while CPU
defaults it to `None`; specs `CentralBodySpec`/`ControlMomentGyroSpec`/`MjoSpec`
are not re-exported by Warp. Code written against the "same public API" claim does
not fully port across backends.

## Concurrency

### C1 — Global sensor-callback chain is racy — **Fixed**
`src/cpp/src/sensors_plugin.cc`.
`g_previous_callback` was a plain (non-atomic) pointer read in the hot callback
path, and `mjcb_sensor` is one process-global shared by all `mjData`. Concurrent
install (a second model) could torn-read the function pointer; two models with
different prior callbacks corrupt each other's sensor writes.
**Fix:** make `g_previous_callback` a `std::atomic`, install once under the
existing mutex with the prior callback captured atomically, and read it
atomically in the callback.

### C2 — Threaded plugin test cannot catch cross-thread interference — **Documented**
`src/cpp/tests/test_threaded_plugin.cc`.
The test loads one shared model and asserts only against a sequential baseline,
so it would not catch a reintroduced shared/static cache or the C1 callback race.
Documented; a stronger test (two models with distinct previous callbacks stepped
concurrently) is follow-up work.

## Python runtime / planner

### R1 — MPPI clamps knots, not executed controls — **Fixed**
`src/mjorbit/planning/mppi.py`.
`np.clip` was applied to the spline knots; the interpolated per-step controls and
the `action()` output were never re-clamped, so a cubic spline overshoots the box.
**Failure:** the cost function and the executed plan can see out-of-bounds
controls.
**Fix:** clamp the interpolated controls before rollout, and clamp `action()`
output to the configured bounds.

### R2 — `rollout` silently copies user output arrays — **Fixed**
`src/mjorbit/rollout.py`.
`_ensure_3d` ran `np.ascontiguousarray(arr, dtype=np.float64)` on the `state=` /
`sensordata=` output arrays, returning a copy for any non-contiguous or
non-float64 buffer; the native fill then wrote the copy and the caller's array
stayed unfilled.
**Failure:** a caller passing a pre-allocated float32 or sliced output array
(MuJoCo-rollout-style in-place fill) gets it never updated.
**Fix:** raise a clear `ValueError` when a provided output array is not a
float64 C-contiguous array, instead of silently copying.

## Viewer

### V1 — `MjOrbitViewer` ECI path reintroduces float32 jitter — **Fixed**
`src/viewer/viewer.py` (`_world_transform`).
Returned `translation = 1000 * R_eci`, placing geometry at absolute ECI
(~6.8e6 m). The sibling `MjOrbitApp` renders chief-centered and translates the
*Earth* by `-R_eci` instead (the float32-jitter fix). `examples/collision_viewer.py`
uses `MjOrbitViewer` with `frame="eci"`.
**Fix:** render chief-centered (translation `None`, origin 0) and offset the Earth
by `-R_eci`, mirroring `MjOrbitApp`.

### V2 — Viewer never stops its viser server — **Fixed**
`src/viewer/app.py` / `src/viewer/viewer.py`.
No `server.stop()` / cleanup, so repeated construction (tests, notebooks, task
switching) leaks server threads and eventually fails to rebind the port.
**Fix:** add a `close()` method and call it from a `finally` in `run()`.

### V3 — `distance_for_fill` uses tan/atan for a sphere — **Fixed**
`src/viewer/framing.py`.
A sphere of radius r at distance d subtends `2*asin(r/d)`, not `2*atan(r/d)`.
**Fix:** use `asin`, with the domain clamped so the sphere is always outside the
camera.

## Branding

### B1 — Package renamed `mujoco_orbit` → `mjorbit` — **Fixed**
The Python packages (`mujoco_orbit`, `mujoco_orbit_warp`), the C++ namespace and
include directory, the CMake variables/targets, the MuJoCo plugin name string
(`mujoco_orbit.orbit` → `mjorbit.orbit`, updated in C++ and the XML models in
lockstep), the distribution name, the test directories, and all docs/imports were
renamed to `mjorbit` / `mjorbit_warp`.

The custom MJCF extension element `<mjorbit>` and the `spec.mjorbit` attribute are
**kept** — they are an independent namespace (a parser-recognized element name,
not the Python package name), so the package rename does not collide with them.
The standalone MuJoCo engine references and the upstream `mujoco` / `mujoco-warp`
dependencies are untouched.
