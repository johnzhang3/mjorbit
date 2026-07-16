# mjorbit

mjorbit is an orbit-aware MuJoCo simulator for coupled orbital and spacecraft
multibody dynamics. The MuJoCo world follows a propagated reference orbit, so
articulated spacecraft feel the correct orbital environment (gravity gradient,
J2, drag, solar-radiation pressure, magnetic field) while MuJoCo solves the
constrained multibody dynamics.

Two backends share one API:

- **CPU reference backend**: `import mjorbit` — float64, low-latency, the
  ground truth.
- **MJWarp GPU backend**: `import mjorbit_warp` — batched simulation
  (`nworld` parallel worlds) for large-scale rollouts and RL training.

Backend choice is explicit by import path.

## Setup

```bash
pixi install
pixi run test
```

Optional environments:

```bash
pixi install -e warp      # GPU backend (linux-64)
pixi run -e warp test-warp

pixi install -e frames    # astropy-based orbit IC frame conversion
pixi run -e frames test-frames

pixi install -e rl        # PPO training examples (torch + rsl-rl, linux-64)
pixi install -e report    # plotting/analysis extras
```

## Minimal Example

```python
import numpy as np

from mjorbit import MjoModel, OrbitInit, mjo_step
from mjorbit.constants import GM_EARTH, R_EARTH
from mjorbit.testdata import FREE_BODY_XML

radius_km = R_EARTH + 400.0
speed_km_s = np.sqrt(GM_EARTH / radius_km)

orbit = OrbitInit(
    R_eci=[radius_km, 0.0, 0.0],
    V_eci=[0.0, speed_km_s, 0.0],
)

model = MjoModel.from_xml_path(FREE_BODY_XML, mj_timestep=0.01)
data = model.make_data(orbit=orbit)

mjo_step(model, data)
print(data.time, data.qpos[:3])
```

For a runnable script with a Clohessy-Wiltshire reference check, see
`examples/free_drift.py`.

Orbit initial conditions can also be given in another named frame with an
absolute epoch (requires the `frames` environment) — inertial realizations like
`TEME` (the SGP4/TLE output frame) or Earth-fixed `ITRF`/`ECEF` states, whose
ω×r velocity term is applied explicitly:

```python
data = model.make_data(
    orbit=OrbitInit(R_eci=r_teme, V_eci=v_teme, frame="TEME", epoch="2024-03-01T12:00:00"),
)
```

## Batched GPU Simulation

The warp backend uses the same API names, plus `nworld` on `make_data(...)`:

```python
from mjorbit_warp import MjoModel, OrbitInit, mjo_step

model = MjoModel.from_xml_path("model.xml")
data = model.make_data(
    orbit=OrbitInit(R_eci=[7000.0, 0.0, 0.0], V_eci=[0.0, 7.5, 0.0]),
    nworld=256,
)

data.actuators.mtq_dipole_cmd[:, 0] = 5.0  # actuator commands auto-sync each step
mjo_step(model, data)
```

On the warp backend the device state is authoritative: actuator command inputs
(`*_cmd`) auto-sync each step, but after editing `qpos`/`qvel`/`ctrl`/`orbit`
call `mjo_upload(...)` (or `mjo_step(..., sync=True)`), and call `mjo_pull(...)`
before reading public buffers back on the host.

## Units and Frames

Orbit-side quantities — `OrbitInit`, `data.orbit`, environment caches, and the
orbital actuator commands (`data.actuators.*_cmd`) — use km, s, kg, rad, T
(torques in kg·km²/s²; forces in kN). The MuJoCo-facing buffers (`qpos`,
`qvel`, `ctrl`, `xfrc_applied`, ...) and XML models stay in MuJoCo's native SI
units (m, s, kg); conversions between the two happen inside the step at the
MuJoCo boundary. The MuJoCo `world` frame is a chief-centered local inertial
frame (SI offsets from the chief) with axes parallel to ECI; the absolute
orbit lives in `data.orbit`, and LVLH quantities are derived via the
`data.frame` helpers. See `CLAUDE.md` for the full frame conventions.

## Interactive Viewer

```bash
pixi run viewer
# or, with options:
pixi run viewer --task arm_reach_mppi --frame lvlh
mjo-viewer --list-tasks
```

Opens a browser viewer (judo-style) with a task dropdown, play/pause/reset,
speed control, and per-task parameter sliders. The camera auto-frames and
tracks the spacecraft with a photoreal, rotating Earth in the background, and
MPPI tasks draw their predicted rollout trajectories. Built-in tasks:
`free_drift`, `arm_reach_mppi`, `capture_stabilize_mppi`. Register your own
with `viewer.tasks.register_task`.

## Examples

The paper's four on-orbit scenarios, plus small API demos:

```bash
pixi run example-free-drift        # free drift vs Clohessy-Wiltshire reference
pixi run example-reorient          # multibody attitude control: dual-arm momentum-exchange slew (MPPI)
pixi run example-docking           # autonomous docking: Soyuz-class chaser -> ISS-class target (MPPI)
pixi run example-mppi-capture      # capture & nadir pointing via gravity-gradient torque (MPPI)
pixi run example-mppi-arm-reach    # arm reach with a floating base (MPPI)
```

GPU RL training (Astrobee detumble-and-grasp, truss nadir-pointing) lives in
`examples/ppo/` — see `examples/ppo/README.md`:

```bash
pixi run -e rl ppo-truss-train
pixi run -e rl python examples/ppo/astrobee_train.py
```

## Paper Experiments

`experiments/` holds the generators for the paper's figures:

- `experiments/frame_study/` — orbital frame choice comparison for a 6-DOF
  rigid-body propagator (standalone Newton–Euler, no MuJoCo dependency).
- `experiments/cross_track_collision/` — orbit-coupled vs zero-g contact
  outcomes for two boxes on a cross-track collision course.
- `experiments/mppi_fidelity/` — MPPI rollout-model fidelity: orbit-coupled vs
  zero-g rollouts on the capture-and-stabilize task.
- `experiments/basilisk_multibody/` — accuracy and throughput comparison
  against Basilisk on a bimanual free-flyer.

`scripts/record/` renders the paper/video clips headlessly from the viewer.

## Development

```bash
pixi run lint
pixi run typecheck
pixi run test
pixi run cpp-test
pixi run -e warp typecheck-warp
```
