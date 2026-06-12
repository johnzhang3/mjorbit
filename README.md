# MuJoCo Orbit

MuJoCo Orbit is an orbit-aware MuJoCo simulator for spacecraft multibody dynamics.
The CPU package is the reference backend, and the optional Warp package targets
batched GPU simulation.

## Backends

- CPU reference backend: `import mujoco_orbit`
- MJWarp backend: `import mujoco_orbit_warp`

Backend choice is explicit by import path. Use `model.make_data(...)` to create
runtime state for both backends.

## Setup

```bash
pixi install
pixi run test
```

Optional Warp environment:

```bash
pixi install -e warp
pixi run -e warp typecheck-warp
pixi run -e warp test-warp
```

Useful development checks:

```bash
pixi run lint
pixi run typecheck
pixi run cpp-test
```

## Minimal Example

```python
import numpy as np

from mujoco_orbit import MjoModel, OrbitInit, mjo_step
from mujoco_orbit.constants import GM_EARTH, R_EARTH
from mujoco_orbit.testdata import FREE_BODY_XML

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

## Interactive Viewer

```bash
pixi run viewer
# or, with options:
pixi run viewer --task arm_reach_mppi --frame lvlh
mjo-viewer --list-tasks
```

Opens a browser viewer (judo-style) with a task dropdown, play/pause/reset,
speed control, and per-task parameter sliders. The camera auto-frames and
tracks the spacecraft with a photoreal, rotating Earth in the background,
and MPPI tasks draw their predicted rollout trajectories (best rollout in
orange, sampled alternatives in purple). Built-in tasks: `free_drift`,
`arm_reach_mppi`, `capture_stabilize_mppi`. Register your own with
`viewer.tasks.register_task`.
