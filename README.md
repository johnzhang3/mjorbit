# mjorbit

**Coupled orbital and multibody dynamics for space robotics.**

mjorbit extends MuJoCo with a propagated reference orbit and spacecraft
interaction with gravity gradients, J2, drag, solar radiation pressure, and
magnetic fields. It supports articulated spacecraft, contact, spacecraft
actuators, and parallel rollouts for control and learning.

[Project page](https://johnzhang3.github.io/mjorbit/) ·
[Documentation](docs/index.md) · [Examples](examples/README.md) ·
[Paper reproduction](experiments/README.md)

## Get started

Install [Pixi](https://pixi.sh/latest/installation/), then run:

```bash
git clone https://github.com/johnzhang3/mjorbit.git
cd mjorbit
pixi install
pixi run viewer --task free_drift
```

Open the printed URL, normally **http://localhost:8080**, to see a dual-arm
spacecraft orbiting Earth. Use **Pause**, **Reset**, and the task dropdown to
explore. The initial installation builds the C++ extension; Pixi supplies the
compiler and build tools. No GPU or trained checkpoint is needed for this demo.

For a numerical example without a browser:

```bash
pixi run example-minimal
pixi run example-free-drift  # comparison with a Clohessy–Wiltshire reference
```

Python **3.11–3.12** is supported. The CPU environments cover Linux x86-64 and
macOS Intel/Apple Silicon. GPU and RL environments target Linux x86-64 with an
NVIDIA GPU. See [installation](docs/installation.md) for optional environments
and instructions for using mjorbit from another project.

## Minimal Python example

```python
import numpy as np

from mjorbit import MjoModel, OrbitInit, mjo_forward, mjo_step
from mjorbit.constants import GM_EARTH, R_EARTH
from mjorbit.testdata import FREE_BODY_XML

radius_km = R_EARTH + 400.0
model = MjoModel.from_xml_path(FREE_BODY_XML, mj_timestep=0.01)
data = model.make_data(
    orbit=OrbitInit(
        R_eci=[radius_km, 0.0, 0.0],
        V_eci=[0.0, np.sqrt(GM_EARTH / radius_km), 0.0],
    )
)
data.qpos[:3] = [10.0, 0.0, 0.0]  # offset from the chief, in meters
mjo_forward(model, data)

for _ in range(100):
    mjo_step(model, data)

print(f"Time: {data.time:.2f} s")
print("Chief position (km):", data.orbit.R_eci)
print("Spacecraft offset (m):", data.qpos[:3])
```

This advances one second. The chief moves along its orbit while the spacecraft
stays approximately 10 m away. The [quick-start tutorial](docs/quickstart.md)
explains the XML model, initial conditions, and output.

**Frames and units:** `OrbitInit` and `data.orbit` use absolute ECI position in
km and velocity in km/s. MuJoCo `world` is a chief-centered local inertial frame
with axes parallel to ECI; `qpos` and `qvel` use SI offsets. LVLH is a derived
rotating frame. Spacecraft actuator commands use SI units (N·m, A·m², N,
rad/s); see [frames and units](docs/frames.md) before applying forces or torques.

## CPU and GPU backends

Choose a backend explicitly by import path. Both use the
`MjoModel` → `model.make_data(...)` → `mjo_step(...)` workflow.

| Capability | `mjorbit` | `mjorbit_warp` |
| --- | --- | --- |
| Simulation | Float64 CPU reference; threaded rollouts | Batched device simulation with `nworld=N` |
| Browser viewer and MPPI planner | Supported | Use the CPU viewer; GPU/RL examples have separate workflows |
| Reaction wheels, magnetorquers, thrusters | Supported | Supported |
| Control moment gyros | Supported | Not implemented |
| Noisy sensor measurement namespace | Supported | Not implemented; native sensor buffers are available |
| Public NumPy state | Live runtime buffers | Host mirrors; explicit upload/pull for state changes and reads |

```bash
pixi install -e warp
pixi run -e warp example-batched
```

The [complete batched example](examples/batched.py) uses a bundled model and
shows `mjo_upload` and `mjo_pull`. The [GPU guide](docs/gpu.md) explains shapes,
precision, synchronization, and current limitations.

## Examples and experiments

| Example | Run |
| --- | --- |
| Free-body drift and analytical comparison | `pixi run example-free-drift` |
| Floating-base arm reach with MPPI | `pixi run example-mppi-arm-reach` |
| Dual-arm attitude reorientation | `pixi run example-reorient` |
| ISS–Soyuz docking | `pixi run example-docking` |
| Capture and gravity-gradient stabilization | `pixi run example-mppi-capture` |
| PPO truss pointing | `pixi run -e rl ppo-truss-train` |

See the [example catalog](examples/README.md) for viewer options, duration,
requirements, and training/playback instructions. [Experiments](experiments/README.md)
contains paper benchmarks and figure generators. [Recording tools](scripts/record/README.md)
produce videos from simulated trajectories. These remain in the source
repository and source distribution; they are not installed as runtime packages.

## Documentation and development

Browse the [documentation sources](docs/index.md), or build a local site:

```bash
pixi run -e docs docs-build
pixi run -e docs docs-serve  # http://localhost:8000
```

```bash
pixi run lint
pixi run typecheck
pixi run test
pixi run cpp-test
pixi run package-check     # isolated wheel and source-distribution installs
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidance and
[RELEASING.md](RELEASING.md) for release checks.

## Citation and license

The paper, *mjorbit: A Simulation Framework for Space Robotics*, has been
submitted to arXiv. Its permanent preprint link will be added when available.
Software citation metadata is in [CITATION.cff](CITATION.cff).

Original project code and documentation are licensed under
[Apache 2.0](LICENSE). Third-party assets retain their own terms; see
[NOTICE](NOTICE) and the [docking asset provenance notes](examples/docking/assets/README.md).
