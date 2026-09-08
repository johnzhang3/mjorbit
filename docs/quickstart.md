# Your first simulation

Run the complete example:

```bash
pixi run example-minimal
```

It loads a bundled 100 kg free body, initializes a circular reference orbit
400 km above Earth, places the spacecraft 10 m from the reference, and advances
one simulated second.

## The model

The bundled `FREE_BODY_XML` contains:

```{literalinclude} ../src/mjorbit/testdata/free_body.xml
:language: xml
```

The free joint permits translation and rotation. The box dimensions, mass,
joint positions, and timestep use MuJoCo's native SI units. Uniform MuJoCo
gravity is zero; the orbit layer supplies the orbital environment.

## The Python program

```{literalinclude} ../examples/minimal.py
:language: python
```

`MjoModel.from_xml_path(...)` compiles the XML into a reusable model.
`model.make_data(orbit=...)` allocates the state of one simulation. The chief
orbit uses kilometers and kilometers per second; the MuJoCo state uses offsets
from that chief in meters and meters per second.

After changing `qpos`, `mjo_forward` refreshes derived positions, frames, and
other runtime quantities. `mjo_step` advances both the coupled spacecraft and
the reference orbit by the model timestep.

Expect output close to:

```text
Time: 1.00 s
Chief position (km): [6778.13265579    7.66855654    0.        ]
Spacecraft offset (m): [1.00000129e+01 7.25404685e-09 0.00000000e+00]
```

The chief moves several kilometers while the spacecraft's local offset remains
approximately 10 m. Last digits may vary with platform and configuration. A
body initialized exactly at the chief may keep zero local displacement while
its absolute orbit advances.

## Initialize in LVLH

LVLH axes follow the orbit: radial, along-track, and cross-track. Use the data
helpers to convert a desired LVLH position and velocity:

```python
position_lvlh = np.array([10.0, 0.0, 5.0])  # m
velocity_lvlh = np.array([0.0, 0.05, 0.0])  # m/s in the rotating frame
data.qpos[:3] = data.world_position_from_lvlh(position_lvlh)
data.qvel[:3] = data.world_velocity_from_lvlh(position_lvlh, velocity_lvlh)
mjo_forward(model, data)
```

The velocity helper includes frame rotation. Rotating the velocity vector
alone is insufficient. See [frames and units](frames.md).

## Next steps

Run `pixi run example-free-drift` for the analytical Clohessy–Wiltshire
comparison, `pixi run example-mppi-arm-reach` for a controller, or
`pixi run -e warp example-batched` for the [GPU tutorial](gpu.md).
