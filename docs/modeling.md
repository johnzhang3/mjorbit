# Build a model

An mjorbit model combines ordinary MuJoCo XML with optional `<mjorbit>`
metadata. Use `MjoModel.from_xml_path(...)` when the file already describes
the desired system, or edit an `MjoSpec` before compilation.

## Edit before compiling

```python
from mjorbit import MjoSpec
from mjorbit.testdata import FREE_BODY_XML

spec = MjoSpec.from_xml_path(FREE_BODY_XML)
spec.mjorbit.use_j2 = False
spec.mjorbit.use_drag = False
spec.mjorbit.use_srp = False
spec.mjorbit.use_magnetic = False
model = spec.compile(mj_timestep=0.01)
```

This configuration retains the two-body reference orbit and differential
gravity. The environment flags select additional models; they do not disable
the orbital dynamics as a whole. The free-body analytical comparison uses this
configuration to match its Clohessy–Wiltshire reference.

`MjoSpec.from_xml_string(xml, assets=...)` accepts in-memory XML and an optional
asset map. Prefer `from_xml_path` for files that include meshes or other XML,
so relative asset paths remain resolvable by the runtime and viewer.
`spec.to_xml()` serializes the editable specification; `spec.copy()` supports
independent variants.

## Spacecraft metadata

`spec.mjorbit` manages the central body, environmental flags, surfaces, magnetic
bodies, reaction wheels, magnetorquers, thrusters, and CMGs. Each element has
an `add_*`, `update_*`, and `remove_*` operation. Static physical values belong
in the corresponding `*Spec` dataclass; runtime commands belong in `data`.

For example, add a wheel to the bundled free body:

```python
from mjorbit import ReactionWheelSpec

spec.mjorbit.add_reaction_wheel(
    ReactionWheelSpec(
        name="wheel_x",
        body_name="spacecraft",
        axis_body=[1.0, 0.0, 0.0],
        inertia=0.01,
        torque_limit=0.02,
    )
)
model = spec.compile(mj_timestep=0.01)
```

See the [API reference](api.md) for exact fields and signatures. Compilation
resolves body names and validates metadata. Recompile after changing a spec;
an existing model and its data retain the previously compiled configuration.

## Runtime ownership

Create each independent run with `model.make_data(orbit=...)`. CPU data exposes
live NumPy views such as `qpos`, `qvel`, `ctrl`, `xpos`, and `sensordata`.
Use `mjo_forward(model, data)` after changing initial state, then advance with
`mjo_step(model, data)`. `data.reset()` restores the initial orbit and model
state; supplying another `OrbitInit` also replaces the stored reset orbit.

The native model/data are owned by mjorbit. The wrappers do not expose
`mj_model` or `mj_data` Python objects; use their supported fields and methods.
For a custom application, keep model construction separate from the control
loop so each episode can allocate or reset data without recompiling XML.
