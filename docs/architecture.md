# Architecture

mjorbit exposes the model/data pattern familiar from MuJoCo. A compiled model
contains static geometry and orbit configuration; each data object owns a
simulation state. Multiple data objects can share a model.

```text
MJCF + <mjorbit> configuration
             |
          MjoSpec                 editable specification
             |
          compile()
             |
          MjoModel                compiled model
             |
       make_data(orbit=...)
             |
           MjoData                one runtime state (or N GPU worlds)
             |
     mjo_forward / mjo_step
```

## CPU runtime

`src/mjorbit/` provides the Python interface. `src/cpp/` owns the compiled model,
per-data allocation, orbital propagation, environmental coupling, sensors,
state packing, and native rollout. Nanobind exposes the native runtime to Python.

The underlying native `mjModel` is owned by C++. `MjoModel` and `MjoData` expose
supported MuJoCo-style fields, but do not expose Python `mujoco.MjModel` and
`mujoco.MjData` objects. Pass the wrappers to mjorbit functions; arbitrary
MuJoCo Python functions do not necessarily accept them.

The MuJoCo world is centered on the chief with inertially fixed axes. The orbit
layer computes differential gravity, environmental loads, and spacecraft
actuator effects; MuJoCo solves the articulated and contact dynamics. Frame
conversions are part of the public data interface.

## GPU runtime

`src/mjorbit_warp/` compiles a host model and prepares MJWarp device structures.
The device kernels implement the supported orbit and coupling calculations.
`model.make_data(..., nworld=N)` allocates independent simulations sharing model
metadata. Public NumPy arrays are host mirrors of device state.

Choose the backend through `import mjorbit` or `import mjorbit_warp`.
The main construction and stepping workflow is shared; the [GPU guide](gpu.md)
documents differences in shapes, precision, configuration, and feature support.

## Control, visualization, and research code

- `mjorbit.rollout` runs batches of CPU trajectories; `mjorbit.planning` builds
  a spline-knot MPPI controller on that rollout.
- `viewer` renders CPU simulation state in a browser. Registered tasks supply
  model construction, controls, parameters, and status.
- `examples/` demonstrates the API and complete control/learning applications.
- `experiments/` contains paper analyses; `scripts/record/` produces media.

These research tools stay in the source repository so a release tag can
identify the simulator and the reproduction code together.
