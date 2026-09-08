# Python API reference

The CPU backend is imported from `mjorbit`. Prefer `model.make_data(...)` to
construct runtime state. See [GPU simulation](gpu.md) for the corresponding
`mjorbit_warp` lifecycle and synchronization functions.

## Model and data

```{eval-rst}
.. autoclass:: mjorbit.MjoModel
   :members: from_xml_path, make_data, body_id, sensor, central_body, backend
   :undoc-members:
```

```{eval-rst}
.. autoclass:: mjorbit.MjoData
   :members: reset, eci_position_from_world, eci_velocity_from_world, world_position_from_lvlh, world_velocity_from_lvlh, lvlh_position_from_world, lvlh_velocity_from_world, contact_force
   :undoc-members:
```

The wrappers forward supported native arrays and dimensions, including
`model.nq`, `model.nv`, `model.nu`, `data.qpos`, `data.qvel`, `data.ctrl`,
`data.time`, `data.orbit`, and `data.actuators`. Array units and frames are
specified in [frames and units](frames.md).

```{eval-rst}
.. autofunction:: mjorbit.mjo_forward
```

```{eval-rst}
.. autofunction:: mjorbit.mjo_step
```

## Configuration

```{eval-rst}
.. autoclass:: mjorbit.OrbitInit
```

```{eval-rst}
.. autoclass:: mjorbit.MjoSpec
   :members: from_xml_path, from_xml_string, from_mj_spec, copy, compile, to_xml
   :undoc-members:
```

```{eval-rst}
.. autoclass:: mjorbit.MjoOrbitSpec
   :members:
   :undoc-members:
```

```{eval-rst}
.. autoclass:: mjorbit.CentralBodySpec
```

```{eval-rst}
.. autoclass:: mjorbit.SurfaceSpec
```

```{eval-rst}
.. autoclass:: mjorbit.MagneticBodySpec
```

```{eval-rst}
.. autoclass:: mjorbit.ReactionWheelSpec
```

```{eval-rst}
.. autoclass:: mjorbit.MagnetorquerSpec
```

```{eval-rst}
.. autoclass:: mjorbit.ThrusterSpec
```

```{eval-rst}
.. autoclass:: mjorbit.ControlMomentGyroSpec
```

## Rollout and planning

```{eval-rst}
.. autofunction:: mjorbit.mjo_state_size
```

```{eval-rst}
.. autofunction:: mjorbit.mjo_control_size
```

```{eval-rst}
.. autofunction:: mjorbit.mjo_get_state
```

```{eval-rst}
.. autofunction:: mjorbit.mjo_set_state
```

```{eval-rst}
.. autofunction:: mjorbit.rollout
```

```{eval-rst}
.. autoclass:: mjorbit.planning.MppiConfig
```

```{eval-rst}
.. autoclass:: mjorbit.planning.MppiPlanner
   :members: reset, update_action, action
   :undoc-members:
```

```{eval-rst}
.. autofunction:: mjorbit.planning.make_spline
```
