# Rollouts and MPPI

The CPU backend provides batched open-loop rollouts and a spline-knot MPPI
planner. The planner samples control sequences, evaluates them with the full
coupled dynamics, and updates a nominal control spline using cost weights.

## Open-loop rollout

Continuing the [quick start](quickstart.md):

```python
from mjorbit import mjo_control_size, mjo_get_state, rollout

initial_state = mjo_get_state(model, data)
controls = np.zeros((8, 100, mjo_control_size(model)))
states, sensors = rollout(
    model, data, initial_state, controls, nthread=2
)
```

This evaluates eight trajectories, each with 100 steps. The free-body model
has no actuators, so the final control dimension is zero. `states` has shape
`(batch, steps, mjo_state_size(model))`; `sensors` has shape
`(batch, steps, model.nsensordata)`. Use `mjo_get_state` / `mjo_set_state` to
pack and restore the complete orbital and multibody state.

Rollout uses the supplied data as workspace. Preserve and restore plant state
if the same data is also being used for a live simulation. `nthread > 1`
allocates additional data instances sharing the compiled model.

The default control layout is:

```text
[MuJoCo ctrl | reaction-wheel torque | magnetorquer dipole | thruster force | CMG rate]
```

Use `mjo_control_size(model)` to allocate controls. Each orbital tail section
uses the [actuator units](actuators-sensors.md); MuJoCo controls retain the
units defined by their actuator configuration.

## Start with arm reach

```bash
pixi run example-mppi-arm-reach --nthread 2
pixi run viewer --task arm_reach_mppi
```

The [MPPI examples](https://github.com/johnzhang3/mjorbit/tree/main/examples/mppi)
show the full cost callback, control assignment, and receding-horizon loop.
Start with arm reach before the much longer capture-and-stabilize scenario.

`MppiConfig` controls the horizon, number of rollouts, spline nodes/order,
sampling noise, temperature, seed, and CPU threads. The cost callback receives
`states`, `sensors`, and `controls`, and returns one scalar per rollout, with
lower values preferred.

Create `MppiPlanner(model, config, cost_fn, ctrl_low=..., ctrl_high=...)`, then
call `planner.reset(data)`. `planner.update_action(data)` replans and returns
rollout costs. Read the resulting command with `planner.action(data.time)`
and assign its components to the plant before stepping.

The planner restores plant physics, orbit, and `data.ctrl` after its rollouts.
Orbital actuator commands and applied-force buffers are reset during rollout;
reapply them before advancing the plant. Sensor values in a rollout are one
step behind packed states, as described in [sensors](actuators-sensors.md).
