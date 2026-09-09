# Actuators and sensors

For gravity-gradient, atmospheric drag, solar pressure, and residual magnetic
loads, see [forces and disturbances](forces-disturbances.md).

Standard MuJoCo actuator controls live in `data.ctrl`. Spacecraft devices use
separate buffers in `data.actuators`, populated in the order the devices were
added to the model specification.

| Device | Runtime command | Units | Limits |
| --- | --- | --- | --- |
| Reaction wheel | `rw_torque_cmd` | N·m | Optional torque and wheel-speed limits |
| Magnetorquer | `mtq_dipole_cmd` | A·m² | Symmetric dipole limit; magnetic model must be enabled |
| Thruster | `thr_force_cmd` | N | Clamped from zero to `force_limit` |
| Control moment gyro | `cmg_gimbal_rate_cmd` | rad/s | Optional gimbal-rate and angle limits |

Commands are held across steps until overwritten or reset. Reaction-wheel
torque changes rotor momentum and applies the opposite torque to the attached
body. Thruster direction and application point are body-frame metadata.
CMGs are currently available only on the CPU backend.

After building the reaction-wheel model from [modeling](modeling.md), create
data with the orbit from the [quick start](quickstart.md), then command it:

```python
data.actuators.rw_torque_cmd[0] = 0.01  # N·m
mjo_step(model, data)
```

Only index a device buffer when that device exists in the compiled model.
A free-body model with no wheel has an empty `rw_torque_cmd` array.

## Sensors

Declare sensors in the MuJoCo XML. `model.sensor(name)` and
`model.sensors.descriptors` expose metadata, while `data.sensordata` contains
the native sensor values. The CPU measurement namespace adds bias/noise
handling:

```python
measurements = data.sensors.measure_all(noisy=False)
noisy_measurements = data.sensors.measure_all(noisy=True)
```

Pass `rng_seed` to `model.make_data(...)` for reproducible runtime noise.
For one configured sensor, use `data.sensors.measure(name, noisy=True)`.
Axis and quaternion measurements use rotational perturbations and remain
normalized. The GPU backend exposes native sensor buffers but does not yet
implement the noisy measurement namespace.

MuJoCo evaluates sensors before integrating a step. Rollout sensor row `k`
therefore corresponds to the start of step `k`, while packed state row `k`
is the state after the step. Account for this offset when mixing states and
sensors in a cost or comparison.
