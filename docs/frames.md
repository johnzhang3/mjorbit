# Frames and units

The chief/reference orbit and the robot simulation use different length units.
Keep the conversion at the interface explicit.

| Quantity | Frame | Units |
| --- | --- | --- |
| `OrbitInit.R_eci`, `data.orbit.R_eci` | Absolute Earth-centered inertial (ECI) | km |
| `OrbitInit.V_eci`, `data.orbit.V_eci` | Absolute ECI | km/s |
| Free-joint `qpos[:3]`, body `xpos` | Chief-centered local inertial, axes parallel to ECI | m |
| Free-joint `qvel[:3]` | Local inertial velocity relative to chief | m/s |
| Free-joint `qvel[3:6]` | Body angular velocity | rad/s |
| `data.eci_position_from_world(...)` | Absolute ECI | **m** |
| `data.eci_velocity_from_world(...)` | Absolute ECI | **m/s** |
| `data.xfrc_applied[:, :3]` / `[:, 3:]` | World force / torque | N / N·m |
| `data.time` | Simulation time | s |
| `data.orbit.t` | Environment clock, referenced to J2000.0 | s |

`qpos[:3]` applies to a model whose first joint is free. For models with more
joints, use their addresses rather than assuming every body has this layout.

## Chief-centered world

MuJoCo `world` has its origin at the propagated chief. Its axes stay parallel
to ECI; they do not rotate with the orbit. XML free-body positions are offsets
from that origin. Do not put the chief's absolute position or velocity in the
MuJoCo initial state.

For a local position $r$ in meters, the absolute position is
$R_{\mathrm{ECI,m}} = 1000 R_{\mathrm{chief,km}} + r$.
The Python ECI conversion helpers return SI values even though `data.orbit`
stores km and km/s.

## LVLH

LVLH is the derived rotating frame: x is radial outward, y is along-track,
and z is the orbit normal. Use the conversion helpers, including the position
argument when converting velocity; it accounts for frame rotation.

Continuing the [quick start](quickstart.md):

```python
position_lvlh_m = np.array([10.0, 0.0, 0.0])
velocity_lvlh_m_s = np.zeros(3)
data.qpos[:3] = data.world_position_from_lvlh(position_lvlh_m)
data.qvel[:3] = data.world_velocity_from_lvlh(position_lvlh_m, velocity_lvlh_m_s)
mjo_forward(model, data)
```

The inverse helpers are `lvlh_position_from_world` and
`lvlh_velocity_from_world`. `data.frame` exposes the current derived frame.

## Body and inertia axes

`xmat` maps body vectors to world vectors. `ximat` maps principal-inertia
vectors to world vectors; these matrices differ when `body_iquat` is nonidentity.
For a body-frame torque, write:

```python
body_id = model.body_id("spacecraft")
tau_body_nm = np.array([0.0, 0.0, 0.01])
data.xfrc_applied[body_id, 3:] = data.xmat[body_id].reshape(3, 3) @ tau_body_nm
```

Use XML `fullinertia` for a nondiagonal inertia tensor. MuJoCo decomposes it
into principal moments and an inertia orientation during compilation; editing
only `body_inertia` or `body_iquat` afterwards does not rebuild the dynamics.

## Actuator and environment units

Spacecraft actuator commands use SI: reaction-wheel torque in N·m,
magnetorquer dipole in A·m², thruster force in N, and CMG gimbal rate in rad/s.
Their `*Spec` limits use the same units. Do not scale these commands to km.
See [actuators and sensors](actuators-sensors.md).

Central-body orbital parameters use km: `gm` is km³/s² and `radius` is km.
Surface positions and areas use m and m²; mass and inertia use kg and kg·m².

## Epoch-aware orbit inputs

The default `OrbitInit(..., frame="ECI", epoch=None)` needs no optional
frame-conversion dependencies. Install the `frames` environment for
epoch-aware GCRF, TEME, or ITRF inputs. Epoch-dependent input frames require
an epoch and are normalized to the canonical inertial state. Supplying an
epoch sets the environment clock to its J2000.0 offset. These conversions do
not change MuJoCo's local world convention. See `OrbitInit` in the
[API reference](api.md) and the frame conversion tests for supported inputs.
