# Forces and disturbances

mjorbit couples an absolute reference orbit to MuJoCo's local multibody dynamics.
The built-in environment includes central gravity, the $J_2$ oblateness term,
gravity-gradient torque, atmospheric drag, solar radiation pressure (SRP), and a
centered magnetic dipole field. This guide describes the implemented equations,
configuration, and approximations. See the [paper](https://arxiv.org/abs/2609.08010)
for the framework and evaluation.

## Select the models

The five `spec.mjorbit.use_*` flags below default to `True` for a new spec;
an XML file can override them. Configure the spec before compiling it.

| Contribution | Switch | Required model information |
| --- | --- | --- |
| Point-mass chief acceleration and differential gravity | Always active | Central-body `gm`; each body's mass and position |
| $J_2$ chief acceleration and differential gravity | `use_j2` | Central-body `j2` and `radius` |
| Gravity-gradient torque | `use_gravity_gradient` | Each body's mass, inertia, position, and orientation |
| Atmospheric drag | `use_drag` | At least one `SurfaceSpec` with `use_drag=True` |
| Solar radiation pressure | `use_srp` | At least one `SurfaceSpec` with `use_srp=True` |
| Residual magnetic and magnetorquer torque | `use_magnetic` | A `MagneticBodySpec` or commanded `MagnetorquerSpec` |

A visible mesh or geom does not automatically become an aerodynamic or optical
surface. Define the panels explicitly. Both the global flag and the individual
surface's flag must be enabled for that surface to receive a load.

This complete CPU example adds two sides of a panel and a residual magnetic
dipole to the bundled free body:

```python
import numpy as np

from mjorbit import MagneticBodySpec, MjoSpec, OrbitInit, SurfaceSpec, mjo_forward, mjo_step
from mjorbit.testdata import FREE_BODY_XML

spec = MjoSpec.from_xml_path(FREE_BODY_XML)
spec.mjorbit.use_j2 = True
spec.mjorbit.use_gravity_gradient = True
spec.mjorbit.use_drag = True
spec.mjorbit.use_srp = True
spec.mjorbit.use_magnetic = True

# These are the default Earth atmosphere parameters.
central = spec.mjorbit.central_body
central.atmosphere_h0 = 400.0             # km above the reference radius
central.atmosphere_rho0 = 2.62e-13        # kg/m^3 at that altitude
central.atmosphere_scale_height = 58.2   # km

for name, normal in [("panel_front", [0.0, 1.0, 0.0]),
                     ("panel_back", [0.0, -1.0, 0.0])]:
    spec.mjorbit.add_surface(SurfaceSpec(
        name=name,
        body_name="spacecraft",
        center_of_pressure_body=[0.0, 0.0, 0.5],  # m from body COM, in body axes
        normal_body=normal,
        area=2.0,       # m^2 per side
        drag_coeff=2.2,
        srp_coeff=1.8,
    ))

spec.mjorbit.add_magnetic_body(MagneticBodySpec(
    name="residual_dipole",
    body_name="spacecraft",
    dipole_body=[0.0, 0.0, 0.05],  # A m^2
))

radius_km = central.radius + 400.0
orbit = OrbitInit(
    R_eci=[radius_km, 0.0, 0.0],
    V_eci=[0.0, np.sqrt(central.gm / radius_km), 0.0],
    t=0.0,  # environment time in seconds since J2000
)
model = spec.compile(mj_timestep=0.01)
data = model.make_data(orbit=orbit)
mjo_forward(model, data)
for _ in range(100):
    mjo_step(model, data)

print("Chief atmospheric density [kg/m^3]:", data.env.atm_density)
print("Chief illumination factor:", data.env.eclipse)
print("Chief magnetic field [T]:", data.env.mag_field_eci)
```

To anchor the environment to a calendar date, install the optional frames extra
and pass an `epoch` to `OrbitInit`; see [epoch-aware inputs](frames.md#epoch-aware-orbit-inputs).
The initial velocity above is the point-mass circular value;
it is not an exact circular solution with $J_2$ and disturbances enabled.

## Coordinates and units

Let $\mathbf R$ and $\mathbf V$ be the chief's absolute ECI position and velocity
in **km** and **km/s**. Let $\mathbf x_i$ be body $i$'s center of mass (COM) in
MuJoCo world coordinates, in **m**. Its absolute position is

```{math}
\mathbf r_i = \mathbf R + 10^{-3}\mathbf x_i.
```

MuJoCo world axes are parallel to ECI; the origin follows the chief. They do not
rotate with LVLH. Forces below are in **N** and torques in **N m**, expressed in
world axes. $Q_i$ denotes `data.xmat[i]`, the world-from-body rotation. See
[frames and units](frames.md) for conversions and inertia-frame conventions.

## Gravity and gravity gradient

### Chief acceleration and differential gravity

For $\mathbf r=(x,y,z)$ in km and $r=\|\mathbf r\|$, the point-mass acceleration
in km/s² is

```{math}
\mathbf g_0(\mathbf r) = -\frac{\mu}{r^3}\mathbf r,
```

where `central_body.gm` is $\mu$ in km³/s². With `use_j2=True`, mjorbit adds

```{math}
\mathbf g_{J_2}(\mathbf r) =
\frac{3J_2\mu a^2}{2r^5}
\begin{bmatrix}
x(5z^2/r^2-1)\\
y(5z^2/r^2-1)\\
z(5z^2/r^2-3)
\end{bmatrix},
```

where $a$ is `central_body.radius` in km. The symmetry axis of this term is
**ECI z**; changing `central_body.omega` does not rotate the gravity field.
Default Earth values are $\mu=398600.4418$ km³/s², $a=6378.137$ km, and
$J_2=1.08262668\times10^{-3}$.

Each positive-mass MuJoCo body receives the COM force

```{math}
\mathbf F_{g,i} = 10^3 m_i\left[\mathbf g(\mathbf r_i)-\mathbf g(\mathbf R)\right].
```

The point-mass difference uses an algebraically equivalent **Encke formulation**
to avoid subtracting nearly equal accelerations when body offsets are small.
The $J_2$ difference is evaluated separately. This is a nonlinear differential
gravity model; it does not use the linear Clohessy–Wiltshire equations to step
the simulation. Compilation sets uniform MuJoCo gravity to zero.

### Torque within each rigid body

The variation of gravity across a body's mass distribution also produces a
torque. With `use_gravity_gradient=True`, each positive-mass body receives

```{math}
\boldsymbol\tau_{gg,i} =
\frac{3\mu}{r_i^3}\,
\hat{\mathbf r}_i\times\left(J_i^W\hat{\mathbf r}_i\right),
\qquad
J_i^W = Q_{I,i}\,\operatorname{diag}(I_{1,i},I_{2,i},I_{3,i})\,Q_{I,i}^{T}.
```

Here $Q_{I,i}$ is **`data.ximat[i]`**, the world-from-principal-inertia rotation,
and the principal moments come from `model.body_inertia[i]` in kg m². The
factor $\mu/r_i^3$ has units s⁻², so this expression directly gives N m with
the orbital km units. It handles non-diagonal body-frame inertia compiled from
XML `fullinertia`.

This is the leading point-mass gravity-gradient torque. Enabling $J_2$ does
**not** add a $J_2$ gravity-gradient torque. The bodywise COM forces above also
produce moments through the articulated structure. Disabling
`use_gravity_gradient` removes only the within-body torque, while differential
gravity forces remain active.

## Surface geometry and application points

Each `SurfaceSpec` is a one-sided flat plate with area $A$, outward normal
$\mathbf n_B$, and center-of-pressure offset $\mathbf c_B$. Compilation requires
positive area, nonnegative drag/SRP coefficients, and a nonzero normal, which it
normalizes. The world normal, lever arm, and absolute surface position are

```{math}
\mathbf n = Q_i\mathbf n_B,\qquad
\mathbf c = Q_i\mathbf c_B,\qquad
\mathbf r_p = \mathbf R + 10^{-3}(\mathbf x_i+\mathbf c).
```

**`center_of_pressure_body` is an offset from the body's COM, expressed in body
axes.** If a point is given relative to the XML body origin, subtract
`model.body_ipos[i]` to obtain this offset. This differs from
`ThrusterSpec.position_body`, which is relative to the body origin.

Drag and SRP forces on a surface are summed, then applied as a force at the
body COM plus the equivalent torque

```{math}
\boldsymbol\tau_p = \mathbf c\times(\mathbf F_D+\mathbf F_{SRP}).
```

Back-facing surfaces contribute zero for the corresponding load. Represent a
plate exposed on both sides with two surfaces having opposite normals, as in
the example. Normals and offsets rotate with their attached MuJoCo body.

## Atmospheric drag

The atmospheric load is a drag force based on **dynamic pressure**. There is
no separate static atmospheric pressure or buoyancy model.

### Density and relative velocity

The atmosphere is a single exponential profile:

```{math}
h = \|\mathbf r_p\|-a,\qquad
\rho(h) = \rho_0\exp\!\left[-\frac{h-h_0}{H}\right].
```

`atmosphere_h0` ($h_0$) and `atmosphere_scale_height` ($H$) are in km;
`atmosphere_rho0` ($\rho_0$) is in kg/m³. Defaults are 400 km, 58.2 km, and
$2.62\times10^{-13}$ kg/m³, respectively. Nonpositive reference density or
scale height yields zero density. Density is evaluated at **each surface's
position**, rather than using the chief's cached density.

The atmosphere co-rotates rigidly with `central_body.omega`, by default
$(0,0,7.292115\times10^{-5})$ rad/s. The implemented surface velocity and its
velocity relative to the atmosphere, both in m/s, are

```{math}
\mathbf v_p = 10^3\mathbf V + \mathbf v_c + \boldsymbol\omega_i\times\mathbf c,
\qquad
\mathbf u = \mathbf v_p - 10^3(\boldsymbol\Omega\times\mathbf r_p),
```

where $\mathbf v_c$ is the linear part of `data.cvel[i]` and
$\boldsymbol\omega_i$ is its angular part, in world axes.

```{note}
The current CPU and GPU surface implementations use the linear `cvel` component
as the body COM velocity. MuJoCo expresses this spatial velocity about the
kinematic tree's COM. For a rotating articulated tree whose body COM differs
from that reference point, the code omits the corresponding velocity shift.
The expression is correct when those points coincide (including the isolated
rigid body in the example), or when the omitted angular cross product is zero.
This is a current implementation limitation for articulated-body drag, not an
additional physical approximation to the atmosphere. See MuJoCo's
[spatial-coordinate convention](https://mujoco.readthedocs.io/en/stable/computation/index.html#general-framework).
```

### Force law

For speed $u=\|\mathbf u\|$ and direction $\hat{\mathbf u}=\mathbf u/u$,

```{math}
A_D = A\max(0,\mathbf n\cdot\hat{\mathbf u}),\qquad
q_D = \tfrac12\rho u^2,\qquad
\mathbf F_D = -q_D C_D A_D\hat{\mathbf u}.
```

`drag_coeff` is $C_D$ (default 2.2). The normal must point toward the direction
of travel relative to the atmosphere to expose the panel. Force opposes that
velocity. At speeds below the implementation threshold of $10^{-10}$ m/s,
drag is zero.

This model has no winds, lift, temperature dependence, gas-surface scattering,
or space-weather variation. The exponential profile is not a layered or
empirical thermosphere model, and its parameters require calibration for the
altitudes and conditions of interest. No aerodynamic shielding by other panels
or meshes is computed.

## Solar radiation pressure

Let $\hat{\mathbf s}$ be the ECI unit vector **toward the Sun**. For illumination
factor $\chi\in\{0,1\}$, mjorbit computes

```{math}
A_S = A\max(0,\mathbf n\cdot\hat{\mathbf s}),\qquad
\mathbf F_{SRP} = -\chi P_\odot C_R A_S\hat{\mathbf s}.
```

`srp_coeff` is the scalar $C_R$ (default 1.8). The pressure is the constant
$P_\odot=4.56\times10^{-6}$ N/m². The minus sign pushes the panel away from the
Sun. A directly illuminated 1 m² panel with $C_R=1.8$ receives 8.208 µN.

The force is always along the Sun-to-spacecraft direction. `srp_coeff` scales
its magnitude; it does not select separate specular and diffuse reflection
directions. There is no Sun-distance scaling, albedo, planetary infrared
pressure, thermal recoil, or spacecraft self-shadowing.

### Sun direction and eclipse

The Sun direction uses an analytic solar-longitude approximation. With $T$ in
Julian centuries since J2000, the angles below are in degrees:

```{math}
\begin{aligned}
T &= t/(36525\cdot86400),\\
L &= 280.460+36000.771T,\\
M &= 357.528+35999.050T,\\
\lambda &= L+1.915\sin M+0.020\sin(2M),\\
\epsilon &= 23.439-0.013T,\\
\hat{\mathbf s} &= (\cos\lambda,\;\sin\lambda\cos\epsilon,\;
\sin\lambda\sin\epsilon).
\end{aligned}
```

The implementation converts degree arguments to radians for trigonometric
functions. It uses `data.orbit.t`, the environment time in seconds since J2000,
rather than the elapsed `data.time`. This is an Earth-oriented approximation,
not a general planetary ephemeris.

Eclipse is a hard-edged cylindrical shadow behind the central body. For each
surface, define

```{math}
d_\perp = \left\|\mathbf r_p-
(\mathbf r_p\cdot\hat{\mathbf s})\hat{\mathbf s}\right\|.
```

The surface is dark ($\chi=0$) when $\mathbf r_p\cdot\hat{\mathbf s}\leq0$ and
$d_\perp<a$; otherwise $\chi=1$. At the cylinder's radial boundary it is lit.
The Sun direction is shared across the scene, while eclipse is evaluated
separately at each surface. There is no penumbra or finite solar disk.

## Magnetic disturbances

The magnetic environment is a centered dipole evaluated at the **chief**:

```{math}
\mathbf B(\mathbf R,t) = B_0\left(\frac{a}{\|\mathbf R\|}\right)^3
\left[3(\hat{\mathbf m}(t)\cdot\hat{\mathbf R})\hat{\mathbf R}
-\hat{\mathbf m}(t)\right].
```

`magnetic_b0` is $B_0$ in tesla (default $3.12\times10^{-5}$ T).
`magnetic_axis` is normalized and interpreted in central-body-fixed axes; the
default is $(0,0,-1)$. It rotates into ECI about `central_body.omega` by
$4.894961212823756+\|\boldsymbol\Omega\|t$ radians. For zero spin it stays fixed.
The default axis is aligned with the spin axis, so the default field has no
rotation-induced time variation at a fixed ECI position.

A residual dipole $\mathbf m_B$ from `MagneticBodySpec.dipole_body`, in A m²,
produces

```{math}
\boldsymbol\tau_B = \mathbf m_B\times(Q_i^T\mathbf B),\qquad
\boldsymbol\tau_W = Q_i\boldsymbol\tau_B.
```

Magnetorquers use the same law with their commanded, limited dipole. This model
applies torque only: it has no magnetic-gradient translational force, eddy
currents, or high-order geomagnetic field model. All bodies use the chief's
field, even when they have different local offsets. Setting `use_magnetic=False`
disables both residual-dipole and magnetorquer torques.

## Coupling back to the reference orbit

Drag, SRP, and thrusters also change the chief orbit. Let $\mathbf F_{ext}$ be
their total world force across the scene and $M_{total}$ the sum of positive
MuJoCo body masses. The reference acceleration includes

```{math}
\mathbf a_{feedback} = 10^{-3}\frac{\mathbf F_{ext}}{M_{total}}
\quad\text{(km/s²)}.
```

The local dynamics receive the corresponding origin-acceleration force
$-10^3m_i\mathbf a_{feedback}$ on every positive-mass body. This accounts for the
chief's acceleration while retaining each body's relative response to its own
loads. Differential gravity and torques do not enter the external-force sum;
manually supplied MuJoCo forces are not automatically included in this sum.
Wheel and CMG momentum exchange, thruster commands, and magnetorquer limits are
described in [actuators and sensors](actuators-sensors.md).

The chief uses RK4 propagation. `spec.mjorbit.orbit_dt` defaults to the MuJoCo
timestep. A smaller value substeps the chief with the current feedback held
constant. A larger value predicts an orbit segment, linearly interpolates its
position and velocity between endpoints, and recomputes the endpoint using
the time-averaged feedback accumulated over the segment. MuJoCo continues to
step at its own timestep; this is not a single RK4 solve of the entire coupled
system. Check convergence with both timesteps for rapidly varying loads.

## Inspecting loads and backend scope

On CPU, call `mjo_forward(model, data)` to refresh derived quantities for the
current state. `data.env` exposes the chief's Sun direction, eclipse flag,
magnetic field, atmospheric rotation, and density. The cached density and
eclipse are diagnostics at the chief; the surface calculations reevaluate them
at their own positions.

`data.wrench_buffer[i, :3]` and `[i, 3:]` contain the accumulated mjorbit world
force and COM torque in N and N m. They include the origin-acceleration
correction and any mjorbit actuator contributions, so they are not an isolated
drag/SRP measurement or a sum of all MuJoCo contact forces. To isolate a model,
compare otherwise identical configurations with its flag enabled and disabled,
and account for the reference-orbit feedback.

The GPU backend implements the same force laws for the default Earth model,
with these current limits:

- Its gravity kernels use fixed Earth `gm`, `j2`, and gravity radius. Custom
  central-body gravity is supported by the CPU backend; do not assume those
  overrides carry over to GPU. GPU density, eclipse, and magnetic kernels do
  use the configured environment parameters.
- Device multibody state, chief position/velocity, and RK4 propagation use
  float32. The absolute epoch anchor, solar/magnetic angle accumulation, and
  cancellation-sensitive $J_2$ differencing use float64. Precision can affect
  altitude-dependent loads and hard eclipse transitions.
- The articulated-body drag velocity limitation described above is shared by
  both backends. CMGs and noisy sensor measurements remain CPU-only.

The built-in gravity field stops at point mass plus $J_2$: there are no higher
spherical harmonics, third-body gravity, relativistic corrections, or mutual
gravitational attraction between spacecraft. Contact and joint forces remain
MuJoCo's responsibility. See the [GPU guide](gpu.md) for synchronization and
execution details.

## Implementation references

The source of truth for the equations and defaults is:

- [Gravity](https://github.com/johnzhang3/mjorbit/blob/main/src/cpp/src/gravity.cc)
  and [environment models](https://github.com/johnzhang3/mjorbit/blob/main/src/cpp/src/environment.cc).
- [Force and torque assembly](https://github.com/johnzhang3/mjorbit/blob/main/src/cpp/src/coupling_passive.cc)
  and [orbit scheduling](https://github.com/johnzhang3/mjorbit/blob/main/src/cpp/src/orbit_schedule.cc).
- [Public surface/dipole specs](https://github.com/johnzhang3/mjorbit/blob/main/src/mjorbit/config.py),
  [central-body configuration](https://github.com/johnzhang3/mjorbit/blob/main/src/mjorbit/spec.py),
  and [GPU kernels](https://github.com/johnzhang3/mjorbit/tree/main/src/mjorbit_warp/device/kernels).

The repository's `test_gravity_gradient.py`, `test_surface_loads.py`,
`test_environment.py`, and `test_inertial_wrenches.py` under `tests/mjorbit/`
exercise analytic torques, panel force scaling, shadow/density behavior, and
relative dynamics. Passing these checks validates those cases; it does not
establish fidelity against a high-resolution atmosphere or ephemeris.
