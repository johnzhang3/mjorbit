# mujoco_orbit

Default NumPy + MuJoCo simulator for coupled orbital dynamics and MuJoCo multibody dynamics.

## Project layout

- `src/mujoco_orbit/` — main package (hatchling src layout)
- `src/mujoco_orbit/` — default simulator (NumPy + standard MuJoCo)
- `tests/mujoco_orbit/` — unit and integration tests for the default path
- `examples/` — runnable example scripts

## Development

```
uv sync
uv run pytest tests/
```

## Units

All physics quantities use km, s, kg, rad, T unless stated otherwise.
- distances: km
- velocities: km/s
- accelerations: km/s²
- forces: kg·km/s² (= kN)
- torques: kg·km²/s²
- magnetic field: T

MuJoCo uses SI (m, s, kg) internally. Conversions happen at the MuJoCo boundary.

## MuJoCo frame conventions

These conventions are critical for correctness — getting them wrong causes silent energy/momentum non-conservation.

- **`qvel[3:6]`** for a free joint is angular velocity in the **body frame** (child frame), NOT the world frame.
- **`xmat`** is the body-frame orientation matrix (world-from-body). Use this for body-frame ↔ world-frame transforms.
- **`ximat`** is the **inertia-frame** orientation matrix (world-from-principal-axes). This equals `xmat` only when `body_iquat` is identity (i.e., diagonal inertia aligned with body axes). **Do NOT use `ximat` as a body rotation matrix.**
- **`xfrc_applied`** torques are in the **world frame**. To apply a body-frame torque: `xfrc[bid, 3:] = xmat @ tau_body`.
- **`body_iquat`**: rotation from body frame to inertia (principal axes) frame. When inertia is diagonal in body frame, this is identity.
- **`fullinertia`** XML attribute: MuJoCo internally eigendecomposes into `body_inertia` (diagonal, possibly reordered) + `body_iquat` (non-trivial rotation). The full tensor is preserved. To set non-diagonal inertia, use `fullinertia` in the XML — do NOT modify `body_inertia`/`body_iquat` post-compilation (changes may not take effect on the dynamics).

## Key design rules

- The default path is the single-world correctness baseline, not a performance simulator.
- Per-body gravity/J2 forces (not a single rigid-body gradient torque formula).
- Explicit flat-plate surface metadata for drag/SRP (not inferred from MuJoCo geoms).
- Reaction wheels, magnetorquers, thrusters are external actuator state — not MuJoCo joints.
- Bidirectional coupling only through net external wrench on the system.
