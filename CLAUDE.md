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

## Key design rules

- The default path is the single-world correctness baseline, not a performance simulator.
- Per-body gravity/J2 forces (not a single rigid-body gradient torque formula).
- Explicit flat-plate surface metadata for drag/SRP (not inferred from MuJoCo geoms).
- Reaction wheels, magnetorquers, thrusters are external actuator state — not MuJoCo joints.
- Bidirectional coupling only through net external wrench on the system.
