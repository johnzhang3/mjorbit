# mjorbit

CPU-first reference simulator for coupled orbital dynamics and MuJoCo multibody dynamics.

## Project layout

- `src/mjorbit/` — main package (hatchling src layout)
- `src/mjorbit/cpu/` — CPU reference simulator (NumPy + standard MuJoCo)
- `tests/cpu/` — unit and integration tests for the CPU path
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

- CPU path is single-world correctness baseline, not a performance simulator.
- Per-body gravity/J2 forces (not a single rigid-body gradient torque formula).
- Explicit flat-plate surface metadata for drag/SRP (not inferred from MuJoCo geoms).
- Reaction wheels, magnetorquers, thrusters are external actuator state — not MuJoCo joints.
- Bidirectional coupling only through net external wrench on the system.
