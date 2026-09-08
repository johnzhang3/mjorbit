# Contributing to mjorbit

For bug reports, include the mjorbit revision, operating system, Python/MuJoCo
versions, backend, and a small runnable example. For dynamics issues, specify
frames, units, timestep, and the expected invariant or reference result.

## Set up

```bash
git clone https://github.com/johnzhang3/mjorbit.git
cd mjorbit
pixi install
pixi run example-minimal
```

Use `pixi` for development commands. Python 3.11 and 3.12 are supported.
Optional environments include `docs`, `report`, `frames`, `warp`, and `rl`;
the GPU and RL environments target Linux x86-64.

## Make a change

Keep the CPU runtime under `src/mjorbit/`, GPU code under `src/mjorbit_warp/`,
native implementation under `src/cpp/`, and browser integration under
`src/viewer/`. See the [architecture guide](docs/architecture.md).

Use explicit type hints, four-space indentation, and focused modules. Ruff
checks imports and the 100-character line limit. Public examples should use
`MjoModel`, `model.make_data(...)`, `mjo_forward`, and `mjo_step`. Keep backend
selection explicit by import path.

Before modifying dynamics, read [frames and units](docs/frames.md). Keep
deterministic numeric tests beside the subsystem in `tests/mjorbit/` or
`tests/mjorbit_warp/`. Runtime API changes also need public model/data test
coverage. Guard optional GPU tests with `pytest.importorskip`.

```bash
pixi run lint
pixi run typecheck
pixi run test
pixi run cpp-test
pixi run -e docs docs-build
```

For packaging changes run `pixi run package-check`. For optional subsystems,
also run `pixi run -e frames test-frames` or `pixi run -e warp test-warp` in
the appropriate environment. Run changed analysis scripts directly and keep
their generated outputs intentional.

## Pull requests

Use a short imperative commit subject. Explain the problem, resulting behavior,
and checks performed; record any hardware or external dependency limitations.
Include screenshots for viewer changes and artifact notes for changes to
paper figures. Keep the documentation and runnable examples consistent with
the implementation.

`examples/`, `experiments/`, and `scripts/` remain in the source repository for
learning and reproduction. Keep generated builds, videos, trajectories, and
training outputs out of commits unless they are deliberate reference artifacts.
When adding third-party assets, include their source, creator, license/terms,
and a description of modifications beside the files.
