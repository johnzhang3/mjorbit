# Installation

## Source checkout with Pixi

Install [Pixi](https://pixi.sh/latest/installation/), then:

```bash
git clone https://github.com/johnzhang3/mjorbit.git
cd mjorbit
pixi install
pixi run example-minimal
pixi run viewer --task free_drift
```

The minimal example prints a simulation time of `1.00 s`. The viewer prints a
local URL, normally `http://localhost:8080`; open it in a browser. Stop the
server with Ctrl+C.

Pixi installs Python, CMake, Ninja, and a C++ compiler, and builds the native
MuJoCo plugin and Python extension. The first install needs network access and
takes longer than subsequent launches. Run commands from the repository root.

## Platform and environment choices

Python 3.11 and 3.12 are supported. The default environment uses 3.12.

| Environment | Purpose | Platforms configured in Pixi |
| --- | --- | --- |
| `default`, `py311`, `py312` | CPU simulator, browser viewer, MPPI, tests | Linux x86-64; macOS Intel and Apple Silicon |
| `frames` | Named-frame and absolute-epoch input through Astropy | Same as CPU |
| `report` | Plotting and analysis | Same as CPU |
| `docs` | Build and preview this guide | Same as CPU |
| `warp` | Batched MJWarp backend, including frame conversions | Linux x86-64; NVIDIA GPU for GPU execution |
| `rl` | PPO training with Torch and rsl-rl | Linux x86-64; NVIDIA GPU |

The configured platform list is distinct from a binary-wheel support promise.
The project currently documents source installation; Windows and a portable
prebuilt-wheel matrix are not established release targets.

```bash
pixi install -e frames
pixi run -e frames test-frames
pixi install -e report
pixi install -e warp
pixi run -e warp example-batched
pixi install -e rl
```

Keep the repository's locked MuJoCo/MJWarp versions when reproducing results.
The CPU package requires MuJoCo `>=3.7,<3.8` and NumPy `>=1.26,<2`.
The [GPU guide](gpu.md) covers additional behavior differences.

## Use from another Python project

With Python 3.11 or 3.12 and a C++17 compiler installed, a source checkout can
be installed into an existing virtual environment:

```bash
python -m pip install /path/to/mjorbit
# Editable development install:
python -m pip install -e /path/to/mjorbit
# Optional features:
python -m pip install '/path/to/mjorbit[frames]'
```

Use `python -m pip` from the environment that will run the simulator. The build
backend obtains MuJoCo headers/libraries and nanobind, and provisions CMake/Ninja
when needed. It still requires the platform compiler and system development
tools. Then run `mjo-viewer --list-tasks` or the [minimal Python program](quickstart.md).

Only `mjorbit`, `mjorbit_warp`, `viewer`, their assets, and native libraries are
installed. Clone the repository for `examples/`, `experiments/`, and `scripts/`.

## Common setup problems

- **Native binding import failure:** rebuild in the environment you are using:
  `pixi run sync-package`. Check Python and MuJoCo versions before reusing a
  binary from another environment. Do this after switching Pixi environments
  if invoking Python directly; the named test, example, viewer, and docs tasks
  synchronize the native build automatically. Avoid running native builds in
  different environments concurrently because they share `build/cpp`.
- **Port already in use:** select `pixi run viewer --port 8081` and open that URL.
- **No browser window appears:** open the URL printed by the viewer manually.
- **GPU environment unavailable on macOS:** use the CPU environment; the Pixi
  GPU environments are restricted to Linux.
- **Plotting import failure:** run the analysis in `pixi run -e report ...`.
- **Missing Astropy:** use `pixi run -e frames ...` when providing a noncanonical
  input frame or an absolute epoch.
