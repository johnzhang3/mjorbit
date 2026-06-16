# Viewer video recorder

Headless, deterministic recording of the `viewer` scenes to mp4 — no manual
browser screen-capture. A Chrome window renders the viser scene with the real
GPU and `client.get_render` pulls frames offscreen; frames are piped straight to
`ffmpeg` (libx264). The simulation is stepped offline, so output is smooth and
reproducible even for expensive MPPI controllers.

A Chrome window opens during a render (it must use the real GPU; macOS headless
falls back to SwiftShader, which drops the Earth). Leave it alone for the ~1 min
it takes.

## Two non-obvious requirements (see `scene.py`)

1. **World scale.** viser's offscreen `get_render` drops geometry beyond a few
   thousand world units, so the 6371 km Earth never appears at true scale. Every
   render coordinate is multiplied by `world≈1e-4`; uniform scaling leaves the
   image unchanged. The spacecraft is separately magnified (`mag`) so it reads
   against the globe.
2. **Visible Chrome window** (`HeadlessRecorder(headless=False)`, default) for
   the real Metal GPU.

## Produce the paper clips

```bash
# Banner — fleet of bimanual robots in LEO (CPU equivalent of banner_viewer_gpu)
pixi run python scripts/record/record_banner.py --out videos/banner.mp4 --nworld 60

# Autonomous docking (paper example b)
pixi run python examples/docking/main_mppi.py --duration 13 --save-traj /tmp/dock_traj.npz
pixi run python scripts/record/record_docking.py --traj /tmp/dock_traj.npz --out videos/docking.mp4

# Grasping under gravity gradient (paper example c)
pixi run python scripts/record/produce_grasp.py --out /tmp/grasp_traj.npz
pixi run python scripts/record/record_grasp.py --traj /tmp/grasp_traj.npz --out videos/grasping.mp4

# RL truss servicing (paper example d) — needs the warp/GPU `rl` env + a trained
# checkpoint (examples/ppo). produce_hug composes scripted fly-in + trained
# hug/stabilize; the spacecraft hugs a large free truss and slews it to Earth.
pixi run -e rl python scripts/record/produce_hug.py \
    --checkpoint examples/ppo/logs/<run>/model_<it>.pt --out /tmp/hug_traj.npz
pixi run -e rl python scripts/record/record_truss.py \
    --traj /tmp/hug_traj.npz --out videos/truss_pointing.mp4 --clip-sim-seconds 0
```

Common knobs: `--seconds`, `--fps`, `--width/--height`, `--mag`, camera offsets
in the per-scene `record_*.py`. Trajectory clips (`docking`, `grasping`,
`truss_pointing`) run the controller/policy once to dump a `(qpos, R_eci)`
trajectory, then `render_traj.py` replays it.

## Files

- `recorder.py` — `HeadlessRecorder` (Chrome + get_render) and `VideoWriter` (ffmpeg).
- `scene.py` — world-scaled single-cluster scene (`SingleScene`) + framing helpers.
- `render_traj.py` — replay a saved trajectory through `SingleScene` to mp4.
- `record_docking.py`, `produce_grasp.py` + `record_grasp.py`, `record_banner.py`,
  `produce_hug.py` + `record_truss.py` — per-scene drivers.

## Not yet covered (paper Fig. example a)

- **(a) Multibody attitude control** — no implementation exists on `main` (paper
  `\todo`); needs a small task built first.
