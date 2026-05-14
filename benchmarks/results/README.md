# Throughput benchmark results

Hardware: i7-12700K (8 P-cores + 4 E-cores; 20 logical), RTX 3080 (10 GiB).
Run config: `--nstep 2000 --nbatch 256`.

## Peak throughput (steps/s)

| Benchmark      | Mode    | CPU pure peak           | CPU orbit peak          | GPU pure peak              | GPU orbit peak             | GPU > CPU |
|----------------|---------|-------------------------|-------------------------|----------------------------|----------------------------|-----------|
| capture_arm    | zero    | 4.71M @ 20 threads      | 2.88M @ 20 threads      | **9.66M** @ 16384 worlds   | **8.58M** @ 16384 worlds   | 2.0× / 3.0× |
| capture_arm    | sin     | 4.60M @ 20 threads      | 2.68M @ 20 threads      | **9.70M** @ 16384 worlds   | **8.88M** @ 16384 worlds   | 2.1× / 3.3× |
| panel_deploy   | passive | 776k @ 20 threads       | 625k @ 20 threads       | **1.67M** @ 4096 worlds    | **1.54M** @ 4096 worlds    | 2.1× / 2.5× |

Numbers are total simulated steps per second (i.e. nbatch × nstep / wall on CPU,
nworld × nstep / wall on GPU).

## What unblocked the GPU numbers

The first version of these benchmarks called `mjw.step()` and `mjo_step()`
directly inside the timing loop, paying full Python-side dispatch cost on each
of MJWarp's ~30–50 kernel launches per step. That capped both backends at
~290k steps/s for capture_arm.

The fix: capture the per-step kernel sequence into a CUDA Graph once, then
replay it via `wp.capture_launch(graph)` on each step. This is the same
pattern MJWarp's own `testspeed.py` benchmark uses.

```python
# Warm up first (kernel JIT, allocations, etc.)
for _ in range(4):
    mjo_warp.mjo_step(model, data)
wp.synchronize()

# Capture
with wp.ScopedCapture() as capture:
    mjo_warp.mjo_step(model, data)
graph = capture.graph

# Hot loop just replays the graph
for k in range(nstep):
    wp.capture_launch(graph)
wp.synchronize()
```

For our orbit-aware step, this works cleanly because `mjo_step` is just
`refresh_core` → `step1` → `assemble_step_and_propagate` → `step2`, all of which
are pure `wp.launch` chains with no host-side branching after the first call.

Speedups vs. the pre–graph-capture results saved earlier:

| Benchmark      | Pure GPU before | Pure GPU after | Speedup | Orbit GPU before | Orbit GPU after | Speedup |
|----------------|-----------------|----------------|---------|------------------|-----------------|---------|
| capture_arm    | 290k            | 9.7M           | 33×     | 284k             | 8.9M            | 31×     |
| panel_deploy   | 244k            | 1.67M          | 6.8×    | 238k             | 1.54M           | 6.5×    |

Capture_arm gets the bigger speedup because its step does less actual work per
launch (smaller scene, fewer DOF), so launch overhead was a larger fraction of
total step time. Panel_deploy has heavier per-step work (12 hinges + denser
contact graph) so launch overhead was always proportionally smaller; even so,
graph capture is a clear win.

## Observations

- **Orbit overlay tax (CPU)**: 30–40% on capture_arm, 20% on panel_deploy.
  The lighter-weight scene has higher *relative* tax because the underlying
  MuJoCo step is cheap, so the per-step orbit work (gravity gradient, drag,
  RW gyroscopic torques, origin compensation) is a bigger fraction.
- **Orbit overlay tax (GPU)**: ~10–25% across the sweep. Higher than CPU's
  proportional cost because the orbit overlay adds two Warp kernels
  (`refresh_core`, `assemble_step_and_propagate`) that don't get to amortize
  across worlds as efficiently as MJWarp's batched solver kernels.
- **CPU vs GPU at small DOF (capture_arm, ~8 DOF)**: GPU wins by 2–3× at the
  full sweep. Below ~512 worlds, CPU is faster; the crossover sits around
  64–128 worlds, where MJWarp's per-step kernel cost amortizes over enough
  worlds to beat 20-thread CPU rollout.
- **CPU vs GPU at moderate DOF (panel_deploy, 18 DOF)**: GPU wins by 2–2.5×
  at peak. Crossover earlier (around 64 worlds) because the heavier per-step
  work tilts the comparison toward GPU's parallelism advantage faster.
- **GPU memory ceiling**: capture_arm fits 16k worlds comfortably (well under
  10 GiB if launched cold-but-warmed). 32k+ OOMs on cold start as MJWarp's
  EPA narrowphase scratch buffer scales with `naccdmax × nworld`. Panel_deploy
  with `njmax=200` peaks at 4096; beyond that the constraint buffers grow
  faster than scene-level throughput improves.

## TODO baselines (for the eventual figure)

The JSON files have stub entries for these — fill in from separate runs:

- **CPU**: Basilisk-MuJoCo on the same scenarios. Single-body Basilisk-MuJoCo
  is already wired up under `comparisons/basilisk_mujoco/`; the multi-body
  capture-arm and panel-deploy ports still need to be written.
- **GPU**: smallsatsim.github.io's flexible-array deploy demo on the same
  hardware. Their solver and frame conventions differ, so a fair comparison
  is total simulated-spacecraft-seconds per wall-second (not raw kernel
  throughput).

## Reproducing

```
pixi run -e warp python benchmarks/capture_arm.py \
    --nstep 2000 --nbatch 256 \
    --threads 1 2 4 8 12 16 20 \
    --nworlds 1 64 512 4096 16384 \
    --modes both \
    --json benchmarks/results/capture_arm_maxed.json

pixi run -e warp python benchmarks/panel_deploy.py \
    --nstep 2000 --nbatch 256 \
    --threads 1 2 4 8 12 16 20 \
    --nworlds 1 64 512 4096 \
    --json benchmarks/results/panel_deploy_maxed.json
```

Sanity-check against MJWarp's own `testspeed.py`:

```
.pixi/envs/warp/bin/python -m mujoco_warp.testspeed \
    benchmarks/capture_arm.xml --nworld 8192 --nstep 1000 --format short
# Expected: ~9M steps/s on a 3080
```
