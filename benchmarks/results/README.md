# Throughput benchmark results

Hardware: i7-12700K (8 P-cores + 4 E-cores; 20 logical), RTX 3080 (10 GiB).
Run config: `--nstep 2000 --nbatch 256`.

| Benchmark      | Mode    | CPU pure peak           | CPU orbit peak          | GPU pure peak             | GPU orbit peak            |
|----------------|---------|-------------------------|-------------------------|---------------------------|---------------------------|
| capture_arm    | zero    | 4.44M @ 20 threads      | 2.58M @ 16 threads      | 290k @ 16384 worlds       | 284k @ 16384 worlds       |
| capture_arm    | sin     | 4.78M @ 20 threads      | 2.63M @ 16 threads      | 287k @ 16384 worlds       | 281k @ 16384 worlds       |
| panel_deploy   | passive | 762k @ 20 threads       | 659k @ 20 threads       | 244k @ 4096 worlds        | 238k @ 4096 worlds        |

Numbers are total simulated steps per second (i.e. nbatch × nstep / wall on CPU,
nworld × nstep / wall on GPU).

## Observations

- **Orbit overlay tax (CPU)**: 15–48% depending on threads + scenario. The free-flight
  scenes with fewer bodies (capture_arm) see a higher relative tax (~30–48%) than
  panel_deploy (~13–25%) because the per-step orbit work is proportionally larger
  when the underlying MuJoCo step is cheap.
- **Orbit overlay tax (GPU)**: ≤5% across the board, often within measurement
  noise. The orbit overlay piggybacks on the existing per-world kernel launches,
  so its added cost is dwarfed by the constant launch overhead of the MJWarp
  pipeline.
- **CPU vs GPU at small DOF**: CPU is ~15× faster than GPU on capture_arm (8 DOF
  + free joint) and ~3× faster on panel_deploy (18 DOF + heavier contact graph).
  MJWarp's per-step launch overhead dominates the actual physics work at this
  problem size; GPU only wins for batched stochastic / RL workloads where the
  CPU side would otherwise need >256 environments simultaneously.
- **CPU thread scaling**: peaks at 16 threads (P-cores fully saturated) for
  capture_arm; panel_deploy's heavier per-step work scales to 20 threads.
- **GPU memory ceiling**: capture_arm runs at 32k+ worlds OOM cold-starting on a
  10 GiB card (EPA narrowphase scratch hits ~1 GB at 64k worlds). Sweeping
  upward incrementally amortizes earlier per-world buffers but also OOMs by
  64k. Panel_deploy with `njmax=200` hits the wall earlier; 4096 is the safe
  ceiling.

## TODO baselines (for the eventual figure)

The JSON files have stub entries for these — fill in from separate runs:

- **CPU**: Basilisk-MuJoCo on the same scenarios. Single-body Basilisk-MuJoCo
  is already wired up under `comparisons/basilisk_mujoco/`; the multi-body
  capture-arm and panel-deploy ports still need to be written.
- **GPU**: smallsatsim.github.io's flexible-array deploy demo on the same hardware.

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
