# Sim-fidelity study: does training in `mjorbit_warp` beat training in bare MuJoCo Warp?

**Question.** We already train RL policies in `mjorbit_warp` (our orbital
backend). If you only had upstream **MuJoCo Warp** — rigid-body multibody, no
orbital dynamics — and trained the *same task with the same reward*, would the
resulting policy do *worse* when deployed in orbit? In other words, is the
orbital fidelity of `mjorbit_warp` worth anything to the learned policy, or could
you have trained in vanilla MJWarp and been fine?

**Setup.** Train policies under two physics backends, then evaluate **all** of
them in the full `mjorbit_warp` orbital environment (our stand-in for "reality").
The observation, action, and reward are *byte-identical* across backends — only
the per-step dynamics differ — so a policy trained under one backend drops
straight into the evaluator with no remapping (see
[Implementation](#implementation)).

This is the orbital analogue of a sim-to-real / domain-gap study: does the
higher-fidelity training simulator transfer better?

## What `mjorbit_warp` adds over bare MuJoCo Warp

`mjo_step` injects, every step, on top of `mujoco_warp.step`:

- **gravity-gradient torque** on each body,
- **differential (tidal) gravity** between bodies and the chief,
- **atmospheric drag** and **J2**,
- the **non-inertial chief-frame** origin-acceleration compensation,
- **propagation of the chief orbit** — so *nadir sweeps* over the episode.

Bare MuJoCo Warp has none of these: bodies are in true free-float
(`opt.gravity == 0`, `xfrc_applied == 0`), and the orbit never advances.

### The magnitude caveat (why the baseline split matters)

At 400 km LEO with this hardware, the orbital *wrenches* are **tiny** relative to
control authority. On the truss, gravity-gradient torque is ~`7e-5` N·m while the
reaction wheels deliver up to `8` N·m — a ~10⁵ ratio. (The smoke test measures
`max|xfrc_applied| ≈ 3e-5` under `mjorbit`, exactly `0` under `mjwarp`.) The
*dominant* orbital signal a nadir-pointing policy must cope with is not the GG
dynamics but the **nadir target sweeping ~7.8° over the 120 s episode**
(orbit rate × horizon).

That observation drives the baseline design. A skeptic could dismiss a naive
"mjwarp can't point at Earth" result as a *reference-signal* problem (the policy
was simply never told nadir moves), not a *physics-fidelity* problem. So we split
the bare-MJWarp baseline into two:

| baseline | dynamics | nadir reference | isolates |
|---|---|---|---|
| **`mjwarp_fair`** | bare (no GG/drag/J2/frame) | **moving** (kinematic Keplerian) | the orbital **dynamics** gap alone |
| **`mjwarp_naive`** | bare | **frozen** at the initial phase | the **total** gap (no orbit model at all) |

The fair baseline is the honest, skeptic-proof test of the hypothesis: it gives
vanilla MJWarp every advantage a competent engineer would add (a time-varying
nadir computed from a Keplerian orbit is trivial and needs no `mjorbit_warp`),
and removes *only* the coupled dynamics. Our kinematic nadir propagation
reproduces the real orbit sweep to 4 decimals (0.0324° drift over a 6-sample
smoke rollout, identical between `mjwarp_fair` and `mjorbit`), so the fair
baseline sees essentially the same reference the evaluator does. (It is a pure
circular Keplerian advance — it omits J2 and chief feedback-acceleration that the
`mjorbit` eval orbit includes, but those shift nadir by <0.05° over the horizon.)

**Matched reset distribution.** All three truss conditions train with
`resample_orbit_on_reset=True`, so every episode reset draws a fresh orbit phase
(hence a fresh initial nadir error). Without this, the frozen-target `naive`
baseline would re-initialize each world to the *same* misalignment every episode
and train on a narrower distribution than the others — conflating "frozen
reference" with "degenerate resets." With it, the *only* differences between
conditions are the intended ones: coupled dynamics (mjorbit) and whether the
target moves *within* an episode (fair vs naive).

**Prediction.** Because the dynamics delta is small vs control authority, we
expect `mjwarp_fair ≈ mjorbit` (small gap) and `mjwarp_naive ≪ mjorbit` (large
gap). Running both makes the result interpretable *whatever* it turns out to be:
it decomposes the value of `mjorbit_warp` into "providing the correct reference"
vs "providing the correct dynamics".

## Tasks

| task | what makes it orbital | conditions |
|---|---|---|
| **truss** (nadir-pointing) | objective *is* the gravity-gradient-stable, nadir-pointing attitude; nadir sweeps with the orbit | `mjorbit`, `mjwarp_fair`, `mjwarp_naive` |
| **astrobee** (detumble + grasp) | **none** — reward/obs are purely relative/body-frame; orbital wrenches negligible at this scale | `mjorbit`, `mjwarp` |

Astrobee is the **predicted-null control**: it has no orbital-frame reference in
its observation or reward, so bare-MJWarp and `mjorbit_warp` training should be
near-indistinguishable. Including it guards against a "we tuned until orbital
fidelity looked important" critique — if the effect is real and physics-driven,
it should appear in the GG-dominated truss task and *not* in the
manipulation-dominated astrobee task.

## Metrics (evaluated in `mjorbit_warp`, paired worlds)

Every policy is scored in the full orbital environment with a **fixed eval seed**,
so all policies face the *identical* set of worlds (orbits / initial conditions).
Auto-reset is disabled, so each world runs exactly one fixed-horizon episode.

- **Truss** — *success* = the hug is retained for the whole episode **and** the
  final-quarter mean nadir-pointing error is below a threshold. We report the
  success rate across thresholds (1–10°), the per-world error distribution, and
  the hug-lost rate.
- **Astrobee** — *success* = grasped-and-held for the final quarter, with no
  fail/crash. We report success rate, grasped fraction, and final spin.

## How to run

```bash
# 1. (one-time) build the native plugin
pixi run -e rl sync-package

# 2. full matrix: 2 tasks x conditions x 3 seeds, train + eval (~few GPU-hours)
pixi run -e rl python experiments/sim_fidelity/run_experiment.py --tasks both --seeds 0 1 2

#    resume after interruption (skips finished train/eval):
pixi run -e rl python experiments/sim_fidelity/run_experiment.py --skip-existing

#    validate the whole pipeline fast (tiny iters/worlds):
pixi run -e rl python experiments/sim_fidelity/run_experiment.py --quick --seeds 0

# 3. aggregate -> results/SUMMARY.md + results/figures/  (report env has matplotlib)
pixi run -e report python experiments/sim_fidelity/plot_results.py
```

Per-policy eval JSONs land in `results/<task>/<condition>_seed<k>.json`; training
checkpoints/tensorboard go to `examples/ppo/logs/fidelity/...` (gitignored).

## Implementation

The backend is a single seam, `examples/ppo/sim_backend.py`, selected by
`TrussEnvCfg.backend` / `AstrobeeEnvCfg.backend` (`"mjorbit"` | `"mjwarp"`):

- Both backends use the **same** `MjoModel`/`MjoData` — same compiled MuJoCo
  model, DOF layout, integrator, and `opt.gravity == 0`. Only the step differs:
  `mjo_step` (full coupling) vs `mujoco_warp.step` (bare rigid body).
- On `mjwarp`, `mujoco_warp` never writes `xfrc_applied`, so the body is in true
  free-float; the chief orbit isn't propagated. The truss env propagates the
  nadir reference itself (`_propagate_target`) when `mjwarp_moving_target=True`
  (fair) and leaves it frozen otherwise (naive).
- Training selects the backend via `train.py --backend {mjorbit,mjwarp}`
  (`--frozen-target` for the naive truss baseline) and
  `astrobee_train.py --backend {mjorbit,mjwarp}`. **Evaluation always uses
  `mjorbit`.**

## Results

Run `plot_results.py`; the generated [`results/SUMMARY.md`](results/SUMMARY.md)
and `results/figures/` hold the tables and plots. Headline figures:

- `results/figures/truss_success_curve.png` — success rate vs pointing-error
  threshold, per training backend.
- `results/figures/truss_align_box.png` — per-world final pointing error.
- `results/figures/astrobee_success_bars.png` — grasp success by backend.

## Caveats & extensions

- **Regime-dependent.** The fair-baseline gap is expected to be small *because*
  control authority ≫ orbital wrenches at 400 km with these actuators. The gap
  would grow under: lower control authority (force the policy to *exploit* GG
  rather than overpower it), lower altitude / larger panels (stronger drag
  torque), longer multi-orbit horizons (GG/tidal effects accumulate), or larger
  inertia asymmetry. These are one-line config changes (`_RW_LIMIT`,
  `altitude_km`, `episode_length_s`) and make good follow-ups.
- **fp32 orbit clock.** Long device-resident runs (≫ 1 h sim) lose orbit-clock
  accuracy; the 120 s / 60 s horizons here are well within budget.
- 3 seeds give mean ± std; bump `--seeds` for tighter intervals.
