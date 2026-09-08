# MPPI sim-fidelity study (grasp & gravity-gradient stabilize)

Does the *dynamics model used inside a sampling-based controller* matter for
long-horizon, gravity-gradient-dominated control? A free-flying servicer with a
slow two-link arm and **no attitude actuators** grasps a payload roughly
along-track (~90° from the gravity-gradient equilibrium), then must reorient the
captured stack to local vertical and hold it, using only arm motion + the
environmental torques. We run one MPPI controller whose internal rollouts use
either the full orbit-coupled `mjorbit` dynamics or a zero-gravity approximation,
and execute **both on the full `mjorbit` plant** (the grasp is shared; only the
rollout model used for the long-horizon stabilization differs).

This is the cheap, no-training precursor to an RL sim-fidelity study: if the
rollout model doesn't matter here, it won't matter for RL.

## Reproduce

```bash
# 5 seeds; each runs the full grasp+stabilize episode (~4 min/seed on CPU)
for s in 0 1 2 3 4; do
  pixi run python examples/mppi/mppi_fidelity_capture.py \
    --backends mjorbit zerog \
    --horizon-a 300 --horizon-b 2400 --duration-b 10000 --rollouts 64 --seed $s \
    --out experiments/mppi_fidelity/out/seed${s}.json
done

# paper figure -> out/mppi_fidelity.tikz (x in hours, mean +/- 1sigma bands)
pixi run -e report python experiments/mppi_fidelity/make_paper_figure.py
```

`out/mppi_fidelity.tikz` is synced into the paper as `figures/mppi_fidelity.tikz`
(see the paper Makefile `PAPER_FIGURES` manifest).

## Result (5 seeds; final-window nadir-pointing error, eval on full mjorbit)

| planner rollout model | final error (mean ± std) | settled |
|---|---|---|
| `mjorbit` (full coupling)        | 2.2 ± 1.2°  | 5/5 |
| `zerog` (gravity disabled, μ≈0)  | 30.2 ± 15.3° | 1/5 |

The orbit-coupled planner settles the stack to local vertical on every seed; the
zero-gravity planner cannot anticipate the gravity-gradient libration and keeps
the stack swinging (one seed is eventually rescued passively by the true plant's
gravity gradient). Simulating the orbital dynamics directly improves long-horizon
control with the same planner and arm hardware.

### Ablation (`--backends ... zerog_fair`)
`zerog_fair` keeps the orbital reference/frame propagation but drops only the
gravity-gradient *torque* from the rollouts. It also stabilizes well, indicating
the essential modeling ingredient is the non-inertial orbital frame rather than
the GG torque inside the planner (the true environment supplies that). This
ablation is available but is not shown in the paper figure.
