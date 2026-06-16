# PPO: hug & point a large free-floating truss

PPO training (via [rsl-rl](https://github.com/leggedrobotics/rsl_rl)) on the
`mjorbit_warp` backend. A bi-manual free-flyer spacecraft (the banner-figure
bus) holds a **large, free-floating truss in a two-arm hug** — each hand closes
its **capsule jaws on the spar** and holds it by **contact + friction** (no
weld, no equality) — and slews it into its gravity-gradient-stable attitude
(long axis along the local vertical, i.e. pointing at Earth). The truss is a
fully independent free body; the only coupling is contact, so the policy must
keep the jaws on it.

## Why this task

- **Body-contact hug, done right.** Each hand is a two-jaw clamp: a fixed lower
  jaw plus an upper jaw on a slide joint (`grip_*`) that closes onto the spar. A
  wrist hinge (`wrist_*`) levels the hand so the jaws straddle the spar's top and
  bottom faces with **shallow surface contact** — no jamming, no interpenetration
  — and can open to release. Collision is restricted to the forearms + jaws vs.
  the truss (`contype`/`conaffinity` bitmasks); the contacting pairs are
  capsule-vs-box / capsule-vs-sphere primitives, which the MJWarp collision
  pipeline supports directly. The wrist and grip are auto-driven (wrist =
  `-(shoulder+elbow)` to keep the hand level; grip held closed), so the policy
  still commands only the 4 arm joints + 6 bus DOFs. A point gripper slips when
  slewing a big truss — the two distributed jaw contacts of a hug hold it.
- **Full 6-DOF spacecraft control.** The policy commands 3 reaction-wheel
  torques + 3 thruster forces (ideal `motor` actuators on the bus free joint,
  thrusters on a station-keeping PD baseline) alongside the 4 arm joints. It
  slews the whole stack with the wheels to aim the truss and station-keeps with
  the thrusters. Wheel/thruster authority is capped so the slew is gentle enough
  not to fling the truss out of the hug.
- **Gravity gradient matters.** The 6 m truss is light but strongly elongated
  (inertia ~18/18/0.1 kg·m²), so its stable equilibrium is the long axis along
  the local vertical — the attitude the policy drives it to. The orbit is in the
  X-Z plane and nadir sweeps with it; gravity gradient, drag, and J2 are enabled.

## Files

- `spacecraft_truss.xml` — MJCF: the banner bus (gold MLI, radiators, antenna,
  framed solar wings) + two arms ending in wrist-leveled two-jaw hands + 6 bus
  `motor` actuators (reaction wheels / thrusters); a 6 m free-floating lattice
  truss. Only the forearms + jaws collide with the truss.
- `truss_env.py` — batched `VecEnv` (rsl-rl ≥ 5.x TensorDict API) over
  `mjorbit_warp`: device-resident stepping with control decimation, per-world
  in-plane orbit phases, per-world resets. A dense pointing-progress reward plus
  a hug-retention term; the episode ends if the truss slips out of the hug.
- `train.py` / `play.py` — PPO training and checkpoint evaluation.
- `../../scripts/record/produce_hug.py` — composes the full clip: a scripted
  fly-in (the bus rises and the jaws close on the truss), the trained policy
  (hug + slew to nadir), rendered by `scripts/record/record_truss.py`.

## Running

```bash
pixi install -e rl
pixi run -e rl sync-package          # first time: build the native plugin
pixi run -e rl python examples/ppo/train.py --num-envs 1024 --max-iterations 440
pixi run -e rl tensorboard --logdir examples/ppo/logs
pixi run -e rl python examples/ppo/play.py --checkpoint examples/ppo/logs/<run>/model_<it>.pt
# full fly-in + hug + stabilize clip:
pixi run -e rl python scripts/record/produce_hug.py \
    --checkpoint examples/ppo/logs/<run>/model_<it>.pt --out /tmp/hug_traj.npz
pixi run -e rl python scripts/record/record_truss.py \
    --traj /tmp/hug_traj.npz --out videos/truss_pointing.mp4 --clip-sim-seconds 0
```

There is also a pixi task `pixi run -e rl ppo-truss-train`.

## Task definition

| | |
|---|---|
| Observation (42) | arm q/q̇ (8), nadir + truss long axis in bus frame (6), truss hub offset / lin-vel / ang-vel (9), bus position / lin-vel / ang-vel (9), previous action (10) |
| Action (10) | 4 arm joint targets (small moves around the hug pose) + 3 reaction-wheel torques + 3 thruster forces (PD station-keep + policy residual). The wrist + grip joints are auto-driven (leveled / held closed), not policy outputs. |
| Reward | hug retention (jaws on the spar) + (gated on the hug) axis alignment + two Gaussian precision bonuses + dense pointing-progress − truss-rate, bus position/rate, action-rate, effort |
| Episode | 120 s, control at 10 Hz (physics 0.01 s × 10 decimation); ends on timeout, hug lost, or non-finite state |
| Resets | truss hugged at identity attitude; random in-plane orbit phase sets the initial pointing error (up to ~85°) |

The truss is end-symmetric, so alignment uses `|cos|` — pointing either tip at
Earth counts.

## Results

From a ~50–85° initial misalignment the policy slews the hugged truss to
**~2–3° mean pointing error** (final-quarter ~1°) in ~440 iterations at 1024
worlds on an RTX 3080 (~25k env-steps/s with contacts), keeping the truss in the
hug essentially every step (hug-lost < 0.1%). A representative rollout: 52° → 1°.

## Notes

- The hold is **contact only** — no `weld`, no equality, no joint *to* the
  truss. The jaws close on the spar (a grip DOF internal to each hand) and hold
  it by surface contact + friction; the wrist keeps the jaws level so the
  contact is shallow rather than jammed.
- The policy is trained starting already hugging the truss (the trainable
  manipulation phase). The fly-in is a scripted bookend in `produce_hug.py`.
  A clean **release + fly-away** is not included by default: even though the jaws
  *can* open, releasing a force-clamped grip on the slender spar imparts a small
  kick to the now-free truss, and gravity gradient is far too slow to re-settle
  it within a clip — so the clip ends with the truss pointed and held. (Pass
  `--release-s > 0` to `produce_hug.py` to script the jaw-open + depart anyway.)
- The chief orbit is not reset between episodes — worlds keep flying, so each
  episode starts at a fresh orbit phase.
