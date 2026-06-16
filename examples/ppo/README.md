# PPO free-flyer manipulation examples

Two PPO examples (via [rsl-rl](https://github.com/leggedrobotics/rsl_rl)) on the
`mjorbit_warp` GPU backend, both manipulating a **free-floating object by
contact** (no weld, no equality — capsule-jaw-vs-box primitive contact the MJWarp
collision pipeline supports directly):

1. **Truss hug & point** (`truss_env.py`, `spacecraft_truss.xml`) — a bi-manual
   free-flyer holds a large free truss in a two-arm hug and slews it to point at
   Earth (gravity-gradient-stable attitude).
2. **Astrobee detumble & grasp** (`astrobee_env.py`, `astrobee_grasp.xml`) — an
   Astrobee-style cube free-flyer detumbles from an initial disturbance, flies to
   a free-floating cargo module, and grasps its grapple bar.

Both share `train.py`'s rsl-rl/PPO config and the `rl` pixi env. Sections below
cover each.

---

# Example 1: hug & point a large free-floating truss

A bi-manual free-flyer spacecraft (the banner-figure bus) holds a **large,
free-floating truss in a two-arm hug** — each hand closes its **capsule jaws on
the spar** and holds it by **contact + friction** (no weld, no equality) — and
slews it into its gravity-gradient-stable attitude (long axis along the local
vertical, i.e. pointing at Earth). The truss is a fully independent free body;
the only coupling is contact, so the policy must keep the jaws on it.

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

---

# Example 2: Astrobee detumble & grasp

An **Astrobee-style cube free-flyer** (NASA's ~32 cm ISS robot) starts tumbling
and drifting from an initial disturbance, **detumbles**, flies to a
**free-floating cargo module**, and **grasps** the cargo's handle bar with its
perching-arm gripper. The cargo is a fully independent free body; the only
coupling is contact, so the policy must bring the open gripper around the bar
and **close it** (a binary grip action) to pinch it.

## Why this task

- **Sequential, by construction.** The approach/grasp reward is gated on the bus
  being detumbled (`exp(-(spin/width)^2)`), so the policy learns to kill its
  angular rate *before* it can collect approach reward — it stabilizes first,
  then flies in and grabs.
- **Parallel-jaw gripper on a crosswise bar.** The gripper is two flat pads on
  opposed slide joints; the cargo carries a **capsule handle bar perpendicular to
  the gripper's approach** (on two end posts). The robot brings the open gripper
  around the bar and the pads close in Z to pinch it — a standard parallel-jaw
  grasp. Contact is restricted to pad-vs-bar, a capsule-vs-box primitive the
  MJWarp pipeline supports directly (the cargo body is non-colliding, so there is
  never box-vs-box). The wide pads make the grasp forgiving in approach depth.
  Validated on CPU and warp: the closed pads hold the cargo through a full tow.
- **Binary grip action.** The grip open/close is its own **policy action**
  (action > 0 → close), not a proximity auto-latch — the policy learns *when* to
  close. A reach reward pulls the gripper to the bar, a grasp bonus rewards
  closing on it, and a small penalty discourages closing on empty space.
- **Full 6-DOF control.** The policy commands 3 reaction-wheel torques + 3
  thruster forces (ideal `motor` actuators on the bus free joint, sized for the
  light ~9 kg cube) plus the 2 perching-arm joints. The cargo floats calmly; the
  disturbance is on the robot.

## Files

- `astrobee_grasp.xml` — MJCF: the ~32 cm cube bus (side propulsion modules,
  touchscreen face, signal lights) + a 2-DOF perching arm with the parallel-jaw
  pad gripper + 6 bus `motor` actuators; a free-floating cargo module with a
  crosswise capsule handle bar. Only the bar collides (pad-vs-bar primitive).
- `astrobee_env.py` — batched `VecEnv` over `mjorbit_warp`: per-world initial
  tumble/drift, a free-floating cargo to fly to, and a detumble + reach +
  grasp/hold reward. A grasp counts when the policy closes the grip with the
  gripper on the bar and the jaws aligned to it.
- `astrobee_train.py` / `astrobee_play.py` — PPO training (with `--cargo-dist-
  min/max` and `--init-spin` curriculum knobs) and checkpoint evaluation.
- `../../scripts/record/produce_astrobee.py` + `record_astrobee.py` — pick a
  clean held-grasp rollout and render it (`videos/astrobee_grasp.mp4`).

## Running

```bash
pixi install -e rl
# A distance curriculum trains the grasp reliably: learn it close, then widen.
pixi run -e rl python examples/ppo/astrobee_train.py --num-envs 1024 \
    --max-iterations 250 --run-name ab-close --cargo-dist-min 0.9 --cargo-dist-max 1.1
pixi run -e rl python examples/ppo/astrobee_train.py --num-envs 1024 \
    --max-iterations 250 --run-name ab-far --resume examples/ppo/logs/ab-close/model_249.pt \
    --cargo-dist-min 1.2 --cargo-dist-max 2.0
pixi run -e rl python examples/ppo/astrobee_play.py \
    --checkpoint examples/ppo/logs/ab-far/model_<it>.pt
# full clip:
pixi run -e rl python scripts/record/produce_astrobee.py \
    --checkpoint examples/ppo/logs/ab-far/model_<it>.pt --out /tmp/astrobee_traj.npz
pixi run -e rl python scripts/record/record_astrobee.py \
    --traj /tmp/astrobee_traj.npz --out videos/astrobee_grasp.mp4
```

## Task definition

| | |
|---|---|
| Observation (38) | arm q/q̇ (4), bus angular + linear velocity (6), gripper→bar vector + bar axis + jaw axis + approach axis in bus frame (12), cargo relative lin/ang velocity (6), grasped flag (1), previous action (9) |
| Action (9) | 2 perching-arm joint targets + 3 reaction-wheel torques + 3 thruster forces + 1 binary grip (action > 0 closes the pads) |
| Reward | detumble (low spin) + (gated on detumble) approach progress + reach (close to the bar) + alignment + grasp + hold bonuses − spin, soft-dock velocity, closing-on-empty, effort, action-rate |
| Episode | 60 s, control at 10 Hz (physics 0.01 s × 10 decimation); ends on timeout, cargo lost (gripper too far), or non-finite state |
| Resets | robot at the origin with a random body-frame tumble + linear drift; cargo floating ahead (random distance/offset), oriented so its bar faces the robot, at rest |

## Results

From a random tumble (up to ~0.4 rad/s/axis) and ~1.2–2.0 m initial separation,
the policy detumbles to **~0.12 rad/s** and grasps + holds the cargo **~80–88% of
episodes** (1024 worlds on an RTX 3080), trained with a close→far distance
curriculum (~1.3k total iterations across stages). A representative rollout:
start 1.4 m apart → grasp at ~6 s → held to the end.

## Notes

- The grip is a **binary policy action** (close when action > 0). The handle bar
  is **perpendicular to the gripper** so the pads close *around* it rather than
  sliding on axially, and a **reach reward** plus a steep proximity term pull the
  gripper onto the bar — together these make the grasp tractable to learn.
- The **distance curriculum matters**: grasping is rare enough at full range that
  a from-scratch run can plateau on a detumble-and-hover local optimum; training
  close first makes grasping common, then the distance is widened.
- Gravity gradient, drag, and J2 are enabled for orbital context, but at this
  body scale / horizon they are minor next to the control authority — the
  challenge is the coupled detumble + precision approach + grasp.
