# PPO free-flyer manipulation examples

Three PPO examples (via [rsl-rl](https://github.com/leggedrobotics/rsl_rl)) on the
`mjorbit_warp` GPU backend, all manipulating a **free-floating object by
contact** (no weld, no equality — capsule-jaw-vs-box primitive contact the MJWarp
collision pipeline supports directly):

1. **Truss hug & point** (`truss_env.py`, `spacecraft_truss.xml`) — a bi-manual
   free-flyer holds a large free truss in a two-arm hug and slews it to point at
   Earth (gravity-gradient-stable attitude).
2. **Astrobee detumble & grasp** (`astrobee_env.py`, `astrobee_grasp.xml`) — an
   Astrobee-style cube free-flyer detumbles from an initial disturbance, flies to
   a free-floating cargo module, and grasps its grapple bar.
3. **Capture & reorient with SGS** (`capture_env.py`, `capture_reorient.xml`) — the
   teaser bi-manual free-flyer captures a drifting, tumbling client satellite by
   its grapple bar and slews the stack until the client's antenna points at
   nadir. Sparse binary success reward; resets chosen by Success Guided Sampling
   (arXiv 2610.12465) with a uniform-sampling baseline.

All share `train.py`'s rsl-rl/PPO config (example 3 changes a few settings, see
below) and the `rl` pixi env. Sections below cover each.

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

---

# Example 3: capture & reorient a free-floating client with SGS

The teaser/banner bi-manual free-flyer **captures a free-floating client
satellite** by its grapple bar with both hands, then **slews the captured stack
with its reaction wheels until the client's antenna points at nadir** and holds
it there. It is trained with **sparse binary success rewards** and the small
control costs of Zhang et al., *A Balanced Data Diet* (arXiv 2610.12465), with
resets chosen by their **Success Guided Sampling (SGS)**; a uniform-sampling run
is the baseline.

## Algorithm (from the paper)

- **SGS** (Sec. 3.2, App. B.2; `sgs.py`). A fixed set of N task configurations is
  drawn before training. Each keeps its last H = 100 Boolean episode outcomes, and
  p_i is their mean. Configurations are scored with the Beta-shaped kernel
  w_i = (p_i+ε)^{κt} (1−p_i+ε)^{κ(1−t)}, floored at ε, and l_i = log(w_i+ε). Each
  reset draws i.i.d. from softmax(l/T) with T_eff = max(T, 1). We use the paper's
  manipulation values t = 0.5, κ = 1, ε = 1e-4, T = 2, H = 100, with N = 8192
  (paper: 32,768). The paper does not say how a configuration with no recorded
  outcomes is scored; here it gets p = 0, so sampling starts uniform (as their
  Fig. 7 notes). `--sampler uniform` draws uniformly from the same set.
- **Reward** (Eq. 3/5, App. A.3.2, the shared manipulation reward, with the
  paper's weights): r = 100·1[success] − 1·1[failure] − 1e-5·C(‖a‖²) −
  1e-4·C(‖a_t − a_{t−1}‖²) − 1e-4·C(‖q̇_arm‖²), where C(x) = min(x, 1e4).
  Success and failure both end the episode. There are no distance, alignment, or
  progress terms.
- **RL**: PPO (rsl-rl). The paper gives no PPO hyperparameter table, only that
  mini-batch count and all other PPO settings stay fixed across runs. Both
  samplers here share one config: the truss example's (γ = 0.997, λ = 0.95,
  5 epochs × 4 mini-batches, adaptive LR) with a [512, 256, 128] MLP, entropy
  bonus 0.005 (not 0.02), std clamped to [0.02, 1], and no value clipping. With
  0.02, every action std inflated to ~2.5 within 150 iterations, because actions
  are clipped. The value clip (0.2) would stall the critic against a reward of 100.

## Task definition

| | |
|---|---|
| Robot | banner bus (380 kg incl. arms/wings) with two 2-link arms; each hand is the truss example's two-jaw clamp (mirrored to point inboard), wrists auto-leveled; **3 reaction wheels** (5 N·m, 15 N·m·s) and **6 RCS thrusters** (±x/±y/±z, 25 N) through mjorbit's native actuator models |
| Client | free 50 kg satellite: 0.8 m box on a 0.84 m grapple bar, antenna boresight = client +Z; held by contact + friction only |
| Orbit | 500 km circular, random plane/phase per world; J2, drag, gravity gradient, differential gravity on |
| Observation (61) | arm q / q̇ / targets, grip slides, bus rate + velocity, wheel speeds, jaw→bar vectors, bar pose / axis / antenna axis / relative velocity and rate, nadir (all in bus frame), grip + on-bar flags, hold progress, previous action |
| Action (11) | 4 arm joint-target increments (≤ 0.5 rad/s), 1 latched grip (close on a > 0.8, release after 3 consecutive a < −0.8), 3 wheel torques, 3 body-frame thrust requests |
| Success | both hands clamped on the bar (grip closed, jaw center within 4.5 cm of the bar axis, upper jaw down) **and** antenna within **10°** of nadir, continuously for **3 s** |
| Failure | client > 4 m away, client box through the bus, arm joint > 6 rad/s, bus > 0.5 rad/s, client > 1 rad/s, or non-finite state |
| Episode | 60 s at 10 Hz control (10 × 0.01 s physics, one captured CUDA graph) |

**Reset distribution** (`sample_task_configs`): a fixed, collision-checked set of
configurations, split evenly across three families (cf. OmniReset's
Near-Goal / Stable-Grasp / Reaching). The servicer's initial attitude puts its +Z,
and so the antenna of a captured client, at a pointing error drawn from U[0°, 180°]
from nadir, defined in the LVLH frame at reset time.

- **held**: the client is already clamped in both hands.
- **pregrasp**: the client floats with its bar inside the open jaws (±3 cm, up to
  8 cm high), ±4° misalignment, ≤ 1 cm/s drift, ≤ 0.5°/s tumble.
- **approach**: the client floats 0.05–1.0 m away on the upper hemisphere, with
  up to ±30° yaw / ±15° tilt, 2 cm/s drift and 1.5°/s tumble. A per-configuration
  difficulty u ~ U[0, 1] scales all of these together from pregrasp levels to
  those maxima.

## Files

- `capture_reorient.xml` — the model (bus, arms, hands, wings, client; RW/RCS in
  the `<mjorbit>` element).
- `sgs.py` — the sampler (SGS and uniform modes).
- `capture_env.py` — batched `VecEnv`: configuration set, resets, success/failure,
  reward, per-iteration CSV log, sampler snapshots.
- `capture_train.py` / `capture_eval.py` / `capture_plot.py` — training, held-out
  evaluation (fresh configurations from the training distribution, one
  deterministic episode each), and curves.
- `../../scripts/record/record_capture_rl.py` — renders a saved rollout.

## Running

```bash
pixi run -e rl python examples/ppo/capture_train.py --sampler sgs --run-name cap-sgs
pixi run -e rl python examples/ppo/capture_train.py --sampler uniform --run-name cap-uni
# held-out success vs environment steps (appends to <run>/eval.csv, resumable)
pixi run -e rl python examples/ppo/capture_eval.py --run examples/ppo/logs/cap-sgs --every 200
pixi run -e report python examples/ppo/capture_plot.py \
    --runs examples/ppo/logs/cap-sgs examples/ppo/logs/cap-uni --labels SGS uniform \
    --out examples/ppo/logs/capture_curves.png \
    --focus-run examples/ppo/logs/cap-sgs --focus-out examples/ppo/logs/capture_sgs_focus.png
# video: dump a successful held-out approach rollout, then render (CPU, Chrome)
pixi run -e rl python examples/ppo/capture_eval.py --run examples/ppo/logs/cap-sgs \
    --checkpoint model_<it>.pt --save-traj videos/capture_traj.npz
pixi run python scripts/record/record_capture_rl.py \
    --traj videos/capture_traj.npz --out videos/capture_reorient_rl.mp4
```

Two concurrent runs at 4096 worlds each take ~2.4 s per PPO iteration on an RTX
3080 (one alone: ~1.5 s, ~100k env-steps/s with contacts).
