# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""MPPI reach-and-grasp with the 5-finger claw (``spacecraft_capture_claw.xml``)
and dump a trajectory for offline rendering.

The bus carries a telescoping boom ending in a 5-finger claw. The planner
samples two continuous control channels jointly each replan, both as splines
over a few knots (the usual bezier-style MPPI control):

* **boom command** — the prismatic ``extend`` actuator: how far to telescope
  toward the payload, in [0, 3.8] m (the claw starts at the bus-payload
  midpoint, so it must extend ~3.55 m to reach the cube);
* **claw phase** — a single scalar in [0, 1] that drives the whole claw from
  fully open (0) to gently closed (1). The two grip tendons are normalized so a
  shared [0, 1] command maps linearly to the finger joint angles (open -> the
  closed-cage joint limits), so one number opens/closes all five fingers.

Contact is enabled and scoped so the robot collides only with the payload, so
the closed claw physically grips the cube (friction). Once the cube is centered
the plant latches the grip shut, freezes the boom, and coasts for a long span:
the captured bus+arm+cube stack is elongated and inertially fixed while local
vertical rotates with the orbit, so the gravity gradient librates it and the
grip holds the cube through the swing.

    pixi run python scripts/record/produce_grasp_claw.py --out /tmp/grasp_claw_traj.npz
    pixi run python scripts/record/record_grasp_claw.py \
        --traj /tmp/grasp_claw_traj.npz --out videos/grasping_claw.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for p in (str(_ROOT / "src"), str(_ROOT / "scripts" / "record")):
    if p not in sys.path:
        sys.path.insert(0, p)

from mjorbit import MjoModel, OrbitInit, mjo_forward, mjo_step  # noqa: E402
from mjorbit.constants import GM_EARTH, R_EARTH  # noqa: E402
from mjorbit.planning.mppi import make_spline  # noqa: E402
from mjorbit.rollout import mjo_get_state, mjo_set_state, rollout  # noqa: E402
from mjorbit.testdata import SPACECRAFT_CAPTURE_CLAW_XML  # noqa: E402

# ---------------------------------------------------------------------------
# Packed mjo state layout for the claw model (nstate = 56):
#   [time, qpos(25), qvel(23), R_eci(3), V_eci(3), orbit_t].
# qpos: bus pos 1:4, bus quat 4:8, arm_slide 8, fingers 9:19, payload pos 19:22.
# qvel block starts at 1+25 = 26; the arm_slide dof is dof 6 -> state index 32.
BUS_POS = slice(1, 4)
ARM_SLIDE_VEL = 32

# Sensor layout (16): ee_pos 0:3, ee_quat 3:7, ee_linvel 7:10,
# payload_pos 10:13, payload_linvel 13:16.
EE_POS = slice(0, 3)
SENS_PAYLOAD_POS = slice(10, 13)

# Control layout (nu = 3): [extend, grip_prox, grip_dist]. Both grip rows take
# the same normalized phase, so the planner decides two channels: boom + phase.
EXTEND = 0
GRIP_PROX = 1
GRIP_DIST = 2
BOOM_CH, PHASE_CH = 0, 1                      # planner channel indices
CTRL_LOW = np.array([0.0, 0.0])              # [boom (m), phase]
CTRL_HIGH = np.array([3.8, 1.0])

# Where the payload should sit relative to the palm: ~0.32 m beyond the palm
# along the boom axis seats the cube against the palm with the fingers wrapping
# and gripping it (measured from the closed-claw pose on the 0.5 m cube).
GRIP_CENTER_OFFSET = 0.32
CENTER_SCALE = 0.15   # m, how tight "centered" is (cube half-size = 0.25)


def claw_center(ee_pos: np.ndarray, bus_pos: np.ndarray) -> np.ndarray:
    """Centre of the claw cage: ``GRIP_CENTER_OFFSET`` beyond the palm along the
    boom axis (``ee - bus``). Keying off the boom direction is robust to the
    ee body's frame convention, unlike its reported orientation."""
    boom = ee_pos - bus_pos
    boom = boom / np.linalg.norm(boom, axis=-1, keepdims=True)
    return ee_pos + GRIP_CENTER_OFFSET * boom


def grasp_distance(states: np.ndarray, sensors: np.ndarray) -> np.ndarray:
    """Distance from the payload to the claw centre, on any (..., n) batch."""
    center = claw_center(sensors[..., EE_POS], states[..., BUS_POS])
    return np.linalg.norm(sensors[..., SENS_PAYLOAD_POS] - center, axis=-1)


def plant_distance(data) -> float:
    """Claw-to-payload distance from the live plant ``data``."""
    sens = np.asarray(data.sensordata)
    center = claw_center(sens[EE_POS], np.asarray(data.qpos[0:3]))
    return float(np.linalg.norm(sens[SENS_PAYLOAD_POS] - center))


def compute_K(model, data) -> tuple[float, np.ndarray]:
    """Gravity-gradient elongation ratio ``K = (I_max - I_min) / I_max`` of the
    whole assembly in its *current* configuration.

    Builds the composite inertia tensor of every body about the system centre of
    mass via the parallel-axis theorem, using the live data MuJoCo exposes:
    ``body_inertia`` are each body's principal moments and ``ximat`` is their
    world orientation (world-from-principal-axes), so
    ``J_world = R diag(I) Rᵀ``. Body 0 (the world) is skipped.

    K is read straight from the compiled model masses and the current body
    frames, so editing the payload mass (or any geometry) in the XML changes the
    returned K automatically. It sets the libration frequency through
    ``omega_lib = omega0 * sqrt(3 K)``.

    Returns ``(K, principal_moments)`` with the three principal moments ascending
    (I_min, I_mid, I_max), in MuJoCo SI units (kg·m²).
    """
    masses = np.asarray(model.body_mass)[1:]                  # drop world body 0
    coms = np.asarray(data.xipos)[1:]                         # (nb-1, 3) world COMs
    inertias = np.asarray(model.body_inertia)[1:]             # (nb-1, 3) principal
    ximat = np.asarray(data.ximat)[1:].reshape(-1, 3, 3)      # world<-principal

    total_mass = float(masses.sum())
    com = (masses[:, None] * coms).sum(axis=0) / total_mass

    inertia_tensor = np.zeros((3, 3))
    for m_i, com_i, principal_i, rot_i in zip(masses, coms, inertias, ximat):
        j_world = rot_i @ np.diag(principal_i) @ rot_i.T
        offset = com_i - com
        inertia_tensor += j_world + m_i * (
            offset @ offset * np.eye(3) - np.outer(offset, offset)
        )

    principal = np.linalg.eigvalsh(inertia_tensor)            # ascending
    i_min, i_max = float(principal[0]), float(principal[-1])
    return (i_max - i_min) / i_max, principal


def make_grasp_cost(num_timesteps: int):
    """Reach the payload, settle the boom, and reward a centered closed claw."""
    term = slice(int(0.9 * num_timesteps), None)

    def cost_fn(states: np.ndarray, sensors: np.ndarray, controls: np.ndarray) -> np.ndarray:
        dist = grasp_distance(states, sensors)               # (R, T)
        boom_vel = states[:, :, ARM_SLIDE_VEL]               # (R, T)
        boom_cmd = controls[:, :, EXTEND]                    # (R, T) target ext (m)
        phase = np.clip(controls[:, :, GRIP_PROX], 0.0, 1.0)  # 0 open .. 1 closed
        centered = np.exp(-((dist / CENTER_SCALE) ** 2))     # 1 when centered
        grasp = centered * phase                             # reward only if both

        # Boom acceleration (change in slide velocity per step)
        boom_accel = np.diff(boom_vel, axis=1)               # (R, T-1)

        # Proximity-gated speed damping: brake as the claw nears the cube 
        near = np.exp(-((dist / (2.0 * CENTER_SCALE)) ** 2))  # (R, T)

        reached = np.maximum.accumulate(boom_cmd, axis=1)    # (R, T) running max
        retract = reached - boom_cmd                         # (R, T) >= 0 if pulled back
        return (
            2.0 * np.mean(dist**2, axis=1)                      # approach (drives boom)
            + 8.0 * np.mean(dist[:, term] ** 2, axis=1)         # be centered at the end
            + 3.0 * np.mean(near * boom_vel**2, axis=1)         # damp speed near the cube
            + 8.0 * np.mean(retract**2, axis=1)                 # never retract the boom
            + 1.0 * np.mean(boom_accel**2, axis=1)              # smooth, jitter-free approach
            + 10.0 * np.mean(phase * (1.0 - centered), axis=1)  # stay open off-target
            - 30.0 * np.mean(grasp[:, term], axis=1)            # reward: centered + closed
        )

    return cost_fn


class ClawGraspMppi:
    """MPPI over two continuous spline channels: boom extension and claw phase.

    The decision variables carried between replans are ``nominal_knots`` of
    shape ``(num_nodes, 2)`` — column 0 is the boom command, column 1 the claw
    phase in [0, 1]. Per replan they are perturbed with a judo-style ramped
    Gaussian, rolled out, and updated with the standard MPPI exponential weight.
    """

    def __init__(self, model, cost_fn, *, horizon, num_nodes, num_rollouts, dt,
                 sigma, temperature, noise_ramp, seed, nthread):
        self.model = model
        self.cost_fn = cost_fn
        self.dt = dt
        self.num_nodes = num_nodes
        self.num_rollouts = num_rollouts
        self.temperature = temperature
        self.nthread = nthread
        self.num_timesteps = int(np.ceil(horizon / dt))
        self._knot_offsets = np.linspace(0.0, horizon, num_nodes, endpoint=True)
        self._step_offsets = dt * np.arange(self.num_timesteps)
        # judo-style noise ramp: distant knots explore more than the imminent one.
        ramp = noise_ramp * np.linspace(1.0 / num_nodes, 1.0, num_nodes)
        self._sigma = ramp[:, None] * np.asarray(sigma, dtype=np.float64)  # (N, 2)
        self._rng = np.random.default_rng(seed)

        self.nominal_knots = np.zeros((num_nodes, 2))
        self.times = self._knot_offsets.copy()
        self._spline = make_spline(self.times, self.nominal_knots, "linear")
        self.last_costs: np.ndarray | None = None

    def reset(self, data, boom0: float = 0.0, phase0: float = 0.0) -> None:
        self.nominal_knots = np.tile([boom0, phase0], (self.num_nodes, 1))
        self.times = float(data.time) + self._knot_offsets
        self._spline = make_spline(self.times, self.nominal_knots, "linear")

    def action(self, time: float) -> np.ndarray:
        """Nominal [boom, phase] at ``time`` (clamped to the control box)."""
        return np.clip(np.asarray(self._spline(time)), CTRL_LOW, CTRL_HIGH)

    def _build_controls(self, knots, times):
        """(R, num_nodes, 2) knots -> (R, T, 3) per-step actuator controls."""
        chan = make_spline(times, knots, "linear")(times[0] + self._step_offsets)  # (R, T, 2)
        chan = np.clip(chan, CTRL_LOW, CTRL_HIGH)
        controls = np.empty((self.num_rollouts, self.num_timesteps, 3))
        controls[:, :, EXTEND] = chan[:, :, BOOM_CH]
        controls[:, :, GRIP_PROX] = chan[:, :, PHASE_CH]   # both grip rows share
        controls[:, :, GRIP_DIST] = chan[:, :, PHASE_CH]   # the same [0,1] phase
        return controls

    def update_action(self, data) -> np.ndarray:
        now = float(data.time)
        new_times = now + self._knot_offsets
        nominal = np.asarray(self._spline(new_times), dtype=np.float64)  # (N, 2)

        # Gaussian knot perturbations; nominal rides as sample 0.
        noise = self._rng.standard_normal((self.num_rollouts - 1, self.num_nodes, 2))
        samples = np.concatenate([nominal[None], nominal[None] + self._sigma * noise], axis=0)
        samples = np.clip(samples, CTRL_LOW, CTRL_HIGH)             # (R, N, 2)

        controls = self._build_controls(samples, new_times)

        initial_state = mjo_get_state(self.model, data)
        initial_ctrl = np.array(data.ctrl, dtype=np.float64)
        try:
            states, sensors = rollout(
                self.model, data, initial_state, controls, nthread=self.nthread
            )
        finally:
            mjo_set_state(self.model, data, initial_state)
            np.copyto(data.ctrl, initial_ctrl)
            mjo_forward(self.model, data)

        costs = np.asarray(self.cost_fn(states, sensors, controls), dtype=np.float64)
        beta = float(np.min(costs))
        weights = np.exp(-(costs - beta) / self.temperature)
        weights /= np.sum(weights)

        self.nominal_knots = np.tensordot(weights, samples, axes=([0], [0]))  # (N, 2)
        self.times = new_times
        self._spline = make_spline(self.times, self.nominal_knots, "linear")
        self.last_costs = costs
        return costs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/tmp/grasp_claw_traj.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollouts", type=int, default=128)
    ap.add_argument("--horizon", type=float, default=10.0,
                    help="planning horizon (s); the stiff dt=0.01 servos reach "
                    "and grip within a few seconds")
    ap.add_argument("--num-nodes", type=int, default=4)
    ap.add_argument("--replan", type=float, default=2.0)
    ap.add_argument("--max-approach-time", type=float, default=30.0)
    ap.add_argument("--hold-seconds", type=float, default=3.0*5560.0,
                    help="post-grasp passive integration (s): hold the grip closed "
                    "and boom fixed, then coast to watch the gravity gradient librate "
                    "the captured stack (~5560 s is one 400 km orbit)")
    ap.add_argument("--hold-frames", type=int, default=2500,
                    help="how many post-grasp samples to keep (sub-sampled over the hold)")
    ap.add_argument("--nthread", type=int, default=max(1, os.cpu_count() or 1))
    args = ap.parse_args()

    alt_km = 400.0
    r_orbit = R_EARTH + alt_km
    omega = float(np.sqrt(GM_EARTH / r_orbit**3))
    v_orbit = float(np.sqrt(GM_EARTH / r_orbit))
    R0 = np.array([r_orbit, 0.0, 0.0])
    V0 = np.array([0.0, v_orbit, 0.0])  # equatorial: orbit plane = world XY

    model = MjoModel.from_xml_path(SPACECRAFT_CAPTURE_CLAW_XML)
    dt = float(model.opt.timestep)

    # Keep the XML's initial layout: bus at the origin, payload at +x 7.1 m
    # (radially outward). Both start at rest in the chief-centered world frame so
    # the boom (fixed along +x) stays pointed at the payload; over this short
    # grasp the payload only drifts ~0.1 m outward under the gravity gradient,
    # which the planner tracks by extending the boom.
    data = model.make_data(orbit=OrbitInit(R_eci=R0, V_eci=V0))
    np.copyto(data.ctrl, [0.0, 0.0, 0.0])          # boom retracted, claw open
    mjo_forward(model, data)

    planner = ClawGraspMppi(
        model, make_grasp_cost(int(np.ceil(args.horizon / dt))),
        horizon=args.horizon, num_nodes=args.num_nodes, num_rollouts=args.rollouts,
        dt=dt, sigma=[0.7, 0.25], 
        temperature=0.05, noise_ramp=2.5,
        seed=args.seed, nthread=args.nthread,
    )
    planner.reset(data, boom0=0.0, phase0=0.0)

    dist0 = plant_distance(data)
    print("=" * 64)
    print("MPPI claw reach-and-grasp")
    print("=" * 64)
    print(f"orbit: {alt_km:.0f} km equatorial, rate {omega:.3e} rad/s")
    print(f"initial claw-to-payload distance {dist0:.2f} m")

    qpos_hist: list[np.ndarray] = []
    R_hist: list[np.ndarray] = []
    t_hist: list[float] = []
    replan_every = max(1, int(round(args.replan / dt)))
    log_every = int(round(20.0 / dt))
    n_max = int(round(args.max_approach_time / dt))

    grasped = False
    boom_grasp = 0.0
    t_latch = float("nan")
    for step in range(n_max):
        if step % replan_every == 0:
            planner.update_action(data)

        dist = plant_distance(data)
        boom, phase = planner.action(float(data.time))
        data.ctrl[:] = [boom, phase, phase]
        mjo_step(model, data)

        qpos_hist.append(np.asarray(data.qpos).copy())
        R_hist.append(np.asarray(data.orbit.R_eci).copy())
        t_hist.append(float(data.time))

        if step % log_every == 0:
            print(f"  t={data.time:6.1f} s  dist={dist:6.3f} m  "
                  f"boom={boom:4.2f}  phase={phase:4.2f}")

        # Latch once the cube is centered in the claw: snap the grip fully shut
        # and stop chasing it (a gripped cube tracks the palm, so the planner's
        # distance signal goes flat and the boom would otherwise wander).
        if dist < 0.08:
            grasped = True
            boom_grasp = float(data.qpos[7])
            t_latch = float(data.time)
            print(f"  GRASP at t={data.time:.1f} s: cube centered in the claw "
                  f"({dist:.3f} m), latching grip closed")
            break

    n_capture = len(qpos_hist)
    if not grasped:
        print("WARN: did not reach a centered grasp; rendering the approach only.")

    # Inertia ratio of the captured stack (boom extended, cube in the claw), read
    # from the live model masses + body frames -> sets the gravity-gradient
    # libration. omega_lib = omega0 * sqrt(3 K); small-amplitude period below.
    K, (i_min, i_mid, i_max) = compute_K(model, data)
    t_orbit = 2.0 * np.pi / omega
    t_lib = t_orbit / np.sqrt(3.0 * K)
    print(f"captured-stack inertia: I_min={i_min:.1f} I_mid={i_mid:.1f} "
          f"I_max={i_max:.1f} kg·m²")
    print(f"  K = (I_max-I_min)/I_max = {K:.4f}  ->  small-amplitude libration "
          f"period {t_lib:.0f} s ({t_lib/t_orbit:.2f} orbit)")

    # Post-grasp: freeze the captured state (grip closed, boom fixed) and coast
    # for a long span. The bus+arm+gripped-cube stack is elongated and inertially
    # fixed while local-vertical rotates with the orbit, so the gravity gradient
    # librates it; the grip holds the cube through the swing. Sub-sample so the
    # long passive tail does not dwarf the file.
    if args.hold_seconds > 0:
        n_hold = int(round(args.hold_seconds / dt))
        stride = max(1, n_hold // max(1, args.hold_frames))
        for k in range(n_hold):
            data.ctrl[:] = [boom_grasp, 1.0, 1.0]
            mjo_step(model, data)
            if k % stride == 0 or k == n_hold - 1:
                qpos_hist.append(np.asarray(data.qpos).copy())
                R_hist.append(np.asarray(data.orbit.R_eci).copy())
                t_hist.append(float(data.time))
        held = plant_distance(data)
        print(f"  coasted {args.hold_seconds:.0f} s with grip closed; "
              f"cube still in claw at {held:.3f} m")

    np.savez(
        args.out,
        qpos=np.asarray(qpos_hist),
        R_eci=np.asarray(R_hist),
        t=np.asarray(t_hist),
        V_eci=V0,
        t_latch=t_latch,
        dt=dt,
        xml_path=str(SPACECRAFT_CAPTURE_CLAW_XML),
        n_capture=n_capture,
    )
    print(f"saved {len(qpos_hist)}-step trajectory to {args.out} (approach={n_capture})")


if __name__ == "__main__":
    main()
