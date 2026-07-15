# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Produce a free-flyer servicing trajectory for rendering: scripted fly-in and
grab, then the trained-policy hug + gravity-gradient stabilize.

The trained policy (examples/ppo) handles the contact-rich part -- holding the
big free-floating truss in a two-arm hug (wrist-leveled jaws clamping the spar
with shallow surface contact) and slewing it to nadir with the reaction wheels.
The approach and grab are scripted bookends so the clip shows the whole
sequence:

    start apart -> fly in (jaws open) -> close the jaws on the spar -> slew it to
    point at Earth.

With `--release-s > 0` the clip also scripts a jaw-open release and a sideways
departure, but opening a force-clamped grip kicks the now-free truss off nadir
(gravity gradient is too slow to re-settle it in a clip), so by default the clip
ends with the truss pointed and held.

    pixi run -e rl python scripts/record/produce_hug.py \
        --checkpoint examples/ppo/logs/<run>/model_<it>.pt --out /tmp/hug_traj.npz
    pixi run -e rl python scripts/record/record_truss.py \
        --traj /tmp/hug_traj.npz --out videos/truss_pointing.mp4 --clip-sim-seconds 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mjorbit  # noqa: F401  (load native bindings before torch)

# isort: split

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

_HERE = Path(__file__).resolve().parent
_PPO = _HERE.parents[1] / "examples" / "ppo"
for p in (str(_PPO), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from train import build_train_cfg  # noqa: E402
from truss_env import _GRIP_CLOSED, TrussEnvCfg, TrussReorientEnv  # noqa: E402

from mjorbit.constants import GM_EARTH, R_EARTH  # noqa: E402
from mjorbit_warp import mjo_forward, mjo_pull, mjo_step, mjo_upload  # noqa: E402

W = 0  # recorded world
_GRIP_OPEN = 0.12


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="/tmp/hug_traj.npz")
    ap.add_argument("--phase-deg", type=float, default=52.0, help="initial pointing error")
    ap.add_argument("--approach-s", type=float, default=5.0)
    ap.add_argument("--grab-s", type=float, default=2.0)
    ap.add_argument("--stabilize-s", type=float, default=55.0)
    ap.add_argument("--release-s", type=float, default=2.0)
    ap.add_argument("--depart-s", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = TrussEnvCfg(num_envs=2, seed=args.seed)
    env = TrussReorientEnv(cfg)
    runner = OnPolicyRunner(env, build_train_cfg(args.seed), log_dir=None, device=cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=cfg.device)

    hug = env._arm_ready  # [sh_a, el_a, sh_b, el_b]; wrist = -(sh+el) levels the hand
    wrist = np.array([-(hug[0] + hug[1]), -(hug[2] + hug[3])])
    bus_q, bus_v, tr_q = env._bus_q, env._bus_v, env._tr_q
    dt = cfg.control_dt
    radius = R_EARTH + cfg.altitude_km
    speed = float(np.sqrt(GM_EARTH / radius))
    phi = np.radians(args.phase_deg)

    # Initial state: bus offset below, arms at the hug pose with jaws OPEN, truss
    # at identity (orbit phase sets the pointing error), at rest.
    env.data.orbit.R_eci[W] = radius * np.array([np.cos(phi), 0.0, np.sin(phi)])
    env.data.orbit.V_eci[W] = speed * np.array([-np.sin(phi), 0.0, np.cos(phi)])
    env.data.qpos[W] = env._qpos0
    env.data.qpos[W, bus_q : bus_q + 3] = [0.0, 0.0, -0.9]
    env.data.qpos[W, bus_q + 3 : bus_q + 7] = [1.0, 0.0, 0.0, 0.0]
    env.data.qpos[W, env._arm_jq] = hug
    env.data.qpos[W, env._wrist_jq] = wrist
    env.data.qpos[W, env._grip_jq] = _GRIP_OPEN
    env.data.qpos[W, tr_q : tr_q + 3] = [0.0, 0.0, 1.0]
    env.data.qpos[W, tr_q + 3 : tr_q + 7] = [1.0, 0.0, 0.0, 0.0]
    env.data.qvel[W] = 0.0
    mjo_upload(env.model, env.data, fields=("qpos", "qvel", "orbit"))
    mjo_forward(env.model, env.data)
    mjo_pull(env.model, env.data, fields=("qpos", "qvel", "xpos", "xmat", "orbit"))

    v_eci0 = np.asarray(env.data.orbit.V_eci[W], dtype=float).copy()
    qpos_hist, r_hist = [], []

    def scripted(grip, bus_target):
        """One control step driving both worlds with the hug arm pose (jaws at
        `grip`) + a PD bus translation (no slew), then record world W."""
        bus_pos = env.data.qpos[:, bus_q : bus_q + 3]
        bus_vel = env.data.qvel[:, bus_v : bus_v + 3]
        thr = np.clip(-250.0 * (bus_pos - bus_target) - 80.0 * bus_vel, -80.0, 80.0)
        env.data.ctrl[:, env._arm_ctrl] = hug[None, :]
        env.data.ctrl[:, env._wrist_ctrl] = wrist[None, :]
        env.data.ctrl[:, env._grip_ctrl] = grip
        env.data.ctrl[:, env._rw_ctrl] = 0.0
        env.data.ctrl[:, env._thrust_ctrl] = thr
        mjo_upload(env.model, env.data, fields=("ctrl",))
        for _ in range(env._decimation):
            mjo_step(env.model, env.data)
        mjo_pull(env.model, env.data, fields=("qpos", "qvel", "xpos", "xmat", "orbit"))
        qpos_hist.append(env.data.qpos[W].copy())
        r_hist.append(np.asarray(env.data.orbit.R_eci[W], dtype=float).copy())

    origin = np.zeros(3)
    # Phase 1: fly in -- bus rises to the truss, jaws open.
    for _ in range(int(round(args.approach_s / dt))):
        scripted(_GRIP_OPEN, origin)
    # Phase 2: grab -- close the jaws on the spar (bus holds).
    ng = int(round(args.grab_s / dt))
    for k in range(ng):
        f = (k + 1) / ng
        scripted((1 - f) * _GRIP_OPEN + f * _GRIP_CLOSED, origin)

    # Phase 3: trained policy hugs + slews the truss to nadir.
    env._last_actions[:] = 0.0
    env._prev_align_err = env.truss_align_error()
    obs = env._compute_obs()
    with torch.inference_mode():
        for _ in range(int(round(args.stabilize_s / dt))):
            a = policy(obs)
            obs, _, _, _ = env.step(a)
            qpos_hist.append(env.data.qpos[W].copy())
            r_hist.append(np.asarray(env.data.orbit.R_eci[W], dtype=float).copy())

    # Phases 4-5 (optional) release + fly away. Opening a force-clamped grip on
    # the round spar imparts a small kick to the now-free truss (and gravity
    # gradient is far too slow to re-settle it within a clip), so by default the
    # clip ends with the truss pointed and held. Enable with --release-s > 0.
    # Note: a scripted hold here would zero the reaction wheels and let the
    # assembly drift off nadir, so when release is disabled the clip simply ends
    # with the policy still actively holding the truss pointed at Earth.
    if args.release_s > 0:
        nr = int(round(args.release_s / dt))
        for k in range(nr):
            f = (k + 1) / nr
            scripted((1 - f) * _GRIP_CLOSED + f * _GRIP_OPEN, origin)
        for k in range(int(round(args.depart_s / dt))):
            f = (k + 1) / int(round(args.depart_s / dt))
            scripted(_GRIP_OPEN, np.array([0.0, 1.6 * f, 0.0]))

    np.savez(
        args.out,
        qpos=np.asarray(qpos_hist),
        R_eci=np.asarray(r_hist),
        V_eci=v_eci0,
        dt=float(dt),
        xml_path=str(_PPO / "spacecraft_truss.xml"),
    )
    print(f"saved {len(qpos_hist)}-step servicing trajectory to {args.out}")


if __name__ == "__main__":
    main()
