# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Produce an Astrobee detumble-and-grasp trajectory for rendering.

Runs the trained policy (examples/ppo/astrobee_*) across a batch of worlds and
dumps the single world that best shows the whole sequence -- start tumbling and
drifting, detumble, fly to the free-floating cargo, and grasp + hold its grapple
bar -- as a ``(qpos, R_eci, V_eci, dt, xml_path)`` npz for the headless recorder.

    pixi run -e rl python scripts/record/produce_astrobee.py \
        --checkpoint examples/ppo/logs/<run>/model_<it>.pt --out /tmp/astrobee_traj.npz
    pixi run -e rl python scripts/record/record_truss.py \
        --traj /tmp/astrobee_traj.npz --out videos/astrobee_grasp.mp4 \
        --clip-sim-seconds 0 --mag 9000 --distance 70
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

from astrobee_env import AstrobeeEnvCfg, AstrobeeGraspEnv  # noqa: E402
from train import build_train_cfg  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="/tmp/astrobee_traj.npz")
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-start", type=float, default=0.8,
                    help="require this initial gripper->bar distance (a clear approach)")
    args = ap.parse_args()

    env = AstrobeeGraspEnv(
        AstrobeeEnvCfg(num_envs=args.num_envs, device="cuda", seed=args.seed)
    )
    runner = OnPolicyRunner(env, build_train_cfg(args.seed), log_dir=None, device="cuda")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device="cuda")

    n = args.num_envs
    start_dist = env.grip_bar_dist().copy()
    qpos_hist = np.zeros((args.steps, n, env.model.mj_model.nq))
    reci_hist = np.zeros((args.steps, n, 3))
    dist_hist = np.zeros((args.steps, n))
    grasp_hist = np.zeros((args.steps, n), dtype=bool)
    spin_hist = np.zeros((args.steps, n))
    bv = env._bus_v

    obs = env.get_observations()
    with torch.inference_mode():
        for t in range(args.steps):
            a = policy(obs)
            obs, _, dones, _ = env.step(a)
            qpos_hist[t] = env.data.qpos
            reci_hist[t] = np.asarray(env.data.orbit.R_eci)
            dist_hist[t] = env._grip_bar_dist()
            grasp_hist[t] = env._grasped
            spin_hist[t] = np.linalg.norm(env.data.qvel[:, bv + 3 : bv + 6], axis=1)
            # Freeze any world that resets so its dumped trajectory stays one episode.
            done = dones.cpu().numpy().astype(bool)
            if t + 1 < args.steps and done.any():
                qpos_hist[t + 1 :, done] = qpos_hist[t, done]
                reci_hist[t + 1 :, done] = reci_hist[t, done]

    # Score worlds: clear approach (started far), grasps, holds to the end, and a
    # quiet final hold (the cargo stays with the gripper).
    first_grasp = np.array([np.argmax(grasp_hist[:, w]) if grasp_hist[:, w].any() else -1
                            for w in range(n)])
    final_grasped = grasp_hist[-30:].all(axis=0)
    final_dist = dist_hist[-30:].mean(axis=0)
    ok = (start_dist > args.min_start) & final_grasped & (final_dist < 0.16) & (first_grasp >= 0)
    if not ok.any():
        ok = final_grasped & (first_grasp >= 0)
    if not ok.any():
        raise SystemExit("no world grasped and held; try another --seed or checkpoint")
    # Prefer the one with the earliest clean grasp among the longest approaches.
    score = np.where(ok, start_dist - 0.02 * first_grasp, -1e9)
    w = int(np.argmax(score))
    grasp_s = first_grasp[w] * env.cfg.control_dt
    print(f"world {w}: start {start_dist[w]:.2f} m, first grasp {grasp_s:.1f} s, "
          f"final dist {final_dist[w]:.3f} m, final spin {spin_hist[-1, w]:.3f} rad/s")

    np.savez(
        args.out,
        qpos=qpos_hist[:, w],
        R_eci=reci_hist[:, w],
        V_eci=np.asarray(env.data.orbit.V_eci[w], dtype=float).copy(),
        dt=float(env.cfg.control_dt),
        control_dt=env.cfg.control_dt,
        xml_path=str(_PPO / "astrobee_grasp.xml"),
        grip_bar_dist=dist_hist[:, w],
    )
    print(f"saved {args.steps}-step world-{w} trajectory to {args.out}")


if __name__ == "__main__":
    main()
