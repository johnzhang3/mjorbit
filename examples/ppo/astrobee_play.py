"""Evaluate a trained Astrobee detumble-and-grasp policy and report stats.

Usage:

    pixi run -e rl python examples/ppo/astrobee_play.py \
        --checkpoint examples/ppo/logs/<run>/model_<it>.pt

Optionally dumps a single-world qpos trajectory to .npz for offline rendering.
"""

from __future__ import annotations

import argparse

# Imported before torch/rsl_rl so the native bindings get the env's libstdc++.
import mjorbit  # noqa: F401

# isort: split

import numpy as np
import torch
from astrobee_env import AstrobeeEnvCfg, AstrobeeGraspEnv
from rsl_rl.runners import OnPolicyRunner
from train import build_train_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-traj", type=str, default=None, help="output .npz path")
    parser.add_argument(
        "--record-world",
        type=str,
        default="median",
        help="world to dump for --save-traj: 'median' (median initial distance), "
             "'hardest' (farthest), or an int index",
    )
    args = parser.parse_args()

    env = AstrobeeGraspEnv(
        AstrobeeEnvCfg(num_envs=args.num_envs, device=args.device, seed=args.seed)
    )
    runner = OnPolicyRunner(env, build_train_cfg(args.seed), log_dir=None, device=args.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=args.device)

    init_dist = env.grip_bar_dist()
    if args.record_world == "hardest":
        rec_w = int(np.argmax(init_dist))
    elif args.record_world == "median":
        rec_w = int(np.argsort(init_dist)[len(init_dist) // 2])
    else:
        rec_w = int(args.record_world)
    if args.save_traj:
        print(f"recording world {rec_w} (initial gripper->bar {init_dist[rec_w]:.2f} m)")

    dist_log, grasped_log, spin_log = [], [], []
    qpos_hist, r_eci_hist = [], []
    v_eci0 = np.asarray(env.data.orbit.V_eci[rec_w], dtype=float).copy()
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(args.steps):
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            dist_log.append(extras["log"]["/metrics/grip_bar_dist"])
            grasped_log.append(extras["log"]["/metrics/grasped_frac"])
            spin_log.append(extras["log"]["/metrics/spin"])
            if args.save_traj:
                if bool(dones[rec_w]):
                    break
                qpos_hist.append(env.data.qpos[rec_w].copy())
                r_eci_hist.append(np.asarray(env.data.orbit.R_eci[rec_w], dtype=float).copy())

    dist_log = np.asarray(dist_log)
    tail = slice(-len(dist_log) // 4, None)
    print(f"steps:                  {args.steps} ({args.steps * env.cfg.control_dt:.0f} s sim)")
    print(f"mean gripper->bar:      {dist_log.mean():.3f} m")
    print(f"final-quarter dist:     {dist_log[tail].mean():.3f} m")
    print(f"final-quarter grasped:  {np.asarray(grasped_log)[tail].mean():.2%}")
    print(f"final-quarter spin:     {np.asarray(spin_log)[tail].mean():.3f} rad/s")

    if args.save_traj:
        from astrobee_env import _XML_PATH

        np.savez(
            args.save_traj,
            qpos=np.asarray(qpos_hist),
            R_eci=np.asarray(r_eci_hist),
            V_eci=v_eci0,
            dt=float(env.cfg.control_dt),
            control_dt=env.cfg.control_dt,
            xml_path=str(_XML_PATH),
            grip_bar_dist=dist_log,
        )
        print(f"saved world-{rec_w} trajectory ({len(qpos_hist)} steps) to {args.save_traj}")


if __name__ == "__main__":
    main()
