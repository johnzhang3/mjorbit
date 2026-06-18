"""Train a PPO policy for the Astrobee detumble-and-grasp task with rsl-rl.

Usage (from the repo root):

    pixi run -e rl python examples/ppo/astrobee_train.py --num-envs 1024 --max-iterations 600

Logs and checkpoints go to ``examples/ppo/logs/<run>``; monitor with
``pixi run -e rl tensorboard --logdir examples/ppo/logs``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

# Imported before torch/rsl_rl so the native bindings get the env's libstdc++.
import mjorbit  # noqa: F401

# isort: split

import torch
from astrobee_env import AstrobeeEnvCfg, AstrobeeGraspEnv
from rsl_rl.runners import OnPolicyRunner
from train import build_train_cfg  # shared rsl-rl runner/PPO config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--max-iterations", type=int, default=600)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint .pt to resume from")
    parser.add_argument("--init-spin", type=float, default=None,
                        help="curriculum: max initial body-frame spin per axis (rad/s)")
    parser.add_argument("--cargo-dist-min", type=float, default=None,
                        help="curriculum: min cargo spawn distance (m)")
    parser.add_argument("--cargo-dist-max", type=float, default=None,
                        help="curriculum: max cargo spawn distance (m)")
    args = parser.parse_args()

    # rsl-rl does not consume cfg["seed"]; seed torch here for reproducible
    # network init and action sampling (the env's numpy RNG is seeded in cfg).
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    env_cfg = AstrobeeEnvCfg(num_envs=args.num_envs, device=args.device, seed=args.seed)
    if args.init_spin is not None:
        env_cfg.init_spin_max = args.init_spin
    lo, hi = env_cfg.cargo_dist
    if args.cargo_dist_min is not None:
        lo = args.cargo_dist_min
    if args.cargo_dist_max is not None:
        hi = args.cargo_dist_max
    env_cfg.cargo_dist = (lo, hi)
    env = AstrobeeGraspEnv(env_cfg)

    run_name = args.run_name or time.strftime("astrobee-%Y%m%d-%H%M%S")
    log_dir = Path(__file__).with_name("logs") / run_name
    runner = OnPolicyRunner(
        env, build_train_cfg(args.seed), log_dir=str(log_dir), device=args.device
    )
    if args.resume is not None:
        runner.load(args.resume)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
