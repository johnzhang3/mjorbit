"""Train a PPO policy for the bi-manual truss earth-pointing task with rsl-rl.

Usage (from the repo root):

    pixi run -e rl python examples/ppo/train.py --num-envs 1024 --max-iterations 600

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
from rsl_rl.runners import OnPolicyRunner
from truss_env import TrussEnvCfg, TrussReorientEnv


def build_train_cfg(seed: int) -> dict:
    """rsl-rl (>= 5.x) runner/algorithm configuration."""
    return {
        "seed": seed,
        "num_steps_per_env": 32,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "logger": "tensorboard",
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.997,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.02,
            "learning_rate": 1.0e-3,
            "max_grad_norm": 1.0,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 128, 64],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 0.4,
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 128, 64],
            "activation": "elu",
            "obs_normalization": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--max-iterations", type=int, default=600)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint .pt to resume from")
    parser.add_argument("--backend", type=str, default="mjorbit",
                        choices=["mjorbit", "mjwarp"],
                        help="physics backend to TRAIN in (evaluation is always mjorbit)")
    parser.add_argument("--frozen-target", action="store_true",
                        help="bare-mjwarp 'naive' baseline: freeze the nadir target "
                             "(no kinematic orbit propagation). Ignored for --backend mjorbit")
    parser.add_argument("--resample-orbit", action="store_true",
                        help="re-randomize the orbit phase on every reset so all "
                             "backends share an identical initial-condition distribution")
    parser.add_argument("--init-offset-min", type=float, default=None,
                        help="curriculum: min initial truss misalignment (rad)")
    parser.add_argument("--init-offset-max", type=float, default=None,
                        help="curriculum: max initial truss misalignment (rad)")
    args = parser.parse_args()

    # rsl-rl does not consume cfg["seed"]; seed torch here for reproducible
    # network init and action sampling (the env's numpy RNG is seeded in cfg).
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    env_cfg = TrussEnvCfg(
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        backend=args.backend,
        mjwarp_moving_target=not args.frozen_target,
        resample_orbit_on_reset=args.resample_orbit,
    )
    if args.init_offset_min is not None and args.init_offset_max is not None:
        env_cfg.init_offset_rad = (args.init_offset_min, args.init_offset_max)
    env = TrussReorientEnv(env_cfg)

    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    log_dir = Path(__file__).with_name("logs") / run_name
    runner = OnPolicyRunner(
        env, build_train_cfg(args.seed), log_dir=str(log_dir), device=args.device
    )
    if args.resume is not None:
        runner.load(args.resume)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
