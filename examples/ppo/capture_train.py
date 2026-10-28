"""Train the bi-manual capture-and-reorient policy with PPO + SGS (rsl-rl).

Sparse binary success reward + the paper's small regularizers (see
``capture_env.py``), resets drawn from a fixed configuration set by Success
Guided Sampling (``--sampler sgs``, default) or uniformly (``--sampler
uniform``, the baseline).

Usage (from the repo root):

    pixi run -e rl python examples/ppo/capture_train.py --sampler sgs --run-name cap-sgs
    pixi run -e rl python examples/ppo/capture_train.py --sampler uniform --run-name cap-uni

Logs and checkpoints go to ``examples/ppo/logs/<run>``: rsl-rl's TensorBoard
events, ``train_metrics.csv`` (one row per PPO iteration: success rates per
reset family, sampler statistics), periodic ``sampler_*.npz`` snapshots of the
per-configuration success estimates, ``env_cfg.json``, and ``model_<it>.pt``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

# Imported before torch/rsl_rl so the native bindings get the env's libstdc++.
import mjorbit  # noqa: F401

# isort: split

import torch
from capture_env import CaptureEnvCfg, CaptureReorientEnv
from rsl_rl.runners import OnPolicyRunner
from train import build_train_cfg


def build_capture_train_cfg(seed: int, save_interval: int) -> dict:
    """The truss example's rsl-rl PPO config with a wider network.

    The SGS paper states PPO but no hyperparameter table; it holds the number of
    mini-batches fixed and all other PPO / optimizer settings constant across
    scales (Sec. 4.3), which is what this keeps fixed between the SGS and
    uniform runs.
    """
    cfg = build_train_cfg(seed)
    cfg["save_interval"] = save_interval
    cfg["actor"]["hidden_dims"] = [512, 256, 128]
    cfg["critic"]["hidden_dims"] = [512, 256, 128]
    # The success reward is 100 (paper weight); rsl-rl's clipped value loss would
    # cap each value update at +-clip_param (0.2) and stall the critic for
    # hundreds of iterations at that return scale.
    cfg["algorithm"]["use_clipped_value_loss"] = False
    # Actions are clipped to [-1, 1], so the return is insensitive to std beyond
    # saturation and the truss example's entropy bonus (0.02) inflated every
    # action std to ~2.5 within 150 iterations (bang-bang noise). A smaller bonus
    # plus a std ceiling keeps exploration noise meaningful.
    cfg["algorithm"]["entropy_coef"] = 0.005
    cfg["actor"]["distribution_cfg"]["std_range"] = (0.02, 1.0)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampler", choices=["sgs", "uniform"], default="sgs")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--num-configs", type=int, default=8192)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--config-seed", type=int, default=0, help="seed of the fixed training configuration set"
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint .pt to resume from")
    parser.add_argument("--no-cuda-graph", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    run_name = args.run_name or time.strftime(f"cap-{args.sampler}-%Y%m%d-%H%M%S")
    log_dir = Path(__file__).with_name("logs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = build_capture_train_cfg(args.seed, args.save_interval)
    env_cfg = CaptureEnvCfg(
        num_envs=args.num_envs,
        num_configs=args.num_configs,
        config_seed=args.config_seed,
        sampler=args.sampler,
        steps_per_log=train_cfg["num_steps_per_env"],
        log_dir=str(log_dir),
        use_cuda_graph=not args.no_cuda_graph,
        device=args.device,
        seed=args.seed,
    )
    (log_dir / "env_cfg.json").write_text(json.dumps(dataclasses.asdict(env_cfg), indent=2))
    (log_dir / "train_cfg.json").write_text(json.dumps(train_cfg, indent=2))
    env = CaptureReorientEnv(env_cfg)

    runner = OnPolicyRunner(env, train_cfg, log_dir=str(log_dir), device=args.device)
    if args.resume is not None:
        runner.load(args.resume)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)


if __name__ == "__main__":
    main()
