"""Evaluate a trained policy in the FULL mjorbit_warp orbital environment.

This is the common evaluator for the sim-fidelity study: every policy -- whether
it was trained under bare ``mjwarp`` or full ``mjorbit`` dynamics -- is scored
here, in the high-fidelity orbital environment that stands in for "reality". A
fixed ``--eval-seed`` makes every policy face the *identical* set of worlds
(orbits / initial conditions), so the comparison is paired.

Auto-reset is disabled, so each world runs exactly one fixed-horizon episode; we
record per-world outcomes and dump them (plus a summary) to JSON for
``plot_results.py``.

    pixi run -e rl python experiments/sim_fidelity/eval_fidelity.py \
        --task truss --checkpoint examples/ppo/logs/.../model_440.pt \
        --tag mjorbit/seed0 --out experiments/sim_fidelity/results/truss/mjorbit_seed0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PPO = Path(__file__).resolve().parents[2] / "examples" / "ppo"
sys.path.insert(0, str(PPO))

import mjorbit  # noqa: F401,E402  (native bindings before torch)

# isort: split

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from train import build_train_cfg  # noqa: E402

# Success thresholds for the truss nadir-pointing task (final-window mean error).
TRUSS_THRESHOLDS_DEG = (1.0, 2.0, 3.0, 5.0, 10.0)
# An astrobee world counts as a success if it is grasped+held for at least this
# fraction of the final window.
ASTROBEE_HOLD_FRAC = 0.5


def _crashed(env) -> np.ndarray:
    qpos = np.asarray(env.data.qpos)
    qvel = np.asarray(env.data.qvel)
    return ~np.isfinite(qpos).all(axis=1) | ~np.isfinite(qvel).all(axis=1)


def eval_truss(checkpoint: str, num_envs: int, eval_seed: int, steps: int | None,
               device: str) -> dict:
    from truss_env import TrussEnvCfg, TrussReorientEnv

    cfg = TrussEnvCfg(num_envs=num_envs, device=device, seed=eval_seed,
                      backend="mjorbit", auto_reset=False)
    env = TrussReorientEnv(cfg)
    runner = OnPolicyRunner(env, build_train_cfg(eval_seed), log_dir=None, device=device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=device)

    steps = steps or env.max_episode_length
    initial_err = np.degrees(env.truss_align_error()).copy()
    align = np.full((steps, num_envs), np.nan)
    ever_lost = np.zeros(num_envs, dtype=bool)
    ever_crashed = np.zeros(num_envs, dtype=bool)

    obs = env.get_observations()
    with torch.inference_mode():
        for t in range(steps):
            obs, _, _, _ = env.step(policy(obs))
            align[t] = np.degrees(env.truss_align_error())
            ever_lost |= env._hug_dist() > cfg.hug_lost_m
            ever_crashed |= _crashed(env)

    win = slice(-(steps // 4), None)
    final_align = np.nanmean(align[win], axis=0)  # per world, deg
    alive = ~ever_lost & ~ever_crashed
    success = {f"success@{d}deg": float(np.mean(alive & (final_align <= d)))
               for d in TRUSS_THRESHOLDS_DEG}

    alive_align = final_align[alive]
    n_alive = alive_align.size
    summary = {
        "n_worlds": int(num_envs),
        "steps": int(steps),
        "alive_frac": float(np.mean(alive)),
        "lost_frac": float(np.mean(ever_lost)),
        "crashed_frac": float(np.mean(ever_crashed)),
        "final_align_mean_deg": float(np.mean(alive_align)) if n_alive else float("nan"),
        "final_align_median_deg": float(np.median(alive_align)) if n_alive else float("nan"),
        "final_align_p90_deg": float(np.quantile(alive_align, 0.9)) if n_alive else float("nan"),
        "initial_err_mean_deg": float(np.mean(initial_err)),
        **success,
    }
    per_world = {
        "final_align_deg": final_align.tolist(),
        "ever_lost": ever_lost.tolist(),
        "ever_crashed": ever_crashed.tolist(),
        "initial_err_deg": initial_err.tolist(),
    }
    return {"summary": summary, "per_world": per_world}


def eval_astrobee(checkpoint: str, num_envs: int, eval_seed: int, steps: int | None,
                  device: str) -> dict:
    from astrobee_env import AstrobeeEnvCfg, AstrobeeGraspEnv

    cfg = AstrobeeEnvCfg(num_envs=num_envs, device=device, seed=eval_seed,
                         backend="mjorbit", auto_reset=False)
    env = AstrobeeGraspEnv(cfg)
    runner = OnPolicyRunner(env, build_train_cfg(eval_seed), log_dir=None, device=device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=device)

    steps = steps or env.max_episode_length
    initial_dist = env.grip_bar_dist().copy()
    grasped = np.zeros((steps, num_envs), dtype=bool)
    spin = np.full((steps, num_envs), np.nan)
    ever_failed = np.zeros(num_envs, dtype=bool)
    ever_crashed = np.zeros(num_envs, dtype=bool)

    obs = env.get_observations()
    with torch.inference_mode():
        for t in range(steps):
            obs, _, _, _ = env.step(policy(obs))
            grasped[t] = env._grasped
            qvel = np.asarray(env.data.qvel)
            spin[t] = np.linalg.norm(qvel[:, env._bus_v + 3 : env._bus_v + 6], axis=1)
            ever_failed |= env.grip_bar_dist() > cfg.fail_radius
            ever_crashed |= _crashed(env)

    win = slice(-(steps // 4), None)
    grasped_frac = grasped[win].mean(axis=0)  # per world
    final_spin = np.nanmean(spin[win], axis=0)
    held = grasped_frac >= ASTROBEE_HOLD_FRAC
    success = held & ~ever_crashed & ~ever_failed

    summary = {
        "n_worlds": int(num_envs),
        "steps": int(steps),
        "success_rate": float(np.mean(success)),
        "grasped_frac_mean": float(np.mean(grasped_frac)),
        "any_grasp_frac": float(np.mean(grasped.any(axis=0))),
        "failed_frac": float(np.mean(ever_failed)),
        "crashed_frac": float(np.mean(ever_crashed)),
        "final_spin_mean": float(np.nanmean(final_spin)),
        "initial_dist_mean_m": float(np.mean(initial_dist)),
    }
    per_world = {
        "grasped_frac": grasped_frac.tolist(),
        "final_spin": final_spin.tolist(),
        "ever_failed": ever_failed.tolist(),
        "ever_crashed": ever_crashed.tolist(),
        "success": success.tolist(),
        "initial_dist_m": initial_dist.tolist(),
    }
    return {"summary": summary, "per_world": per_world}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["truss", "astrobee"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tag", required=True, help="label for this policy, e.g. 'mjwarp_fair/seed0'")
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--eval-seed", type=int, default=20240617,
                   help="fixed across all policies so worlds are identical (paired eval)")
    p.add_argument("--steps", type=int, default=None, help="default: one full episode")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed_all(args.eval_seed)

    fn = eval_truss if args.task == "truss" else eval_astrobee
    result = fn(args.checkpoint, args.num_envs, args.eval_seed, args.steps, args.device)
    result["meta"] = {
        "task": args.task,
        "tag": args.tag,
        "checkpoint": str(args.checkpoint),
        "eval_seed": args.eval_seed,
        "num_envs": args.num_envs,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    s = result["summary"]
    print(f"[{args.tag}] {args.task} -> {args.out}")
    for k, v in s.items():
        print(f"    {k:26s} {v}")


if __name__ == "__main__":
    main()
