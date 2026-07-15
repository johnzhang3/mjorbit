"""Evaluate a trained truss earth-pointing policy and report alignment stats.

Usage:

    pixi run -e rl python examples/ppo/play.py --checkpoint examples/ppo/logs/<run>/model_<it>.pt

Optionally dumps a single-world qpos trajectory to .npz for offline plotting.
"""

from __future__ import annotations

import argparse

# Imported before torch/rsl_rl so the native bindings get the env's libstdc++.
import mjorbit  # noqa: F401

# isort: split

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from train import build_train_cfg
from truss_env import TrussEnvCfg, TrussReorientEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-traj", type=str, default=None, help="output .npz path")
    parser.add_argument(
        "--record-world",
        type=str,
        default="target",
        help="world to dump for --save-traj: 'target' (initial misalignment nearest "
             "--record-target-deg), 'hardest' (max), or an int index",
    )
    parser.add_argument("--record-target-deg", type=float, default=55.0,
                        help="for --record-world target: desired initial misalignment (deg)")
    args = parser.parse_args()

    env = TrussReorientEnv(TrussEnvCfg(num_envs=args.num_envs, device=args.device, seed=args.seed))
    runner = OnPolicyRunner(env, build_train_cfg(args.seed), log_dir=None, device=args.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=args.device)

    # Which world to dump for the recording. 'target' picks a large-but-
    # survivable initial misalignment (the extreme worlds tend to drop the
    # grasp); 'hardest' picks the most misaligned.
    init_err_deg = np.degrees(env.truss_align_error())
    if args.record_world == "hardest":
        rec_w = int(np.argmax(init_err_deg))
    elif args.record_world == "target":
        rec_w = int(np.argmin(np.abs(init_err_deg - args.record_target_deg)))
    else:
        rec_w = int(args.record_world)
    if args.save_traj:
        print(f"recording world {rec_w} (initial misalignment "
              f"{np.degrees(env.truss_align_error()[rec_w]):.1f} deg)")

    align_deg = []
    qpos_hist = []
    r_eci_hist = []
    v_eci0 = np.asarray(env.data.orbit.V_eci[rec_w], dtype=float).copy()
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(args.steps):
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            align_deg.append(extras["log"]["/metrics/align_err_deg"])
            if args.save_traj:
                # Stop before the recorded world auto-resets so the dump is one
                # clean episode.
                if bool(dones[rec_w]):
                    break
                qpos_hist.append(env.data.qpos[rec_w].copy())
                r_eci_hist.append(np.asarray(env.data.orbit.R_eci[rec_w], dtype=float).copy())

    align_deg = np.asarray(align_deg)
    tail = align_deg[-len(align_deg) // 4 :]
    print(f"steps:                  {args.steps} ({args.steps * env.cfg.control_dt:.0f} s sim)")
    print(f"mean align error:       {align_deg.mean():.2f} deg")
    print(f"final-quarter mean:     {tail.mean():.2f} deg")
    print(f"final-quarter p90:      {np.quantile(tail, 0.9):.2f} deg")

    if args.save_traj:
        # Format consumed by scripts/record/render_traj.py: per-step world-0
        # qpos + chief R_eci, a constant V_eci for camera framing, the sample
        # spacing (control_dt), and the XML so the recorder can build a CPU model.
        from truss_env import _XML_PATH

        np.savez(
            args.save_traj,
            qpos=np.asarray(qpos_hist),
            R_eci=np.asarray(r_eci_hist),
            V_eci=v_eci0,
            dt=float(env.cfg.control_dt),
            control_dt=env.cfg.control_dt,
            xml_path=str(_XML_PATH),
            align_err_deg=align_deg,
        )
        print(f"saved world-{rec_w} trajectory ({len(qpos_hist)} steps) to {args.save_traj}")


if __name__ == "__main__":
    main()
