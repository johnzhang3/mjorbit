"""Evaluate capture-and-reorient checkpoints on held-out task configurations.

Held-out resets are freshly drawn from the training distribution with a
different seed (as in the SGS paper's evaluation, Sec. 4.3), balanced across the
three reset families. Every world runs exactly one episode from its own
configuration with the deterministic (mean) policy; an episode counts as a
success if the binary success predicate fires before the 60 s timeout.

    # one checkpoint, with a rollout dump for the video
    pixi run -e rl python examples/ppo/capture_eval.py \
        --run examples/ppo/logs/cap-sgs --checkpoint model_2999.pt \
        --save-traj videos/capture_traj.npz
    # success vs environment steps over a run's checkpoints (every 200 iterations)
    pixi run -e rl python examples/ppo/capture_eval.py --run examples/ppo/logs/cap-sgs --every 200

Results go to ``<run>/eval.csv`` (one row per checkpoint).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import re
from pathlib import Path

# Imported before torch/rsl_rl so the native bindings get the env's libstdc++.
import mjorbit  # noqa: F401

# isort: split

import numpy as np
import torch
from capture_env import FAMILIES, XML_PATH, CaptureEnvCfg, CaptureReorientEnv
from capture_train import build_capture_train_cfg
from rsl_rl.runners import OnPolicyRunner

_ERR_BINS = (0, 45, 90, 135, 180)


def _checkpoints(run: Path, every: int, max_iter: int | None) -> list[tuple[int, Path]]:
    """Checkpoints every ``every`` iterations (plus the last), skipping any
    iteration already in ``eval.csv`` so a sweep can be resumed."""
    found = sorted(
        (int(m.group(1)), p)
        for p in run.glob("model_*.pt")
        if (m := re.match(r"model_(\d+)\.pt", p.name))
    )
    if max_iter is not None:
        found = [(it, p) for it, p in found if it <= max_iter]
    last = found[-1][0] if found else -1
    done = set()
    if (run / "eval.csv").exists():
        done = {int(r["iteration"]) for r in csv.DictReader((run / "eval.csv").open())}
    return [(it, p) for it, p in found if (it % every == 0 or it == last) and it not in done]


def run_episodes(
    env: CaptureReorientEnv,
    policy,
    *,
    record: np.ndarray | None = None,
    post_success_s: float = 0.0,
) -> dict[str, np.ndarray]:
    """One episode per world from configs 0..num_envs-1. Optionally records the
    state of the ``record`` worlds every control step (continuing for
    ``post_success_s`` after a success, to show the hold)."""
    n = env.num_envs
    obs = env.reset_to_configs(np.arange(n))
    outcome = np.full(n, -1)  # -1 running, 0 timeout, 1 success, 2 failure
    t_end = np.zeros(n, dtype=int)
    post_steps = int(round(post_success_s / env.cfg.control_dt))
    trace: dict[str, list] = {
        k: [] for k in ("qpos", "R_eci", "V_eci", "point_err", "held", "grip")
    }
    t = 0
    with torch.inference_mode():
        while True:
            obs, _, _, _ = env.step(policy(obs))
            t += 1
            newly = outcome < 0
            for code, flag in ((1, env.last_success), (2, env.last_fail), (0, env.last_timeout)):
                hit = newly & flag
                outcome[hit], t_end[hit] = code, t
                newly &= ~hit
            if record is not None:
                trace["qpos"].append(env.data.qpos[record].copy())
                trace["R_eci"].append(np.asarray(env.data.orbit.R_eci, dtype=float)[record].copy())
                trace["V_eci"].append(np.asarray(env.data.orbit.V_eci, dtype=float)[record].copy())
                trace["point_err"].append(np.degrees(env.last_point_err[record]))
                trace["held"].append(env.last_held[record].copy())
                trace["grip"].append(env._grip_closed[record].copy())
            done = outcome >= 0
            if record is not None:
                # Keep stepping while a recorded world is inside its post-success window.
                linger = (outcome[record] == 1) & (t < t_end[record] + post_steps)
                if done.all() and not linger.any():
                    break
            elif done.all():
                break
            if t >= env.max_episode_length + post_steps:
                break
    out = {"outcome": outcome, "t_end": t_end}
    if record is not None:
        out.update({k: np.asarray(v) for k, v in trace.items()})
    return out


def summarize(env: CaptureReorientEnv, res: dict[str, np.ndarray]) -> dict[str, float]:
    fam = env.configs["family"]
    err0 = np.degrees(env.configs["point_err0"])
    succ = res["outcome"] == 1
    row = {
        "success": float(succ.mean()),
        "fail": float((res["outcome"] == 2).mean()),
        "timeout": float((res["outcome"] == 0).mean()),
        "t_success_s": float(res["t_end"][succ].mean() * env.cfg.control_dt)
        if succ.any()
        else float("nan"),
    }
    for f, name in enumerate(FAMILIES):
        row[f"success_{name}"] = float(succ[fam == f].mean())
    for lo, hi in zip(_ERR_BINS[:-1], _ERR_BINS[1:]):
        sel = (err0 >= lo) & (err0 < hi)
        row[f"success_err{lo}_{hi}"] = float(succ[sel].mean())
    # Approach success by difficulty tercile (graded resets).
    u = env.configs["difficulty"]
    for lo, hi in ((0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01)):
        sel = (fam == FAMILIES.index("approach")) & (u >= lo) & (u < hi)
        row[f"success_app_u{int(lo * 3)}"] = float(succ[sel].mean()) if sel.any() else float("nan")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=str, required=True, help="training log dir")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="model_<it>.pt in --run (default: sweep)"
    )
    parser.add_argument("--every", type=int, default=200, help="sweep: evaluate every N iterations")
    parser.add_argument(
        "--max-iter", type=int, default=None, help="sweep: only up to this iteration"
    )
    parser.add_argument("--num-eval", type=int, default=3072)
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--save-traj",
        type=str,
        default=None,
        help="dump one successful rollout (render_traj.py format) to this .npz",
    )
    parser.add_argument("--traj-family", choices=FAMILIES, default="approach")
    parser.add_argument(
        "--traj-min-err-deg",
        type=float,
        default=90.0,
        help="prefer rollouts whose initial pointing error exceeds this",
    )
    parser.add_argument("--post-success-s", type=float, default=5.0)
    args = parser.parse_args()

    run = Path(args.run)
    train_env = json.loads((run / "env_cfg.json").read_text())
    # Same task / reset distribution as training, fresh configurations.
    known = {f.name for f in dataclasses.fields(CaptureEnvCfg)}
    cfg = CaptureEnvCfg(**{k: v for k, v in train_env.items() if k in known})
    cfg = dataclasses.replace(
        cfg,
        num_envs=args.num_eval,
        num_configs=args.num_eval,
        config_seed=args.eval_seed,
        sampler="uniform",
        auto_reset=False,
        log_dir=None,
        device=args.device,
        seed=args.eval_seed,
    )
    env = CaptureReorientEnv(cfg)
    runner = OnPolicyRunner(env, build_capture_train_cfg(0, 50), log_dir=None, device=args.device)
    steps_per_it = train_env["num_envs"] * train_env["steps_per_log"]

    if args.checkpoint is not None:
        ckpts = [
            (
                int(re.search(r"(\d+)", Path(args.checkpoint).stem).group(1)),
                run / Path(args.checkpoint).name,
            )
        ]
    else:
        ckpts = _checkpoints(run, args.every, args.max_iter)
        if not ckpts:
            print("all selected checkpoints already evaluated")
            return

    record = None
    if args.save_traj:
        fam_id = FAMILIES.index(args.traj_family)
        cand = np.flatnonzero(env.configs["family"] == fam_id)
        err0 = np.degrees(env.configs["point_err0"][cand])
        # Record up to 48 worlds of the family, large initial pointing errors first.
        record = cand[np.argsort(err0 < args.traj_min_err_deg, kind="stable")][:48]

    out_csv = run / "eval.csv"
    for it, path in ckpts:
        runner.load(str(path))
        policy = runner.get_inference_policy(device=args.device)
        res = run_episodes(
            env,
            policy,
            record=record if it == ckpts[-1][0] else None,
            post_success_s=args.post_success_s,
        )
        row = {"iteration": it, "env_steps": (it + 1) * steps_per_it, **summarize(env, res)}
        print(
            json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}),
            flush=True,
        )
        new = not out_csv.exists()
        with out_csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)

    if args.save_traj and record is not None:
        ok = np.flatnonzero(res["outcome"][record] == 1)
        if ok.size == 0:
            print(
                f"no successful {args.traj_family} rollout among the recorded worlds; nothing saved"
            )
            return
        # Most dramatic success: largest initial pointing error.
        err0 = env.configs["point_err0"][record[ok]]
        j = int(ok[np.argmax(err0)])
        w = int(record[j])
        t_stop = min(
            len(res["qpos"]),
            int(res["t_end"][w]) + int(round(args.post_success_s / cfg.control_dt)),
        )
        Path(args.save_traj).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_traj,
            qpos=res["qpos"][:t_stop, j],
            R_eci=res["R_eci"][:t_stop, j],
            V_eci=res["V_eci"][0, j],
            V_eci_per_frame=res["V_eci"][:t_stop, j],
            dt=float(cfg.control_dt),
            xml_path=str(XML_PATH),
            point_err_deg=res["point_err"][:t_stop, j],
            held=res["held"][:t_stop, j],
            grip_closed=res["grip"][:t_stop, j],
            t_success_s=float(res["t_end"][w] * cfg.control_dt),
            point_err0_deg=float(np.degrees(env.configs["point_err0"][w])),
            family=args.traj_family,
            checkpoint=str(ckpts[-1][1]),
        )
        print(
            f"saved world {w} ({args.traj_family}, initial pointing error "
            f"{np.degrees(env.configs['point_err0'][w]):.0f} deg, success at "
            f"{res['t_end'][w] * cfg.control_dt:.1f} s) -> {args.save_traj}"
        )


if __name__ == "__main__":
    main()
