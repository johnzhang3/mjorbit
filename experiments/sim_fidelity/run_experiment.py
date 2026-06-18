"""Orchestrate the sim-fidelity training/eval matrix.

For each (task, training-backend condition, seed) this trains a PPO policy and
then evaluates it in the FULL mjorbit_warp orbital environment (the common
"reality"). Training runs are launched as subprocesses (one fresh CUDA context
each, so GPU memory is released between runs); evaluation writes a per-policy
JSON under ``results/<task>/``.

Conditions
----------
truss (nadir-pointing; gravity-gradient + sweeping nadir is the orbital signal):
    mjorbit       - trained in full orbital dynamics (the home-sim policy)
    mjwarp_fair   - bare mjwarp dynamics, but the nadir target is propagated
                    kinematically so the policy still sees a MOVING reference
                    (isolates the orbital DYNAMICS gap)
    mjwarp_naive  - bare mjwarp dynamics AND a frozen nadir target (no orbit
                    model at all -- the literal vanilla-MJWarp baseline)

astrobee (detumble + grasp; no orbital-frame reference -> predicted null control):
    mjorbit       - trained in full orbital dynamics
    mjwarp        - bare mjwarp dynamics

Usage:
    pixi run -e rl python experiments/sim_fidelity/run_experiment.py \
        --tasks both --seeds 0 1 2
    # smoke the whole pipeline fast:
    pixi run -e rl python experiments/sim_fidelity/run_experiment.py --quick --seeds 0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PPO = ROOT / "examples" / "ppo"
RESULTS = HERE / "results"
RUN_LOGS = RESULTS / "logs"

TRUSS_CONDITIONS = {
    "mjorbit": {"backend": "mjorbit", "extra": []},
    "mjwarp_fair": {"backend": "mjwarp", "extra": []},
    "mjwarp_naive": {"backend": "mjwarp", "extra": ["--frozen-target"]},
}
ASTROBEE_CONDITIONS = {"mjorbit": "mjorbit", "mjwarp": "mjwarp"}


def latest_ckpt(run_dir: Path) -> Path | None:
    ckpts = list(run_dir.glob("model_*.pt"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(p.stem.split("_")[1]))


def sh(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"    $ {' '.join(cmd)}")
    print(f"      (log -> {log_path})")
    t0 = time.time()
    with log_path.open("w") as f:
        r = subprocess.run(cmd, cwd=str(PPO), stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if r.returncode != 0:
        tail = "\n".join(log_path.read_text().splitlines()[-25:])
        raise RuntimeError(f"command failed ({r.returncode}) after {dt:.0f}s:\n{tail}")
    print(f"      done in {dt:.0f}s")


def train_truss(cond: str, seed: int, iters: int, num_envs: int, device: str,
                skip_existing: bool) -> Path:
    spec = TRUSS_CONDITIONS[cond]
    run_name = f"fidelity/truss/{cond}/seed{seed}"
    run_dir = PPO / "logs" / run_name
    ck = latest_ckpt(run_dir)
    if skip_existing and ck is not None:
        print(f"    [skip] truss/{cond}/seed{seed} already trained -> {ck.name}")
        return ck
    cmd = [sys.executable, str(PPO / "train.py"),
           "--num-envs", str(num_envs), "--max-iterations", str(iters),
           "--device", device, "--seed", str(seed),
           "--backend", spec["backend"], "--run-name", run_name, *spec["extra"]]
    sh(cmd, RUN_LOGS / f"truss_{cond}_seed{seed}.log")
    ck = latest_ckpt(run_dir)
    if ck is None:
        raise RuntimeError(f"no checkpoint produced in {run_dir}")
    return ck


def train_astrobee(cond: str, seed: int, iters: int, num_envs: int, device: str,
                   skip_existing: bool) -> Path:
    backend = ASTROBEE_CONDITIONS[cond]
    close_name = f"fidelity/astrobee/{cond}/seed{seed}/close"
    far_name = f"fidelity/astrobee/{cond}/seed{seed}/far"
    close_dir = PPO / "logs" / close_name
    far_dir = PPO / "logs" / far_name

    far_ck = latest_ckpt(far_dir)
    if skip_existing and far_ck is not None:
        print(f"    [skip] astrobee/{cond}/seed{seed} already trained -> far/{far_ck.name}")
        return far_ck

    # Stage 1: close-range distance curriculum (grasping must be discoverable).
    close_ck = latest_ckpt(close_dir)
    if not (skip_existing and close_ck is not None):
        sh([sys.executable, str(PPO / "astrobee_train.py"),
            "--num-envs", str(num_envs), "--max-iterations", str(iters),
            "--device", device, "--seed", str(seed), "--backend", backend,
            "--run-name", close_name,
            "--cargo-dist-min", "0.9", "--cargo-dist-max", "1.1"],
           RUN_LOGS / f"astrobee_{cond}_seed{seed}_close.log")
        close_ck = latest_ckpt(close_dir)
    else:
        print(f"    [skip] astrobee/{cond}/seed{seed} stage-close -> {close_ck.name}")
    if close_ck is None:
        raise RuntimeError(f"no close checkpoint in {close_dir}")

    # Stage 2: resume at full range.
    sh([sys.executable, str(PPO / "astrobee_train.py"),
        "--num-envs", str(num_envs), "--max-iterations", str(iters),
        "--device", device, "--seed", str(seed), "--backend", backend,
        "--run-name", far_name, "--resume", str(close_ck),
        "--cargo-dist-min", "1.2", "--cargo-dist-max", "2.0"],
       RUN_LOGS / f"astrobee_{cond}_seed{seed}_far.log")
    far_ck = latest_ckpt(far_dir)
    if far_ck is None:
        raise RuntimeError(f"no far checkpoint in {far_dir}")
    return far_ck


def evaluate(task: str, cond: str, seed: int, ckpt: Path, eval_envs: int,
             eval_seed: int, device: str, skip_existing: bool) -> None:
    out = RESULTS / task / f"{cond}_seed{seed}.json"
    if skip_existing and out.exists():
        print(f"    [skip] eval {task}/{cond}/seed{seed} exists")
        return
    sh([sys.executable, str(HERE / "eval_fidelity.py"),
        "--task", task, "--checkpoint", str(ckpt), "--tag", f"{cond}/seed{seed}",
        "--out", str(out), "--num-envs", str(eval_envs),
        "--eval-seed", str(eval_seed), "--device", device],
       RUN_LOGS / f"eval_{task}_{cond}_seed{seed}.log")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", choices=["truss", "astrobee", "both"], default="both")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--eval-envs", type=int, default=1024)
    p.add_argument("--eval-seed", type=int, default=20240617)
    p.add_argument("--truss-iters", type=int, default=440)
    p.add_argument("--astrobee-iters", type=int, default=250, help="per curriculum stage")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-existing", action="store_true",
                   help="resume: skip runs whose checkpoint/result already exists")
    p.add_argument("--quick", action="store_true",
                   help="tiny smoke run: few iters, few worlds (validates the pipeline)")
    args = p.parse_args()

    if args.quick:
        args.num_envs = min(args.num_envs, 64)
        args.eval_envs = min(args.eval_envs, 64)
        args.truss_iters = 4
        args.astrobee_iters = 3

    tasks = ["truss", "astrobee"] if args.tasks == "both" else [args.tasks]
    t_start = time.time()
    for seed in args.seeds:
        for task in tasks:
            conds = TRUSS_CONDITIONS if task == "truss" else ASTROBEE_CONDITIONS
            for cond in conds:
                print(f"\n=== {task} / {cond} / seed {seed} "
                      f"(elapsed {(time.time()-t_start)/60:.1f} min) ===")
                if task == "truss":
                    ck = train_truss(cond, seed, args.truss_iters, args.num_envs,
                                     args.device, args.skip_existing)
                else:
                    ck = train_astrobee(cond, seed, args.astrobee_iters, args.num_envs,
                                        args.device, args.skip_existing)
                evaluate(task, cond, seed, ck, args.eval_envs, args.eval_seed,
                         args.device, args.skip_existing)

    print(f"\nALL DONE in {(time.time()-t_start)/60:.1f} min. Results in {RESULTS}")


if __name__ == "__main__":
    main()
