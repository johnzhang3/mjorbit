"""Parameter sweep for the MPPI docking demo.

Runs ``main_mppi.py`` headless across a grid of planner settings and scores each
run by whether the attitude actually *converges* rather than limit-cycling.

The default aligned dock (no slew) is easy — the normalised cost holds attitude
and seats the ports in ~10 s. The hard case is a commanded slew: with only
~0.005 rad/s^2 of reaction-wheel authority a braking slew takes far longer than
a short MPPI horizon, so a short-horizon planner overshoots the target attitude
and limit-cycles instead of settling. This sweep commands a slew (``--slew``,
passed through as ``--target-angle``) and searches the planner knobs that govern
whether the slew can brake in time: ``horizon``, ``num-nodes``, and the
angular-rate damping weights.

Scoring (lower is better), all read from the per-second ``att=`` / ``sep=`` log:
  * ``att_max_tail`` — worst attitude error over the final ``--tail`` seconds.
    A limit cycle keeps a large max even if it momentarily dips near zero, so
    this is the primary convergence metric.
  * ``att_mean_tail`` / ``sep_mean_tail`` — tail means, reported for context.

Usage:
    pixi run python examples/docking/sweep.py
    pixi run python examples/docking/sweep.py --duration 160 --tail 40
    pixi run python examples/docking/sweep.py --quick   # tiny grid, fast
"""

from __future__ import annotations

import argparse
import itertools
import re
import subprocess
import sys
from pathlib import Path

MAIN = Path(__file__).with_name("main_mppi.py")

# Per-second log line, e.g.
#   t=  28.0 s   sep= 2.606 m   att= 40.28 deg   v=0.0892 m/s   w= 0.0121 rad/s ...
LOG_RE = re.compile(
    r"t=\s*([\d.]+)\s*s\s+sep=\s*([\d.]+)\s*m\s+att=\s*([\d.]+)\s*deg"
)


def run_one(params: dict, duration: float, tail: float, slew: float) -> dict:
    """Run main_mppi.py with ``params`` and return tail convergence metrics."""
    cmd = [sys.executable, str(MAIN), "--duration", str(duration)]
    if slew != 0.0:
        cmd += ["--target-angle", str(slew)]
    for k, v in params.items():
        cmd += [f"--{k}", str(v)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"ok": False, "err": proc.stderr.strip()[-400:]}

    ts, seps, atts = [], [], []
    for m in LOG_RE.finditer(proc.stdout):
        ts.append(float(m.group(1)))
        seps.append(float(m.group(2)))
        atts.append(float(m.group(3)))
    if not ts:
        return {"ok": False, "err": "no log lines parsed"}

    t_end = ts[-1]
    tail_idx = [i for i, t in enumerate(ts) if t >= t_end - tail]
    att_tail = [atts[i] for i in tail_idx]
    sep_tail = [seps[i] for i in tail_idx]
    return {
        "ok": True,
        "att_max_tail": max(att_tail),
        "att_mean_tail": sum(att_tail) / len(att_tail),
        "att_min_tail": min(att_tail),
        "sep_mean_tail": sum(sep_tail) / len(sep_tail),
        "att_final": atts[-1],
        "sep_final": seps[-1],
    }


def main() -> None:
    ps = argparse.ArgumentParser(description=__doc__)
    ps.add_argument("--duration", type=float, default=160.0)
    ps.add_argument("--tail", type=float, default=40.0,
                    help="final window (s) used to judge convergence")
    ps.add_argument("--slew", type=float, default=-45.0,
                    help="commanded --target-angle (deg about world +z) the sweep "
                    "tries to make converge. The Soyuz starts at ~-90 deg about z, "
                    "so the default is a ~45 deg slew; 0 reproduces the easy "
                    "aligned dock.")
    ps.add_argument("--quick", action="store_true", help="tiny grid for a fast check")
    args = ps.parse_args()

    if args.quick:
        grid = {
            "horizon": [6.0, 12.0],
            "w-term-rate": [40, 100],
        }
    else:
        # Whether a low-authority slew can brake in time is governed by three
        # coupled knobs, so they are swept together rather than one at a time
        # (weights are on the normalised, dimensionless scale of main_mppi.py):
        #   * horizon — must be long enough for the planner to foresee the
        #     braking burn (a y/z slew brakes at only ~0.005 rad/s^2).
        #   * num-nodes — a longer horizon needs more spline knots, or the
        #     control resolution gets too coarse to brake cleanly (raising
        #     horizon alone actually made attitude worse).
        #   * w-term-rate / w-rate — bleed off angular momentum before the
        #     target so the slew arrives stopped, not passing through.
        grid = {
            "horizon": [8.0, 12.0],
            "num-nodes": [6, 10],
            "w-term-rate": [40, 100],
            "w-rate": [6, 18],
        }

    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f"sweeping {len(combos)} combos  (duration={args.duration:g}s, tail={args.tail:g}s)")
    print("baseline knobs not listed use main_mppi.py defaults\n")

    results = []
    for i, vals in enumerate(combos, 1):
        params = dict(zip(keys, vals))
        label = "  ".join(f"{k}={v}" for k, v in params.items())
        print(f"[{i:2d}/{len(combos)}] {label} ... ", end="", flush=True)
        r = run_one(params, args.duration, args.tail, args.slew)
        if not r["ok"]:
            print(f"FAILED: {r['err']}")
            continue
        print(
            f"att_max={r['att_max_tail']:6.2f}  att_mean={r['att_mean_tail']:6.2f}  "
            f"sep_mean={r['sep_mean_tail']:5.2f}"
        )
        results.append((params, r))

    if not results:
        print("\nno successful runs")
        return

    results.sort(key=lambda pr: pr[1]["att_max_tail"])
    print("\n" + "=" * 72)
    print("Ranked by worst-case attitude error over the final window (converged = low):")
    print("=" * 72)
    for params, r in results:
        label = "  ".join(f"{k}={v}" for k, v in params.items())
        flag = "  <-- converged" if r["att_max_tail"] < 5.0 else ""
        print(
            f"att_max={r['att_max_tail']:6.2f}  att_mean={r['att_mean_tail']:6.2f}  "
            f"sep_mean={r['sep_mean_tail']:5.2f}   {label}{flag}"
        )


if __name__ == "__main__":
    main()
