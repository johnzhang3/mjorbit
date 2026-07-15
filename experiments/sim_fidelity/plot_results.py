"""Aggregate sim-fidelity eval JSONs into a summary table + figures.

Reads ``results/<task>/<condition>_seed<k>.json`` (written by eval_fidelity.py),
aggregates across seeds, and emits:
  - results/SUMMARY.md            markdown tables
  - results/figures/*.png         comparison plots

Run under the `report` env (matplotlib lives there; this script needs no GPU):
    pixi run -e report python experiments/sim_fidelity/plot_results.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGS = RESULTS / "figures"

# Display order + colors per condition.
TRUSS_ORDER = ["mjorbit", "mjwarp_fair", "mjwarp_naive"]
ASTROBEE_ORDER = ["mjorbit", "mjwarp"]
COLORS = {
    "mjorbit": "#1b7837",       # green  (home sim)
    "mjwarp_fair": "#762a83",   # purple (bare dynamics, moving target)
    "mjwarp_naive": "#b35806",  # orange (bare dynamics, frozen target)
    "mjwarp": "#762a83",
}
LABELS = {
    "mjorbit": "mjorbit-warp\n(full orbital)",
    "mjwarp_fair": "mjwarp fair\n(bare dyn, moving nadir)",
    "mjwarp_naive": "mjwarp naive\n(bare dyn, frozen nadir)",
    "mjwarp": "mjwarp\n(bare dynamics)",
}
TRUSS_THRESHOLDS = [1.0, 2.0, 3.0, 5.0, 10.0]


def load(task: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    d = RESULTS / task
    if not d.exists():
        return out
    for f in sorted(d.glob("*_seed*.json")):
        rec = json.loads(f.read_text())
        cond = rec["meta"]["tag"].split("/")[0]
        out[cond].append(rec)
    return out


def mstd(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    return float(np.nanmean(a)), float(np.nanstd(a))


def fmt(m: float, s: float, pct: bool = False) -> str:
    if pct:
        return f"{100*m:5.1f} ± {100*s:4.1f}%"
    return f"{m:6.3f} ± {s:5.3f}"


# ----------------------------------------------------------------------------
# Truss
# ----------------------------------------------------------------------------
def summarize_truss(data: dict[str, list[dict]], md: list[str]) -> None:
    if not data:
        return
    md += ["## Truss (nadir-pointing)\n",
           "Success = hug retained for the whole episode **and** final-quarter "
           "mean nadir error below the threshold. Eval in full mjorbit_warp, "
           "identical worlds across all policies.\n",
           "| condition | seeds | alive | lost | final err (deg) | succ@3° | succ@5° | succ@10° |",
           "|---|---|---|---|---|---|---|---|"]
    for cond in TRUSS_ORDER:
        recs = data.get(cond)
        if not recs:
            continue
        s = [r["summary"] for r in recs]
        row = [
            LABELS[cond].replace("\n", " "),
            str(len(recs)),
            fmt(*mstd([x["alive_frac"] for x in s]), pct=True),
            fmt(*mstd([x["lost_frac"] for x in s]), pct=True),
            fmt(*mstd([x["final_align_mean_deg"] for x in s])),
            fmt(*mstd([x["success@3.0deg"] for x in s]), pct=True),
            fmt(*mstd([x["success@5.0deg"] for x in s]), pct=True),
            fmt(*mstd([x["success@10.0deg"] for x in s]), pct=True),
        ]
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    _truss_success_curve(data)
    _truss_align_box(data)


def _truss_success_curve(data: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond in TRUSS_ORDER:
        recs = data.get(cond)
        if not recs:
            continue
        curves = []
        for r in recs:
            fa = np.asarray(r["per_world"]["final_align_deg"], dtype=float)
            lost = np.asarray(r["per_world"]["ever_lost"], dtype=bool)
            crashed = np.asarray(r["per_world"]["ever_crashed"], dtype=bool)
            alive = ~lost & ~crashed
            thr = np.linspace(0.5, 20, 80)
            curves.append([np.mean(alive & (fa <= t)) for t in thr])
        thr = np.linspace(0.5, 20, 80)
        cur = np.asarray(curves)
        m, sd = cur.mean(0), cur.std(0)
        ax.plot(thr, 100 * m, color=COLORS[cond], label=LABELS[cond].replace("\n", " "))
        ax.fill_between(thr, 100 * (m - sd), 100 * (m + sd), color=COLORS[cond], alpha=0.15)
    ax.set_xlabel("nadir-pointing error threshold (deg)")
    ax.set_ylabel("success rate (%)")
    ax.set_title("Truss: success rate vs pointing-error threshold\n(evaluated in mjorbit-warp)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "truss_success_curve.png", dpi=140)
    plt.close(fig)


def _truss_align_box(data: dict[str, list[dict]]) -> None:
    conds = [c for c in TRUSS_ORDER if data.get(c)]
    if not conds:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    box_data, labels, colors = [], [], []
    for cond in conds:
        pooled = []
        for r in data[cond]:
            fa = np.asarray(r["per_world"]["final_align_deg"], dtype=float)
            lost = np.asarray(r["per_world"]["ever_lost"], dtype=bool)
            crashed = np.asarray(r["per_world"]["ever_crashed"], dtype=bool)
            pooled.append(fa[~lost & ~crashed])
        box_data.append(np.concatenate(pooled) if pooled else np.array([np.nan]))
        labels.append(LABELS[cond])
        colors.append(COLORS[cond])
    bp = ax.boxplot(box_data, tick_labels=labels, showfliers=False, patch_artist=True,
                    medianprops=dict(color="black"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.set_ylabel("final-quarter nadir error per world (deg)")
    ax.set_title("Truss: per-world pointing error (alive worlds, evaluated in mjorbit-warp)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "truss_align_box.png", dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Astrobee
# ----------------------------------------------------------------------------
def summarize_astrobee(data: dict[str, list[dict]], md: list[str]) -> None:
    if not data:
        return
    md += ["## Astrobee (detumble + grasp) — predicted-null control\n",
           "Success = grasped+held for the final quarter, no fail/crash. No "
           "orbital-frame reference in this task, so mjwarp and mjorbit should "
           "be close.\n",
           "| condition | seeds | success | grasped frac | final spin (rad/s) | failed |",
           "|---|---|---|---|---|---|"]
    for cond in ASTROBEE_ORDER:
        recs = data.get(cond)
        if not recs:
            continue
        s = [r["summary"] for r in recs]
        row = [
            LABELS[cond].replace("\n", " "),
            str(len(recs)),
            fmt(*mstd([x["success_rate"] for x in s]), pct=True),
            fmt(*mstd([x["grasped_frac_mean"] for x in s]), pct=True),
            fmt(*mstd([x["final_spin_mean"] for x in s])),
            fmt(*mstd([x["failed_frac"] for x in s]), pct=True),
        ]
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    conds = [c for c in ASTROBEE_ORDER if data.get(c)]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    xs = np.arange(len(conds))
    means = [mstd([r["summary"]["success_rate"] for r in data[c]])[0] for c in conds]
    stds = [mstd([r["summary"]["success_rate"] for r in data[c]])[1] for c in conds]
    ax.bar(xs, [100 * m for m in means], yerr=[100 * s for s in stds],
           color=[COLORS[c] for c in conds], alpha=0.75, capsize=5)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[c] for c in conds], fontsize=8)
    ax.set_ylabel("grasp+hold success rate (%)")
    ax.set_title("Astrobee: success rate by training backend\n(evaluated in mjorbit-warp)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "astrobee_success_bars.png", dpi=140)
    plt.close(fig)


def main() -> None:
    md = ["# Sim-fidelity study — results\n",
          "_Training backend on the x-axis; every policy evaluated in the full "
          "mjorbit_warp orbital environment._\n"]
    summarize_truss(load("truss"), md)
    summarize_astrobee(load("astrobee"), md)
    (RESULTS).mkdir(parents=True, exist_ok=True)
    (RESULTS / "SUMMARY.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {RESULTS/'SUMMARY.md'} and figures in {FIGS}")


if __name__ == "__main__":
    main()
