"""Plot SGS vs uniform training curves for the capture-and-reorient task.

Reads ``eval.csv`` (held-out success per checkpoint, from ``capture_eval.py``)
and ``train_metrics.csv`` (per-iteration training log) from each run:

    pixi run -e report python examples/ppo/capture_plot.py \
        --runs examples/ppo/logs/cap-sgs examples/ppo/logs/cap-uni \
        --labels SGS uniform --out examples/ppo/logs/capture_curves.png

Top row: held-out success rate vs environment steps, overall and per reset
family. Bottom row: training-time quantities -- the mean rolling success
estimate over the fixed training configuration set (~ success under uniform
sampling of that set) and the share of finished episodes from the approach
family (where the sampler spends experience).

``--focus-run <sgs run> --focus-out <png>`` additionally plots where SGS puts
its sampling mass over training (cf. the paper's Fig. 7), from the
``sampler_*.npz`` snapshots: relative sampling weight (probability x N) vs the
initial pointing error (held, pregrasp) or the client's start distance
(approach), one line per snapshot.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import offset_copy

# Reference categorical palette, slots 1-2 (blue, orange); recessive ink + grid.
_SERIES = ("#2a78d6", "#eb6834")
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_GRID = "#e4e3df"
# Sequential blue ramp (light -> dark = early -> late snapshot).
_RAMP = ("#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def _read(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open()))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else {}


def _smooth(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or y.size < k:
        return y
    y = np.where(np.isnan(y), 0.0, y)
    return np.convolve(y, np.ones(k) / k, mode="valid")


def _style(ax, title: str, ylabel: str | None) -> None:
    ax.set_facecolor(_SURFACE)
    ax.set_title(title, color=_INK, fontsize=11, loc="left")
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK_2)
    ax.tick_params(colors=_INK_2, labelsize=9)
    ax.set_xlabel("environment steps (millions)", color=_INK_2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=_INK_2, fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smooth", type=int, default=25, help="moving-average window (iterations)")
    parser.add_argument("--focus-run", default=None, help="SGS run dir with sampler_*.npz")
    parser.add_argument("--focus-out", default=None)
    args = parser.parse_args()
    if args.focus_run and args.focus_out:
        plot_focus(Path(args.focus_run), Path(args.focus_out))
    assert len(args.runs) == len(args.labels) <= len(_SERIES)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), facecolor=_SURFACE)
    eval_panels = (
        ("success", "Held-out success: all resets"),
        ("success_held", "held (already clamped)"),
        ("success_pregrasp", "pregrasp (bar in open jaws)"),
        ("success_approach", "approach (fly in + capture)"),
    )
    ends: dict[int, tuple] = {}
    for run, label, color in zip(args.runs, args.labels, _SERIES):
        ev = _read(Path(run) / "eval.csv")
        tr = _read(Path(run) / "train_metrics.csv")
        if ev:
            order = np.argsort(ev["env_steps"])
            x = ev["env_steps"][order] / 1e6
            for ax, (key, _) in zip(axes[0], eval_panels):
                y = ev[key][order] * 100
                ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4, label=label)
                ends.setdefault(id(ax), (ax, []))[1].append((y[-1], x[-1], label))
        if tr:
            x = _smooth(tr["env_steps"], args.smooth) / 1e6
            for ax, key in zip(axes[1, :2], ("sgs_p_mean_visited", "frac_approach")):
                ax.plot(
                    x, _smooth(tr[key], args.smooth) * 100, color=color, linewidth=2, label=label
                )
            for ax, key in zip(axes[1, 2:], ("success_rate", "fail_rate")):
                ax.plot(
                    x, _smooth(tr[key], args.smooth) * 100, color=color, linewidth=2, label=label
                )

    for ax, (_, title) in zip(axes[0], eval_panels):
        _style(ax, title, "success rate (%)" if ax is axes[0, 0] else None)
        ax.set_ylim(-3, 103)
    # End-of-curve value labels: under the curve end near the 100% ceiling, and
    # stacked downward so converged curves stay legible.
    for ax, pts in ends.values():
        placed: list[float] = []
        for y, x, label in sorted(pts, reverse=True):
            yy = y - 7.0 if y > 90.0 else y
            if placed:
                yy = min(yy, placed[-1] - 7.0)
            placed.append(yy)
            ax.text(
                x,
                yy,
                f"{label} {y:.0f}%",
                color=_INK,
                fontsize=8,
                va="center",
                transform=offset_copy(ax.transData, fig=fig, x=6, y=0, units="points"),
            )
            ax.margins(x=0.18)
    titles = (
        "Rolling success over training configs",
        "Share of episodes from approach family",
        "Training-episode success (as sampled)",
        "Training-episode failure (as sampled)",
    )
    for ax, title in zip(axes[1], titles):
        _style(ax, title, "%" if ax is axes[1, 0] else None)
    axes[0, 0].legend(frameon=False, fontsize=9, labelcolor=_INK)
    axes[1, 0].legend(frameon=False, fontsize=9, labelcolor=_INK)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=_SURFACE)
    fig.savefig(Path(args.out).with_suffix(".pdf"), facecolor=_SURFACE)
    print(f"wrote {args.out}")


def plot_focus(run: Path, out: Path, n_lines: int = len(_RAMP)) -> None:
    snaps = sorted(run.glob("sampler_*.npz"))
    if len(snaps) > n_lines:
        snaps = [snaps[i] for i in np.linspace(0, len(snaps) - 1, n_lines).round().astype(int)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), facecolor=_SURFACE)
    panels = (
        (0, "held: weight vs initial pointing error", "point_err0", "deg"),
        (1, "pregrasp: weight vs initial pointing error", "point_err0", "deg"),
        (2, "approach: weight vs client start distance", "dist", "m"),
    )
    for snap, color in zip(snaps, _RAMP[-len(snaps) :]):
        z = np.load(snap)
        it = int(snap.stem.split("_")[1])
        weight = z["probs"] * z["probs"].size  # 1 = uniform
        for ax, (fam, _, key, _) in zip(axes, panels):
            sel = z["cfg_family"] == fam
            if key == "point_err0":
                x, edges = np.degrees(z["cfg_point_err0"][sel]), np.linspace(0, 180, 13)
            else:
                x = np.linalg.norm(z["cfg_client_dp"][sel], axis=1)
                edges = np.linspace(x.min(), x.max(), 8)
            bins = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
            mean = np.array([weight[sel][bins == b].mean() for b in range(len(edges) - 1)])
            centers = 0.5 * (edges[1:] + edges[:-1])
            ax.plot(
                centers,
                mean,
                color=color,
                linewidth=2,
                marker="o",
                markersize=4,
                label=f"iter {it}",
            )
    for ax, (_, title, _, unit) in zip(axes, panels):
        _style(ax, title, "sampling weight (1 = uniform)" if ax is axes[0] else None)
        ax.set_xlabel(unit, color=_INK_2, fontsize=9)
        ax.axhline(1.0, color=_INK_2, linewidth=0.8, linestyle="--")
    axes[-1].legend(
        frameon=False, fontsize=8, labelcolor=_INK, loc="upper left", bbox_to_anchor=(1.02, 1.0)
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=_SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
