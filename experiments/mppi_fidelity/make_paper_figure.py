"""Generate the paper TikZ figure for the MPPI sim-fidelity study.

Reads the per-seed JSONs produced by
``examples/mppi/mppi_fidelity_capture.py --out experiments/mppi_fidelity/out/seedN.json``
and emits a PGFPlots figure: nadir-pointing error vs time since capture for an
MPPI controller whose internal rollouts use different dynamics models, all
executed on the full-fidelity mjorbit plant. Mean over seeds with a +/- 1 sigma
band.

    python experiments/mppi_fidelity/make_paper_figure.py
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).with_name("out")

# (json backend key, legend label, color style, line style). The paper figure
# contrasts the full orbit-coupled model against the zero-gravity approximation;
# the seed JSONs also contain a "zerog_fair" ablation that is not plotted here.
CURVES = (
    ("mjorbit", "\\mjorbit{} (orbit-coupled)", "fidMjorbit", "solid"),
    ("zerog", "zero-g (gravity disabled)", "fidZerog", "solid"),
)


def coordinates_block(x: np.ndarray, y: np.ndarray) -> str:
    rows = [f"({xv:.5g},{yv:.4g})" for xv, yv in zip(x, y, strict=True)]
    return "\n".join(f"        {r}" for r in rows)


def band_block(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> str:
    # lower edge left->right, then upper edge right->left; \closedcycle closes it.
    fwd = [f"({xv:.5g},{yv:.4g})" for xv, yv in zip(x, lo, strict=True)]
    rev = [f"({xv:.5g},{yv:.4g})" for xv, yv in zip(x[::-1], hi[::-1], strict=True)]
    return "\n".join(f"        {r}" for r in fwd + rev)


def load(out_dir: Path, max_points: int):
    """Load per-seed (time-in-hours, nadir-error) traces and average them on a
    common grid (capture times differ slightly across seeds, so we interpolate)."""
    files = sorted(glob.glob(str(out_dir / "seed*.json")))
    if not files:
        raise FileNotFoundError(f"no seed*.json in {out_dir}")
    n_seeds = len(files)
    raw: dict[str, list] = {key: [] for key, *_ in CURVES}
    for f in files:
        r = json.loads(Path(f).read_text())["results"]
        for key, *_ in CURVES:
            if key in r:
                t = np.asarray(r[key]["t_s"], dtype=float) / 3600.0  # seconds -> hours
                e = np.asarray(r[key]["err_deg"], dtype=float)
                raw[key].append((t, e))
    t_max = min(t[-1] for traces in raw.values() if traces for t, _ in traces)
    grid = np.linspace(0.0, t_max, max_points)
    series: dict[str, dict[str, np.ndarray]] = {}
    for key, *_ in CURVES:
        if not raw[key]:
            continue
        stack = np.array([np.interp(grid, t, e) for t, e in raw[key]])
        mean, std = stack.mean(0), stack.std(0)
        series[key] = {"t": grid, "mean": mean,
                       "lo": np.clip(mean - std, 0.0, None), "hi": mean + std}
    return series, n_seeds, float(t_max)


def tikz_document(series, n_seeds: int, t_max: float) -> str:
    lines = [
        "\\begin{tikzpicture}",
        "\\pgfplotsset{",
        "  fidMjorbit/.style={color={rgb,255:red,0;green,58;blue,125}},",   # plotDarkBlue
        "  fidZerog/.style={color={rgb,255:red,216;green,48;blue,52}},",     # plotRed
        "  fidFair/.style={color={rgb,255:red,255;green,157;blue,58}},",     # plotOrange
        "}",
        "\\begin{axis}[",
        "  width=\\columnwidth, height=0.62\\columnwidth,",
        f"  xmin=0, xmax={t_max:.2f}, ymin=0, ymax=95,",
        "  xlabel={time [hours]},",
        "  ylabel={nadir-pointing error [deg]},",
        "  grid=both,",
        "  minor grid style={draw=gray!12}, major grid style={draw=gray!30},",
        "  tick align=outside, tick pos=left,",
        "  label style={font=\\footnotesize}, tick label style={font=\\scriptsize},",
        "  legend pos=north east, legend cell align=left,",
        "  legend style={font=\\scriptsize, draw=none, fill=white, fill opacity=0.7,"
        " text opacity=1},",
        "]",
    ]
    # bands first (behind), then mean lines (with legend)
    for key, _, color, _ in CURVES:
        s = series.get(key)
        if s is None:
            continue
        lines += [
            f"  \\addplot[{color}, fill, draw=none, fill opacity=0.35, forget plot] coordinates {{",
            band_block(s["t"], s["lo"], s["hi"]),
            "  } \\closedcycle;",
        ]
    for key, label, color, style in CURVES:
        s = series.get(key)
        if s is None:
            continue
        lines += [
            f"  \\addplot[{color}, {style}, line width=2.2pt, mark=none] coordinates {{",
            coordinates_block(s["t"], s["mean"]),
            "  };",
            f"  \\addlegendentry{{{label}}}",
        ]
    lines += ["\\end{axis}", "\\end{tikzpicture}", ""]
    return "\n".join(lines)


def standalone(tikz_name: str) -> str:
    return "\n".join([
        "\\documentclass[tikz,border=2pt]{standalone}",
        "\\usepackage{pgfplots}", "\\pgfplotsset{compat=1.18}",
        "\\newcommand{\\mjorbit}{mjorbit}",
        "\\setlength{\\columnwidth}{3.45in}",
        "\\begin{document}", f"\\input{{{tikz_name}}}", "\\end{document}", "",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--max-points", type=int, default=140)
    args = ap.parse_args()

    series, n_seeds, t_max = load(args.out_dir, args.max_points)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tikz_path = args.out_dir / "mppi_fidelity.tikz"
    tikz_path.write_text(tikz_document(series, n_seeds, t_max), encoding="utf-8")
    (args.out_dir / "mppi_fidelity_standalone.tex").write_text(
        standalone(tikz_path.name), encoding="utf-8")
    print(f"wrote {tikz_path} ({n_seeds} seeds)")
    for key, label, *_ in CURVES:
        s = series.get(key)
        if s is not None:
            print(f"  {label:32s} final mean {s['mean'][-1]:.1f} deg")


if __name__ == "__main__":
    main()
