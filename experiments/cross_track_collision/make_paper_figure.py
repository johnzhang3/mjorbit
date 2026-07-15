"""Generate the paper TikZ figure for the zero-g vs. orbit-coupled comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zero_g_compare import ComparisonResult, run_both  # noqa: E402

OUT_DIR = Path(__file__).with_name("out")

CURVES = (
    ("a_orbit", "body A, mujoco\\_orbit", "bodyA", "solid"),
    ("b_orbit", "body B, mujoco\\_orbit", "bodyB", "solid"),
    ("a_naive", "body A, naive zero-g", "bodyA", "densely dashed"),
    ("b_naive", "body B, naive zero-g", "bodyB", "densely dashed"),
)


def _select_curve(result: ComparisonResult, name: str) -> np.ndarray:
    return {
        "a_orbit": result.cross_track_a_orbit,
        "b_orbit": result.cross_track_b_orbit,
        "a_naive": result.cross_track_a_naive,
        "b_naive": result.cross_track_b_naive,
    }[name]


def _decimate(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform-stride decimation, preserving endpoints."""
    if x.size <= max_points:
        return x, y
    idx = np.linspace(0, x.size - 1, max_points).astype(np.int64)
    idx = np.unique(idx)
    return x[idx], y[idx]


def _trim_to_band(
    x: np.ndarray, y: np.ndarray, y_band: float
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the leading prefix where |y| <= y_band, plus the first sample beyond."""
    inside = np.abs(y) <= y_band
    if inside.all():
        return x, y
    first_outside = int(np.argmin(inside))
    end = min(first_outside + 1, x.size)
    return x[: end + 1], y[: end + 1]


def _build_panel_curves(
    result: ComparisonResult,
    t_min: float,
    t_max: float,
    y_band: float,
    max_points: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    mask = (result.time >= t_min) & (result.time <= t_max)
    time = result.time[mask]
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, _, _, _ in CURVES:
        y = _select_curve(result, name)[mask]
        x_trim, y_trim = _trim_to_band(time, y, y_band)
        curves[name] = _decimate(x_trim, y_trim, max_points)
    return curves


def _coordinates_block(x: np.ndarray, y: np.ndarray) -> str:
    rows = [f"({xi:.5g},{yi:.4g})" for xi, yi in zip(x, y, strict=True)]
    return "\n".join(f"        {row}" for row in rows)


def _axis_block(
    title: str,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    xlabel: str,
    add_legend: bool,
    extras: str = "",
) -> str:
    options = [
        f"title={{{title}}}",
        f"xmin={xmin:g}, xmax={xmax:g}",
        f"ymin={ymin:g}, ymax={ymax:g}",
        f"xlabel={{{xlabel}}}",
    ]
    if add_legend:
        options.append("legend to name=zeroGLegend")

    lines = [f"  \\nextgroupplot[{', '.join(options)}]"]
    for name, legend_label, color_name, line_style in CURVES:
        x, y = curves[name]
        legend = f"\\addlegendentry{{{legend_label}}}" if add_legend else ""
        plot_header = (
            f"    \\addplot[mark=none, {color_name}, {line_style}, "
            "line width=1.0pt] coordinates {"
        )
        lines.extend(
            [
                plot_header,
                _coordinates_block(x, y),
                "    };",
                f"    {legend}",
            ]
        )
    if extras:
        lines.append(extras)
    return "\n".join(lines)


def _tikz_document(
    short_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    long_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    short_xmax: float,
    long_xmax: float,
    short_yband: float,
    long_yband: float,
    half_orbit: float,
    label_y_offset: float,
) -> str:
    half_orbit_marker = (
        "    \\draw[gray!70, dashed, line width=0.5pt] "
        f"(axis cs:{half_orbit:.6g},{-long_yband:.6g}) -- "
        f"(axis cs:{half_orbit:.6g},{long_yband:.6g});"
    )
    half_orbit_label = (
        "    \\node[gray!85, font=\\scriptsize, anchor=south west] "
        f"at (axis cs:{half_orbit:.6g},{label_y_offset:.6g}) {{$T/2$}};"
    )
    return "\n".join(
        [
            "\\begin{tikzpicture}",
            "\\pgfplotsset{",
            "  zeroGAxis/.style={",
            "    width=\\columnwidth,",
            "    height=0.55\\columnwidth,",
            "    grid=both,",
            "    minor grid style={draw=gray!15},",
            "    major grid style={draw=gray!30},",
            "    tick align=outside,",
            "    tick pos=left,",
            "    xlabel near ticks,",
            "    ylabel={cross-track position [m]},",
            "    title style={font=\\footnotesize},",
            "    label style={font=\\footnotesize},",
            "    tick label style={font=\\scriptsize},",
            "    legend columns=2,",
            "    legend cell align=left,",
            "    legend style={",
            "      font=\\scriptsize,",
            "      /tikz/every even column/.append style={column sep=0.45cm},",
            "      draw=none,",
            "    },",
            "  },",
            "}",
            "\\pgfplotsset{",
            "  bodyA/.style={color={rgb,255:red,0;green,58;blue,125}},",
            "  bodyB/.style={color={rgb,255:red,216;green,48;blue,52}},",
            "}",
            "\\begin{groupplot}[",
            "  zeroGAxis,",
            "  group style={group size=1 by 2, vertical sep=1.55cm},",
            "]",
            _axis_block(
                "Short horizon: methods agree across the contact",
                short_curves,
                xmin=0.0,
                xmax=short_xmax,
                ymin=-short_yband,
                ymax=short_yband,
                xlabel="time [s]",
                add_legend=True,
            ),
            _axis_block(
                "Full chief orbit: naive zero-g drifts; mujoco\\_orbit returns",
                long_curves,
                xmin=0.0,
                xmax=long_xmax,
                ymin=-long_yband,
                ymax=long_yband,
                xlabel="time [s]",
                add_legend=False,
                extras="\n".join([half_orbit_marker, half_orbit_label]),
            ),
            "\\end{groupplot}",
            "\\node[anchor=north, yshift=-1.05cm] at (group c1r2.south)",
            "  {\\pgfplotslegendfromname{zeroGLegend}};",
            "\\end{tikzpicture}",
            "",
        ]
    )


def _standalone_document(tikz_filename: str) -> str:
    return "\n".join(
        [
            "\\documentclass[tikz,border=2pt]{standalone}",
            "\\usepackage{pgfplots}",
            "\\pgfplotsset{compat=1.18}",
            "\\usepgfplotslibrary{groupplots}",
            "\\setlength{\\columnwidth}{3.45in}",
            "\\begin{document}",
            f"\\input{{{tikz_filename}}}",
            "\\end{document}",
            "",
        ]
    )


def _write_outputs(out_dir: Path, tikz: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tikz_path = out_dir / "zero_g_comparison.tikz"
    standalone_path = out_dir / "zero_g_comparison_standalone.tex"
    tikz_path.write_text(tikz, encoding="utf-8")
    standalone_path.write_text(_standalone_document(tikz_path.name), encoding="utf-8")
    return tikz_path, standalone_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--short-horizon-s",
        type=float,
        default=3.0,
        help="x-axis upper bound for the top (short-horizon) panel.",
    )
    parser.add_argument(
        "--short-yband",
        type=float,
        default=8.0,
        help="symmetric y-axis half-extent (m) for the top panel.",
    )
    parser.add_argument(
        "--long-yband",
        type=float,
        default=2600.0,
        help="symmetric y-axis half-extent (m) for the bottom panel.",
    )
    parser.add_argument("--short-max-points", type=int, default=80)
    parser.add_argument("--long-max-points", type=int, default=150)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_both()

    short_curves = _build_panel_curves(
        result,
        t_min=0.0,
        t_max=args.short_horizon_s,
        y_band=max(args.short_yband, args.long_yband) * 1.5,
        max_points=args.short_max_points,
    )
    long_curves = _build_panel_curves(
        result,
        t_min=0.0,
        t_max=result.orbit_period,
        y_band=args.long_yband * 1.05,
        max_points=args.long_max_points,
    )

    tikz = _tikz_document(
        short_curves,
        long_curves,
        short_xmax=args.short_horizon_s,
        long_xmax=result.orbit_period,
        short_yband=args.short_yband,
        long_yband=args.long_yband,
        half_orbit=0.5 * result.orbit_period,
        label_y_offset=-0.92 * args.long_yband,
    )
    tikz_path, standalone_path = _write_outputs(args.out_dir, tikz)

    print("Zero-g vs. orbit-coupled paper figure")
    print(f"  TikZ           : {tikz_path}")
    print(f"  Standalone TeX : {standalone_path}")
    print(f"  orbit period T : {result.orbit_period:.2f} s")
    print(f"  short panel    : {sum(c[0].size for c in short_curves.values())} samples")
    print(f"  long  panel    : {sum(c[0].size for c in long_curves.values())} samples")


if __name__ == "__main__":
    main()
