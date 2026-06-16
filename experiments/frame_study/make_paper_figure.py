"""Generate the paper TikZ figure for the ECI/local-frame precision study."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).with_name("out")

CURVES = (
    ("ECI + implicit", "ECI, implicit", "frameEci", "solid"),
    ("local chief + implicit", "chief-inertial, implicit", "frameChief", "solid"),
    ("LVLH + implicit", "LVLH, implicit", "frameLvlh", "solid"),
    ("ECI + RK4", "ECI, RK4\\hphantom{implt}", "frameEci", "densely dashed"),
    ("local chief + RK4", "chief-inertial, RK4\\hphantom{implt}", "frameChief", "densely dashed"),
    ("LVLH + RK4", "LVLH, RK4\\hphantom{implt}", "frameLvlh", "densely dashed"),
)


def format_float_for_filename(value: float) -> str:
    """Format a float the same way as run.py output stems."""
    text = f"{value:g}".replace("-", "m").replace("+", "")
    return text.replace(".", "p")


def orbit_suffix(n_orbits: float) -> str:
    """Return the orbit suffix used by run.py."""
    suffix = "1_orbit" if np.isclose(n_orbits, 1.0) else f"{n_orbits:g}_orbits"
    return suffix.replace(".", "p")


def velocity_suffix(rel_vel_bias_lvlh: np.ndarray) -> str:
    """Return the optional velocity suffix used by run.py."""
    if np.allclose(rel_vel_bias_lvlh, 0.0):
        return ""
    values = "_".join(format_float_for_filename(float(value)) for value in rel_vel_bias_lvlh)
    return f"_dv_{values}"


def sample_path(
    out_dir: Path,
    scenario: str,
    n_orbits: float,
    dt: float,
    precision: str,
    rel_vel_bias_lvlh: np.ndarray,
    length_unit_m: float,
) -> Path:
    """Build the expected .npz path for one study run."""
    stem = (
        f"{scenario}_{orbit_suffix(n_orbits)}_dt_{format_float_for_filename(dt)}s"
        f"_{precision}{velocity_suffix(rel_vel_bias_lvlh)}"
        f"_unit_{format_float_for_filename(length_unit_m)}m_eci_vs_local_integrators"
    )
    return out_dir / f"{stem}.npz"


def downsample_curve(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample a positive finite curve while preserving endpoints and maximum."""
    valid = np.flatnonzero(np.isfinite(x) & np.isfinite(y) & (y > 0.0))
    if len(valid) == 0:
        raise ValueError("curve has no positive finite samples")

    x_valid = x[valid]
    y_valid = y[valid]
    if len(valid) <= max_points:
        keep = np.arange(len(valid))
    else:
        keep = np.unique(
            np.concatenate(
                [
                    np.linspace(0, len(valid) - 1, max_points - 1, dtype=np.int64),
                    np.asarray([int(np.argmax(y_valid))], dtype=np.int64),
                ]
            )
        )
    return x_valid[keep], y_valid[keep]


def coordinates_block(x: np.ndarray, y: np.ndarray) -> str:
    """Format coordinates for an inline PGFPlots coordinate table."""
    rows = [f"({x_value:.5g},{y_value:.4e})" for x_value, y_value in zip(x, y, strict=True)]
    return "\n".join(f"        {row}" for row in rows)


def load_curves(
    path: Path,
    n_orbits: float,
    max_points: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, float]]:
    """Load and downsample the four paper curves from a study sample file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing study output: {path}\n"
            "Run experiments/frame_study/run.py with the matching precision and dt first."
        )

    data = np.load(path)
    labels = list(data["labels"])
    times = np.asarray(data["times_s"], dtype=np.float64)
    elapsed_orbits = times / times[-1] * n_orbits
    position_error_m = np.asarray(data["position_error_m"], dtype=np.float64)
    finite = np.asarray(data["finite"], dtype=bool)
    if not np.all(finite):
        raise ValueError(f"Non-finite study run in {path}")

    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    maxima: dict[str, float] = {}
    for source_label, _, _, _ in CURVES:
        index = labels.index(source_label)
        x = elapsed_orbits[1:]
        y = position_error_m[index, 1:]
        maxima[source_label] = float(np.max(y))
        curves[source_label] = downsample_curve(x, y, max_points)
    return curves, maxima


def axis_block(
    precision_title: str,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    add_legend: bool,
    show_xlabel: bool,
) -> str:
    """Return one PGFPlots axis worth of curves."""
    options = [f"title={{{precision_title}}}"]
    if add_legend:
        options.append("legend to name=frameStudyLegend")
    if not show_xlabel:
        options.append("xticklabels=\\empty")
    else:
        options.append("xlabel={time [orbits]}")

    lines = [f"  \\nextgroupplot[{', '.join(options)}]"]
    for source_label, legend_label, color_name, line_style in CURVES:
        x, y = curves[source_label]
        legend = f"\\addlegendentry{{{legend_label}}}" if add_legend else ""
        lines.extend(
            [
                (
                    f"    \\addplot[mark=none, {color_name}, {line_style}, "
                    "line width=2.3pt] coordinates {"
                ),
                coordinates_block(x, y),
                "    };",
                f"    {legend}",
            ]
        )
    return "\n".join(lines)


def tikz_document(
    float64_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    float32_curves: dict[str, tuple[np.ndarray, np.ndarray]],
) -> str:
    """Build the copyable TikZ/PGFPlots snippet."""
    return "\n".join(
        [
            "\\begin{tikzpicture}",
            "\\pgfplotsset{",
            "  frameStudyAxis/.style={",
            "    width=\\columnwidth,",
            "    height=0.5\\columnwidth,",
            "    xmin=0, xmax=3,",
            "    ymin=1e-9, ymax=1e4,",
            "    ymode=log,",
            "    grid=both,",
            "    minor grid style={draw=gray!15},",
            "    major grid style={draw=gray!30},",
            "    tick align=outside,",
            "    tick pos=left,",
            "    xlabel near ticks,",
            "    ylabel={position error [m]},",
            "    title style={font=\\footnotesize},",
            "    label style={font=\\footnotesize},",
            "    tick label style={font=\\scriptsize},",
            "    legend columns=3,",
            "    legend cell align=left,",
            "    legend style={",
            "      font=\\scriptsize,",
            "      /tikz/every even column/.append style={column sep=0.05cm},",
            "      draw=none,",
            "    },",
            "  },",
            "}",
            "\\pgfplotsset{",
            # Match the paper palette: plotMedBlue (008DFF), plotOrange (FF9D3A),
            # plotPurple (C701FF) so the figure agrees with the caption colors.
            "  frameEci/.style={color={rgb,255:red,0;green,141;blue,255}},",
            "  frameChief/.style={color={rgb,255:red,255;green,157;blue,58}},",
            "  frameLvlh/.style={color={rgb,255:red,199;green,1;blue,255}},",
            "}",
            "\\begin{groupplot}[",
            "  frameStudyAxis,",
            "  group style={group size=1 by 2, vertical sep=0.82cm},",
            "]",
            axis_block("Double precision", float64_curves, add_legend=True, show_xlabel=False),
            axis_block("Single precision", float32_curves, add_legend=False, show_xlabel=True),
            "\\end{groupplot}",
            "\\node[anchor=north, xshift=-0.35cm, yshift=-1.0cm] at (group c1r2.south)",
            "  {\\pgfplotslegendfromname{frameStudyLegend}};",
            "\\end{tikzpicture}",
            "",
        ]
    )


def standalone_document(tikz_filename: str) -> str:
    """Build a standalone TeX wrapper for local rendering checks."""
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


def write_outputs(
    out_dir: Path,
    tikz: str,
) -> tuple[Path, Path]:
    """Write the TikZ snippet and standalone wrapper."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tikz_path = out_dir / "frame_comparison.tikz"
    standalone_path = out_dir / "frame_comparison_standalone.tex"
    tikz_path.write_text(tikz, encoding="utf-8")
    standalone_path.write_text(standalone_document(tikz_path.name), encoding="utf-8")
    return tikz_path, standalone_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="circular_equatorial")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--study-orbits", type=float, default=3.0)
    parser.add_argument(
        "--study-rel-vel-lvlh",
        type=float,
        nargs=3,
        default=(0.01, 0.0, 0.0),
        metavar=("VX", "VY", "VZ"),
    )
    parser.add_argument(
        "--orbit-length-unit-m",
        "--study-length-unit-m",
        dest="study_length_unit_m",
        type=float,
        default=1000.0,
    )
    parser.add_argument("--max-points", type=int, default=250)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Generate the frame-study paper figure."""
    args = parse_args()
    rel_vel_bias_lvlh = np.asarray(args.study_rel_vel_lvlh, dtype=np.float64)

    paths = {
        precision: sample_path(
            args.out_dir,
            args.scenario,
            args.study_orbits,
            args.dt,
            precision,
            rel_vel_bias_lvlh,
            args.study_length_unit_m,
        )
        for precision in ("float64", "float32")
    }

    float64_curves, float64_maxima = load_curves(
        paths["float64"], args.study_orbits, args.max_points
    )
    float32_curves, float32_maxima = load_curves(
        paths["float32"], args.study_orbits, args.max_points
    )

    tikz_path, standalone_path = write_outputs(
        args.out_dir,
        tikz_document(float64_curves, float32_curves),
    )

    print("Frame-study paper figure")
    print(f"TikZ: {tikz_path}")
    print(f"Standalone TeX: {standalone_path}")
    print()
    print("case                    float64 max [m]  float32 max [m]")
    print("---------------------------------------------------------")
    for source_label, legend_label, _, _ in CURVES:
        print(
            f"{legend_label:22s} "
            f"{float64_maxima[source_label]:15.6e} "
            f"{float32_maxima[source_label]:15.6e}"
        )


if __name__ == "__main__":
    main()
