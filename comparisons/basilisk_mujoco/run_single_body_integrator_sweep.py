"""Compare single-body parity with different Basilisk MJScene integrators."""

from __future__ import annotations

import argparse

from .cases import run_single_body_mujoco_orbit
from .common import ensure_out_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alt-km", type=float, default=400.0)
    parser.add_argument("--inc-deg", type=float, default=51.6)
    parser.add_argument("--n-steps", type=int, default=8000)
    parser.add_argument("--orbit-dt", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument(
        "--integrators",
        nargs="+",
        choices=("euler", "rk2", "rk4", "rkf45", "rkf78"),
        default=("euler", "rkf45"),
    )
    args = parser.parse_args()

    summaries = []
    print("=" * 72)
    print("Basilisk-MuJoCo comparison: single-body integrator sweep")
    print("=" * 72)
    for integrator in args.integrators:
        result = run_single_body_mujoco_orbit(
            alt_km=args.alt_km,
            inc_deg=args.inc_deg,
            n_steps=args.n_steps,
            orbit_dt=args.orbit_dt,
            max_samples=args.max_samples,
            basilisk_integrator=integrator,
        )
        summaries.append(result.summary)
        basilisk = result.summary["basilisk_direct"]
        print()
        print(f"Basilisk integrator: {integrator}")
        print(
            "mujoco_orbit final position error: "
            f"{result.summary['final_position_error_m']:.6e} m"
        )
        if isinstance(basilisk, dict):
            print(f"Basilisk ran: {basilisk.get('ran')}")
            print(f"Basilisk message: {basilisk.get('message')}")
            if basilisk.get("ran"):
                print(
                    "Ours vs Basilisk max position: "
                    f"{basilisk['ours_vs_basilisk_max_position_error_m']:.6e} m"
                )
                print(
                    "Ours vs Basilisk max velocity: "
                    f"{basilisk['ours_vs_basilisk_max_velocity_error_m_s']:.6e} m/s"
                )
                print(
                    "Basilisk analytic max position: "
                    f"{basilisk['max_position_error_m']:.6e} m"
                )

    out_dir = ensure_out_dir()
    write_json(
        out_dir / "single_body_integrator_sweep_summary.json",
        {
            "case": "single_body_integrator_sweep",
            "summaries": summaries,
        },
    )


if __name__ == "__main__":
    main()
