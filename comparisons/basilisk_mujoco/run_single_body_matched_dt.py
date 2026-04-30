"""Run the single-body comparison with matched 0.1 s Basilisk and mjorbit steps."""

from __future__ import annotations

import argparse

import numpy as np

from .cases import run_single_body_mujoco_orbit
from .common import ensure_out_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alt-km", type=float, default=400.0)
    parser.add_argument("--inc-deg", type=float, default=51.6)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--orbit-dt", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument(
        "--omega-body",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.01),
        metavar=("WX", "WY", "WZ"),
        help="Initial body-frame angular velocity in rad/s.",
    )
    parser.add_argument(
        "--integrators",
        nargs="+",
        choices=("euler", "rk2", "rk4", "rkf45", "rkf78"),
        default=("euler", "rkf45"),
    )
    args = parser.parse_args()

    omega_body = np.asarray(args.omega_body, dtype=np.float64)
    summaries = []
    print("=" * 72)
    print("Basilisk-MuJoCo comparison: matched 0.1 s single rigid body")
    print("=" * 72)
    print(f"mujoco_orbit: mj_timestep={args.dt:.6g} s, orbit_dt={args.orbit_dt:.6g} s")
    print(f"Basilisk: task dt={args.dt:.6g} s")
    print(f"Initial omega_BN_B={omega_body.tolist()} rad/s")

    for integrator in args.integrators:
        result = run_single_body_mujoco_orbit(
            alt_km=args.alt_km,
            inc_deg=args.inc_deg,
            dt_s=args.dt,
            orbit_dt=args.orbit_dt,
            max_samples=args.max_samples,
            basilisk_integrator=integrator,
            initial_omega_body_rad_s=omega_body,
        )
        summaries.append(result.summary)
        basilisk = result.summary["basilisk_direct"]
        print()
        print(f"Basilisk integrator: {integrator}")
        print(f"Duration: {result.summary['duration_s']:.6f} s")
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
                    "Ours vs Basilisk final attitude: "
                    f"{basilisk['ours_vs_basilisk_final_attitude_error_rad']:.6e} rad"
                )
                print(
                    "Ours vs Basilisk max attitude: "
                    f"{basilisk['ours_vs_basilisk_max_attitude_error_rad']:.6e} rad"
                )
                print(
                    "Basilisk analytic max position: "
                    f"{basilisk['max_position_error_m']:.6e} m"
                )

    out_dir = ensure_out_dir()
    write_json(
        out_dir / "single_body_matched_dt_summary.json",
        {
            "case": "single_body_matched_dt",
            "summaries": summaries,
        },
    )


if __name__ == "__main__":
    main()
