"""Run the single rigid-body Basilisk-MuJoCo comparison case."""

from __future__ import annotations

import argparse

from .cases import run_single_body_mujoco_orbit
from .common import ensure_out_dir, save_npz, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alt-km", type=float, default=400.0)
    parser.add_argument("--inc-deg", type=float, default=51.6)
    parser.add_argument("--n-steps", type=int, default=8000)
    parser.add_argument("--orbit-dt", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument(
        "--basilisk-integrator",
        choices=("euler", "rk2", "rk4", "rkf45", "rkf78"),
        default="rkf45",
    )
    args = parser.parse_args()

    result = run_single_body_mujoco_orbit(
        alt_km=args.alt_km,
        inc_deg=args.inc_deg,
        n_steps=args.n_steps,
        orbit_dt=args.orbit_dt,
        max_samples=args.max_samples,
        basilisk_integrator=args.basilisk_integrator,
    )

    out_dir = ensure_out_dir()
    write_json(out_dir / "single_body_orbit_summary.json", result.summary)
    samples = {
        "times_s": result.times_s,
        "r_eci_km": result.r_eci_km,
        "v_eci_km_s": result.v_eci_km_s,
        "quat_world_body": result.quat_world_body,
        "r_ref_eci_km": result.r_ref_eci_km,
        "v_ref_eci_km_s": result.v_ref_eci_km_s,
    }
    if (
        result.basilisk_times_s is not None
        and result.basilisk_r_eci_km is not None
        and result.basilisk_v_eci_km_s is not None
        and result.basilisk_quat_world_body is not None
    ):
        samples.update(
            {
                "basilisk_times_s": result.basilisk_times_s,
                "basilisk_r_eci_km": result.basilisk_r_eci_km,
                "basilisk_v_eci_km_s": result.basilisk_v_eci_km_s,
                "basilisk_quat_world_body": result.basilisk_quat_world_body,
            }
        )
    save_npz(
        out_dir / "single_body_orbit_samples.npz",
        **samples,
    )

    _print_summary(result.summary)
    if result.summary["final_position_error_m"] > 1.0:
        raise SystemExit("single-body final position error exceeded 1 m")
    if result.summary["final_velocity_error_m_s"] > 1.0e-3:
        raise SystemExit("single-body final velocity error exceeded 1 mm/s")


def _print_summary(summary: dict[str, object]) -> None:
    print("=" * 72)
    print("Basilisk-MuJoCo comparison: single rigid body")
    print("=" * 72)
    print(f"Orbit: {summary['alt_km']:.0f} km circular, inc={summary['inc_deg']:.1f} deg")
    print(f"Duration: {summary['period_s']:.6f} s, dt={summary['dt_s']:.6f} s")
    print(f"Samples written: {summary['samples']}")
    print()
    print(f"Final position error: {summary['final_position_error_m']:.6e} m")
    print(f"Max position error:   {summary['max_position_error_m']:.6e} m")
    print(f"Final velocity error: {summary['final_velocity_error_m_s']:.6e} m/s")
    print(f"Max velocity error:   {summary['max_velocity_error_m_s']:.6e} m/s")
    basilisk = summary["basilisk_direct"]
    if isinstance(basilisk, dict):
        print()
        print(f"Basilisk direct leg: {basilisk['message']}")
        if basilisk.get("integrator"):
            print(f"Basilisk integrator: {basilisk['integrator']}")
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
                "Ours vs Basilisk max attitude: "
                f"{basilisk['ours_vs_basilisk_max_attitude_error_rad']:.6e} rad"
            )


if __name__ == "__main__":
    main()
