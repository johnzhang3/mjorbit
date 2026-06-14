"""Compare two-arm free drift in raw absolute-ECI MuJoCo versus mjorbit."""

from __future__ import annotations

import argparse

from .common import ensure_out_dir, save_npz, write_json
from .eci_frame import run_two_arm_eci_frame_check


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alt-km", type=float, default=400.0)
    parser.add_argument("--inc-deg", type=float, default=51.6)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration in seconds. Defaults to one circular orbit.",
    )
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--mj-integrator",
        choices=("Euler", "RK4", "implicit", "implicitfast"),
        default="RK4",
        help="MuJoCo integrator for both raw ECI and mjorbit legs.",
    )
    parser.add_argument(
        "--gravity-application",
        choices=("callback", "xfrc"),
        default="callback",
        help=(
            "How the raw ECI leg applies point-mass gravity. The callback mode "
            "updates gravity inside MuJoCo dynamics evaluations; xfrc mirrors the "
            "original experiments/frame_study applied-force style."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=2048)
    args = parser.parse_args()

    result = run_two_arm_eci_frame_check(
        alt_km=args.alt_km,
        inc_deg=args.inc_deg,
        duration_s=args.duration,
        dt_s=args.dt,
        mj_integrator=args.mj_integrator,
        gravity_application=args.gravity_application,
        max_samples=args.max_samples,
    )

    out_dir = ensure_out_dir()
    suffix = args.mj_integrator.lower().replace(" ", "_")
    gravity_suffix = args.gravity_application.lower().replace(" ", "_")
    write_json(
        out_dir / f"two_arm_eci_frame_{suffix}_{gravity_suffix}_summary.json",
        result.summary,
    )
    write_json(out_dir / "two_arm_eci_frame_summary.json", result.summary)
    save_npz(
        out_dir / f"two_arm_eci_frame_{suffix}_{gravity_suffix}_samples.npz",
        times_s=result.times_s,
        local_hinge_angles_rad=result.local_hinge_angles_rad,
        eci_hinge_angles_rad=result.eci_hinge_angles_rad,
        local_hinge_rates_rad_s=result.local_hinge_rates_rad_s,
        eci_hinge_rates_rad_s=result.eci_hinge_rates_rad_s,
        local_hub_quat_world_body=result.local_hub_quat_world_body,
        eci_hub_quat_world_body=result.eci_hub_quat_world_body,
        local_body_r_eci_km=result.local_body_r_eci_km,
        eci_body_r_eci_km=result.eci_body_r_eci_km,
        local_body_v_eci_km_s=result.local_body_v_eci_km_s,
        eci_body_v_eci_km_s=result.eci_body_v_eci_km_s,
        local_system_com_eci_km=result.local_system_com_eci_km,
        eci_system_com_eci_km=result.eci_system_com_eci_km,
    )

    _print_summary(result.summary)


def _print_summary(summary: dict[str, object]) -> None:
    print("=" * 72)
    print("Two-arm ECI-frame check: raw MuJoCo vs mjorbit")
    print("=" * 72)
    print(f"Duration: {summary['duration_s']:.6f} s")
    print(f"dt: {summary['dt_s']:.6g} s, integrator={summary['mj_integrator']}")
    print(f"Raw ECI gravity application: {summary['gravity_application']}")
    print(f"Samples written: {summary['samples']}")
    print()
    print(f"Max hinge angle error: {summary['max_hinge_angle_error_rad']:.6e} rad")
    print(f"Max hinge rate error:  {summary['max_hinge_rate_error_rad_s']:.6e} rad/s")
    print(f"Max hub attitude error: {summary['max_hub_attitude_error_rad']:.6e} rad")
    print(f"Max hub position error: {summary['max_hub_position_error_m']:.6e} m")
    print(f"Max body position error: {summary['max_body_position_error_m']:.6e} m")
    print(
        "Max body-relative position error: "
        f"{summary['max_body_relative_position_error_m']:.6e} m"
    )
    print(f"Max system COM error: {summary['max_system_com_position_error_m']:.6e} m")
    print()
    print(f"mjorbit final hinges: {summary['local_final_hinges_rad']}")
    print(f"raw ECI final hinges:     {summary['eci_final_hinges_rad']}")


if __name__ == "__main__":
    main()
