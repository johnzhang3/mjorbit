"""Run the passive two-arm one-orbit Basilisk-MuJoCo comparison case."""

from __future__ import annotations

import argparse

import numpy as np

from .cases import TwoArmFreeDriftRun, run_two_arm_free_drift_mujoco_orbit
from .common import ensure_out_dir, save_npz, write_json


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
    parser.add_argument("--orbit-dt", type=float, default=0.1)
    parser.add_argument(
        "--mj-integrator",
        choices=("Euler", "RK4", "implicit", "implicitfast"),
        default="Euler",
        help="MuJoCo integrator used by the mujoco_orbit leg.",
    )
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument(
        "--hinge-angles",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("Q1", "Q2"),
        help="Initial hinge angles in radians.",
    )
    parser.add_argument(
        "--hinge-rates",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("DQ1", "DQ2"),
        help="Initial hinge rates in radians per second.",
    )
    parser.add_argument(
        "--omega-body",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("WX", "WY", "WZ"),
        help="Initial hub body-frame angular velocity in rad/s.",
    )
    parser.add_argument(
        "--integrators",
        nargs="+",
        choices=("euler", "rk2", "rk4", "rkf45", "rkf78"),
        default=("rkf45",),
        help="Basilisk integrators to run against the same mujoco_orbit setup.",
    )
    args = parser.parse_args()

    out_dir = ensure_out_dir()
    summaries = []
    print("=" * 72)
    print("Basilisk-MuJoCo comparison: passive two-arm free drift")
    print("=" * 72)
    print(
        "mujoco_orbit: "
        f"mj_timestep={args.dt:.6g} s, orbit_dt={args.orbit_dt:.6g} s, "
        f"integrator={args.mj_integrator}"
    )
    print(f"Basilisk: task dt={args.dt:.6g} s")
    print(f"Initial hinge angles: {list(args.hinge_angles)} rad")
    print(f"Initial hinge rates: {list(args.hinge_rates)} rad/s")

    mj_suffix = args.mj_integrator.lower().replace(" ", "_")
    for integrator in args.integrators:
        result = run_two_arm_free_drift_mujoco_orbit(
            alt_km=args.alt_km,
            inc_deg=args.inc_deg,
            duration_s=args.duration,
            dt_s=args.dt,
            orbit_dt=args.orbit_dt,
            mj_integrator=args.mj_integrator,
            max_samples=args.max_samples,
            basilisk_integrator=integrator,
            initial_hinge_angles_rad=np.asarray(args.hinge_angles, dtype=np.float64),
            initial_hinge_rates_rad_s=np.asarray(args.hinge_rates, dtype=np.float64),
            initial_omega_body_rad_s=np.asarray(args.omega_body, dtype=np.float64),
        )
        summaries.append(result.summary)
        save_npz(
            out_dir / f"two_arm_free_drift_{mj_suffix}_{integrator}_samples.npz",
            times_s=result.times_s,
            qpos=result.qpos,
            qvel=result.qvel,
            hinge_angles_rad=result.hinge_angles_rad,
            hinge_rates_rad_s=result.hinge_rates_rad_s,
            hub_r_eci_km=result.hub_r_eci_km,
            hub_v_eci_km_s=result.hub_v_eci_km_s,
            hub_quat_world_body=result.hub_quat_world_body,
            body_r_eci_km=result.body_r_eci_km,
            body_v_eci_km_s=result.body_v_eci_km_s,
            system_com_world_m=result.system_com_world_m,
            system_com_eci_km=result.system_com_eci_km,
            r_ref_eci_km=result.r_ref_eci_km,
            v_ref_eci_km_s=result.v_ref_eci_km_s,
            **_basilisk_sample_arrays(result),
        )
        _print_integrator_summary(integrator, result.summary)
        if not result.summary["all_finite"]:
            raise SystemExit("two-arm free-drift run produced non-finite mujoco_orbit state")

    payload = {
        "case": "two_arm_free_drift",
        "mj_integrator": args.mj_integrator,
        "summaries": summaries,
    }
    write_json(out_dir / f"two_arm_free_drift_{mj_suffix}_summary.json", payload)
    write_json(out_dir / "two_arm_free_drift_summary.json", payload)


def _basilisk_sample_arrays(result: TwoArmFreeDriftRun) -> dict[str, np.ndarray]:
    arrays = {}
    for attr in (
        "basilisk_times_s",
        "basilisk_hinge_angles_rad",
        "basilisk_hinge_rates_rad_s",
        "basilisk_hub_quat_world_body",
        "basilisk_body_r_eci_km",
        "basilisk_body_v_eci_km_s",
        "basilisk_system_com_eci_km",
    ):
        value = getattr(result, attr)
        if value is not None:
            arrays[attr] = value
    return arrays


def _print_integrator_summary(integrator: str, summary: dict[str, object]) -> None:
    print()
    print(f"Basilisk integrator: {integrator}")
    print(f"Duration: {summary['duration_s']:.6f} s")
    print(f"Samples written: {summary['samples']}")
    print(
        "Final hinges: "
        f"[{summary['final_hinge_1_rad']:.6f}, {summary['final_hinge_2_rad']:.6f}] rad"
    )
    print(f"Max hinge rate: {summary['max_abs_hinge_rate_rad_s']:.6e} rad/s")
    print(
        "Chief final reference error: "
        f"{summary['chief_final_reference_position_error_m']:.6e} m"
    )
    print(f"System COM world drift: {summary['system_com_world_drift_m']:.6e} m")

    basilisk = summary["basilisk_direct"]
    if isinstance(basilisk, dict):
        print(f"Basilisk ran: {basilisk.get('ran')}")
        print(f"Basilisk message: {basilisk.get('message')}")
        if basilisk.get("ran"):
            print(
                "Basilisk final hinges: "
                f"[{basilisk['final_hinge_1_rad']:.6f}, "
                f"{basilisk['final_hinge_2_rad']:.6f}] rad"
            )
            print(
                "Ours vs Basilisk max hub position: "
                f"{basilisk['ours_vs_basilisk_max_hub_position_error_m']:.6e} m"
            )
            print(
                "Ours vs Basilisk max hub attitude: "
                f"{basilisk['ours_vs_basilisk_max_hub_attitude_error_rad']:.6e} rad"
            )
            print(
                "Ours vs Basilisk max body position: "
                f"{basilisk['ours_vs_basilisk_max_body_position_error_m']:.6e} m"
            )
            print(
                "Ours vs Basilisk max hinge angle: "
                f"{basilisk['ours_vs_basilisk_max_hinge_angle_error_rad']:.6e} rad"
            )
            print(
                "Basilisk hub reference max position: "
                f"{basilisk['hub_max_reference_position_error_m']:.6e} m"
            )


if __name__ == "__main__":
    main()
