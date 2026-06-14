"""Run the articulated hinge Basilisk-MuJoCo comparison case."""

from __future__ import annotations

import argparse

from .cases import run_articulated_hinges_mjorbit
from .common import ensure_out_dir, save_npz, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alt-km", type=float, default=400.0)
    parser.add_argument("--inc-deg", type=float, default=51.6)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--orbit-dt", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument(
        "--basilisk-integrator",
        choices=("euler", "rk2", "rk4", "rkf45", "rkf78"),
        default="rkf45",
    )
    args = parser.parse_args()

    result = run_articulated_hinges_mjorbit(
        alt_km=args.alt_km,
        inc_deg=args.inc_deg,
        duration_s=args.duration,
        dt_s=args.dt,
        orbit_dt=args.orbit_dt,
        max_samples=args.max_samples,
        basilisk_integrator=args.basilisk_integrator,
    )

    out_dir = ensure_out_dir()
    write_json(out_dir / "articulated_hinges_summary.json", result.summary)
    samples = {
        "times_s": result.times_s,
        "qpos": result.qpos,
        "qvel": result.qvel,
        "hinge_targets_rad": result.hinge_targets_rad,
        "hub_r_eci_km": result.hub_r_eci_km,
        "hub_v_eci_km_s": result.hub_v_eci_km_s,
        "system_com_world_m": result.system_com_world_m,
    }
    if (
        result.basilisk_times_s is not None
        and result.basilisk_hinge_angles_rad is not None
        and result.basilisk_hinge_rates_rad_s is not None
        and result.basilisk_hub_r_eci_km is not None
        and result.basilisk_hub_v_eci_km_s is not None
    ):
        samples.update(
            {
                "basilisk_times_s": result.basilisk_times_s,
                "basilisk_hinge_angles_rad": result.basilisk_hinge_angles_rad,
                "basilisk_hinge_rates_rad_s": result.basilisk_hinge_rates_rad_s,
                "basilisk_hub_r_eci_km": result.basilisk_hub_r_eci_km,
                "basilisk_hub_v_eci_km_s": result.basilisk_hub_v_eci_km_s,
            }
        )
    save_npz(
        out_dir / "articulated_hinges_samples.npz",
        **samples,
    )

    _print_summary(result.summary)
    if not result.summary["all_finite"]:
        raise SystemExit("articulated hinge run produced non-finite state")
    if result.summary["final_joint_error_norm_rad"] > 5.0e-2:
        raise SystemExit("articulated hinge final joint error exceeded 0.05 rad")
    if result.summary["chief_delta_vs_passive_m"] > 1.0e-6:
        raise SystemExit("internal hinge motion changed the chief orbit position")


def _print_summary(summary: dict[str, object]) -> None:
    print("=" * 72)
    print("Basilisk-MuJoCo comparison: articulated hinges")
    print("=" * 72)
    print(f"Duration: {summary['duration_s']:.2f} s, dt={summary['dt_s']:.4f} s")
    print(f"Samples written: {summary['samples']}")
    print()
    print(
        "Final hinges: "
        f"[{summary['final_hinge_1_rad']:.6f}, {summary['final_hinge_2_rad']:.6f}] rad"
    )
    print(
        "Targets:      "
        f"[{summary['final_target_hinge_1_rad']:.6f}, "
        f"{summary['final_target_hinge_2_rad']:.6f}] rad"
    )
    print(f"Final joint error norm: {summary['final_joint_error_norm_rad']:.6e} rad")
    print(f"Chief delta vs passive: {summary['chief_delta_vs_passive_m']:.6e} m")
    print(f"Orbit energy rel change: {summary['orbit_energy_rel_change']:.6e}")
    basilisk = summary["basilisk_direct"]
    if isinstance(basilisk, dict):
        print()
        print(f"Basilisk direct leg: {basilisk['message']}")
        if basilisk.get("integrator"):
            print(f"Basilisk integrator: {basilisk['integrator']}")
        if basilisk.get("ran"):
            print(
                "Ours vs Basilisk hinge angle max: "
                f"{basilisk['ours_vs_basilisk_max_hinge_angle_error_rad']:.6e} rad"
            )
            print(
                "Ours vs Basilisk hub position max: "
                f"{basilisk['ours_vs_basilisk_max_hub_position_error_m']:.6e} m"
            )


if __name__ == "__main__":
    main()
