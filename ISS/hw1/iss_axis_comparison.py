"""ISS spin stability comparison across all three principal axes.

For each principal axis, simulates the ISS with the same angular momentum
magnitude as 10 RPM about the major axis (Z). Runs both unperturbed and
perturbed (1% transverse kick) cases to demonstrate:

  - Z-axis (Izz = 201 M): STABLE   (maximum moment — major axis)
  - Y-axis (Iyy = 107 M): STABLE   (minimum moment — minor axis)
  - X-axis (Ixx = 128 M): UNSTABLE (intermediate moment — tennis racket theorem)

The perturbed X-axis case should exhibit the Dzhanibekov / intermediate axis
instability, with transverse rates growing dramatically.

Usage:
    pixi run python ISS/hw1/iss_axis_comparison.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mujoco_orbit import MjoData, MjoModel, OrbitInit, SurfaceSpec, mjo_forward, mjo_step
from mujoco_orbit.constants import R_EARTH
from tests.mujoco_orbit.reference.orbit.elements import keplerian_to_cartesian

# ---------------------------------------------------------------------------
# ISS parameters
# ---------------------------------------------------------------------------

ISS_IXX = 128e6  # kg·m²
ISS_IYY = 107e6  # kg·m²
ISS_IZZ = 201e6  # kg·m²
INERTIAS = np.array([ISS_IXX, ISS_IYY, ISS_IZZ])

ALT_KM = 410.0
INC_DEG = 51.6
MODEL_XML = str(pathlib.Path(__file__).resolve().parents[1] / "iss_model.xml")

# Reference angular momentum: 10 RPM about Z
OMEGA_REF = 10.0 * 2.0 * np.pi / 60.0  # rad/s
L_MAG = ISS_IZZ * OMEGA_REF  # kg·m²/s

PERTURBATION_FRAC = 0.01  # 1% transverse kick


def _make_surfaces() -> list[SurfaceSpec]:
    """Reuse the ISS surface model from iss_spin_stability.py."""
    surfaces: list[SurfaceSpec] = []
    sa_area = 375.0
    sa_y_offsets = [-45.0, -35.0, 35.0, 45.0]
    for y_off in sa_y_offsets:
        surfaces.append(SurfaceSpec(
            body_name="iss",
            center_of_pressure_body=np.array([0.0, y_off, 0.5]),
            normal_body=np.array([0.0, 0.0, 1.0]),
            area=sa_area,
        ))
        surfaces.append(SurfaceSpec(
            body_name="iss",
            center_of_pressure_body=np.array([0.0, y_off, -0.5]),
            normal_body=np.array([0.0, 0.0, -1.0]),
            area=sa_area,
        ))
    rad_area = 75.0
    for y_off in [-25.0, -15.0, 15.0, 20.0, 25.0, 30.0]:
        surfaces.append(SurfaceSpec(
            body_name="iss",
            center_of_pressure_body=np.array([0.0, y_off, 3.0]),
            normal_body=np.array([0.0, 0.0, 1.0]),
            area=rad_area,
            srp_coeff=1.5,
        ))
    mod_area = 180.0
    surfaces.append(SurfaceSpec(
        body_name="iss",
        center_of_pressure_body=np.array([36.5, 0.0, 0.0]),
        normal_body=np.array([1.0, 0.0, 0.0]),
        area=mod_area,
    ))
    surfaces.append(SurfaceSpec(
        body_name="iss",
        center_of_pressure_body=np.array([-36.5, 0.0, 0.0]),
        normal_body=np.array([-1.0, 0.0, 0.0]),
        area=mod_area,
    ))
    truss_area = 250.0
    surfaces.append(SurfaceSpec(
        body_name="iss",
        center_of_pressure_body=np.array([0.0, 54.25, 0.0]),
        normal_body=np.array([0.0, 1.0, 0.0]),
        area=truss_area,
        srp_coeff=1.5,
    ))
    surfaces.append(SurfaceSpec(
        body_name="iss",
        center_of_pressure_body=np.array([0.0, -54.25, 0.0]),
        normal_body=np.array([0.0, -1.0, 0.0]),
        area=truss_area,
        srp_coeff=1.5,
    ))
    return surfaces


def run_sim(
    omega0_body: np.ndarray,
    t_total: float = 300.0,
    dt: float = 0.01,
) -> dict:
    """Run one ISS spin simulation and return recorded data."""
    a_km = R_EARTH + ALT_KM
    R_eci, V_eci = keplerian_to_cartesian(
        a=a_km, e=0.0, inc=np.deg2rad(INC_DEG),
        raan=0.0, argp=0.0, nu=0.0,
    )
    model = MjoModel.from_xml_path(
        MODEL_XML,
        mj_timestep=dt,
        surfaces=_make_surfaces(),
        use_j2=True, use_drag=True, use_srp=True, use_magnetic=False,
    )
    data = MjoData(model, orbit=OrbitInit(R_eci=R_eci, V_eci=V_eci))

    # Set initial angular velocity (body = world at t=0)
    data.qvel[3:6] = omega0_body
    mjo_forward(model, data)

    n_steps = int(t_total / dt)
    record_every = int(0.1 / dt)
    n_records = n_steps // record_every + 1

    times = np.zeros(n_records)
    omega_body = np.zeros((n_records, 3))
    L_body = np.zeros((n_records, 3))

    def record(idx: int, t: float) -> None:
        times[idx] = t
        w_body = data.qvel[3:6].copy()
        omega_body[idx] = w_body
        L_body[idx] = INERTIAS * w_body

    record(0, 0.0)
    rec_idx = 1
    for i in range(n_steps):
        mjo_step(model, data)
        if (i + 1) % record_every == 0:
            record(rec_idx, (i + 1) * dt)
            rec_idx += 1

    n = rec_idx
    return {
        "times": times[:n],
        "omega_body": omega_body[:n],
        "L_body": L_body[:n],
    }


def main() -> None:
    axis_names = ["X (roll, Ixx)", "Y (pitch, Iyy)", "Z (yaw, Izz)"]
    axis_labels = ["X", "Y", "Z"]
    axis_stability = ["UNSTABLE (intermediate)", "STABLE (minor)", "STABLE (major)"]

    # Spin rates for same |L| about each axis
    omega_per_axis = L_MAG / INERTIAS  # [ωx, ωy, ωz]

    print("ISS Spin Stability — All Axes Comparison")
    print("=" * 60)
    print(f"|L| = {L_MAG:.3e} kg·m²/s  (= Izz × {OMEGA_REF:.4f} rad/s)")
    for i in range(3):
        print(f"  {axis_names[i]}: ω = {omega_per_axis[i]:.4f} rad/s "
              f"({omega_per_axis[i]*60/2/np.pi:.2f} RPM) — {axis_stability[i]}")
    print(f"Perturbation: {PERTURBATION_FRAC*100:.0f}% transverse kick")
    print("=" * 60)

    results = {}
    cases = []
    for ax_idx in range(3):
        for perturbed in [False, True]:
            omega0 = np.zeros(3)
            omega0[ax_idx] = omega_per_axis[ax_idx]
            if perturbed:
                # Add perturbation in the two transverse axes
                perturb_axes = [j for j in range(3) if j != ax_idx]
                for j in perturb_axes:
                    omega0[j] = PERTURBATION_FRAC * omega_per_axis[ax_idx]
            label = f"{axis_labels[ax_idx]}_{'perturbed' if perturbed else 'pure'}"
            cases.append((label, omega0, ax_idx, perturbed))

    for label, omega0, ax_idx, perturbed in cases:
        tag = "perturbed" if perturbed else "pure"
        print(f"\nRunning: {axis_names[ax_idx]} ({tag})  "
              f"ω₀ = [{omega0[0]:.4f}, {omega0[1]:.4f}, {omega0[2]:.4f}] rad/s ...")
        results[label] = run_sim(omega0)
        # Quick summary
        w_final = results[label]["omega_body"][-1]
        print(f"  Final ω = [{w_final[0]:+.4f}, {w_final[1]:+.4f}, {w_final[2]:+.4f}] rad/s")

    # --- Plotting ---
    plot_dir = pathlib.Path(__file__).parent / "plots"
    plot_dir.mkdir(exist_ok=True)

    comp_labels = ["ωx (roll)", "ωy (pitch)", "ωz (yaw)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    fig.suptitle(
        "ISS Spin Stability: All Three Principal Axes\n"
        f"|L| = {L_MAG:.2e} kg·m²/s (equivalent to 10 RPM about major axis)",
        fontsize=13,
    )

    for ax_idx in range(3):
        for col, perturbed in enumerate([False, True]):
            label = f"{axis_labels[ax_idx]}_{'perturbed' if perturbed else 'pure'}"
            data = results[label]
            t = data["times"]
            w = data["omega_body"]

            ax = axes[ax_idx, col]
            for k in range(3):
                ax.plot(t, w[:, k], label=comp_labels[k], color=colors[k], linewidth=0.8)

            tag = "Pure" if not perturbed else "1% Perturbation"
            ax.set_title(
                f"{axis_names[ax_idx]} — {tag}\n{axis_stability[ax_idx]}",
                fontsize=10,
            )
            ax.set_ylabel("ω (rad/s)")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 1].set_xlabel("Time (s)")
    plt.tight_layout()

    out_path = plot_dir / "iss_axis_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")

    # --- Zoomed view of intermediate axis instability ---
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    data_x_pert = results["X_perturbed"]
    t = data_x_pert["times"]
    w = data_x_pert["omega_body"]
    for k in range(3):
        ax2.plot(t, w[:, k], label=comp_labels[k], color=colors[k], linewidth=1.0)
    ax2.set_title(
        "Intermediate Axis Instability (X-axis spin, perturbed)\n"
        "Tennis Racket Theorem / Dzhanibekov Effect",
        fontsize=12,
    )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("ω (rad/s)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path2 = plot_dir / "iss_intermediate_axis_instability.png"
    fig2.savefig(out_path2, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {out_path2}")
    plt.close("all")


if __name__ == "__main__":
    main()
