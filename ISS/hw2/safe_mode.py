"""HW2 Section 1 — Safe Mode: Gyrostat spin stabilization.

Designs a passively stable spin configuration for the ISS that keeps the
solar panels pointed in an inertially fixed direction (toward the sun).

Steps:
  1. Perturb the ISS inertia matrix (eigenvalues +/- few %, axes rotated a few deg)
  2. Desired spin: 10 RPM about the solar panel normal (body +Z)
  3. Superspin + dynamic balance -> rotor momentum h with inertia ratio >= 1.2
  4. Simulate the gyrostat equation:  J*omega_dot + omega x (J*omega + h) = 0
  5. Perturbed ICs to demonstrate stability

Usage:
    uv run python ISS/hw2/safe_mode.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import (
    INERTIA_RATIO_MIN,
    ISS_IXX,
    ISS_IYY,
    ISS_IZZ,
    J_NOMINAL,
    OMEGA_RAD_S,
    OMEGA_RPM,
    SOLAR_NORMAL,
    compute_rotor_momentum,
    perturb_inertia,
    quat_normalize,
    quat_rotate,
)
from scipy.integrate import solve_ivp

np.random.seed(42)


# ---------------------------------------------------------------------------
# Gyrostat simulation
# ---------------------------------------------------------------------------

def gyrostat_eom(
    t: float,
    state: np.ndarray,
    J: np.ndarray,
    J_inv: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    """Gyrostat equations of motion (torque-free).

    J*omega_dot + omega x (J*omega + h) = 0
    State = [q0, q1, q2, q3, wx, wy, wz]  (scalar-first quaternion)
    """
    q = state[:4]
    omega = state[4:]

    H = J @ omega + h
    omega_dot = J_inv @ (-np.cross(omega, H))

    q0, q1, q2, q3 = q
    wx, wy, wz = omega
    q_dot = 0.5 * np.array([
        -q1*wx - q2*wy - q3*wz,
         q0*wx + q2*wz - q3*wy,
         q0*wy + q3*wx - q1*wz,
         q0*wz + q1*wy - q2*wx,
    ])

    return np.concatenate([q_dot, omega_dot])


def compute_pointing_error(q: np.ndarray, sun_eci: np.ndarray) -> float:
    """Angle between solar panel normal (+Z body) and sun vector (deg)."""
    panel_normal_eci = quat_rotate(q, SOLAR_NORMAL)
    cos_angle = np.clip(np.dot(panel_normal_eci, sun_eci), -1, 1)
    return np.rad2deg(np.arccos(abs(cos_angle)))


def run_simulation(
    J: np.ndarray, h: np.ndarray, omega0: np.ndarray,
    q0: np.ndarray, t_total: float, label: str = "",
) -> dict:
    """Integrate gyrostat equations and return history."""
    J_inv = np.linalg.inv(J)
    state0 = np.concatenate([q0, omega0])

    def eom(t, y):
        return gyrostat_eom(t, y, J, J_inv, h)

    sol = solve_ivp(
        eom, [0, t_total], state0,
        method="DOP853", rtol=1e-12, atol=1e-14, max_step=0.05,
        dense_output=True,
    )

    dt_sample = 0.05
    t_eval = np.arange(0, t_total, dt_sample)
    states = sol.sol(t_eval).T

    for i in range(len(states)):
        states[i, :4] = quat_normalize(states[i, :4])

    return {"times": t_eval, "quat": states[:, :4], "omega": states[:, 4:], "label": label}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("HW2 Section 1 — Safe Mode: Gyrostat Spin Stabilization")
    print("=" * 65)

    # --- Step 1: Perturb inertia ---
    J = perturb_inertia(J_NOMINAL)

    print("\nNominal inertia (diagonal, kg*m^2):")
    print(f"  Ixx = {ISS_IXX:.3e}")
    print(f"  Iyy = {ISS_IYY:.3e}")
    print(f"  Izz = {ISS_IZZ:.3e}")

    D_pert, V_pert = np.linalg.eigh(J)
    print("\nPerturbed inertia eigenvalues (kg*m^2):")
    for i, val in enumerate(D_pert):
        print(f"  I_{i+1} = {val:.6e}")
    print("\nPerturbed principal axes (columns of V):")
    for i in range(3):
        print(f"  e_{i+1} = [{V_pert[0,i]:+.6f}, {V_pert[1,i]:+.6f}, {V_pert[2,i]:+.6f}]")
    print("\nFull perturbed inertia matrix (kg*m^2):")
    for row in J:
        print(f"  [{row[0]:+.6e}, {row[1]:+.6e}, {row[2]:+.6e}]")

    # --- Step 2: Desired angular velocity ---
    omega_desired = OMEGA_RAD_S * SOLAR_NORMAL
    print(f"\nDesired spin: {OMEGA_RPM} RPM about solar panel normal (+Z)")
    print(
        f"  omega = [{omega_desired[0]:.6f}, {omega_desired[1]:.6f}, "
        f"{omega_desired[2]:.6f}] rad/s"
    )

    J_omega = J @ omega_desired
    J_omega_hat = J_omega / np.linalg.norm(J_omega)
    omega_hat = omega_desired / np.linalg.norm(omega_desired)
    misalignment = np.rad2deg(np.arccos(np.clip(abs(np.dot(J_omega_hat, omega_hat)), 0, 1)))
    print(f"  J*omega misalignment from omega: {misalignment:.2f} deg (0 = principal axis)")

    # --- Step 3: Compute rotor momentum ---
    h, lam, I_trans_max, I_trans = compute_rotor_momentum(J, omega_desired, INERTIA_RATIO_MIN)

    actual_ratio = lam / I_trans_max
    print("\nRotor momentum (dynamic balance + superspin):")
    print(f"  h = [{h[0]:+.6e}, {h[1]:+.6e}, {h[2]:+.6e}] kg*m^2/s")
    print(f"  |h| = {np.linalg.norm(h):.6e} kg*m^2/s")
    print(f"  Effective spin-axis inertia (lam):     {lam:.6e} kg*m^2")
    print(f"  Max transverse effective inertia:       {I_trans_max:.6e} kg*m^2")
    print(f"  Transverse eigenvalues:                 [{I_trans[0]:.6e}, {I_trans[1]:.6e}]")
    print(
        "  Inertia ratio (lam / I_trans_max):      "
        f"{actual_ratio:.4f}  (required >= {INERTIA_RATIO_MIN})"
    )

    residual = np.cross(omega_desired, J @ omega_desired + h)
    print(f"  Dynamic balance check |omega x (J*omega + h)| = {np.linalg.norm(residual):.2e}")

    # --- Step 4-5: Simulate ---
    I_spin = lam
    I_trans_avg = np.mean(I_trans)
    T_nutation = 2.0 * np.pi / OMEGA_RAD_S * I_spin / abs(I_spin - I_trans_avg)
    print(f"\n  Estimated nutation period: {T_nutation:.2f} s")

    t_total = max(10.0 * T_nutation, 120.0)
    print(f"  Simulation duration: {t_total:.1f} s ({t_total/60:.1f} min)")

    q0 = np.array([1.0, 0.0, 0.0, 0.0])

    print("\n--- Case A: Unperturbed IC (exact equilibrium) ---")
    result_exact = run_simulation(J, h, omega_desired.copy(), q0.copy(), t_total, "Exact IC")

    perturb_frac = 0.01
    omega_pert = omega_desired.copy()
    omega_pert[0] += perturb_frac * OMEGA_RAD_S
    omega_pert[1] += perturb_frac * OMEGA_RAD_S
    print("--- Case B: Perturbed IC (1% transverse kick) ---")
    print(f"  omega_0 = [{omega_pert[0]:.6f}, {omega_pert[1]:.6f}, {omega_pert[2]:.6f}] rad/s")
    result_pert1 = run_simulation(J, h, omega_pert.copy(), q0.copy(), t_total, "1% perturbation")

    perturb_frac_5 = 0.05
    omega_pert5 = omega_desired.copy()
    omega_pert5[0] += perturb_frac_5 * OMEGA_RAD_S
    omega_pert5[1] += perturb_frac_5 * OMEGA_RAD_S
    print("--- Case C: Perturbed IC (5% transverse kick) ---")
    print(f"  omega_0 = [{omega_pert5[0]:.6f}, {omega_pert5[1]:.6f}, {omega_pert5[2]:.6f}] rad/s")
    result_pert5 = run_simulation(J, h, omega_pert5.copy(), q0.copy(), t_total, "5% perturbation")

    print("--- Case D: No rotor (h=0) with 1% perturbation ---")
    result_no_rotor = run_simulation(
        J,
        np.zeros(3),
        omega_pert.copy(),
        q0.copy(),
        t_total,
        "No rotor, 1% pert",
    )

    # --- Plotting ---
    plot_dir = pathlib.Path(__file__).parent / "plots"
    plot_dir.mkdir(exist_ok=True)
    sun_eci = np.array([1.0, 0.0, 0.0])

    # Figure 1: Angular velocity components
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig1.suptitle(
        "Safe Mode Gyrostat — Angular Velocity Components (Body Frame)\n"
        f"Perturbed ISS inertia, 10 RPM about solar panel normal (+Z), "
        f"inertia ratio = {actual_ratio:.2f}", fontsize=11)
    comp_labels = ["wx", "wy", "wz"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ax, result in zip(axes1.flat, [result_exact, result_pert1, result_pert5, result_no_rotor]):
        t = result["times"]
        w = result["omega"]
        for k in range(3):
            ax.plot(t, w[:, k], label=comp_labels[k], color=colors[k], linewidth=0.6)
        ax.set_title(result["label"], fontsize=10)
        ax.set_ylabel("omega (rad/s)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes1[1, 0].set_xlabel("Time (s)")
    axes1[1, 1].set_xlabel("Time (s)")
    plt.tight_layout()
    fig1.savefig(plot_dir / "safe_mode_omega.png", dpi=200, bbox_inches="tight")

    # Figure 2: Pointing error
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig2.suptitle("Safe Mode Gyrostat — Solar Panel Pointing Error\n"
                  "Angle between panel normal and sun direction (+X ECI)", fontsize=11)

    for ax, result in zip(axes2.flat, [result_exact, result_pert1, result_pert5, result_no_rotor]):
        t = result["times"]
        errors = np.array(
            [compute_pointing_error(result["quat"][i], sun_eci) for i in range(len(t))]
        )
        ax.plot(t, errors, color="#d62728", linewidth=0.6)
        ax.set_title(result["label"], fontsize=10)
        ax.set_ylabel("Pointing error (deg)")
        ax.grid(True, alpha=0.3)

    axes2[1, 0].set_xlabel("Time (s)")
    axes2[1, 1].set_xlabel("Time (s)")
    plt.tight_layout()
    fig2.savefig(plot_dir / "safe_mode_pointing.png", dpi=200, bbox_inches="tight")

    # Figure 3: Quaternion components
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig3.suptitle("Safe Mode Gyrostat — Attitude Quaternion Components\n"
                  "q = [q0, q1, q2, q3] (scalar-first)", fontsize=11)
    q_labels = ["q0", "q1", "q2", "q3"]
    q_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for ax, result in zip(axes3.flat, [result_exact, result_pert1, result_pert5, result_no_rotor]):
        t = result["times"]
        q = result["quat"]
        for k in range(4):
            ax.plot(t, q[:, k], label=q_labels[k], color=q_colors[k], linewidth=0.6)
        ax.set_title(result["label"], fontsize=10)
        ax.set_ylabel("Quaternion component")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)

    axes3[1, 0].set_xlabel("Time (s)")
    axes3[1, 1].set_xlabel("Time (s)")
    plt.tight_layout()
    fig3.savefig(plot_dir / "safe_mode_quaternion.png", dpi=200, bbox_inches="tight")

    # Figure 4: Transverse rate magnitude
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.set_title("Transverse Angular Velocity Magnitude\n"
                  "sqrt(wx^2 + wy^2) — should remain bounded for stable gyrostat", fontsize=11)
    for result in [result_exact, result_pert1, result_pert5, result_no_rotor]:
        t = result["times"]
        w = result["omega"]
        ax4.plot(t, np.sqrt(w[:, 0]**2 + w[:, 1]**2), label=result["label"], linewidth=0.8)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("|omega_transverse| (rad/s)")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    fig4.savefig(plot_dir / "safe_mode_transverse.png", dpi=200, bbox_inches="tight")

    plt.close("all")
    print(f"\nAll plots saved to {plot_dir}/")


if __name__ == "__main__":
    main()
