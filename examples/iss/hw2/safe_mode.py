"""HW2 Section 1 — Safe Mode: Gyrostat spin stabilization.

Designs a passively stable spin configuration for the ISS that keeps the
solar panels pointed in an inertially fixed direction (toward the sun).

Steps:
  1. Perturb the ISS inertia matrix (eigenvalues ± few %, axes rotated a few deg)
  2. Desired spin: 10 RPM about the solar panel normal (body +Z)
  3. Superspin + dynamic balance → rotor momentum h with inertia ratio ≥ 1.2
  4. Simulate the gyrostat equation:  J·ω̇ + ω × (J·ω + h) = 0
  5. Perturbed ICs to demonstrate stability

Usage:
    uv run python examples/iss/hw2/safe_mode.py
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


# ---------------------------------------------------------------------------
# ISS parameters (from HW1)
# ---------------------------------------------------------------------------

ISS_IXX = 128e6   # kg·m²
ISS_IYY = 107e6   # kg·m²
ISS_IZZ = 201e6   # kg·m²
J_NOMINAL = np.diag([ISS_IXX, ISS_IYY, ISS_IZZ])

# Solar panel normal in body frame = +Z (panels are flat in XY plane)
SOLAR_NORMAL = np.array([0.0, 0.0, 1.0])

OMEGA_RPM = 10.0
OMEGA_RAD_S = OMEGA_RPM * 2.0 * np.pi / 60.0  # ≈ 1.047 rad/s

INERTIA_RATIO_MIN = 1.2  # minimum required

np.random.seed(42)


# ---------------------------------------------------------------------------
# 1. Perturb inertia matrix
# ---------------------------------------------------------------------------

def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from 3-vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def perturb_inertia(
    J: np.ndarray,
    eigenvalue_sigma: float = 0.03,
    axis_sigma_deg: float = 3.0,
) -> np.ndarray:
    """Perturb inertia via eigendecomposition.

    J = V D V^T  →  J̃ = Ṽ D̃ Ṽ^T

    where D̃ = D(I + diag(d)), d ~ N(0, σ²)
    and Ṽ = V exp(v̂), v ~ N(0, σ_axis²)
    """
    D_vals, V = np.linalg.eigh(J)

    # Perturb eigenvalues by a few percent
    d = np.random.randn(3) * eigenvalue_sigma
    D_tilde = np.diag(D_vals * (1.0 + d))

    # Perturb principal axis directions by a small rotation
    v = np.random.randn(3) * np.deg2rad(axis_sigma_deg)
    V_tilde = V @ expm(skew(v))

    J_tilde = V_tilde @ D_tilde @ V_tilde.T
    return J_tilde


# ---------------------------------------------------------------------------
# 2–3. Superspin + dynamic balance
# ---------------------------------------------------------------------------

def compute_rotor_momentum(
    J: np.ndarray,
    omega: np.ndarray,
    inertia_ratio: float = 1.2,
) -> np.ndarray:
    """Compute rotor angular momentum h for dynamic balance + superspin.

    Dynamic balance requires:  ω × (J·ω + h) = 0
    i.e. J·ω + h is parallel to ω.

    So h = λ·ω - J·ω for some scalar λ.

    Superspin requires the effective spin-axis inertia (λ) to exceed
    the largest transverse effective inertia by at least `inertia_ratio`.

    The effective inertia about ω is λ = (J·ω + h)·ω / |ω|².
    The transverse effective inertias are eigenvalues of the projection
    of J onto the plane perpendicular to ω.
    """
    omega_hat = omega / np.linalg.norm(omega)
    omega_mag = np.linalg.norm(omega)

    # Current angular momentum component along spin axis
    J_omega = J @ omega
    lambda_no_rotor = np.dot(J_omega, omega_hat) / omega_mag

    # Transverse effective inertias: project J into plane ⊥ ω
    # Build orthonormal basis for the transverse plane
    if abs(omega_hat[0]) < 0.9:
        e1 = np.cross(omega_hat, np.array([1, 0, 0]))
    else:
        e1 = np.cross(omega_hat, np.array([0, 1, 0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(omega_hat, e1)

    # 2×2 projected inertia matrix in transverse plane
    P = np.array([e1, e2])  # (2,3)
    J_perp = P @ J @ P.T    # (2,2)
    transverse_eigenvalues = np.linalg.eigvalsh(J_perp)
    I_trans_max = np.max(transverse_eigenvalues)

    # We need λ ≥ inertia_ratio * I_trans_max
    lambda_required = inertia_ratio * I_trans_max

    # If the natural λ is already sufficient, no rotor needed (unlikely after perturbation)
    lam = max(lambda_required, lambda_no_rotor)

    # h = λ·ω̂ * |ω| - J·ω  →  but we parameterize as h = λ·ω/|ω| - J·ω/|ω| times |ω|
    # Actually: J·ω + h = λ · ω̂ · |ω|  →  h = λ * omega - J @ omega
    h = lam * omega - J_omega

    return h, lam, I_trans_max, transverse_eigenvalues


# ---------------------------------------------------------------------------
# 4–5. Gyrostat simulation
# ---------------------------------------------------------------------------

def gyrostat_eom(t: float, state: np.ndarray, J: np.ndarray, J_inv: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Gyrostat equations of motion (torque-free).

    J·ω̇ + ω × (J·ω + h) = 0
    ω̇ = J⁻¹ · [-(ω × (J·ω + h))]

    Also propagate quaternion: q̇ = 0.5 * q ⊗ [0, ω]
    State = [q0, q1, q2, q3, wx, wy, wz]  (scalar-first quaternion)
    """
    q = state[:4]
    omega = state[4:]

    # Angular velocity dynamics
    H = J @ omega + h  # total angular momentum in body frame
    omega_dot = J_inv @ (-np.cross(omega, H))

    # Quaternion kinematics: q̇ = 0.5 * Ω(ω) · q
    q0, q1, q2, q3 = q
    wx, wy, wz = omega
    q_dot = 0.5 * np.array([
        -q1*wx - q2*wy - q3*wz,
         q0*wx + q2*wz - q3*wy,
         q0*wy + q3*wx - q1*wz,
         q0*wz + q1*wy - q2*wx,
    ])

    return np.concatenate([q_dot, omega_dot])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (scalar-first)."""
    q0, q1, q2, q3 = q
    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)],
    ])
    return R @ v


def compute_pointing_error(q: np.ndarray, sun_eci: np.ndarray) -> float:
    """Compute angle between solar panel normal and sun vector.

    Solar panel normal is +Z in body frame. Rotate to inertial and
    compute angle to sun direction.
    """
    # Body Z-axis in inertial frame
    panel_normal_eci = quat_rotate(q, SOLAR_NORMAL)
    cos_angle = np.clip(np.dot(panel_normal_eci, sun_eci), -1, 1)
    return np.rad2deg(np.arccos(abs(cos_angle)))


def run_simulation(
    J: np.ndarray,
    h: np.ndarray,
    omega0: np.ndarray,
    q0: np.ndarray,
    t_total: float,
    label: str = "",
) -> dict:
    """Integrate gyrostat equations and return history."""
    J_inv = np.linalg.inv(J)

    state0 = np.concatenate([q0, omega0])

    def eom(t, y):
        return gyrostat_eom(t, y, J, J_inv, h)

    # Event to normalize quaternion periodically (via dense output + post-process)
    sol = solve_ivp(
        eom, [0, t_total], state0,
        method="DOP853", rtol=1e-12, atol=1e-14, max_step=0.05,
        dense_output=True,
    )

    # Sample at uniform times
    dt_sample = 0.05  # 20 Hz
    t_eval = np.arange(0, t_total, dt_sample)
    states = sol.sol(t_eval).T  # (n_times, 7)

    # Normalize quaternions
    for i in range(len(states)):
        states[i, :4] = quat_normalize(states[i, :4])

    return {
        "times": t_eval,
        "quat": states[:, :4],
        "omega": states[:, 4:],
        "label": label,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("HW2 Section 1 — Safe Mode: Gyrostat Spin Stabilization")
    print("=" * 65)

    # --- Step 1: Perturb inertia ---
    J = perturb_inertia(J_NOMINAL)

    print("\nNominal inertia (diagonal, kg·m²):")
    print(f"  Ixx = {ISS_IXX:.3e}")
    print(f"  Iyy = {ISS_IYY:.3e}")
    print(f"  Izz = {ISS_IZZ:.3e}")

    D_pert, V_pert = np.linalg.eigh(J)
    print(f"\nPerturbed inertia eigenvalues (kg·m²):")
    for i, val in enumerate(D_pert):
        print(f"  I_{i+1} = {val:.6e}")
    print(f"\nPerturbed principal axes (columns of V):")
    for i in range(3):
        print(f"  e_{i+1} = [{V_pert[0,i]:+.6f}, {V_pert[1,i]:+.6f}, {V_pert[2,i]:+.6f}]")
    print(f"\nFull perturbed inertia matrix (kg·m²):")
    for row in J:
        print(f"  [{row[0]:+.6e}, {row[1]:+.6e}, {row[2]:+.6e}]")

    # --- Step 2: Desired angular velocity ---
    omega_desired = OMEGA_RAD_S * SOLAR_NORMAL  # 10 RPM about +Z (solar panel normal)
    print(f"\nDesired spin: {OMEGA_RPM} RPM about solar panel normal (+Z)")
    print(f"  ω = [{omega_desired[0]:.6f}, {omega_desired[1]:.6f}, {omega_desired[2]:.6f}] rad/s")

    # Check: is +Z a principal axis of perturbed J?
    J_omega = J @ omega_desired
    J_omega_hat = J_omega / np.linalg.norm(J_omega)
    omega_hat = omega_desired / np.linalg.norm(omega_desired)
    misalignment = np.rad2deg(np.arccos(np.clip(abs(np.dot(J_omega_hat, omega_hat)), 0, 1)))
    print(f"  J·ω misalignment from ω: {misalignment:.2f}° (0° would mean principal axis)")

    # --- Step 3: Compute rotor momentum ---
    h, lam, I_trans_max, I_trans = compute_rotor_momentum(J, omega_desired, INERTIA_RATIO_MIN)

    actual_ratio = lam / I_trans_max
    print(f"\nRotor momentum (dynamic balance + superspin):")
    print(f"  h = [{h[0]:+.6e}, {h[1]:+.6e}, {h[2]:+.6e}] kg·m²/s")
    print(f"  |h| = {np.linalg.norm(h):.6e} kg·m²/s")
    print(f"  Effective spin-axis inertia (λ):     {lam:.6e} kg·m²")
    print(f"  Max transverse effective inertia:     {I_trans_max:.6e} kg·m²")
    print(f"  Transverse eigenvalues:               [{I_trans[0]:.6e}, {I_trans[1]:.6e}]")
    print(f"  Inertia ratio (λ / I_trans_max):      {actual_ratio:.4f}  (required ≥ {INERTIA_RATIO_MIN})")

    # Verify dynamic balance: ω × (J·ω + h) should be zero
    residual = np.cross(omega_desired, J @ omega_desired + h)
    print(f"  Dynamic balance check |ω × (Jω + h)| = {np.linalg.norm(residual):.2e}")

    # --- Step 4-5: Simulate ---
    # Nutation period estimate: T_nut ≈ 2π / |ω_spin| * I_spin / |I_spin - I_trans|
    I_spin = lam
    I_trans_avg = np.mean(I_trans)
    T_nutation = 2.0 * np.pi / OMEGA_RAD_S * I_spin / abs(I_spin - I_trans_avg)
    print(f"\n  Estimated nutation period: {T_nutation:.2f} s")

    t_total = max(10.0 * T_nutation, 120.0)  # at least 10 nutation periods or 2 min
    print(f"  Simulation duration: {t_total:.1f} s ({t_total/60:.1f} min)")

    # Identity quaternion (body aligned with inertial at t=0)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])

    # Case A: unperturbed ICs (exact equilibrium)
    print(f"\n--- Case A: Unperturbed IC (exact equilibrium) ---")
    result_exact = run_simulation(J, h, omega_desired.copy(), q0.copy(), t_total, "Exact IC")

    # Case B: perturbed ICs — 1% transverse kick
    perturb_frac = 0.01
    omega_pert = omega_desired.copy()
    omega_pert[0] += perturb_frac * OMEGA_RAD_S
    omega_pert[1] += perturb_frac * OMEGA_RAD_S
    print(f"--- Case B: Perturbed IC (1% transverse kick) ---")
    print(f"  ω₀ = [{omega_pert[0]:.6f}, {omega_pert[1]:.6f}, {omega_pert[2]:.6f}] rad/s")
    result_pert1 = run_simulation(J, h, omega_pert.copy(), q0.copy(), t_total, "1% perturbation")

    # Case C: 5% transverse kick
    perturb_frac_5 = 0.05
    omega_pert5 = omega_desired.copy()
    omega_pert5[0] += perturb_frac_5 * OMEGA_RAD_S
    omega_pert5[1] += perturb_frac_5 * OMEGA_RAD_S
    print(f"--- Case C: Perturbed IC (5% transverse kick) ---")
    print(f"  ω₀ = [{omega_pert5[0]:.6f}, {omega_pert5[1]:.6f}, {omega_pert5[2]:.6f}] rad/s")
    result_pert5 = run_simulation(J, h, omega_pert5.copy(), q0.copy(), t_total, "5% perturbation")

    # Case D: no rotor (h=0) with perturbed IC — should be unstable
    print(f"--- Case D: No rotor (h=0) with 1% perturbation ---")
    result_no_rotor = run_simulation(J, np.zeros(3), omega_pert.copy(), q0.copy(), t_total, "No rotor, 1% pert")

    # --- Plotting ---
    plot_dir = pathlib.Path(__file__).parent / "plots"
    plot_dir.mkdir(exist_ok=True)

    sun_eci = np.array([1.0, 0.0, 0.0])  # sun along +X in ECI

    # ---- Figure 1: Angular velocity components ----
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig1.suptitle(
        "Safe Mode Gyrostat — Angular Velocity Components (Body Frame)\n"
        f"Perturbed ISS inertia, 10 RPM about solar panel normal (+Z), "
        f"inertia ratio = {actual_ratio:.2f}",
        fontsize=11,
    )
    comp_labels = ["ωx", "ωy", "ωz"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ax, result in zip(axes1.flat, [result_exact, result_pert1, result_pert5, result_no_rotor]):
        t = result["times"]
        w = result["omega"]
        for k in range(3):
            ax.plot(t, w[:, k], label=comp_labels[k], color=colors[k], linewidth=0.6)
        ax.set_title(result["label"], fontsize=10)
        ax.set_ylabel("ω (rad/s)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes1[1, 0].set_xlabel("Time (s)")
    axes1[1, 1].set_xlabel("Time (s)")
    plt.tight_layout()
    out1 = plot_dir / "safe_mode_omega.png"
    fig1.savefig(out1, dpi=200, bbox_inches="tight")
    print(f"\nPlot saved: {out1}")

    # ---- Figure 2: Pointing error ----
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig2.suptitle(
        "Safe Mode Gyrostat — Solar Panel Pointing Error\n"
        f"Angle between panel normal and sun direction (+X ECI)",
        fontsize=11,
    )

    for ax, result in zip(axes2.flat, [result_exact, result_pert1, result_pert5, result_no_rotor]):
        t = result["times"]
        errors = np.array([compute_pointing_error(result["quat"][i], sun_eci) for i in range(len(t))])
        ax.plot(t, errors, color="#d62728", linewidth=0.6)
        ax.set_title(result["label"], fontsize=10)
        ax.set_ylabel("Pointing error (deg)")
        ax.grid(True, alpha=0.3)

    axes2[1, 0].set_xlabel("Time (s)")
    axes2[1, 1].set_xlabel("Time (s)")
    plt.tight_layout()
    out2 = plot_dir / "safe_mode_pointing.png"
    fig2.savefig(out2, dpi=200, bbox_inches="tight")
    print(f"Plot saved: {out2}")

    # ---- Figure 3: Quaternion components ----
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig3.suptitle(
        "Safe Mode Gyrostat — Attitude Quaternion Components\n"
        f"q = [q₀, q₁, q₂, q₃] (scalar-first)",
        fontsize=11,
    )
    q_labels = ["q₀", "q₁", "q₂", "q₃"]
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
    out3 = plot_dir / "safe_mode_quaternion.png"
    fig3.savefig(out3, dpi=200, bbox_inches="tight")
    print(f"Plot saved: {out3}")

    # ---- Figure 4: Transverse rate magnitude (stability metric) ----
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.set_title(
        "Transverse Angular Velocity Magnitude\n"
        "√(ωx² + ωy²) — should remain bounded for stable gyrostat",
        fontsize=11,
    )

    for result in [result_exact, result_pert1, result_pert5, result_no_rotor]:
        t = result["times"]
        w = result["omega"]
        w_trans = np.sqrt(w[:, 0]**2 + w[:, 1]**2)
        ax4.plot(t, w_trans, label=result["label"], linewidth=0.8)

    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("|ω_transverse| (rad/s)")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    out4 = plot_dir / "safe_mode_transverse.png"
    fig4.savefig(out4, dpi=200, bbox_inches="tight")
    print(f"Plot saved: {out4}")

    plt.close("all")
    print(f"\nAll plots saved to {plot_dir}/")


if __name__ == "__main__":
    main()
