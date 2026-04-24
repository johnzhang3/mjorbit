"""HW2 Section 4 — Static Attitude Estimation (Wahba's Problem).

Performs static attitude estimation from simulated sensor vector observations
by solving Wahba's problem using two algorithms:

  1. **q-method** (Davenport) — eigenvalue-based optimal quaternion
  2. **Convex relaxation** (SDP) — semidefinite programming via CVXPY

Monte Carlo comparison (N=5000 trials) with randomly generated measurements.
Compares accuracy (rotation error in arcsec) and computational cost (wall time).

Sensor observations used for each trial:
  - Star tracker:    2 star directions (most accurate)
  - Sun sensor:      1 sun direction
  - Earth horizon:   1 nadir direction
  - Magnetometer:    1 magnetic field direction

Usage:
    pixi run python ISS/hw2/attitude_estimation.py
"""

from __future__ import annotations

import pathlib
import time as pytime

import matplotlib

matplotlib.use("Agg")
import cvxpy as cp
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from attitude_sensors import (
    HORIZON_SENSOR_RAD,
    STAR_CROSS_RAD,
    SUN_SENSOR_RAD,
    magnetometer_direction_variance,
    measure_horizon_sensor,
    measure_magnetometer,
    measure_star_tracker,
    measure_sun_sensor,
)
from common import (
    J_NOMINAL,
    OMEGA_RAD_S,
    SOLAR_NORMAL,
    build_model_data,
    compute_rotor_momentum,
    perturb_inertia,
    set_sun_pointing_attitude,
)

from mujoco_orbit import mjo_forward, mjo_step

np.random.seed(42)
plt.rcParams.update({
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 16,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


# ===================================================================
# WAHBA'S PROBLEM
# ===================================================================
#
# Given n vector observations b_i (body frame) and reference directions
# r_i (inertial/ECI frame), find the rotation matrix R that minimizes:
#
#     L(R) = sum_i  a_i * || b_i - R * r_i ||^2
#
# where a_i = 1 / sigma_i^2 are the observation weights.
#
# Equivalently, maximize:
#
#     J(R) = sum_i  a_i * b_i^T * R * r_i  =  tr(R * B^T)
#
# where B = sum_i  a_i * b_i * r_i^T  is the 3x3 attitude profile matrix.
#
# ===================================================================


def build_attitude_profile_matrix(
    body_vecs: list[np.ndarray],
    ref_vecs: list[np.ndarray],
    weights: list[float],
) -> np.ndarray:
    """Build Wahba's attitude profile matrix B = sum a_i * b_i * r_i^T.

    Weights are normalized to sum to 1 for numerical stability.
    This does not affect the optimal rotation.
    """
    w = np.array(weights)
    w = w / w.sum()
    B = np.zeros((3, 3))
    for b, r, a in zip(body_vecs, ref_vecs, w):
        B += a * np.outer(b, r)
    return B


# -------------------------------------------------------------------
# Algorithm 1: q-method (Davenport's eigenvalue method)
# -------------------------------------------------------------------
#
# Converts Wahba's cost to a quadratic form in the quaternion q:
#     J(q) = q^T K q
#
# where K is the 4x4 Davenport matrix built from the profile matrix B.
# The optimal quaternion is the eigenvector of K corresponding to the
# maximum eigenvalue.
#
# Computational complexity: O(1) for 4x4 eigenvalue problem (after
# building B, which is O(n)).  Very fast.
# -------------------------------------------------------------------

def q_method(
    body_vecs: list[np.ndarray],
    ref_vecs: list[np.ndarray],
    weights: list[float],
) -> np.ndarray:
    """Solve Wahba's problem via Davenport's q-method.

    Args:
        body_vecs: list of unit vectors in body frame.
        ref_vecs: list of unit vectors in reference (ECI) frame.
        weights: list of scalar weights a_i = 1/sigma_i^2.

    Returns:
        q_opt: optimal quaternion [w, x, y, z] (scalar-first).
    """
    B = build_attitude_profile_matrix(body_vecs, ref_vecs, weights)

    # Build Davenport matrix K (4x4)
    S = B + B.T
    sigma = np.trace(B)
    Z = np.array([B[1, 2] - B[2, 1],
                  B[2, 0] - B[0, 2],
                  B[0, 1] - B[1, 0]])

    K = np.zeros((4, 4))
    K[0, 0] = sigma
    K[0, 1:] = Z
    K[1:, 0] = Z
    K[1:, 1:] = S - sigma * np.eye(3)

    # Maximum eigenvalue eigenvector
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    q_opt = eigenvectors[:, -1]  # eigenvector for largest eigenvalue

    # Ensure scalar-first and positive scalar
    if q_opt[0] < 0:
        q_opt = -q_opt

    return q_opt


# -------------------------------------------------------------------
# Algorithm 2: Convex optimization (SDP relaxation)
# -------------------------------------------------------------------
#
# Wahba's problem can be written as:
#     maximize   tr(R * B^T)
#     subject to R^T R = I,  det(R) = +1
#
# The orthogonality constraint is non-convex, but we can relax it
# to a semidefinite program (SDP). We use the quaternion formulation:
#
#     maximize   q^T K q
#     subject to ||q||^2 = 1
#
# This is equivalent to maximizing tr(K * Q) where Q = q*q^T is rank-1
# PSD. The SDP relaxation drops the rank-1 constraint:
#
#     maximize   tr(K * Q)
#     subject to Q >= 0,  tr(Q) = 1
#
# For Wahba's problem this relaxation is known to be tight (the
# optimal Q is always rank-1), so the SDP gives the exact solution.
#
# Computational complexity: O(n^3.5) for interior-point SDP solvers
# with n=4, so practically O(1) but with higher constant than q-method.
# -------------------------------------------------------------------

def wahba_sdp(
    body_vecs: list[np.ndarray],
    ref_vecs: list[np.ndarray],
    weights: list[float],
) -> np.ndarray:
    """Solve Wahba's problem via SDP relaxation.

    Args:
        body_vecs: list of unit vectors in body frame.
        ref_vecs: list of unit vectors in reference (ECI) frame.
        weights: list of scalar weights a_i = 1/sigma_i^2.

    Returns:
        q_opt: optimal quaternion [w, x, y, z] (scalar-first).
    """
    B = build_attitude_profile_matrix(body_vecs, ref_vecs, weights)

    # Build Davenport matrix K
    S = B + B.T
    sigma = np.trace(B)
    Z = np.array([B[1, 2] - B[2, 1],
                  B[2, 0] - B[0, 2],
                  B[0, 1] - B[1, 0]])

    K = np.zeros((4, 4))
    K[0, 0] = sigma
    K[0, 1:] = Z
    K[1:, 0] = Z
    K[1:, 1:] = S - sigma * np.eye(3)

    # SDP: maximize tr(K @ Q), s.t. Q >> 0, tr(Q) = 1
    Q = cp.Variable((4, 4), symmetric=True)
    constraints = [Q >> 0, cp.trace(Q) == 1]
    prob = cp.Problem(cp.Maximize(cp.trace(K @ Q)), constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    Q_val = Q.value
    if Q_val is None:
        # Fallback: if solver fails, use q-method
        return q_method(body_vecs, ref_vecs, weights)

    # Extract quaternion from rank-1 solution
    eigenvalues, eigenvectors = np.linalg.eigh(Q_val)
    q_opt = eigenvectors[:, -1] * np.sqrt(max(eigenvalues[-1], 0.0))
    norm = np.linalg.norm(q_opt)
    if norm < 1e-10:
        return q_method(body_vecs, ref_vecs, weights)
    q_opt /= norm

    if q_opt[0] < 0:
        q_opt = -q_opt

    return q_opt


# -------------------------------------------------------------------
# Attitude error metric
# -------------------------------------------------------------------

def rotation_error_rad(q_est: np.ndarray, q_true: np.ndarray) -> float:
    """Total rotation angle between two quaternions, in radians."""
    q_true_inv = np.array([q_true[0], -q_true[1], -q_true[2], -q_true[3]])
    w1, x1, y1, z1 = q_est
    w2, x2, y2, z2 = q_true_inv
    dw = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return 2.0 * np.arccos(np.clip(abs(dw), 0.0, 1.0))


# ===================================================================
# MAIN — Monte Carlo comparison
# ===================================================================

def main() -> None:
    print("HW2 Section 4 — Static Attitude Estimation (Wahba's Problem)")
    print("=" * 65)

    # ---- Build scenario ----
    J = perturb_inertia(J_NOMINAL)
    omega_desired = OMEGA_RAD_S * SOLAR_NORMAL
    h, lam, I_trans_max, _ = compute_rotor_momentum(J, omega_desired, 1.2)

    dt = 0.002
    model, data, _ = build_model_data(J, h, dt=dt, use_magnetic=True)
    bid = model.body_id("iss")

    sun_eci = data.env.sun_vector_eci.copy()
    set_sun_pointing_attitude(model, data, sun_eci)
    data.qvel[3:6] = omega_desired
    mjo_forward(model, data)

    # Run briefly to settle
    for _ in range(100):
        mjo_step(model, data)
    mjo_forward(model, data)

    # ---- Reference vectors (inertial/ECI) ----
    R_eci_sc = data.orbit.R_eci
    nadir_eci = -R_eci_sc / np.linalg.norm(R_eci_sc)
    B_eci = data.env.mag_field_eci
    sun_eci = data.env.sun_vector_eci
    B_mag = np.linalg.norm(B_eci)

    # Two catalog stars
    stars_eci = []
    for ra_deg, dec_deg in [(96.0, -53.0), (213.0, 19.0)]:  # Canopus, Arcturus
        ra, dec = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
        stars_eci.append(np.array([
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec)]))

    # True body rotation
    R_wb = data.xmat[bid].reshape(3, 3).copy()
    C_IL = data.frame.C_IL
    R_eci_body = C_IL @ R_wb  # ECI-from-body
    q_true = np.zeros(4)
    mujoco.mju_mat2Quat(q_true, R_eci_body.flatten())
    if q_true[0] < 0:
        q_true = -q_true

    print(f"True attitude quaternion: [{q_true[0]:.6f}, {q_true[1]:.6f}, "
          f"{q_true[2]:.6f}, {q_true[3]:.6f}]")

    # ---- Observation weights for Wahba's problem ----
    # sigma^2 for each sensor type (angular variance per axis)
    sigma2_star = STAR_CROSS_RAD**2
    sigma2_sun = SUN_SENSOR_RAD**2
    sigma2_nadir = HORIZON_SENSOR_RAD**2
    sigma2_mag = magnetometer_direction_variance(B_mag)

    print("\nObservation weights (1/sigma^2):")
    print(
        "  Star tracker:  "
        f"{1 / sigma2_star:.3e} "
        f"(sigma = {np.rad2deg(STAR_CROSS_RAD) * 3600:.1f} arcsec)"
    )
    print(f"  Sun sensor:    {1/sigma2_sun:.3e} (sigma = {SUN_SENSOR_DEG:.3f} deg)")
    print(f"  Horizon:       {1/sigma2_nadir:.3e} (sigma = {HORIZON_SENSOR_DEG:.2f} deg)")
    print(
        "  Magnetometer:  "
        f"{1 / sigma2_mag:.3e} "
        f"(sigma = {np.rad2deg(np.sqrt(sigma2_mag)):.3f} deg)"
    )

    # ---- Monte Carlo ----
    N_MC = 5000
    rng = np.random.default_rng(seed=99)

    errors_q = np.zeros(N_MC)
    errors_sdp = np.zeros(N_MC)
    time_q = 0.0
    time_sdp = 0.0

    print(f"\nRunning Monte Carlo (N = {N_MC})...")

    for i in range(N_MC):
        # Generate noisy observations
        body_vecs = []
        ref_vecs = []
        weights = []

        # Two star observations
        for star in stars_eci:
            b, _ = measure_star_tracker(R_wb, C_IL, star, rng)
            body_vecs.append(b)
            ref_vecs.append(star)
            weights.append(1.0 / sigma2_star)

        # Sun sensor
        s_meas, _ = measure_sun_sensor(R_wb, C_IL, sun_eci, rng)
        body_vecs.append(s_meas)
        ref_vecs.append(sun_eci / np.linalg.norm(sun_eci))
        weights.append(1.0 / sigma2_sun)

        # Earth horizon sensor
        n_meas, _ = measure_horizon_sensor(R_wb, C_IL, nadir_eci, rng)
        body_vecs.append(n_meas)
        ref_vecs.append(nadir_eci / np.linalg.norm(nadir_eci))
        weights.append(1.0 / sigma2_nadir)

        # Magnetometer (direction only)
        B_meas, _ = measure_magnetometer(R_wb, C_IL, B_eci, rng)
        B_meas_hat = B_meas / np.linalg.norm(B_meas)
        B_eci_hat = B_eci / np.linalg.norm(B_eci)
        body_vecs.append(B_meas_hat)
        ref_vecs.append(B_eci_hat)
        weights.append(1.0 / sigma2_mag)

        # q-method
        t0 = pytime.perf_counter()
        q_est_q = q_method(body_vecs, ref_vecs, weights)
        time_q += pytime.perf_counter() - t0
        errors_q[i] = rotation_error_rad(q_est_q, q_true)

        # SDP
        t0 = pytime.perf_counter()
        q_est_sdp = wahba_sdp(body_vecs, ref_vecs, weights)
        time_sdp += pytime.perf_counter() - t0
        errors_sdp[i] = rotation_error_rad(q_est_sdp, q_true)

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{N_MC} trials done...")

    # ---- Results ----
    errors_q_arcsec = np.rad2deg(errors_q) * 3600
    errors_sdp_arcsec = np.rad2deg(errors_sdp) * 3600

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)

    print(f"\n{'Metric':<30} {'q-method':>15} {'SDP':>15}")
    print("-" * 60)
    print(
        f"{'Mean error (arcsec)':<30} "
        f"{np.mean(errors_q_arcsec):>15.2f} "
        f"{np.mean(errors_sdp_arcsec):>15.2f}"
    )
    print(
        f"{'Median error (arcsec)':<30} "
        f"{np.median(errors_q_arcsec):>15.2f} "
        f"{np.median(errors_sdp_arcsec):>15.2f}"
    )
    print(
        f"{'RMS error (arcsec)':<30} "
        f"{np.sqrt(np.mean(errors_q_arcsec**2)):>15.2f} "
        f"{np.sqrt(np.mean(errors_sdp_arcsec**2)):>15.2f}"
    )
    print(
        f"{'Max error (arcsec)':<30} "
        f"{np.max(errors_q_arcsec):>15.2f} "
        f"{np.max(errors_sdp_arcsec):>15.2f}"
    )
    print(f"{'Total time (s)':<30} {time_q:>15.4f} {time_sdp:>15.4f}")
    print(f"{'Time per solve (ms)':<30} {time_q/N_MC*1000:>15.4f} {time_sdp/N_MC*1000:>15.4f}")
    print(f"{'Speedup (SDP/q-method)':<30} {time_sdp/time_q:>15.1f}x {'':>15}")

    # ---- Plotting ----
    plot_dir = pathlib.Path(__file__).parent / "plots"
    plot_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Error histograms
    ax = axes[0]
    bins = np.linspace(0, np.percentile(errors_q_arcsec, 99.5), 60)
    ax.hist(errors_q_arcsec, bins=bins, alpha=0.6, density=True, label="q-method", color="C0")
    ax.hist(errors_sdp_arcsec, bins=bins, alpha=0.6, density=True, label="SDP", color="C1")
    ax.axvline(np.mean(errors_q_arcsec), color="C0", ls="--", lw=1.5,
               label=f"q-method mean = {np.mean(errors_q_arcsec):.1f}\"")
    ax.axvline(np.mean(errors_sdp_arcsec), color="C1", ls="--", lw=1.5,
               label=f"SDP mean = {np.mean(errors_sdp_arcsec):.1f}\"")
    ax.set_xlabel("Attitude error (arcsec)")
    ax.set_ylabel("Density")
    ax.set_title("Attitude Estimation Error Distribution", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. CDF comparison
    ax = axes[1]
    for errors, label, color in [(errors_q_arcsec, "q-method", "C0"),
                                  (errors_sdp_arcsec, "SDP", "C1")]:
        sorted_err = np.sort(errors)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        ax.plot(sorted_err, cdf * 100, label=label, color=color, linewidth=1.5)
    ax.set_xlabel("Attitude error (arcsec)")
    ax.set_ylabel("Cumulative probability (%)")
    ax.set_title("CDF of Attitude Estimation Error", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    # 3. Computational cost bar chart
    ax = axes[2]
    methods = ["q-method", "SDP"]
    times_ms = [time_q / N_MC * 1000, time_sdp / N_MC * 1000]
    bars = ax.bar(methods, times_ms, color=["C0", "C1"], alpha=0.7, edgecolor="black")
    ax.set_ylabel("Time per solve (ms)")
    ax.set_title("Computational Cost", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, t in zip(bars, times_ms):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f"{t:.3f} ms", ha="center", va="bottom", fontsize=10)

    fig.suptitle(f"Wahba's Problem: q-method vs SDP (Monte Carlo, N={N_MC})",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = plot_dir / "attitude_estimation.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nPlot saved: {out_path}")
    plt.close("all")

    print("\nDone.")


# Avoid circular import issues — only use SUN_SENSOR_DEG and
# HORIZON_SENSOR_DEG from attitude_sensors for printing
SUN_SENSOR_DEG = np.rad2deg(SUN_SENSOR_RAD)
HORIZON_SENSOR_DEG = np.rad2deg(HORIZON_SENSOR_RAD)


if __name__ == "__main__":
    main()
