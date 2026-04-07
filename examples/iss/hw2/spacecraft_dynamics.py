"""HW2 Section 2 — Spacecraft Dynamics: Gyrostat + Orbit coupled simulation.

Adds gyrostat dynamics to the orbit simulation from HW1.  The state now
includes attitude quaternion, angular velocity, and rotor momentum in
addition to orbital position and velocity.

Uses the perturbed inertia and rotor momentum from Section 1 (safe mode).
Simulates the full coupled spacecraft dynamics (orbit + attitude + gyrostat)
using mjorbit's reaction wheel infrastructure.

Sun vector is assumed parallel to +X in ECI coordinates.

Plots:
  - Attitude quaternion components
  - Solar panel normal pointing error (degrees)

Usage:
    uv run python examples/iss/hw2/spacecraft_dynamics.py
"""

from __future__ import annotations

import pathlib
import time as pytime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from mujoco_orbit import step

from common import (
    J_NOMINAL,
    SOLAR_NORMAL,
    OMEGA_RPM,
    OMEGA_RAD_S,
    INERTIA_RATIO_MIN,
    perturb_inertia,
    compute_rotor_momentum,
    quat_to_rotmat,
    build_scenario,
    set_sun_pointing_attitude,
)

np.random.seed(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pointing_error_deg(
    quat_wxyz: np.ndarray, C_IL: np.ndarray, sun_eci: np.ndarray,
) -> float:
    """Angle between solar panel normal (+Z body) and sun direction."""
    R_body_lvlh = quat_to_rotmat(quat_wxyz)
    panel_eci = C_IL @ R_body_lvlh @ SOLAR_NORMAL
    cos_a = np.clip(np.dot(panel_eci, sun_eci), -1, 1)
    return np.rad2deg(np.arccos(abs(cos_a)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("HW2 Section 2 — Spacecraft Dynamics (Gyrostat + Orbit)")
    print("=" * 65)

    # --- Perturb inertia (same RNG state as safe_mode.py) ---
    J = perturb_inertia(J_NOMINAL)
    omega_desired = OMEGA_RAD_S * SOLAR_NORMAL
    h, lam, I_trans_max, _ = compute_rotor_momentum(J, omega_desired, INERTIA_RATIO_MIN)

    print(f"Perturbed inertia (diagonal of eigendecomp, kg*m^2):")
    D_pert = np.linalg.eigvalsh(J)
    for i, val in enumerate(D_pert):
        print(f"  I_{i+1} = {val:.6e}")
    print(f"Rotor momentum h = [{h[0]:+.4e}, {h[1]:+.4e}, {h[2]:+.4e}] kg*m^2/s")
    print(f"|h| = {np.linalg.norm(h):.4e} kg*m^2/s")
    print(f"Inertia ratio = {lam / I_trans_max:.4f}")

    # --- Build scenario ---
    dt = 0.002
    scenario, rw_speeds = build_scenario(J, h, dt=dt)
    bid = scenario.body_id("iss")

    for i, speed in enumerate(rw_speeds):
        print(f"  RW-{['X','Y','Z'][i]}: speed = {speed:.2f} rad/s "
              f"({speed*60/2/np.pi:.1f} RPM), h = {h[i]:+.4e} kg*m^2/s")

    sun_eci = np.array([1.0, 0.0, 0.0])

    # --- Set initial attitude: panel normal (+Z body) -> sun (+X ECI) ---
    set_sun_pointing_attitude(scenario, sun_eci)

    # --- Set initial angular velocity (perturbed) ---
    perturb_frac = 0.01
    omega0 = omega_desired.copy()
    omega0[0] += perturb_frac * OMEGA_RAD_S
    omega0[1] += perturb_frac * OMEGA_RAD_S
    scenario.mjd.qvel[3:6] = omega0
    mujoco.mj_forward(scenario.mjm, scenario.mjd)
    print(f"\nPerturbed IC: omega_0 = [{omega0[0]:.6f}, {omega0[1]:.6f}, {omega0[2]:.6f}] rad/s")

    # --- Nutation period and sim duration ---
    I_trans_avg = 0.5 * (D_pert[0] + D_pert[1])
    T_nutation = 2.0 * np.pi / OMEGA_RAD_S * lam / abs(lam - I_trans_avg)
    n_nutation_periods = 10
    t_total = n_nutation_periods * T_nutation
    print(f"Nutation period ~ {T_nutation:.2f} s")
    print(f"Simulating {n_nutation_periods} nutation periods = {t_total:.1f} s ({t_total/60:.1f} min)")

    # --- Run simulation ---
    n_steps = int(t_total / dt)
    record_every = int(0.05 / dt)
    n_records = n_steps // record_every + 1

    times = np.zeros(n_records)
    quat_hist = np.zeros((n_records, 4))
    omega_body = np.zeros((n_records, 3))
    pointing_err = np.zeros(n_records)
    rw_speed_hist = np.zeros((n_records, 3))

    def record(idx: int, t: float) -> None:
        times[idx] = t
        q = scenario.mjd.qpos[3:7].copy()
        quat_hist[idx] = q
        omega_body[idx] = scenario.mjd.qvel[3:6].copy()
        C_IL = scenario.frame_cache.C_IL
        pointing_err[idx] = pointing_error_deg(q, C_IL, sun_eci)
        rw_speed_hist[idx] = scenario.actuator_state.rw_speed.copy()

    record(0, 0.0)

    print(f"\nRunning coupled simulation ({n_steps:,} steps, dt={dt}s)...")
    wall_t0 = pytime.perf_counter()

    rec_idx = 1
    for i in range(n_steps):
        step(scenario)
        if (i + 1) % record_every == 0:
            record(rec_idx, (i + 1) * dt)
            rec_idx += 1

    wall_elapsed = pytime.perf_counter() - wall_t0
    n_rec = rec_idx
    print(f"Completed in {wall_elapsed:.1f}s wall time")

    print(f"\nFinal state (t = {times[n_rec-1]:.1f} s):")
    print(f"  q  = [{quat_hist[n_rec-1,0]:+.6f}, {quat_hist[n_rec-1,1]:+.6f}, "
          f"{quat_hist[n_rec-1,2]:+.6f}, {quat_hist[n_rec-1,3]:+.6f}]")
    print(f"  omega  = [{omega_body[n_rec-1,0]:+.6f}, {omega_body[n_rec-1,1]:+.6f}, "
          f"{omega_body[n_rec-1,2]:+.6f}] rad/s")
    print(f"  Pointing error: {pointing_err[n_rec-1]:.4f} deg")
    print(f"  Max pointing error: {np.max(pointing_err[:n_rec]):.4f} deg")

    # --- Plotting ---
    plot_dir = pathlib.Path(__file__).parent / "plots"
    plot_dir.mkdir(exist_ok=True)
    t_plot = times[:n_rec]

    # Figure 1: Attitude quaternion
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    q_labels = ["q0 (w)", "q1 (x)", "q2 (y)", "q3 (z)"]
    q_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for k in range(4):
        ax1.plot(t_plot, quat_hist[:n_rec, k], label=q_labels[k], color=q_colors[k], linewidth=0.8)
    for n in range(n_nutation_periods + 1):
        ax1.axvline(n * T_nutation, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Quaternion component")
    ax1.set_title(f"Attitude Quaternion — Coupled Gyrostat + Orbit Simulation\n"
                  f"Perturbed ISS, 10 RPM about +Z, 1% IC perturbation, T_nut ~ {T_nutation:.1f} s",
                  fontsize=11)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3); ax1.set_ylim(-1.1, 1.1)
    plt.tight_layout()
    fig1.savefig(plot_dir / "spacecraft_dynamics_quaternion.png", dpi=200, bbox_inches="tight")

    # Figure 2: Pointing error
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(t_plot, pointing_err[:n_rec], color="#d62728", linewidth=0.8)
    for n in range(n_nutation_periods + 1):
        ax2.axvline(n * T_nutation, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Pointing error (deg)")
    ax2.set_title(f"Solar Panel Normal Pointing Error\n"
                  f"Angle between body +Z and sun direction (+X ECI), T_nut ~ {T_nutation:.1f} s",
                  fontsize=11)
    ax2.grid(True, alpha=0.3); plt.tight_layout()
    fig2.savefig(plot_dir / "spacecraft_dynamics_pointing.png", dpi=200, bbox_inches="tight")

    # Figure 3: Angular velocity
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    w_labels = ["wx (roll)", "wy (pitch)", "wz (yaw)"]
    w_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for k in range(3):
        ax3.plot(t_plot, omega_body[:n_rec, k], label=w_labels[k], color=w_colors[k], linewidth=0.8)
    for n in range(n_nutation_periods + 1):
        ax3.axvline(n * T_nutation, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("omega (rad/s)")
    ax3.set_title("Angular Velocity (Body Frame) — Coupled Simulation\n"
                  "Nutation period markers shown as dashed lines", fontsize=11)
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3); plt.tight_layout()
    fig3.savefig(plot_dir / "spacecraft_dynamics_omega.png", dpi=200, bbox_inches="tight")

    plt.close("all")
    print(f"\nAll plots saved to {plot_dir}/")


if __name__ == "__main__":
    main()
