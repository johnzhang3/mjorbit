"""ISS angular momentum sphere (Poinsot construction).

Plots trajectories of the angular momentum vector on the constant-|L| sphere
for the ISS inertia tensor. The six equilibrium points (±X, ±Y, ±Z) are
shown, with trajectories at various energy levels demonstrating:

  - Stable equilibria at major axis (±Z, Izz max) and minor axis (±Y, Iyy min)
  - Unstable equilibria at intermediate axis (±X, Ixx) — separatrix

The trajectories are computed by integrating Euler's torque-free rigid body
equations directly (no mjorbit needed — this is a purely analytical result).

Usage:
    pixi run python ISS/hw1/iss_momentum_sphere.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# ISS principal moments of inertia
ISS_IXX = 128e6  # kg·m² — intermediate axis (UNSTABLE)
ISS_IYY = 107e6  # kg·m² — minor axis (minimum, stable)
ISS_IZZ = 201e6  # kg·m² — major axis (maximum, stable)

# Reference angular momentum magnitude (10 RPM about Z)
OMEGA_REF = 10.0 * 2.0 * np.pi / 60.0
L_MAG = ISS_IZZ * OMEGA_REF


def euler_equations(t: float, state: np.ndarray) -> np.ndarray:
    """Torque-free Euler equations for a rigid body."""
    wx, wy, wz = state
    dwx = (ISS_IYY - ISS_IZZ) / ISS_IXX * wy * wz
    dwy = (ISS_IZZ - ISS_IXX) / ISS_IYY * wz * wx
    dwz = (ISS_IXX - ISS_IYY) / ISS_IZZ * wx * wy
    return np.array([dwx, dwy, dwz])


def integrate_trajectory(omega0: np.ndarray, t_span: float = 600.0) -> np.ndarray:
    """Integrate Euler equations, return angular momentum in body frame."""
    sol = solve_ivp(
        euler_equations,
        [0, t_span],
        omega0,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
        max_step=0.05,
    )
    # Convert to angular momentum
    L = np.zeros((3, len(sol.t)))
    L[0] = ISS_IXX * sol.y[0]
    L[1] = ISS_IYY * sol.y[1]
    L[2] = ISS_IZZ * sol.y[2]
    return L


def make_trajectories() -> list[np.ndarray]:
    """Generate trajectories sampling various energy levels on the |L| sphere."""
    trajectories = []

    # Strategy: parameterize initial conditions on the L-sphere.
    # L = (Lx, Ly, Lz) with |L| = L_MAG.
    # Use spherical coordinates: Lx = L*sin(θ)*cos(φ), Ly = L*sin(θ)*sin(φ), Lz = L*cos(θ)
    # Convert back to ω: ωi = Li/Ii

    # Near Z-axis (major axis, stable) — small circles around pole
    for theta_deg in [5, 15, 30, 45]:
        theta = np.deg2rad(theta_deg)
        for phi_deg in [0, 90, 180, 270]:
            phi = np.deg2rad(phi_deg)
            Lx = L_MAG * np.sin(theta) * np.cos(phi)
            Ly = L_MAG * np.sin(theta) * np.sin(phi)
            Lz = L_MAG * np.cos(theta)
            omega0 = np.array([Lx / ISS_IXX, Ly / ISS_IYY, Lz / ISS_IZZ])
            trajectories.append(integrate_trajectory(omega0, t_span=600.0))

    # Near −Z pole
    for theta_deg in [175, 165, 150, 135]:
        theta = np.deg2rad(theta_deg)
        for phi_deg in [0, 90, 180, 270]:
            phi = np.deg2rad(phi_deg)
            Lx = L_MAG * np.sin(theta) * np.cos(phi)
            Ly = L_MAG * np.sin(theta) * np.sin(phi)
            Lz = L_MAG * np.cos(theta)
            omega0 = np.array([Lx / ISS_IXX, Ly / ISS_IYY, Lz / ISS_IZZ])
            trajectories.append(integrate_trajectory(omega0, t_span=600.0))

    # Near Y-axis (minor axis, stable) — small circles around ±Y poles
    for theta_y_deg in [5, 15, 30, 45]:
        theta_y = np.deg2rad(theta_y_deg)
        for phi_deg in [0, 90, 180, 270]:
            phi = np.deg2rad(phi_deg)
            # Parameterize near +Y: Ly = L*cos(θ), Lx = L*sin(θ)*cos(φ), Lz = L*sin(θ)*sin(φ)
            Ly = L_MAG * np.cos(theta_y)
            Lx = L_MAG * np.sin(theta_y) * np.cos(phi)
            Lz = L_MAG * np.sin(theta_y) * np.sin(phi)
            omega0 = np.array([Lx / ISS_IXX, Ly / ISS_IYY, Lz / ISS_IZZ])
            trajectories.append(integrate_trajectory(omega0, t_span=600.0))

    # Near −Y
    for theta_y_deg in [175, 165, 150, 135]:
        theta_y = np.deg2rad(theta_y_deg)
        for phi_deg in [0, 90, 180, 270]:
            phi = np.deg2rad(phi_deg)
            Ly = L_MAG * np.cos(theta_y)
            Lx = L_MAG * np.sin(theta_y) * np.cos(phi)
            Lz = L_MAG * np.sin(theta_y) * np.sin(phi)
            omega0 = np.array([Lx / ISS_IXX, Ly / ISS_IYY, Lz / ISS_IZZ])
            trajectories.append(integrate_trajectory(omega0, t_span=600.0))

    # Near the separatrix (near ±X, intermediate axis) — these are the unstable ones
    for theta_x_deg in [3, 10, 20, 80, 87]:
        theta_x = np.deg2rad(theta_x_deg)
        for phi_deg in [0, 90, 180, 270]:
            phi = np.deg2rad(phi_deg)
            Lx_val = L_MAG * np.cos(theta_x)
            Ly = L_MAG * np.sin(theta_x) * np.cos(phi)
            Lz = L_MAG * np.sin(theta_x) * np.sin(phi)
            omega0 = np.array([Lx_val / ISS_IXX, Ly / ISS_IYY, Lz / ISS_IZZ])
            trajectories.append(integrate_trajectory(omega0, t_span=800.0))

    # Near −X
    for theta_x_deg in [177, 170, 160, 100, 93]:
        theta_x = np.deg2rad(theta_x_deg)
        for phi_deg in [0, 90, 180, 270]:
            phi = np.deg2rad(phi_deg)
            Lx_val = L_MAG * np.cos(theta_x)
            Ly = L_MAG * np.sin(theta_x) * np.cos(phi)
            Lz = L_MAG * np.sin(theta_x) * np.sin(phi)
            omega0 = np.array([Lx_val / ISS_IXX, Ly / ISS_IYY, Lz / ISS_IZZ])
            trajectories.append(integrate_trajectory(omega0, t_span=800.0))

    return trajectories


def main() -> None:
    print("Generating ISS momentum sphere trajectories...")
    trajectories = make_trajectories()
    print(f"  {len(trajectories)} trajectories computed")

    plot_dir = pathlib.Path(__file__).parent / "plots"
    plot_dir.mkdir(exist_ok=True)

    # --- 3D Momentum Sphere ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Draw wireframe sphere
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = L_MAG * np.outer(np.cos(u), np.sin(v))
    ys = L_MAG * np.outer(np.sin(u), np.sin(v))
    zs = L_MAG * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, alpha=0.05, color="lightgray")

    # Classify trajectories by region for coloring
    # Color based on which stable equilibrium they're closest to
    for traj in trajectories:
        # Normalize for plotting
        Ln = traj / L_MAG  # unit sphere

        # Determine color based on initial point proximity
        L0 = traj[:, 0]
        near_z = abs(L0[2]) / L_MAG  # how close to ±Z (major, stable)
        near_y = abs(L0[1]) / L_MAG  # how close to ±Y (minor, stable)

        if near_z > 0.7:
            color = "#2ca02c"  # green — near Z (major axis, stable)
            alpha = 0.6
        elif near_y > 0.7:
            color = "#1f77b4"  # blue — near Y (minor axis, stable)
            alpha = 0.6
        else:
            color = "#ff7f0e"  # orange — separatrix region (near X, intermediate)
            alpha = 0.4

        ax.plot(Ln[0], Ln[1], Ln[2], color=color, alpha=alpha, linewidth=0.5)

    # Plot equilibrium points
    eq_size = 100
    # Major axis ±Z (stable)
    ax.scatter([0, 0], [0, 0], [1, -1], color="#2ca02c", s=eq_size,
               zorder=5, edgecolors="black", linewidths=0.5)
    # Minor axis ±Y (stable)
    ax.scatter([0, 0], [1, -1], [0, 0], color="#1f77b4", s=eq_size,
               zorder=5, edgecolors="black", linewidths=0.5)
    # Intermediate axis ±X (unstable)
    ax.scatter([1, -1], [0, 0], [0, 0], color="#d62728", s=eq_size,
               marker="x", zorder=5, linewidths=2)

    # Labels
    offset = 1.15
    ax.text(0, 0, offset, "+Z (major)\nSTABLE", ha="center", fontsize=8, color="#2ca02c")
    ax.text(0, 0, -offset, "−Z (major)\nSTABLE", ha="center", fontsize=8, color="#2ca02c")
    ax.text(offset, 0, 0, "+X (inter.)\nUNSTABLE", ha="center", fontsize=8, color="#d62728")
    ax.text(-offset, 0, 0, "−X (inter.)\nUNSTABLE", ha="center", fontsize=8, color="#d62728")
    ax.text(0, offset, 0, "+Y (minor)\nSTABLE", ha="center", fontsize=8, color="#1f77b4")
    ax.text(0, -offset, 0, "−Y (minor)\nSTABLE", ha="center", fontsize=8, color="#1f77b4")

    ax.set_xlabel("Lx / |L|")
    ax.set_ylabel("Ly / |L|")
    ax.set_zlabel("Lz / |L|")
    ax.set_title(
        "ISS Angular Momentum Sphere (Poinsot Construction)\n"
        f"Ixx={ISS_IXX:.0e}, Iyy={ISS_IYY:.0e}, Izz={ISS_IZZ:.0e} kg·m²\n"
        f"|L| = {L_MAG:.2e} kg·m²/s",
        fontsize=11,
    )

    # Equal aspect ratio
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(-1.3, 1.3)
    ax.set_box_aspect([1, 1, 1])

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#2ca02c", linewidth=2, label="Near major axis (Z) — stable"),
        Line2D([0], [0], color="#1f77b4", linewidth=2, label="Near minor axis (Y) — stable"),
        Line2D([0], [0], color="#ff7f0e", linewidth=2, label="Near separatrix (X) — unstable"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c",
               markersize=8, label="Stable equilibrium"),
        Line2D([0], [0], marker="x", color="#d62728", linewidth=0,
               markersize=8, markeredgewidth=2, label="Unstable equilibrium"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

    ax.view_init(elev=25, azim=135)

    out_path = plot_dir / "iss_momentum_sphere.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"3D plot saved to {out_path}")

    # --- 2D projections ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    proj_labels = [
        ("Lx / |L|", "Lz / |L|", 0, 2, "View along Y-axis (front)"),
        ("Ly / |L|", "Lz / |L|", 1, 2, "View along X-axis (side)"),
        ("Lx / |L|", "Ly / |L|", 0, 1, "View along Z-axis (top)"),
    ]

    for ax2, (xlabel, ylabel, i, j, title) in zip(axes2, proj_labels):
        # Draw unit circle
        theta = np.linspace(0, 2 * np.pi, 200)
        ax2.plot(np.cos(theta), np.sin(theta), "k-", linewidth=0.5, alpha=0.3)

        for traj in trajectories:
            Ln = traj / L_MAG
            L0 = traj[:, 0]
            near_z = abs(L0[2]) / L_MAG
            near_y = abs(L0[1]) / L_MAG
            if near_z > 0.7:
                color, alpha = "#2ca02c", 0.5
            elif near_y > 0.7:
                color, alpha = "#1f77b4", 0.5
            else:
                color, alpha = "#ff7f0e", 0.4
            ax2.plot(Ln[i], Ln[j], color=color, alpha=alpha, linewidth=0.5)

        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.set_title(title, fontsize=10)
        ax2.set_aspect("equal")
        ax2.set_xlim(-1.3, 1.3)
        ax2.set_ylim(-1.3, 1.3)
        ax2.grid(True, alpha=0.2)

    fig2.suptitle(
        "ISS Momentum Sphere — 2D Projections\n"
        "Green: stable (major axis Z), Blue: stable (minor axis Y), "
        "Orange: separatrix (intermediate axis X)",
        fontsize=11,
    )
    plt.tight_layout()

    out_path2 = plot_dir / "iss_momentum_sphere_projections.png"
    fig2.savefig(out_path2, dpi=200, bbox_inches="tight")
    print(f"2D projections saved to {out_path2}")
    plt.close("all")


if __name__ == "__main__":
    main()
