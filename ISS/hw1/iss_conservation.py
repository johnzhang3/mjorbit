"""ISS conservation sanity check — torque-free multi-orbit simulation.

With drag, SRP, and magnetic effects disabled, the only coupling force is
the gravity gradient (translational only, no torque on a single rigid body
at the LVLH origin). Angular momentum and rotational kinetic energy should
be conserved to machine precision.

Runs for 3 full orbits (~277 min) with the ISS spinning at 10 RPM about
the major axis (Z).

Usage:
    pixi run python ISS/hw1/iss_conservation.py
"""

from __future__ import annotations

import pathlib
import time as pytime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mujoco_orbit import MjoData, MjoModel, OrbitInit, mjo_forward, mjo_step
from mujoco_orbit.constants import GM_EARTH, R_EARTH
from mujoco_orbit.orbit.elements import keplerian_to_cartesian

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

ISS_IXX = 128e6  # kg·m²
ISS_IYY = 107e6  # kg·m²
ISS_IZZ = 201e6  # kg·m²
INERTIAS = np.array([ISS_IXX, ISS_IYY, ISS_IZZ])

ALT_KM = 410.0
INC_DEG = 51.6
MODEL_XML = str(pathlib.Path(__file__).resolve().parents[1] / "iss_model.xml")

OMEGA_REF = 10.0 * 2.0 * np.pi / 60.0  # 10 RPM in rad/s


def main() -> None:
    a_km = R_EARTH + ALT_KM
    orbital_period = 2.0 * np.pi * np.sqrt(a_km**3 / GM_EARTH)  # seconds
    n_orbits = 3
    t_total = n_orbits * orbital_period

    R_eci, V_eci = keplerian_to_cartesian(
        a=a_km, e=0.0, inc=np.deg2rad(INC_DEG),
        raan=0.0, argp=0.0, nu=0.0,
    )

    dt = 0.01  # s
    model = MjoModel.from_xml_path(
        MODEL_XML,
        mj_timestep=dt,
        surfaces=[],
        use_j2=True,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    data = MjoData(model, orbit=OrbitInit(R_eci=R_eci, V_eci=V_eci))

    # Initial spin: 10 RPM about Z (major axis)
    data.qvel[3] = 0.0
    data.qvel[4] = 0.0
    data.qvel[5] = OMEGA_REF
    mjo_forward(model, data)

    n_steps = int(t_total / dt)
    record_every = int(1.0 / dt)  # record at 1 Hz
    n_records = n_steps // record_every + 1

    times = np.zeros(n_records)
    L_world = np.zeros((n_records, 3))   # angular momentum in LVLH/world frame
    L_body = np.zeros((n_records, 3))    # angular momentum in body frame
    L_mag = np.zeros(n_records)          # |L|
    T_rot = np.zeros(n_records)          # rotational kinetic energy
    orbit_alt = np.zeros(n_records)
    orbit_energy = np.zeros(n_records)   # specific orbital energy

    bid = model.body_id("iss")

    def record(idx: int, t: float) -> None:
        times[idx] = t

        # Angular velocity in world (LVLH) frame
        w_body = data.qvel[3:6].copy()
        R_wb = data.xmat[bid].reshape(3, 3)

        # Angular momentum in body frame: L_body = I * w_body
        Lb = INERTIAS * w_body
        L_body[idx] = Lb
        L_mag[idx] = np.linalg.norm(Lb)

        # Angular momentum in world frame
        L_world[idx] = R_wb @ Lb

        # Rotational KE
        T_rot[idx] = 0.5 * np.dot(INERTIAS * w_body, w_body)

        # Orbit
        r = np.linalg.norm(data.orbit.R_eci)
        v = np.linalg.norm(data.orbit.V_eci)
        orbit_alt[idx] = r - R_EARTH
        orbit_energy[idx] = 0.5 * v**2 - GM_EARTH / r  # km²/s²

    record(0, 0.0)

    print("ISS Conservation Check — Torque-Free Multi-Orbit")
    print("=" * 60)
    print(f"Orbit:      {ALT_KM:.0f} km, {INC_DEG}° inc, period = {orbital_period:.1f} s")
    print(f"Duration:   {n_orbits} orbits = {t_total:.1f} s ({t_total/60:.1f} min)")
    print(f"Timestep:   {dt} s ({n_steps:,} steps)")
    print(f"Initial ω:  10 RPM about Z = {OMEGA_REF:.4f} rad/s")
    print("Env forces: drag=OFF, SRP=OFF, magnetic=OFF, J2=ON")
    print("=" * 60)

    wall_t0 = pytime.perf_counter()
    rec_idx = 1
    for i in range(n_steps):
        mjo_step(model, data)
        if (i + 1) % record_every == 0:
            record(rec_idx, (i + 1) * dt)
            rec_idx += 1
    wall_elapsed = pytime.perf_counter() - wall_t0
    n_rec = rec_idx

    print(f"\nSimulation completed in {wall_elapsed:.1f}s wall time")

    # --- Conservation metrics ---
    L0 = L_mag[0]
    T0 = T_rot[0]
    E0 = orbit_energy[0]

    dL_rel = (L_mag[:n_rec] - L0) / L0
    dT_rel = (T_rot[:n_rec] - T0) / T0
    dE_rel = (orbit_energy[:n_rec] - E0) / abs(E0)

    print("\nInitial values:")
    print(f"  |L|   = {L0:.6e} kg·m²/s")
    print(f"  T_rot = {T0:.6e} J")
    print(f"  E_orb = {E0:.6e} km²/s²")

    print(f"\nConservation over {n_orbits} orbits:")
    print(f"  |L| max relative drift:    {np.max(np.abs(dL_rel)):.3e}")
    print(f"  T_rot max relative drift:  {np.max(np.abs(dT_rel)):.3e}")
    print(f"  E_orb max relative drift:  {np.max(np.abs(dE_rel)):.3e}")
    print(f"  Altitude drift:            {orbit_alt[n_rec-1] - orbit_alt[0]:.6f} km")

    threshold = 1e-6
    L_ok = np.max(np.abs(dL_rel)) < threshold
    T_ok = np.max(np.abs(dT_rel)) < threshold

    print(f"\n{'=' * 60}")
    if L_ok:
        print(f"PASS: |L| conserved within {threshold:.0e}")
    else:
        print(f"FAIL: |L| drift {np.max(np.abs(dL_rel)):.3e} exceeds {threshold:.0e}")
    if T_ok:
        print(f"PASS: T_rot conserved within {threshold:.0e}")
    else:
        print(f"FAIL: T_rot drift {np.max(np.abs(dT_rel)):.3e} exceeds {threshold:.0e}")

    # --- Plotting ---
    plot_dir = pathlib.Path(__file__).parent / "plots"
    plot_dir.mkdir(exist_ok=True)

    t_min = times[:n_rec] / 60.0  # minutes
    orbit_marks = [i * orbital_period / 60.0 for i in range(n_orbits + 1)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"ISS Conservation Check — {n_orbits} Orbits, No Environmental Torques\n"
        f"10 RPM about major axis (Z), dt = {dt}s",
        fontsize=12,
    )

    # (a) |L| relative drift
    ax = axes[0, 0]
    ax.plot(t_min, dL_rel, color="#1f77b4", linewidth=0.8)
    for m in orbit_marks:
        ax.axvline(m, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.set_ylabel("(|L| - L₀) / L₀")
    ax.set_title("|L| Relative Drift")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax.grid(True, alpha=0.2)

    # (b) T_rot relative drift
    ax = axes[0, 1]
    ax.plot(t_min, dT_rel, color="#2ca02c", linewidth=0.8)
    for m in orbit_marks:
        ax.axvline(m, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.set_ylabel("(T - T₀) / T₀")
    ax.set_title("Rotational KE Relative Drift")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax.grid(True, alpha=0.2)

    # (c) L components in body frame
    ax = axes[1, 0]
    labels = ["Lx (body)", "Ly (body)", "Lz (body)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for k in range(3):
        ax.plot(t_min, L_body[:n_rec, k], label=labels[k], color=colors[k], linewidth=0.5)
    for m in orbit_marks:
        ax.axvline(m, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("L (kg·m²/s)")
    ax.set_title("Angular Momentum Components (Body Frame)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # (d) Orbital energy relative drift
    ax = axes[1, 1]
    ax.plot(t_min, dE_rel, color="#d62728", linewidth=0.8)
    for m in orbit_marks:
        ax.axvline(m, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("(E - E₀) / |E₀|")
    ax.set_title("Specific Orbital Energy Relative Drift")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = plot_dir / "iss_conservation.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.close("all")


if __name__ == "__main__":
    main()
