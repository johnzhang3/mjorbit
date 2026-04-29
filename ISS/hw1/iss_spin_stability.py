"""ISS spin stability analysis using mjorbit.

Simulates the International Space Station as a single rigid body in LEO
with an initial angular velocity of 10 RPM about the major axis (Z / yaw,
Izz = 201×10⁶ kg·m²).

Spin about the axis of maximum moment of inertia should be stable.
This serves as a sanity check that:
  - MuJoCo attitude dynamics are correct (Euler equations)
  - Gravity gradient torques couple properly from the orbit layer
  - Drag and SRP surface forces produce the expected environmental torques

ISS Parameters (post-assembly-complete):
  Mass:        420,000 kg
  Ixx (roll):  128×10⁶ kg·m²  (along-track)
  Iyy (pitch): 107×10⁶ kg·m²  (orbit-normal)
  Izz (yaw):   201×10⁶ kg·m²  (nadir-pointing)  ← major axis
  Orbit:       ~410 km circular, 51.6° inclination

Usage:
    pixi run python ISS/hw1/iss_spin_stability.py
"""

from __future__ import annotations

import pathlib

import numpy as np

from mujoco_orbit import MjoData, MjoModel, OrbitInit, SurfaceSpec, mjo_forward, mjo_step
from mujoco_orbit.constants import R_EARTH
from tests.mujoco_orbit.reference.orbit.elements import keplerian_to_cartesian

# ---------------------------------------------------------------------------
# ISS physical parameters
# ---------------------------------------------------------------------------

ISS_MASS = 420_000.0  # kg
ISS_IXX = 128e6  # kg·m² (roll, along-track)
ISS_IYY = 107e6  # kg·m² (pitch, orbit-normal)
ISS_IZZ = 201e6  # kg·m² (yaw, nadir-pointing) — major axis

ALT_KM = 410.0  # km
INC_DEG = 51.6  # deg

# MuJoCo model path
MODEL_XML = str(pathlib.Path(__file__).resolve().parents[1] / "iss_model.xml")

# ---------------------------------------------------------------------------
# Surface definitions for drag and SRP
#
# The ISS is decomposed into flat-plate panels.  Each panel has:
#   - body_name:  MuJoCo body it belongs to
#   - center_of_pressure_body:  offset from body COM [m] in body frame
#   - normal_body:  outward-facing unit normal in body frame
#   - area:  panel area [m²]
#   - drag_coeff / srp_coeff
#
# Body frame: X = along-track, Y = cross-track, Z = nadir
# ---------------------------------------------------------------------------


def _make_surfaces() -> list[SurfaceSpec]:
    """Build flat-plate surface model for the ISS."""
    surfaces: list[SurfaceSpec] = []

    # --- Solar Arrays ---
    # 8 wings total, 4 port (−Y) and 4 starboard (+Y).
    # Each wing: ~34 m × 12 m ≈ 375 m².  Normals face ±Z (sun-tracking
    # simplified as fixed nadir/zenith-facing for this analysis).
    sa_area = 375.0  # m² per wing
    sa_y_offsets = [-45.0, -35.0, 35.0, 45.0]  # m, approximate truss positions
    for y_off in sa_y_offsets:
        # +Z facing wing
        surfaces.append(SurfaceSpec(
            body_name="iss",
            center_of_pressure_body=np.array([0.0, y_off, 0.5]),
            normal_body=np.array([0.0, 0.0, 1.0]),
            area=sa_area,
            drag_coeff=2.2,
            srp_coeff=1.8,
        ))
        # −Z facing wing (backside)
        surfaces.append(SurfaceSpec(
            body_name="iss",
            center_of_pressure_body=np.array([0.0, y_off, -0.5]),
            normal_body=np.array([0.0, 0.0, -1.0]),
            area=sa_area,
            drag_coeff=2.2,
            srp_coeff=1.8,
        ))

    # --- Radiator Panels ---
    # 6 radiator panels on the truss, each ~75 m². Normals face ±Z.
    rad_area = 75.0  # m² per panel
    rad_y_offsets = [-25.0, -15.0, 15.0, 20.0, 25.0, 30.0]
    for y_off in rad_y_offsets:
        surfaces.append(SurfaceSpec(
            body_name="iss",
            center_of_pressure_body=np.array([0.0, y_off, 3.0]),
            normal_body=np.array([0.0, 0.0, 1.0]),
            area=rad_area,
            drag_coeff=2.2,
            srp_coeff=1.5,
        ))

    # --- Pressurized Modules (ram-facing) ---
    # Frontal area of module stack: ~180 m². Normals face ±X.
    mod_area = 180.0  # m²
    surfaces.append(SurfaceSpec(
        body_name="iss",
        center_of_pressure_body=np.array([36.5, 0.0, 0.0]),
        normal_body=np.array([1.0, 0.0, 0.0]),
        area=mod_area,
        drag_coeff=2.2,
        srp_coeff=1.8,
    ))
    surfaces.append(SurfaceSpec(
        body_name="iss",
        center_of_pressure_body=np.array([-36.5, 0.0, 0.0]),
        normal_body=np.array([-1.0, 0.0, 0.0]),
        area=mod_area,
        drag_coeff=2.2,
        srp_coeff=1.8,
    ))

    # --- Truss Cross-section (broadside to cross-track) ---
    # Truss projected area per side: ~250 m². Normals face ±Y.
    truss_area = 250.0
    surfaces.append(SurfaceSpec(
        body_name="iss",
        center_of_pressure_body=np.array([0.0, 54.25, 0.0]),
        normal_body=np.array([0.0, 1.0, 0.0]),
        area=truss_area,
        drag_coeff=2.2,
        srp_coeff=1.5,
    ))
    surfaces.append(SurfaceSpec(
        body_name="iss",
        center_of_pressure_body=np.array([0.0, -54.25, 0.0]),
        normal_body=np.array([0.0, -1.0, 0.0]),
        area=truss_area,
        drag_coeff=2.2,
        srp_coeff=1.5,
    ))

    return surfaces


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def main() -> None:
    # ISS orbit (circular LEO)
    a_km = R_EARTH + ALT_KM
    R_eci, V_eci = keplerian_to_cartesian(
        a=a_km, e=0.0, inc=np.deg2rad(INC_DEG),
        raan=0.0, argp=0.0, nu=0.0,
    )

    dt = 0.002  # s — small timestep for accuracy with fast spin
    model = MjoModel.from_xml_path(
        MODEL_XML,
        mj_timestep=dt,
        surfaces=_make_surfaces(),
        use_j2=True,
        use_drag=True,
        use_srp=True,
        use_magnetic=False,
    )
    data = MjoData(model, orbit=OrbitInit(R_eci=R_eci, V_eci=V_eci))

    # --- Set initial angular velocity: 10 RPM about major axis (Z) ---
    omega_rpm = 10.0
    omega_rad_s = omega_rpm * 2.0 * np.pi / 60.0  # ≈ 1.047 rad/s

    # MuJoCo free joint: qpos = [x, y, z, qw, qx, qy, qz]
    #                    qvel = [vx, vy, vz, wx, wy, wz]
    # Angular velocity is in world (LVLH) frame.
    # Body starts aligned with LVLH, so body Z = world Z.
    data.qvel[3] = 0.0
    data.qvel[4] = 0.0
    data.qvel[5] = omega_rad_s
    mjo_forward(model, data)

    # --- Run simulation ---
    t_total = 300.0  # 5 minutes (≈ 50 spin revolutions)
    n_steps = int(t_total / dt)
    record_every = int(0.1 / dt)  # record at 10 Hz

    n_records = n_steps // record_every + 1
    times = np.zeros(n_records)
    omega_body = np.zeros((n_records, 3))  # angular velocity in body frame
    quat_hist = np.zeros((n_records, 4))
    kinetic_rot = np.zeros(n_records)
    orbit_alt = np.zeros(n_records)

    # Helper: world angular velocity → body frame
    def record(idx: int, t: float) -> None:
        times[idx] = t
        quat_hist[idx] = data.qpos[3:7]
        w_body = data.qvel[3:6].copy()
        omega_body[idx] = w_body

        # Rotational kinetic energy: T = 0.5 * (Ixx*wx² + Iyy*wy² + Izz*wz²)
        kinetic_rot[idx] = 0.5 * (
            ISS_IXX * w_body[0] ** 2
            + ISS_IYY * w_body[1] ** 2
            + ISS_IZZ * w_body[2] ** 2
        )

        # Orbit altitude
        r_eci = np.linalg.norm(data.orbit.R_eci)
        orbit_alt[idx] = r_eci - R_EARTH

    record(0, 0.0)

    print("ISS Spin Stability Analysis")
    print("=" * 60)
    print(f"Mass:       {ISS_MASS:,.0f} kg")
    print(f"Ixx (roll): {ISS_IXX:.3e} kg·m²")
    print(f"Iyy (pitch):{ISS_IYY:.3e} kg·m²")
    print(f"Izz (yaw):  {ISS_IZZ:.3e} kg·m²  ← major axis")
    print(f"Orbit:      {ALT_KM:.0f} km circular, {INC_DEG}° inclination")
    print(f"Initial ω:  {omega_rpm} RPM about Z (yaw)")
    print(f"            = {omega_rad_s:.4f} rad/s")
    print(f"Duration:   {t_total:.0f} s ({t_total/60:.1f} min)")
    print(f"Timestep:   {dt} s ({n_steps} steps)")
    print(f"Surfaces:   {len(model.surfaces)} flat-plate panels")
    print("=" * 60)
    print()

    rec_idx = 1
    for i in range(n_steps):
        mjo_step(model, data)
        if (i + 1) % record_every == 0:
            record(rec_idx, (i + 1) * dt)
            rec_idx += 1

    n_rec = rec_idx  # actual number of records

    # --- Analysis ---
    print("Results")
    print("-" * 60)

    # Angular velocity components in body frame
    wx = omega_body[:n_rec, 0]
    wy = omega_body[:n_rec, 1]
    wz = omega_body[:n_rec, 2]

    print("\nAngular velocity (body frame) at t=0:")
    print(f"  ωx = {wx[0]:+.6f} rad/s")
    print(f"  ωy = {wy[0]:+.6f} rad/s")
    print(f"  ωz = {wz[0]:+.6f} rad/s")

    print(f"\nAngular velocity (body frame) at t={t_total:.0f}s:")
    print(f"  ωx = {wx[n_rec-1]:+.6f} rad/s")
    print(f"  ωy = {wy[n_rec-1]:+.6f} rad/s")
    print(f"  ωz = {wz[n_rec-1]:+.6f} rad/s")

    # Stability metrics
    # For major-axis spin, transverse rates (wx, wy) should stay near zero
    max_transverse = max(np.max(np.abs(wx)), np.max(np.abs(wy)))
    wz_mean = np.mean(wz)
    wz_std = np.std(wz)
    wz_drift = abs(wz[-1] - wz[0])

    print("\nStability metrics:")
    print(f"  Max transverse rate (ωx, ωy): {max_transverse:.6e} rad/s")
    print(f"  ωz mean:  {wz_mean:.6f} rad/s")
    print(f"  ωz std:   {wz_std:.6e} rad/s")
    print(f"  ωz drift: {wz_drift:.6e} rad/s over {t_total:.0f}s")

    # Rotational kinetic energy conservation
    T0 = kinetic_rot[0]
    T_final = kinetic_rot[n_rec - 1]
    T_rel_change = abs(T_final - T0) / T0

    print("\nRotational kinetic energy:")
    print(f"  T(0)     = {T0:.6e} J")
    print(f"  T(final) = {T_final:.6e} J")
    print(f"  Relative change: {T_rel_change:.6e}")

    # Orbit altitude
    print("\nOrbit altitude:")
    print(f"  h(0)     = {orbit_alt[0]:.4f} km")
    print(f"  h(final) = {orbit_alt[n_rec-1]:.4f} km")
    print(f"  Δh       = {orbit_alt[n_rec-1] - orbit_alt[0]:.6f} km")

    # Pass/fail criteria
    print(f"\n{'=' * 60}")
    stable = max_transverse < 0.01 * abs(wz_mean)  # transverse < 1% of spin rate
    energy_ok = T_rel_change < 0.01  # energy conserved within 1%

    if stable:
        print("PASS: Major-axis spin is STABLE (transverse rates < 1% of spin rate)")
    else:
        ratio = max_transverse / abs(wz_mean) * 100
        print(f"FAIL: Transverse rates grew to {ratio:.2f}% of spin rate")

    if energy_ok:
        print(f"PASS: Rotational KE conserved within {T_rel_change*100:.4f}%")
    else:
        print(f"WARN: Rotational KE changed by {T_rel_change*100:.2f}%")
        print("      (expected with environmental torques — drag/SRP)")

    # --- Save data for plotting ---
    output_dir = pathlib.Path(__file__).parent
    data_file = output_dir / "iss_spin_data.npz"
    np.savez(
        data_file,
        times=times[:n_rec],
        omega_body=omega_body[:n_rec],
        quat_hist=quat_hist[:n_rec],
        kinetic_rot=kinetic_rot[:n_rec],
        orbit_alt=orbit_alt[:n_rec],
    )
    print(f"\nData saved to {data_file}")


if __name__ == "__main__":
    main()
