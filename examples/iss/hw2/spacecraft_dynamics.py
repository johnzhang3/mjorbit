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
from scipy.linalg import expm

import tempfile

from mujoco_orbit import compile, step
from mujoco_orbit.constants import R_EARTH
from mujoco_orbit.core.config import (
    MuJoCoCfg,
    OrbitCfg,
    ReactionWheelCfg,
    ScenarioCfg,
    SurfaceCfg,
)
from mujoco_orbit.orbit.elements import keplerian_to_cartesian

# ---------------------------------------------------------------------------
# ISS parameters (from HW1)
# ---------------------------------------------------------------------------

ISS_MASS = 420_000.0  # kg
ISS_IXX = 128e6   # kg·m²
ISS_IYY = 107e6   # kg·m²
ISS_IZZ = 201e6   # kg·m²
J_NOMINAL = np.diag([ISS_IXX, ISS_IYY, ISS_IZZ])

ALT_KM = 410.0
INC_DEG = 51.6
MODEL_XML = str(pathlib.Path(__file__).parents[1] / "iss_model.xml")

SOLAR_NORMAL = np.array([0.0, 0.0, 1.0])  # body +Z

OMEGA_RPM = 10.0
OMEGA_RAD_S = OMEGA_RPM * 2.0 * np.pi / 60.0

INERTIA_RATIO_MIN = 1.2

np.random.seed(42)  # same seed as safe_mode.py for reproducibility


# ---------------------------------------------------------------------------
# Inertia perturbation (same as safe_mode.py)
# ---------------------------------------------------------------------------

def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def perturb_inertia(J: np.ndarray) -> np.ndarray:
    D_vals, V = np.linalg.eigh(J)
    d = np.random.randn(3) * 0.03
    D_tilde = np.diag(D_vals * (1.0 + d))
    v = np.random.randn(3) * np.deg2rad(3.0)
    V_tilde = V @ expm(skew(v))
    return V_tilde @ D_tilde @ V_tilde.T


def compute_rotor_momentum(J: np.ndarray, omega: np.ndarray, ratio: float = 1.2):
    omega_hat = omega / np.linalg.norm(omega)
    omega_mag = np.linalg.norm(omega)
    J_omega = J @ omega

    if abs(omega_hat[0]) < 0.9:
        e1 = np.cross(omega_hat, np.array([1, 0, 0]))
    else:
        e1 = np.cross(omega_hat, np.array([0, 1, 0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(omega_hat, e1)
    P = np.array([e1, e2])
    J_perp = P @ J @ P.T
    I_trans = np.linalg.eigvalsh(J_perp)
    I_trans_max = np.max(I_trans)

    lam = max(ratio * I_trans_max, np.dot(J_omega, omega_hat) / omega_mag)
    h = lam * omega - J_omega
    return h, lam, I_trans_max


# ---------------------------------------------------------------------------
# Surface model (same as HW1)
# ---------------------------------------------------------------------------

def _make_surfaces() -> list[SurfaceCfg]:
    surfaces: list[SurfaceCfg] = []
    sa_area = 375.0
    for y_off in [-45.0, -35.0, 35.0, 45.0]:
        surfaces.append(SurfaceCfg(
            body_name="iss",
            center_of_pressure_body=np.array([0.0, y_off, 0.5]),
            normal_body=np.array([0.0, 0.0, 1.0]),
            area=sa_area, drag_coeff=2.2, srp_coeff=1.8,
        ))
        surfaces.append(SurfaceCfg(
            body_name="iss",
            center_of_pressure_body=np.array([0.0, y_off, -0.5]),
            normal_body=np.array([0.0, 0.0, -1.0]),
            area=sa_area, drag_coeff=2.2, srp_coeff=1.8,
        ))
    for y_off in [-25.0, -15.0, 15.0, 20.0, 25.0, 30.0]:
        surfaces.append(SurfaceCfg(
            body_name="iss",
            center_of_pressure_body=np.array([0.0, y_off, 3.0]),
            normal_body=np.array([0.0, 0.0, 1.0]),
            area=75.0, drag_coeff=2.2, srp_coeff=1.5,
        ))
    surfaces.append(SurfaceCfg(
        body_name="iss",
        center_of_pressure_body=np.array([36.5, 0.0, 0.0]),
        normal_body=np.array([1.0, 0.0, 0.0]),
        area=180.0, drag_coeff=2.2, srp_coeff=1.8,
    ))
    surfaces.append(SurfaceCfg(
        body_name="iss",
        center_of_pressure_body=np.array([-36.5, 0.0, 0.0]),
        normal_body=np.array([-1.0, 0.0, 0.0]),
        area=180.0, drag_coeff=2.2, srp_coeff=1.8,
    ))
    surfaces.append(SurfaceCfg(
        body_name="iss",
        center_of_pressure_body=np.array([0.0, 54.25, 0.0]),
        normal_body=np.array([0.0, 1.0, 0.0]),
        area=250.0, drag_coeff=2.2, srp_coeff=1.5,
    ))
    surfaces.append(SurfaceCfg(
        body_name="iss",
        center_of_pressure_body=np.array([0.0, -54.25, 0.0]),
        normal_body=np.array([0.0, -1.0, 0.0]),
        area=250.0, drag_coeff=2.2, srp_coeff=1.5,
    ))
    return surfaces


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Quaternion [w,x,y,z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)],
    ])


def pointing_error_deg(
    quat_wxyz: np.ndarray,
    C_IL: np.ndarray,
    sun_eci: np.ndarray,
) -> float:
    """Angle between solar panel normal (+Z body) and sun direction.

    MuJoCo quaternion gives body→LVLH. Must chain with C_IL (LVLH→ECI)
    to get the panel normal in ECI.
    """
    R_body_lvlh = quat_to_rotmat(quat_wxyz)  # body → LVLH
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
    h, lam, I_trans_max = compute_rotor_momentum(J, omega_desired, INERTIA_RATIO_MIN)

    print(f"Perturbed inertia (diagonal of eigendecomp, kg·m²):")
    D_pert = np.linalg.eigvalsh(J)
    for i, val in enumerate(D_pert):
        print(f"  I_{i+1} = {val:.6e}")
    print(f"Rotor momentum h = [{h[0]:+.4e}, {h[1]:+.4e}, {h[2]:+.4e}] kg·m²/s")
    print(f"|h| = {np.linalg.norm(h):.4e} kg·m²/s")
    print(f"Inertia ratio = {lam / I_trans_max:.4f}")

    # --- Configure reaction wheels to produce h ---
    # Use 3 wheels along body X, Y, Z axes.
    # Each wheel: h_i = I_w * Omega_w * axis_i
    # Choose a reasonable wheel inertia, compute required speed.
    RW_INERTIA = 50.0  # kg·m² per wheel (large CMG-class for ISS)
    rw_configs = []
    rw_speeds = []
    axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    for i, axis in enumerate(axes):
        speed = h[i] / RW_INERTIA  # rad/s
        rw_configs.append(ReactionWheelCfg(
            body_name="iss",
            axis_body=axis.astype(float),
            inertia=RW_INERTIA,
        ))
        rw_speeds.append(speed)
        print(f"  RW-{['X','Y','Z'][i]}: speed = {speed:.2f} rad/s "
              f"({speed*60/2/np.pi:.1f} RPM), h = {h[i]:+.4e} kg·m²/s")

    # --- Set up perturbed initial conditions (from Q1.4) ---
    perturb_frac = 0.01
    omega0 = omega_desired.copy()
    omega0[0] += perturb_frac * OMEGA_RAD_S
    omega0[1] += perturb_frac * OMEGA_RAD_S
    print(f"\nPerturbed IC: ω₀ = [{omega0[0]:.6f}, {omega0[1]:.6f}, {omega0[2]:.6f}] rad/s")

    # --- Nutation period and sim duration ---
    I_trans_avg = 0.5 * (D_pert[0] + D_pert[1])  # approximate
    T_nutation = 2.0 * np.pi / OMEGA_RAD_S * lam / abs(lam - I_trans_avg)
    n_nutation_periods = 10
    t_total = n_nutation_periods * T_nutation
    print(f"Nutation period ≈ {T_nutation:.2f} s")
    print(f"Simulating {n_nutation_periods} nutation periods = {t_total:.1f} s ({t_total/60:.1f} min)")

    # --- Build scenario ---
    # Note: MuJoCo model has the nominal (diagonal) inertia.
    # The perturbed inertia must be set in the MuJoCo model.
    a_km = R_EARTH + ALT_KM
    R_eci, V_eci = keplerian_to_cartesian(
        a=a_km, e=0.0, inc=np.deg2rad(INC_DEG),
        raan=0.0, argp=0.0, nu=0.0,
    )

    # --- Generate MuJoCo XML with perturbed inertia ---
    # MuJoCo's fullinertia attribute takes 6 elements: M11 M22 M33 M12 M13 M23
    fi = f"{J[0,0]} {J[1,1]} {J[2,2]} {J[0,1]} {J[0,2]} {J[1,2]}"
    xml_str = f"""<mujoco model="iss_perturbed">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="iss" pos="0 0 0">
      <freejoint name="base"/>
      <inertial pos="0 0 0" mass="{ISS_MASS}" fullinertia="{fi}"/>
      <geom name="modules" type="box" size="36.5 2.1 2.1" rgba="0.8 0.8 0.8 1" mass="0"/>
      <geom name="truss" type="box" size="2.3 54.25 2.3" rgba="0.6 0.6 0.6 1" mass="0"/>
      <geom name="sa_port" type="box" size="6.0 17.0 0.05" pos="0 -37.25 0" rgba="0.2 0.2 0.5 0.7" mass="0"/>
      <geom name="sa_starboard" type="box" size="6.0 17.0 0.05" pos="0 37.25 0" rgba="0.2 0.2 0.5 0.7" mass="0"/>
      <geom name="rad_port" type="box" size="1.7 11.5 0.03" pos="0 -20.0 3.0" rgba="0.9 0.9 0.9 0.6" mass="0"/>
      <geom name="rad_starboard" type="box" size="1.7 11.5 0.03" pos="0 20.0 3.0" rgba="0.9 0.9 0.9 0.6" mass="0"/>
    </body>
  </worldbody>
</mujoco>"""
    perturbed_xml = tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False)
    perturbed_xml.write(xml_str)
    perturbed_xml.flush()

    dt = 0.002  # s
    cfg = ScenarioCfg(
        orbit=OrbitCfg(R_eci=R_eci, V_eci=V_eci),
        mujoco=MuJoCoCfg(xml_path=perturbed_xml.name, dt=dt),
        surfaces=_make_surfaces(),
        reaction_wheels=rw_configs,
        use_j2=True,
        use_drag=True,
        use_srp=True,
        use_magnetic=False,
    )
    scenario = compile(cfg)
    bid = scenario.body_id("iss")

    sun_eci = np.array([1.0, 0.0, 0.0])  # sun along +X ECI

    # --- Set initial wheel speeds ---
    for i, speed in enumerate(rw_speeds):
        scenario.actuator_state.rw_speed[i] = speed
    scenario.actuator_state.update_rw_momentum()

    # --- Set initial attitude: panel normal (+Z body) → sun (+X ECI) ---
    # At t=0, C_IL (LVLH→ECI) is known from the frame cache.
    # We want: C_IL @ R_body_lvlh @ [0,0,1] = [1,0,0] (sun dir in ECI)
    # So R_body_lvlh = C_LI @ R_body_eci
    # Choose R_body_eci: body +Z → ECI +X, body +X → ECI +Y, body +Y → ECI +Z
    C_LI = scenario.frame_cache.C_LI
    R_body_eci = np.array([
        [0.0, 0.0, 1.0],   # body X in ECI = ECI Z
        [1.0, 0.0, 0.0],   # body Y in ECI = ECI X
        [0.0, 1.0, 0.0],   # body Z in ECI = ECI Y
    ]).T  # columns are body axes in ECI
    # Actually: we want body +Z → sun (+X ECI).  Let's be cleaner:
    # R_body_eci columns = [body_x_eci, body_y_eci, body_z_eci]
    body_z_eci = sun_eci / np.linalg.norm(sun_eci)  # [1,0,0]
    # Choose body_x to be perpendicular (e.g., along ECI Z)
    body_x_eci = np.array([0.0, 0.0, 1.0])
    body_y_eci = np.cross(body_z_eci, body_x_eci)
    body_y_eci /= np.linalg.norm(body_y_eci)
    body_x_eci = np.cross(body_y_eci, body_z_eci)
    R_body_eci = np.column_stack([body_x_eci, body_y_eci, body_z_eci])

    R_body_lvlh = C_LI @ R_body_eci  # body → LVLH

    # Convert to quaternion for MuJoCo [w, x, y, z]
    q_init = np.zeros(4)
    mujoco.mju_mat2Quat(q_init, R_body_lvlh.flatten())
    scenario.mjd.qpos[3:7] = q_init

    # --- Set initial angular velocity (perturbed) ---
    # MuJoCo free-joint qvel[3:6] is in body frame
    scenario.mjd.qvel[3:6] = omega0
    mujoco.mj_forward(scenario.mjm, scenario.mjd)

    # --- Run simulation ---
    n_steps = int(t_total / dt)
    record_every = int(0.05 / dt)  # 20 Hz
    n_records = n_steps // record_every + 1

    times = np.zeros(n_records)
    quat_hist = np.zeros((n_records, 4))   # [w, x, y, z]
    omega_body = np.zeros((n_records, 3))
    pointing_err = np.zeros(n_records)
    rw_speed_hist = np.zeros((n_records, 3))

    def record(idx: int, t: float) -> None:
        times[idx] = t
        q = scenario.mjd.qpos[3:7].copy()  # [w, x, y, z]
        quat_hist[idx] = q

        # Free-joint qvel[3:6] is already in body frame
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

    # --- Summary ---
    print(f"\nFinal state (t = {times[n_rec-1]:.1f} s):")
    print(f"  q  = [{quat_hist[n_rec-1,0]:+.6f}, {quat_hist[n_rec-1,1]:+.6f}, "
          f"{quat_hist[n_rec-1,2]:+.6f}, {quat_hist[n_rec-1,3]:+.6f}]")
    print(f"  ω  = [{omega_body[n_rec-1,0]:+.6f}, {omega_body[n_rec-1,1]:+.6f}, "
          f"{omega_body[n_rec-1,2]:+.6f}] rad/s")
    print(f"  Pointing error: {pointing_err[n_rec-1]:.4f}°")
    print(f"  Max pointing error: {np.max(pointing_err[:n_rec]):.4f}°")
    print(f"  RW speeds: [{rw_speed_hist[n_rec-1,0]:.2f}, {rw_speed_hist[n_rec-1,1]:.2f}, "
          f"{rw_speed_hist[n_rec-1,2]:.2f}] rad/s")

    # --- Plotting ---
    plot_dir = pathlib.Path(__file__).parent / "plots"
    plot_dir.mkdir(exist_ok=True)
    t_plot = times[:n_rec]

    # ---- Figure 1: Attitude quaternion components ----
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    q_labels = ["q₀ (w)", "q₁ (x)", "q₂ (y)", "q₃ (z)"]
    q_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for k in range(4):
        ax1.plot(t_plot, quat_hist[:n_rec, k], label=q_labels[k],
                 color=q_colors[k], linewidth=0.8)

    # Mark nutation periods
    for n in range(n_nutation_periods + 1):
        ax1.axvline(n * T_nutation, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Quaternion component")
    ax1.set_title(
        "Attitude Quaternion — Coupled Gyrostat + Orbit Simulation\n"
        f"Perturbed ISS, 10 RPM about +Z, 1% IC perturbation, "
        f"T_nut ≈ {T_nutation:.1f} s",
        fontsize=11,
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.1, 1.1)
    plt.tight_layout()
    out1 = plot_dir / "spacecraft_dynamics_quaternion.png"
    fig1.savefig(out1, dpi=200, bbox_inches="tight")
    print(f"\nPlot saved: {out1}")

    # ---- Figure 2: Pointing error ----
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(t_plot, pointing_err[:n_rec], color="#d62728", linewidth=0.8)
    for n in range(n_nutation_periods + 1):
        ax2.axvline(n * T_nutation, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Pointing error (deg)")
    ax2.set_title(
        "Solar Panel Normal Pointing Error\n"
        f"Angle between body +Z and sun direction (+X ECI), "
        f"T_nut ≈ {T_nutation:.1f} s",
        fontsize=11,
    )
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = plot_dir / "spacecraft_dynamics_pointing.png"
    fig2.savefig(out2, dpi=200, bbox_inches="tight")
    print(f"Plot saved: {out2}")

    # ---- Figure 3: Angular velocity (body frame) ----
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    w_labels = ["ωx (roll)", "ωy (pitch)", "ωz (yaw)"]
    w_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for k in range(3):
        ax3.plot(t_plot, omega_body[:n_rec, k], label=w_labels[k],
                 color=w_colors[k], linewidth=0.8)
    for n in range(n_nutation_periods + 1):
        ax3.axvline(n * T_nutation, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("ω (rad/s)")
    ax3.set_title(
        "Angular Velocity (Body Frame) — Coupled Simulation\n"
        f"Nutation period markers shown as dashed lines",
        fontsize=11,
    )
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    out3 = plot_dir / "spacecraft_dynamics_omega.png"
    fig3.savefig(out3, dpi=200, bbox_inches="tight")
    print(f"Plot saved: {out3}")

    plt.close("all")
    print(f"\nAll plots saved to {plot_dir}/")


if __name__ == "__main__":
    main()
