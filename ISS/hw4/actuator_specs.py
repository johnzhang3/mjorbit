"""HW4 Q1 — Actuator specifications for the ISS.

Models:
  - 4-CMG pyramid cluster (skew β = 54.73°) representing the Z1-truss CMGs
    with Honeywell M1000-class double-gimbal units, reduced to an SGCMG
    approximation for cluster-level analysis.
  - Zvezda Service Module RCS: 8 attitude thrusters in two rings around the
    aft section plus 2 aft-facing reboost engines.

Computes:
  - Body-frame torque and momentum for the CMG cluster
  - CMG output Jacobian A(θ) ∈ R^{3×4} and momentum H(θ) ∈ R^3
  - Thruster wrench Jacobian B ∈ R^{6×N_thr} (force/torque in body frame)
  - CMG momentum envelope (2D slice, exhaustive θ sampling)

Body frame convention (LVLH-aligned at rest):
  +X along-track (roll)
  +Y orbit-normal (pitch)
  +Z nadir (yaw, pointing at Earth)

Pyramid apex is taken along the body +Z axis; this corresponds to the
CMG cluster being mounted on the top-side truss (Z1) with the symmetry
axis of the pyramid pointing toward the main body of the station.

Usage:
    uv run python ISS/hw4/actuator_specs.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mujoco_orbit import (
    ControlMomentGyroSpec,
    MjoData,
    MjoModel,
    OrbitInit,
    ThrusterSpec,
)
from mujoco_orbit.constants import R_EARTH
from mujoco_orbit.orbit.elements import keplerian_to_cartesian

ISS_XML = str(pathlib.Path(__file__).resolve().parents[1] / "iss_model.xml")
PLOT_DIR = pathlib.Path(__file__).resolve().parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# CMG cluster constants (Honeywell M1000 / ISS Z1 CMGs).
# ---------------------------------------------------------------------------
# References:
#   - Bedrossian, Bhatt, Lebsock, Wong, "International Space Station
#     Zero-Propellant Maneuver," JGCD 2009.
#   - Honeywell M1000 CMG datasheet, via Wie (2008) ch. 7.
CMG_MOMENTUM = 4760.0  # N·m·s per rotor
CMG_OUTPUT_TORQUE_MAX = 258.0  # N·m per CMG (peak output)
CMG_GIMBAL_RATE_MAX = CMG_OUTPUT_TORQUE_MAX / CMG_MOMENTUM  # ≈ 0.054 rad/s
CMG_SKEW_DEG = np.rad2deg(np.arccos(1.0 / np.sqrt(3.0)))  # 54.7356°
BETA = np.deg2rad(CMG_SKEW_DEG)
PHI_ARRAY = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])

# ---------------------------------------------------------------------------
# Zvezda RCS thruster constants.
# ---------------------------------------------------------------------------
# References:
#   - NASA SSP 50235 ISS ECLSS/ACS; Ivanov (2012) analysis of Zvezda DPO/DKD.
#   - Small (DPO) attitude thrusters: ~130 N each (13 kgf class).
#   - Large (DKD) reboost engines: ~3070 N each.
RCS_ATTITUDE_THRUST = 130.0  # N per small thruster
RCS_REBOOST_THRUST = 3070.0  # N per large engine
# Zvezda service module aft compartment, approximate distance from ISS CM.
ZVEZDA_X = -30.0  # m along -X (aft)
ZVEZDA_Y_OFFSET = 3.0  # m lateral (ring radius)
ZVEZDA_Z_OFFSET = 3.0  # m vertical
ZVEZDA_XY_OFFSET = 1.0  # m longitudinal separation between fwd/aft rings


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def cmg_axes(phi: float) -> tuple[np.ndarray, np.ndarray]:
    """Gimbal and spin-at-θ=0 axes for a pyramid CMG at azimuth φ.

    Apex along +Z body; gimbal axis tilted by β from +Z in azimuthal plane.
    Spin axis at θ=0 chosen tangential to the azimuth so Σ ŝ_i(0) = 0 (neutral
    zero-momentum equilibrium).
    """
    g = np.array([np.sin(BETA) * np.cos(phi), np.sin(BETA) * np.sin(phi), np.cos(BETA)])
    s0 = np.array([-np.sin(phi), np.cos(phi), 0.0])
    return g, s0


def build_cmg_specs() -> list[ControlMomentGyroSpec]:
    specs = []
    for phi in PHI_ARRAY:
        g, s0 = cmg_axes(phi)
        specs.append(
            ControlMomentGyroSpec(
                body_name="iss",
                gimbal_axis_body=g,
                spin_axis_body_0=s0,
                rotor_momentum=CMG_MOMENTUM,
                gimbal_rate_limit=CMG_GIMBAL_RATE_MAX,
            )
        )
    return specs


def cmg_momentum_and_jacobian(
    thetas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return net momentum H(θ) ∈ R^3 and output Jacobian A(θ) ∈ R^{3×4}.

    The body-frame torque from commanded gimbal rates θ̇ ∈ R^4 is
    τ_body = A(θ) θ̇  where  A[:, i] = -h · (ĝ_i × ŝ_i(θ_i)) = -h · t̂_i(θ_i).
    """
    H = np.zeros(3)
    A = np.zeros((3, 4))
    for i, (theta, phi) in enumerate(zip(thetas, PHI_ARRAY)):
        g, s0 = cmg_axes(phi)
        t0 = np.cross(g, s0)
        s = np.cos(theta) * s0 + np.sin(theta) * t0
        t = np.cross(g, s)
        H += CMG_MOMENTUM * s
        A[:, i] = -CMG_MOMENTUM * t
    return H, A


# ---------------------------------------------------------------------------
# Zvezda RCS thruster layout
# ---------------------------------------------------------------------------
def build_thruster_specs() -> tuple[list[ThrusterSpec], list[str]]:
    """Return (specs, labels).

    Simplified Zvezda PAO layout: 8 attitude thrusters around the aft ring
    (2 firing +Y, 2 firing -Y, 2 firing +Z, 2 firing -Z) plus 2 aft-facing
    reboost engines.  Positions are approximate.
    """
    thr: list[ThrusterSpec] = []
    lbl: list[str] = []

    def add(pos, dir_, label, force):
        thr.append(
            ThrusterSpec(
                body_name="iss",
                position_body=np.asarray(pos, dtype=float),
                direction_body=np.asarray(dir_, dtype=float),
                force_limit=force,
            )
        )
        lbl.append(label)

    # Aft ring (x = ZVEZDA_X - 0.5), radius ~3 m: paired couples in Y and Z
    # Pairs chosen so that firing both members of a pair gives pure torque
    # (no net force).
    # --- Yaw / Roll couples (firing ±Y) ---
    add([ZVEZDA_X, +ZVEZDA_Y_OFFSET, +ZVEZDA_Z_OFFSET], [0, +1, 0], "ACS_+Y_top", RCS_ATTITUDE_THRUST)
    add([ZVEZDA_X, -ZVEZDA_Y_OFFSET, +ZVEZDA_Z_OFFSET], [0, -1, 0], "ACS_-Y_top", RCS_ATTITUDE_THRUST)
    add([ZVEZDA_X, +ZVEZDA_Y_OFFSET, -ZVEZDA_Z_OFFSET], [0, +1, 0], "ACS_+Y_bot", RCS_ATTITUDE_THRUST)
    add([ZVEZDA_X, -ZVEZDA_Y_OFFSET, -ZVEZDA_Z_OFFSET], [0, -1, 0], "ACS_-Y_bot", RCS_ATTITUDE_THRUST)
    # --- Pitch couples (firing ±Z) ---
    add([ZVEZDA_X + ZVEZDA_XY_OFFSET, 0, +ZVEZDA_Z_OFFSET], [0, 0, +1], "ACS_+Z_fwd", RCS_ATTITUDE_THRUST)
    add([ZVEZDA_X - ZVEZDA_XY_OFFSET, 0, +ZVEZDA_Z_OFFSET], [0, 0, +1], "ACS_+Z_aft", RCS_ATTITUDE_THRUST)
    add([ZVEZDA_X + ZVEZDA_XY_OFFSET, 0, -ZVEZDA_Z_OFFSET], [0, 0, -1], "ACS_-Z_fwd", RCS_ATTITUDE_THRUST)
    add([ZVEZDA_X - ZVEZDA_XY_OFFSET, 0, -ZVEZDA_Z_OFFSET], [0, 0, -1], "ACS_-Z_aft", RCS_ATTITUDE_THRUST)
    # --- Reboost (DKD) engines, aft-facing (+X thrust to raise orbit) ---
    add([ZVEZDA_X - 1.5, +0.8, 0], [+1, 0, 0], "DKD_starboard", RCS_REBOOST_THRUST)
    add([ZVEZDA_X - 1.5, -0.8, 0], [+1, 0, 0], "DKD_port", RCS_REBOOST_THRUST)
    return thr, lbl


def thruster_wrench_jacobian(
    thrusters: list[ThrusterSpec],
) -> np.ndarray:
    """Return B ∈ R^{6×N} so that [F; τ]_body = B · u where u_i ∈ [0, F_max,i].

    Rows 0:3 are body-frame force, rows 3:6 are body-frame torque about the CM.
    """
    N = len(thrusters)
    B = np.zeros((6, N))
    for i, t in enumerate(thrusters):
        d = t.direction_body / np.linalg.norm(t.direction_body)
        B[0:3, i] = d
        B[3:6, i] = np.cross(t.position_body, d)
    return B


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------
def print_matrix(name: str, M: np.ndarray, fmt: str = "{: 10.3f}") -> None:
    print(f"{name} (shape {M.shape}):")
    for row in np.atleast_2d(M):
        print("  [" + ", ".join(fmt.format(v) for v in row) + "]")


def print_cmg_specs() -> None:
    print("=" * 70)
    print("CMG Cluster (4-CMG pyramid, Honeywell M1000-class)")
    print("=" * 70)
    print(f"Rotor momentum per CMG:     h = {CMG_MOMENTUM:.0f} N·m·s")
    print(f"Max output torque per CMG:  τ_max = {CMG_OUTPUT_TORQUE_MAX:.0f} N·m")
    print(f"Max gimbal rate:            θ̇_max = {CMG_GIMBAL_RATE_MAX:.4f} rad/s "
          f"({np.rad2deg(CMG_GIMBAL_RATE_MAX):.2f} deg/s)")
    print(f"Pyramid skew angle:         β = {CMG_SKEW_DEG:.4f}°  "
          "(= arccos(1/√3), optimal symmetric envelope)")
    print()
    print("CMG geometry (body frame):")
    for i, phi in enumerate(PHI_ARRAY):
        g, s0 = cmg_axes(phi)
        t0 = np.cross(g, s0)
        print(
            f"  CMG {i+1} (φ = {np.rad2deg(phi):6.1f}°):  "
            f"ĝ = [{g[0]:+.3f}, {g[1]:+.3f}, {g[2]:+.3f}], "
            f"ŝ₀ = [{s0[0]:+.3f}, {s0[1]:+.3f}, {s0[2]:+.3f}], "
            f"t̂₀ = [{t0[0]:+.3f}, {t0[1]:+.3f}, {t0[2]:+.3f}]"
        )
    print()

    H0, A0 = cmg_momentum_and_jacobian(np.zeros(4))
    print(f"Net momentum at θ = 0:        H(0) = {H0}   (expect ~0)")
    print_matrix("Output Jacobian A(0) [N·m per rad/s]", A0, fmt="{: 12.3f}")

    # Maximum body-frame torque for a single instantaneous θ̇ = θ̇_max on all CMGs:
    # technically depends on the direction; here we just report the per-axis peak.
    u_max = CMG_GIMBAL_RATE_MAX * np.ones(4)
    tau_peak = A0 @ u_max
    print(f"\n|τ| with all gimbals at +rate_max at θ=0: ||A(0)·θ̇_max·1|| = "
          f"{np.linalg.norm(tau_peak):.1f} N·m "
          f"(component-wise: {tau_peak})")

    # Max cluster momentum (spherical): all 4 spin axes aligned.
    print(f"Peak cluster angular momentum: 4h = {4 * CMG_MOMENTUM:.0f} N·m·s "
          "(upper bound; realizable subset ≈ 2h·cos(β/2)·...).")
    print()


def print_thruster_specs(thr: list[ThrusterSpec], lbl: list[str]) -> None:
    print("=" * 70)
    print("Zvezda RCS Thrusters")
    print("=" * 70)
    print(f"{'Label':<18}{'position [m]':<28}{'direction':<22}{'F_max [N]':>10}")
    for t, name in zip(thr, lbl):
        pos_str = f"[{t.position_body[0]:+6.1f}, {t.position_body[1]:+5.1f}, {t.position_body[2]:+5.1f}]"
        dir_str = f"[{t.direction_body[0]:+.2f}, {t.direction_body[1]:+.2f}, {t.direction_body[2]:+.2f}]"
        print(f"{name:<18}{pos_str:<28}{dir_str:<22}{t.force_limit:>10.0f}")
    print()

    B = thruster_wrench_jacobian(thr)
    print_matrix("Thruster wrench Jacobian B (6×N) — top 3 rows are force, bottom 3 are torque",
                 B, fmt="{:+8.2f}")

    # Per-axis pure-torque capability using Moore-Penrose-based signed decomposition
    F_max = np.array([t.force_limit for t in thr])
    print("\nMaximum achievable body torque for pure-torque command (box-constrained):")
    for axis_name, axis in zip("XYZ", np.eye(3)):
        # Heuristic upper bound: fire every thruster whose torque contribution
        # along this axis is positive, at max force.
        u = np.zeros(len(thr))
        for i in range(len(thr)):
            contrib = B[3:6, i] @ axis
            if contrib > 0:
                u[i] = F_max[i]
        tau_axis = B[3:6, :] @ u
        print(f"  +{axis_name}:  τ = {tau_axis}   |τ·{axis_name}̂| = {tau_axis @ axis:8.1f} N·m")


# ---------------------------------------------------------------------------
# Envelope plot
# ---------------------------------------------------------------------------
def plot_momentum_envelope() -> None:
    """Sample the 4-D gimbal-angle space and scatter H(θ) to visualize the
    CMG cluster momentum envelope (boundary of reachable angular momentum).
    """
    rng = np.random.default_rng(0)
    n = 40000
    thetas = rng.uniform(-np.pi, np.pi, size=(n, 4))
    H_samples = np.zeros((n, 3))
    for i in range(n):
        H_samples[i], _ = cmg_momentum_and_jacobian(thetas[i])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (a, b, labels) in zip(
        axes,
        [(0, 1, ("H_x", "H_y")), (0, 2, ("H_x", "H_z")), (1, 2, ("H_y", "H_z"))],
    ):
        ax.scatter(H_samples[:, a], H_samples[:, b], s=0.3, alpha=0.3, color="C0")
        ax.set_xlabel(f"{labels[0]} [N·m·s]")
        ax.set_ylabel(f"{labels[1]} [N·m·s]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    fig.suptitle("ISS CMG cluster momentum envelope (random-θ Monte Carlo)")
    fig.tight_layout()
    out = PLOT_DIR / "cmg_momentum_envelope.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"Saved momentum envelope plot → {out.relative_to(PLOT_DIR.parents[2])}")


def plot_output_torque_sphere() -> None:
    """At θ=0, plot the set of body torques achievable by unit-norm gimbal
    rate commands (i.e., the image of the unit sphere under A(0)).
    This is a 3D ellipsoid."""
    _, A0 = cmg_momentum_and_jacobian(np.zeros(4))

    # Sample unit sphere in R^4 (gimbal-rate directions) and push through A(0)
    rng = np.random.default_rng(1)
    d = rng.standard_normal((10000, 4))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    tau = d @ A0.T * CMG_GIMBAL_RATE_MAX  # scale by max gimbal rate

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (a, b, labels) in zip(
        axes,
        [(0, 1, ("τ_x", "τ_y")), (0, 2, ("τ_x", "τ_z")), (1, 2, ("τ_y", "τ_z"))],
    ):
        ax.scatter(tau[:, a], tau[:, b], s=0.3, alpha=0.3, color="C1")
        ax.set_xlabel(f"{labels[0]} [N·m]")
        ax.set_ylabel(f"{labels[1]} [N·m]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        "CMG body-frame torque image of ||θ̇||₂ = θ̇_max (at θ=0, pyramid apex ∥ +Z)"
    )
    fig.tight_layout()
    out = PLOT_DIR / "cmg_torque_image.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"Saved torque-image plot → {out.relative_to(PLOT_DIR.parents[2])}")


# ---------------------------------------------------------------------------
# Smoke test: instantiate MjoModel + MjoData with the full actuator loadout
# ---------------------------------------------------------------------------
def smoke_test_instantiation(thr_specs: list[ThrusterSpec]) -> None:
    """Make sure the ISS XML + CMG + thruster specs compile together."""
    a = R_EARTH + 408.0  # ISS altitude ≈ 408 km
    r_eci, v_eci = keplerian_to_cartesian(
        a=a, e=0.0, inc=np.deg2rad(51.6), raan=0.0, argp=0.0, nu=0.0
    )
    model = MjoModel.from_xml_path(
        ISS_XML,
        cmgs=build_cmg_specs(),
        thrusters=thr_specs,
        mj_timestep=0.01,
    )
    data = MjoData(model, orbit=OrbitInit(R_eci=r_eci, V_eci=v_eci))
    print(f"Instantiated ISS model: {len(model.cmgs)} CMGs, "
          f"{len(model.thrusters)} thrusters, "
          f"body mass = {model.body_mass[model.body_id('iss')]:.0f} kg.")
    assert data.actuators.cmg_rotor_momentum.shape == (4,)
    assert data.actuators.thr_force_cmd.shape == (len(thr_specs),)


# ---------------------------------------------------------------------------
def main() -> None:
    print("ISS ADCS HW4 Q1 — Actuator Specifications")
    print()

    print_cmg_specs()

    thr, lbl = build_thruster_specs()
    print_thruster_specs(thr, lbl)

    smoke_test_instantiation(thr)

    plot_momentum_envelope()
    plot_output_torque_sphere()


if __name__ == "__main__":
    main()
