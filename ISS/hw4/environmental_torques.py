"""HW4 Q2 - ISS environmental torque experiments.

The homework asks for free multi-orbit simulations with and without
environmental perturbations.  This script therefore runs two free-response
experiment sets:

1. nominal LVLH attitude, which is the modeled ISS operating attitude;
2. a stated off-nominal initial attitude, used only to make the
   gravity-gradient dynamics visible.

For the orbit-averaged torque and one-day momentum estimates, the script also
computes disturbance torques along fixed reference attitudes.  That sizing
calculation represents the torque a controller/CMG cluster would need to reject
while holding a commanded attitude; it is separate from the free-response runs.

Usage:
    pixi run python ISS/hw4/environmental_torques.py
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from mujoco_orbit import MjoData, MjoModel, OrbitInit, SurfaceSpec, mjo_forward, mjo_step
from mujoco_orbit.constants import GM_EARTH, P_SUN, R_EARTH
from mujoco_orbit.coupling.gravity_gradient import apply_gravity_gradient_torques
from mujoco_orbit.coupling.surfaces import apply_surface_wrenches
from mujoco_orbit.orbit.elements import keplerian_to_cartesian
from mujoco_orbit.orbit.environment import atm_density

ROOT = pathlib.Path(__file__).resolve().parents[2]
HW4_DIR = pathlib.Path(__file__).resolve().parent
ISS_XML = ROOT / "ISS" / "iss_model.xml"
PLOT_DIR = HW4_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

ISS_INERTIA_BODY = np.diag([128e6, 107e6, 201e6])
ALT_KM = 408.0
INC_DEG = 51.6

CMG_ROTOR_MOMENTUM = 4760.0
CMG_TOTAL_MOMENTUM = 4.0 * CMG_ROTOR_MOMENTUM
CMG_INSCRIBED_MOMENTUM = 2.0 * CMG_ROTOR_MOMENTUM * np.cos(
    0.5 * np.arccos(1.0 / np.sqrt(3.0))
)

DT = 10.0
N_FREE_ORBITS = 3.0
DAY_SECONDS = 24.0 * 3600.0
OFF_NOMINAL_ERROR_DEG = np.array([5.0, -3.0, 2.0])

PERTURBATION_CASES = {
    "none": (False, False),
    "gravity gradient": (True, False),
    "drag": (False, True),
    "both": (True, True),
}


@dataclass
class TorqueHistory:
    time_s: np.ndarray
    tau_gg_body: np.ndarray
    tau_drag_body: np.ndarray


def make_surfaces() -> list[SurfaceSpec]:
    """ISS flat-plate model used for drag torque."""
    surfaces: list[SurfaceSpec] = []

    sa_area = 375.0
    for y_off in [-45.0, -35.0, 35.0, 45.0]:
        surfaces.append(
            SurfaceSpec(
                body_name="iss",
                center_of_pressure_body=np.array([0.0, y_off, 0.5]),
                normal_body=np.array([0.0, 0.0, 1.0]),
                area=sa_area,
                drag_coeff=2.2,
                srp_coeff=1.8,
            )
        )
        surfaces.append(
            SurfaceSpec(
                body_name="iss",
                center_of_pressure_body=np.array([0.0, y_off, -0.5]),
                normal_body=np.array([0.0, 0.0, -1.0]),
                area=sa_area,
                drag_coeff=2.2,
                srp_coeff=1.8,
            )
        )

    for y_off in [-25.0, -15.0, 15.0, 20.0, 25.0, 30.0]:
        surfaces.append(
            SurfaceSpec(
                body_name="iss",
                center_of_pressure_body=np.array([0.0, y_off, 3.0]),
                normal_body=np.array([0.0, 0.0, 1.0]),
                area=75.0,
                drag_coeff=2.2,
                srp_coeff=1.5,
            )
        )

    surfaces.extend(
        [
            SurfaceSpec(
                body_name="iss",
                center_of_pressure_body=np.array([36.5, 0.0, 0.0]),
                normal_body=np.array([1.0, 0.0, 0.0]),
                area=180.0,
                drag_coeff=2.2,
                srp_coeff=1.8,
            ),
            SurfaceSpec(
                body_name="iss",
                center_of_pressure_body=np.array([-36.5, 0.0, 0.0]),
                normal_body=np.array([-1.0, 0.0, 0.0]),
                area=180.0,
                drag_coeff=2.2,
                srp_coeff=1.8,
            ),
            SurfaceSpec(
                body_name="iss",
                center_of_pressure_body=np.array([0.0, 54.25, 0.0]),
                normal_body=np.array([0.0, 1.0, 0.0]),
                area=250.0,
                drag_coeff=2.2,
                srp_coeff=1.5,
            ),
            SurfaceSpec(
                body_name="iss",
                center_of_pressure_body=np.array([0.0, -54.25, 0.0]),
                normal_body=np.array([0.0, -1.0, 0.0]),
                area=250.0,
                drag_coeff=2.2,
                srp_coeff=1.5,
            ),
        ]
    )
    return surfaces


def orbit_init() -> OrbitInit:
    radius_km = R_EARTH + ALT_KM
    r_eci, v_eci = keplerian_to_cartesian(
        a=radius_km,
        e=0.0,
        inc=np.deg2rad(INC_DEG),
        raan=0.0,
        argp=0.0,
        nu=0.0,
    )
    return OrbitInit(R_eci=r_eci, V_eci=v_eci)


def orbit_period_s() -> float:
    radius_km = R_EARTH + ALT_KM
    return 2.0 * np.pi * np.sqrt(radius_km**3 / GM_EARTH)


def rotation_x(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation_y(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_z(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def nominal_lvlh_attitude_matrix() -> np.ndarray:
    """World-from-body matrix for the report's ISS body-frame convention.

    MuJoCo world/LVLH axes are +x radial-out, +y along-track, +z orbit-normal.
    The attitude convention used for this simulation points body +X along-track
    and body +Z nadir. Body +Y then completes a right-handed frame, so it points
    opposite the simulator's +z orbit-normal direction.
    """
    return np.array(
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )


def attitude_quat(error_deg: np.ndarray) -> np.ndarray:
    """Return nominal LVLH attitude plus a body-frame roll/pitch/yaw offset."""
    roll, pitch, yaw = np.deg2rad(error_deg)
    error_body = rotation_x(roll) @ rotation_y(pitch) @ rotation_z(yaw)
    r_wb = nominal_lvlh_attitude_matrix() @ error_body
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, r_wb.ravel())
    return quat / np.linalg.norm(quat)


def quat_error_angle(q: np.ndarray, q_ref: np.ndarray) -> float:
    q = q / np.linalg.norm(q)
    q_ref = q_ref / np.linalg.norm(q_ref)
    return float(2.0 * np.arccos(np.clip(abs(np.dot(q, q_ref)), 0.0, 1.0)))


def build_model(*, use_gravity_gradient: bool, use_drag: bool, dt: float = DT) -> MjoModel:
    return MjoModel.from_xml_path(
        str(ISS_XML),
        surfaces=make_surfaces(),
        mj_timestep=dt,
        orbit_dt=dt,
        use_j2=True,
        use_drag=use_drag,
        use_srp=False,
        use_magnetic=False,
        use_gravity_gradient=use_gravity_gradient,
    )


def set_reference_state(model: MjoModel, data: MjoData, quat: np.ndarray) -> None:
    data.qpos[:3] = 0.0
    data.qpos[3:7] = quat
    data.qvel[:] = 0.0
    mjo_forward(model, data)


def measure_component_torques(model: MjoModel, data: MjoData) -> tuple[np.ndarray, np.ndarray]:
    """Measure GG and drag torques at the current state, returned in body frame."""
    body_id = model.body_id("iss")
    r_wb = data.xmat[body_id].reshape(3, 3)

    old_use_gg = model.use_gravity_gradient
    model.use_gravity_gradient = True
    data.clear_wrench_buffer()
    apply_gravity_gradient_torques(model, data)
    tau_gg_world = data.wrench_buffer[body_id, 3:].copy()
    model.use_gravity_gradient = old_use_gg

    old_use_drag = model.use_drag
    old_use_srp = model.use_srp
    model.use_drag = True
    model.use_srp = False
    data.clear_wrench_buffer()
    apply_surface_wrenches(model, data)
    tau_drag_world = data.wrench_buffer[body_id, 3:].copy()
    model.use_drag = old_use_drag
    model.use_srp = old_use_srp

    data.clear_wrench_buffer()
    return r_wb.T @ tau_gg_world, r_wb.T @ tau_drag_world


def reference_torque_history(
    error_deg: np.ndarray,
    duration_s: float = DAY_SECONDS,
    dt: float = DT,
) -> TorqueHistory:
    """Torque history along a fixed commanded attitude for ADCS sizing."""
    model = build_model(use_gravity_gradient=True, use_drag=True, dt=dt)
    data = MjoData(model, orbit=orbit_init())
    quat = attitude_quat(error_deg)

    steps = int(np.ceil(duration_s / dt))
    times = np.zeros(steps + 1)
    tau_gg = np.zeros((steps + 1, 3))
    tau_drag = np.zeros((steps + 1, 3))

    for k in range(steps + 1):
        set_reference_state(model, data, quat)
        times[k] = min(k * dt, duration_s)
        tau_gg[k], tau_drag[k] = measure_component_torques(model, data)
        if k < steps:
            mjo_step(model, data)

    return TorqueHistory(time_s=times, tau_gg_body=tau_gg, tau_drag_body=tau_drag)


def cumulative_integral(time_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values)
    dt = np.diff(time_s)
    increments = 0.5 * (values[1:] + values[:-1]) * dt[:, None]
    out[1:] = np.cumsum(increments, axis=0)
    return out


def torque_summary(time_s: np.ndarray, tau_body: np.ndarray, orbit_period: float) -> dict:
    mag = np.linalg.norm(tau_body, axis=1)
    orbit_mask = time_s <= orbit_period
    orbit_time = time_s[orbit_mask]
    orbit_tau = tau_body[orbit_mask]
    orbit_avg = np.trapz(orbit_tau, orbit_time, axis=0) / orbit_time[-1]
    cumulative = cumulative_integral(time_s, tau_body)
    cumulative_norm = np.linalg.norm(cumulative, axis=1)

    return {
        "max_torque_nm": float(np.max(mag)),
        "mean_torque_magnitude_nm": float(np.mean(mag)),
        "orbit_avg_torque_body_nm": orbit_avg.tolist(),
        "orbit_avg_torque_magnitude_nm": float(np.linalg.norm(orbit_avg)),
        "max_cumulative_momentum_nms_day": float(np.max(cumulative_norm)),
        "final_cumulative_momentum_nms_day": float(cumulative_norm[-1]),
        "secular_momentum_from_orbit_avg_nms_day": float(
            np.linalg.norm(orbit_avg) * DAY_SECONDS
        ),
    }


def simulate_free_response(
    *,
    initial_error_deg: np.ndarray,
    use_gravity_gradient: bool,
    use_drag: bool,
    duration_s: float,
    dt: float = DT,
) -> dict[str, np.ndarray]:
    model = build_model(use_gravity_gradient=use_gravity_gradient, use_drag=use_drag, dt=dt)
    data = MjoData(model, orbit=orbit_init())
    quat0 = attitude_quat(initial_error_deg)
    data.qpos[3:7] = quat0
    data.qvel[:] = 0.0
    mjo_forward(model, data)

    steps = int(np.ceil(duration_s / dt))
    stride = max(1, int(round(60.0 / dt)))
    records = steps // stride + 2
    time_s = np.zeros(records)
    attitude_drift_deg = np.zeros(records)
    nominal_error_deg = np.zeros(records)
    rate_norm_deg_s = np.zeros(records)
    quat_nominal = attitude_quat(np.zeros(3))

    rec = 0
    for k in range(steps + 1):
        if k % stride == 0 or k == steps:
            time_s[rec] = min(k * dt, duration_s)
            attitude_drift_deg[rec] = np.rad2deg(quat_error_angle(data.qpos[3:7], quat0))
            nominal_error_deg[rec] = np.rad2deg(
                quat_error_angle(data.qpos[3:7], quat_nominal)
            )
            rate_norm_deg_s[rec] = np.rad2deg(np.linalg.norm(data.qvel[3:6]))
            rec += 1
        if k < steps:
            mjo_step(model, data)

    return {
        "time_s": time_s[:rec],
        "attitude_drift_deg": attitude_drift_deg[:rec],
        "nominal_error_deg": nominal_error_deg[:rec],
        "rate_norm_deg_s": rate_norm_deg_s[:rec],
    }


def run_free_response_set(
    initial_error_deg: np.ndarray,
    duration_s: float,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        label: simulate_free_response(
            initial_error_deg=initial_error_deg,
            use_gravity_gradient=use_gg,
            use_drag=use_drag,
            duration_s=duration_s,
        )
        for label, (use_gg, use_drag) in PERTURBATION_CASES.items()
    }


def gravity_gradient_global_bound() -> float:
    eig = np.linalg.eigvalsh(ISS_INERTIA_BODY)
    radius_km = R_EARTH + ALT_KM
    return float(1.5 * GM_EARTH / radius_km**3 * (eig[-1] - eig[0]))


def disturbance_environment_notes() -> dict[str, float]:
    radius_km = R_EARTH + ALT_KM
    rho = atm_density(np.array([radius_km, 0.0, 0.0]))
    v_circ_m_s = np.sqrt(GM_EARTH / radius_km) * 1000.0
    q_drag = 0.5 * rho * v_circ_m_s**2
    return {
        "rho_kg_m3": float(rho),
        "circular_speed_m_s": float(v_circ_m_s),
        "dynamic_pressure_n_m2": float(q_drag),
        "drag_pressure_with_cd22_n_m2": float(q_drag * 2.2),
        "srp_pressure_with_cr18_n_m2": float(P_SUN * 1.8),
    }


def plot_torque_summary_bars(reference_summaries: dict[str, dict[str, dict]]) -> None:
    labels = ["Nominal LVLH", "(5,-3,2) deg offset"]
    gg_values = [
        reference_summaries["nominal_lvlh"]["gravity_gradient"]["max_torque_nm"],
        reference_summaries["off_nominal"]["gravity_gradient"]["max_torque_nm"],
    ]
    drag_values = [
        reference_summaries["nominal_lvlh"]["drag"]["max_torque_nm"],
        reference_summaries["off_nominal"]["drag"]["max_torque_nm"],
    ]
    x = np.arange(len(labels))
    width = 0.34
    floor = 1e-4

    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.bar(x - width / 2, np.maximum(gg_values, floor), width, label="Gravity gradient")
    ax.bar(x + width / 2, np.maximum(drag_values, floor), width, label="Drag")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("Max torque magnitude [N m]")
    ax.set_title("Environmental Torque Magnitudes")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.text(x[0] - width / 2, floor * 1.5, "~0", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "environmental_torque_magnitude.png", dpi=180)
    plt.close(fig)


def plot_momentum_accumulation(
    nominal: TorqueHistory,
    off_nominal: TorqueHistory,
) -> None:
    h_nominal_combined = cumulative_integral(
        nominal.time_s,
        nominal.tau_gg_body + nominal.tau_drag_body,
    )
    h_offset_gg = cumulative_integral(off_nominal.time_s, off_nominal.tau_gg_body)
    h_offset_combined = cumulative_integral(
        off_nominal.time_s,
        off_nominal.tau_gg_body + off_nominal.tau_drag_body,
    )
    x = nominal.time_s / 3600.0

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.plot(x, np.linalg.norm(h_nominal_combined, axis=1), label="Nominal, GG+drag")
    ax.plot(x, np.linalg.norm(h_offset_gg, axis=1), label="(5,-3,2), GG")
    ax.plot(
        x,
        np.linalg.norm(h_offset_combined, axis=1),
        "--",
        label="(5,-3,2), GG+drag",
    )
    ax.axhline(CMG_INSCRIBED_MOMENTUM, color="k", linestyle=":", linewidth=1.2)
    ax.text(
        0.4,
        CMG_INSCRIBED_MOMENTUM * 1.03,
        "CMG inscribed momentum",
        fontsize=8,
        va="bottom",
    )
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Cumulative momentum norm [N m s]")
    ax.set_yscale("log")
    ax.set_title("Momentum Accumulation")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "environmental_momentum_accumulation.png", dpi=180)
    plt.close(fig)


def plot_free_response_grid(
    nominal_responses: dict[str, dict[str, np.ndarray]],
    offset_responses: dict[str, dict[str, np.ndarray]],
    orbit_period: float,
) -> None:
    styles = {
        "none": dict(color="C0", linestyle="-"),
        "gravity gradient": dict(color="C1", linestyle="--"),
        "drag": dict(color="C2", linestyle="-"),
        "both": dict(color="C3", linestyle=":"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 4.8), sharex=True)
    rows = [
        ("Nominal LVLH initial attitude", nominal_responses),
        ("(5,-3,2) deg off-nominal initial attitude", offset_responses),
    ]
    for row, (title, responses) in enumerate(rows):
        for label, response in responses.items():
            x = response["time_s"] / orbit_period
            axes[row, 0].plot(
                x,
                response["nominal_error_deg"],
                label=label,
                linewidth=1.4,
                **styles[label],
            )
            axes[row, 1].plot(
                x,
                response["rate_norm_deg_s"],
                label=label,
                linewidth=1.4,
                **styles[label],
            )
        axes[row, 0].set_ylabel("Error from nominal [deg]")
        axes[row, 0].set_title(title, fontsize=8)
        axes[row, 1].set_ylabel("Rate norm [deg/s]")
        axes[row, 0].set_yscale("symlog", linthresh=1e-2)
        axes[row, 1].set_yscale("symlog", linthresh=1e-7)
        axes[row, 0].set_ylim(0.0, 2.0e2)
        axes[row, 1].set_ylim(0.0, 2.0e-1)
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Orbit number")
    axes[1, 1].set_xlabel("Orbit number")
    axes[0, 1].legend(fontsize=6, loc="upper right")
    fig.suptitle("Free Attitude Response With Environmental Perturbations", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "environmental_attitude_response.png", dpi=180)
    plt.close(fig)


def response_summary(response: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "final_nominal_error_deg": float(response["nominal_error_deg"][-1]),
        "max_nominal_error_deg": float(np.max(response["nominal_error_deg"])),
        "final_drift_from_initial_deg": float(response["attitude_drift_deg"][-1]),
        "max_drift_from_initial_deg": float(np.max(response["attitude_drift_deg"])),
        "final_rate_norm_deg_s": float(response["rate_norm_deg_s"][-1]),
        "max_rate_norm_deg_s": float(np.max(response["rate_norm_deg_s"])),
    }


def add_desaturation_estimates(summary_group: dict[str, dict]) -> None:
    for key in ("gravity_gradient", "drag", "combined"):
        secular = summary_group[key]["secular_momentum_from_orbit_avg_nms_day"]
        if secular > 1e-12:
            days = CMG_INSCRIBED_MOMENTUM / secular
        else:
            days = None
        summary_group[key]["days_to_cmg_inscribed_momentum"] = days


def main() -> None:
    period = orbit_period_s()
    free_duration = N_FREE_ORBITS * period

    nominal_reference = reference_torque_history(
        error_deg=np.zeros(3),
        duration_s=DAY_SECONDS,
        dt=DT,
    )
    offset_reference = reference_torque_history(
        error_deg=OFF_NOMINAL_ERROR_DEG,
        duration_s=DAY_SECONDS,
        dt=DT,
    )

    reference_summaries = {
        "nominal_lvlh": {
            "gravity_gradient": torque_summary(
                nominal_reference.time_s,
                nominal_reference.tau_gg_body,
                period,
            ),
            "drag": torque_summary(
                nominal_reference.time_s,
                nominal_reference.tau_drag_body,
                period,
            ),
            "combined": torque_summary(
                nominal_reference.time_s,
                nominal_reference.tau_gg_body + nominal_reference.tau_drag_body,
                period,
            ),
        },
        "off_nominal": {
            "gravity_gradient": {
                **torque_summary(offset_reference.time_s, offset_reference.tau_gg_body, period),
                "global_upper_bound_nm": gravity_gradient_global_bound(),
            },
            "drag": torque_summary(offset_reference.time_s, offset_reference.tau_drag_body, period),
            "combined": torque_summary(
                offset_reference.time_s,
                offset_reference.tau_gg_body + offset_reference.tau_drag_body,
                period,
            ),
        },
    }
    add_desaturation_estimates(reference_summaries["nominal_lvlh"])
    add_desaturation_estimates(reference_summaries["off_nominal"])

    nominal_responses = run_free_response_set(np.zeros(3), free_duration)
    offset_responses = run_free_response_set(OFF_NOMINAL_ERROR_DEG, free_duration)

    plot_torque_summary_bars(reference_summaries)
    plot_momentum_accumulation(nominal_reference, offset_reference)
    plot_free_response_grid(nominal_responses, offset_responses, period)

    summary = {
        "orbit": {
            "altitude_km": ALT_KM,
            "inclination_deg": INC_DEG,
            "period_s": period,
            "period_min": period / 60.0,
        },
        "simulation": {
            "dt_s": DT,
            "free_response_orbits": N_FREE_ORBITS,
            "reference_sizing_duration_s": DAY_SECONDS,
            "off_nominal_error_deg_body_xyz": OFF_NOMINAL_ERROR_DEG.tolist(),
        },
        "environment_notes": disturbance_environment_notes(),
        "reference_torque_sizing": reference_summaries,
        "free_response": {
            "nominal_lvlh_initial": {
                label: response_summary(response)
                for label, response in nominal_responses.items()
            },
            "off_nominal_initial": {
                label: response_summary(response)
                for label, response in offset_responses.items()
            },
        },
        "cmg": {
            "total_momentum_nms": CMG_TOTAL_MOMENTUM,
            "inscribed_momentum_nms": CMG_INSCRIBED_MOMENTUM,
        },
    }

    out = HW4_DIR / "environmental_torques_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")

    nominal_drag = reference_summaries["nominal_lvlh"]["drag"]
    offset_gg = reference_summaries["off_nominal"]["gravity_gradient"]
    print("ISS HW4 Q2 environmental torque summary")
    print(f"Orbit period: {period / 60.0:.2f} min")
    print(f"Off-nominal free-response IC: {OFF_NOMINAL_ERROR_DEG} deg")
    print(f"Nominal LVLH drag max: {nominal_drag['max_torque_nm']:.3f} N m")
    print(
        "Off-nominal gravity-gradient max: "
        f"{offset_gg['max_torque_nm']:.3f} N m "
        f"(global bound {gravity_gradient_global_bound():.3f} N m)"
    )
    print(
        "Nominal daily accumulated momentum: "
        f"{reference_summaries['nominal_lvlh']['combined']['max_cumulative_momentum_nms_day']:.1f} "
        "N m s"
    )
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
