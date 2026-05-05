"""HW4 Q3 - LQR attitude regulation for ISS earth pointing.

This script implements the perfect-state baseline for Question 3:

1. Linearize the attitude dynamics about the nominal LVLH/nadir attitude.
2. Compute a continuous-time LQR torque law.
3. Map the commanded body torque to the 4-CMG pyramid by least-norm
   gimbal-rate steering.
4. Test random initial attitude errors out to 90 deg.
5. Test a three-orbit disturbed hold with gravity gradient and drag.

The estimator-in-the-loop version can reuse the same ``lqr_torque`` function by
replacing the perfect state with the HW3 MEKF state estimate.

Usage:
    pixi run python ISS/hw4/attitude_regulation.py
"""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from actuator_specs import (
    CMG_GIMBAL_RATE_MAX,
    CMG_OUTPUT_TORQUE_MAX,
    build_cmg_specs,
    cmg_momentum_and_jacobian,
)
from environmental_torques import (
    ALT_KM,
    CMG_INSCRIBED_MOMENTUM,
    INC_DEG,
    ISS_INERTIA_BODY,
    make_surfaces,
    nominal_lvlh_attitude_matrix,
    orbit_init,
    orbit_period_s,
)
from scipy.linalg import solve_continuous_are

from mujoco_orbit import MjoData, MjoModel, mjo_forward, mjo_step

ROOT = pathlib.Path(__file__).resolve().parents[2]
HW4_DIR = pathlib.Path(__file__).resolve().parent
ISS_XML = ROOT / "ISS" / "iss_model.xml"
PLOT_DIR = HW4_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

DT = 2.0
RANDOM_CASES = 8
RANDOM_CASE_ORBITS = 2.0
DISTURBED_ORBITS = 3.0
RECORD_STRIDE_S = 10.0
ESTIMATOR_ATTITUDE_PROCESS_FLOOR_DEG = 0.01

# LQR Bryson-style scales.  The torque scale is intentionally conservative
# relative to the CMG envelope so the first perfect-state test is not just a
# saturation test.
ATTITUDE_SCALE_DEG = 30.0
RATE_SCALE_DEG_S = 0.05
TORQUE_SCALE_NM = 250.0

TORQUE_LIMIT_BODY_NM = np.array([300.0, 300.0, 700.0])
SETTLE_ATTITUDE_DEG = 0.1
SETTLE_RATE_DEG_S = 0.01


@dataclass(frozen=True)
class LqrDesign:
    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    K: np.ndarray


def skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def rotvec_to_matrix(v: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3) + skew(v)
    axis = v / angle
    K = skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    cos_angle = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    vee = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]
    )
    if angle < 1e-10:
        return 0.5 * vee
    return angle / (2.0 * np.sin(angle)) * vee


def matrix_to_quat(R_wb: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R_wb.ravel())
    return quat / np.linalg.norm(quat)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1.0 - 2.0 * (y**2 + z**2), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x**2 + z**2), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x**2 + y**2)],
        ]
    )


def attitude_quat_from_rotvec(error_body_rad: np.ndarray) -> np.ndarray:
    R_wb = nominal_lvlh_attitude_matrix() @ rotvec_to_matrix(error_body_rad)
    return matrix_to_quat(R_wb)


def ensure_hw3_import_path() -> None:
    hw3_dir = pathlib.Path(__file__).resolve().parents[1] / "hw3"
    hw2_dir = pathlib.Path(__file__).resolve().parents[1] / "hw2"
    for path in (hw3_dir, hw2_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def build_model(*, use_environment: bool) -> MjoModel:
    return MjoModel.from_xml_path(
        str(ISS_XML),
        surfaces=make_surfaces(),
        cmgs=build_cmg_specs(),
        mj_timestep=DT,
        orbit_dt=DT,
        use_j2=True,
        use_drag=use_environment,
        use_srp=False,
        use_magnetic=False,
        use_gravity_gradient=use_environment,
    )


def build_sensor_model(*, use_environment: bool) -> tuple[MjoModel, pathlib.Path]:
    ensure_hw3_import_path()
    from sensors_revisited import make_iss_sensor_xml

    xml_path = pathlib.Path(make_iss_sensor_xml(ISS_INERTIA_BODY))
    model = MjoModel.from_xml_path(
        str(xml_path),
        surfaces=make_surfaces(),
        cmgs=build_cmg_specs(),
        mj_timestep=DT,
        orbit_dt=DT,
        use_j2=True,
        use_drag=use_environment,
        use_srp=False,
        use_magnetic=True,
        use_gravity_gradient=use_environment,
    )
    return model, xml_path


def linearize_attitude_dynamics() -> tuple[np.ndarray, np.ndarray]:
    J_inv = np.linalg.inv(ISS_INERTIA_BODY)
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    B = np.zeros((6, 3))
    B[3:, :] = J_inv
    return A, B


def design_lqr() -> LqrDesign:
    A, B = linearize_attitude_dynamics()
    q_att = 1.0 / np.deg2rad(ATTITUDE_SCALE_DEG) ** 2
    q_rate = 1.0 / np.deg2rad(RATE_SCALE_DEG_S) ** 2
    r_tau = 1.0 / TORQUE_SCALE_NM**2
    Q = np.diag([q_att, q_att, q_att, q_rate, q_rate, q_rate])
    R = np.diag([r_tau, r_tau, r_tau])
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return LqrDesign(A=A, B=B, Q=Q, R=R, K=K)


def attitude_error_body(data: MjoData, body_id: int) -> np.ndarray:
    R_current = data.xmat[body_id].reshape(3, 3)
    R_error = nominal_lvlh_attitude_matrix().T @ R_current
    return matrix_to_rotvec(R_error)


def earth_pointing_error_deg(data: MjoData, body_id: int) -> float:
    R_current = data.xmat[body_id].reshape(3, 3)
    z_body_world = R_current @ np.array([0.0, 0.0, 1.0])
    nadir_world = np.array([-1.0, 0.0, 0.0])
    cos_angle = np.clip(float(np.dot(z_body_world, nadir_world)), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_angle)))


def lqr_torque(
    design: LqrDesign,
    attitude_error_rad: np.ndarray,
    rate_body_rad_s: np.ndarray,
) -> np.ndarray:
    state = np.concatenate([attitude_error_rad, rate_body_rad_s])
    tau = -design.K @ state
    return np.clip(tau, -TORQUE_LIMIT_BODY_NM, TORQUE_LIMIT_BODY_NM)


def lqr_torque_from_estimate(
    design: LqrDesign,
    data: MjoData,
    q_eci_body_est: np.ndarray,
    omega_eci_body_est: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R_eci_body_est = quat_to_matrix(q_eci_body_est)
    R_wb_est = data.frame.C_LI @ R_eci_body_est
    attitude_error_est = matrix_to_rotvec(nominal_lvlh_attitude_matrix().T @ R_wb_est)
    omega_lvlh_body_est = omega_eci_body_est - R_wb_est.T @ data.frame.omega_lvlh
    tau = lqr_torque(design, attitude_error_est, omega_lvlh_body_est)
    return tau, attitude_error_est, omega_lvlh_body_est


def command_cmg_torque(data: MjoData, tau_body_cmd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, A_cmg = cmg_momentum_and_jacobian(data.actuators.cmg_gimbal_angle)
    theta_dot = np.linalg.pinv(A_cmg, rcond=1e-6) @ tau_body_cmd
    max_ratio = np.max(np.abs(theta_dot) / CMG_GIMBAL_RATE_MAX)
    if max_ratio > 1.0:
        theta_dot = theta_dot / max_ratio
    data.actuators.cmg_gimbal_rate_cmd[:] = theta_dot
    tau_achieved = A_cmg @ theta_dot
    return theta_dot, tau_achieved


def initialize_state(
    model: MjoModel,
    data: MjoData,
    attitude_error_rad: np.ndarray,
    rate_body_rad_s: np.ndarray,
) -> None:
    data.qpos[:3] = 0.0
    data.qpos[3:7] = attitude_quat_from_rotvec(attitude_error_rad)
    data.qvel[:] = 0.0
    data.qvel[3:6] = rate_body_rad_s
    data.actuators.cmg_gimbal_angle[:] = 0.0
    data.actuators.cmg_gimbal_rate_cmd[:] = 0.0
    mjo_forward(model, data)


def sample_random_initial_conditions(
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    cases: list[tuple[np.ndarray, np.ndarray]] = []
    angles = np.deg2rad(np.linspace(15.0, 90.0, RANDOM_CASES))
    for angle in angles:
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        attitude_error = angle * axis
        rate = rng.normal(0.0, np.deg2rad(0.003), size=3)
        cases.append((attitude_error, rate))
    return cases


def settling_time_s(
    time_s: np.ndarray,
    attitude_deg: np.ndarray,
    rate_deg_s: np.ndarray,
) -> float | None:
    settled = (attitude_deg < SETTLE_ATTITUDE_DEG) & (rate_deg_s < SETTLE_RATE_DEG_S)
    for idx in np.flatnonzero(settled):
        if np.all(settled[idx:]):
            return float(time_s[idx])
    return None


def run_case(
    *,
    design: LqrDesign,
    initial_attitude_error_rad: np.ndarray,
    initial_rate_body_rad_s: np.ndarray,
    duration_s: float,
    use_environment: bool,
) -> dict[str, np.ndarray | float | None]:
    model = build_model(use_environment=use_environment)
    data = MjoData(model, orbit=orbit_init())
    body_id = model.body_id("iss")
    initialize_state(model, data, initial_attitude_error_rad, initial_rate_body_rad_s)

    steps = int(np.ceil(duration_s / DT))
    stride = max(1, int(round(RECORD_STRIDE_S / DT)))
    records = steps // stride + 2

    time_s = np.zeros(records)
    attitude_error_deg = np.zeros(records)
    earth_pointing_deg = np.zeros(records)
    rate_norm_deg_s = np.zeros(records)
    torque_cmd_body = np.zeros((records, 3))
    torque_achieved_body = np.zeros((records, 3))
    gimbal_rate_cmd = np.zeros((records, 4))
    cmg_momentum_norm = np.zeros(records)

    rec = 0
    last_tau = np.zeros(3)
    last_tau_achieved = np.zeros(3)
    last_theta_dot = np.zeros(4)

    for step in range(steps + 1):
        attitude_error = attitude_error_body(data, body_id)
        rate_body = data.qvel[3:6].copy()
        tau_cmd = lqr_torque(design, attitude_error, rate_body)
        theta_dot, tau_achieved = command_cmg_torque(data, tau_cmd)
        last_tau = tau_cmd
        last_tau_achieved = tau_achieved
        last_theta_dot = theta_dot

        if step % stride == 0 or step == steps:
            cmg_momentum, _ = cmg_momentum_and_jacobian(data.actuators.cmg_gimbal_angle)
            time_s[rec] = min(step * DT, duration_s)
            attitude_error_deg[rec] = np.rad2deg(np.linalg.norm(attitude_error))
            earth_pointing_deg[rec] = earth_pointing_error_deg(data, body_id)
            rate_norm_deg_s[rec] = np.rad2deg(np.linalg.norm(rate_body))
            torque_cmd_body[rec] = last_tau
            torque_achieved_body[rec] = last_tau_achieved
            gimbal_rate_cmd[rec] = last_theta_dot
            cmg_momentum_norm[rec] = np.linalg.norm(cmg_momentum)
            rec += 1

        if step < steps:
            mjo_step(model, data)

    time_s = time_s[:rec]
    attitude_error_deg = attitude_error_deg[:rec]
    earth_pointing_deg = earth_pointing_deg[:rec]
    rate_norm_deg_s = rate_norm_deg_s[:rec]
    torque_cmd_body = torque_cmd_body[:rec]
    torque_achieved_body = torque_achieved_body[:rec]
    gimbal_rate_cmd = gimbal_rate_cmd[:rec]
    cmg_momentum_norm = cmg_momentum_norm[:rec]

    tail_mask = time_s >= max(0.0, duration_s - orbit_period_s())
    return {
        "time_s": time_s,
        "attitude_error_deg": attitude_error_deg,
        "earth_pointing_deg": earth_pointing_deg,
        "rate_norm_deg_s": rate_norm_deg_s,
        "torque_cmd_body": torque_cmd_body,
        "torque_achieved_body": torque_achieved_body,
        "gimbal_rate_cmd": gimbal_rate_cmd,
        "cmg_momentum_norm": cmg_momentum_norm,
        "settling_time_s": settling_time_s(time_s, attitude_error_deg, rate_norm_deg_s),
        "final_attitude_error_deg": float(attitude_error_deg[-1]),
        "max_attitude_error_deg": float(np.max(attitude_error_deg)),
        "rms_pointing_error_deg": float(np.sqrt(np.mean(earth_pointing_deg**2))),
        "tail_rms_pointing_error_deg": float(np.sqrt(np.mean(earth_pointing_deg[tail_mask] ** 2))),
        "max_torque_cmd_nm": float(np.max(np.linalg.norm(torque_cmd_body, axis=1))),
        "max_torque_achieved_nm": float(np.max(np.linalg.norm(torque_achieved_body, axis=1))),
        "max_gimbal_rate_rad_s": float(np.max(np.abs(gimbal_rate_cmd))),
        "max_cmg_momentum_norm_nms": float(np.max(cmg_momentum_norm)),
    }


def run_estimator_case(
    *,
    design: LqrDesign,
    initial_attitude_error_rad: np.ndarray,
    initial_rate_body_rad_s: np.ndarray,
    duration_s: float,
) -> dict[str, np.ndarray | float | None]:
    ensure_hw3_import_path()
    from recursive_estimation import (
        MEKF,
        get_body_rate_eci_body,
        get_true_quat,
        quat_error_angle,
    )
    from sensors_revisited import (
        GYRO_ARW,
        GYRO_SIGMA_U,
        HOR_SIGMA_RAD,
        MAG_SIGMA,
        R_STAR_BODY,
        SUN_SIGMA_RAD,
        GyroSimulator,
        calibrate_gyro,
        calibrate_horizon,
        calibrate_mag,
        calibrate_sun,
        measure_horizon_affine,
        measure_magnetometer_affine,
        measure_star_tracker_quat,
        measure_sun_affine,
        quat_mult,
        rotvec_to_quat,
    )

    rng = np.random.default_rng(11)
    model, xml_path = build_sensor_model(use_environment=True)
    data = MjoData(model, orbit=orbit_init())
    body_id = model.body_id("iss")
    initialize_state(model, data, initial_attitude_error_rad, initial_rate_body_rad_s)

    q_true0 = get_true_quat(data, body_id)
    q0_est = quat_mult(q_true0, rotvec_to_quat(rng.normal(0.0, np.deg2rad(0.25), size=3)))
    q0_est /= np.linalg.norm(q0_est)
    P0 = np.diag([np.deg2rad(2.0) ** 2] * 3 + [1e-12] * 3)
    mekf = MEKF(
        q0_est,
        np.zeros(3),
        P0,
        GYRO_ARW,
        GYRO_SIGMA_U,
        attitude_process_floor=np.deg2rad(ESTIMATOR_ATTITUDE_PROCESS_FLOOR_DEG),
    )
    gyro_sim = GyroSimulator(rng, DT)

    steps = int(np.ceil(duration_s / DT))
    stride = max(1, int(round(RECORD_STRIDE_S / DT)))
    records = steps // stride + 2

    time_s = np.zeros(records)
    attitude_error_deg = np.zeros(records)
    earth_pointing_deg = np.zeros(records)
    estimator_attitude_error_deg = np.zeros(records)
    estimated_control_error_deg = np.zeros(records)
    rate_norm_deg_s = np.zeros(records)
    torque_achieved_body = np.zeros((records, 3))
    cmg_momentum_norm = np.zeros(records)

    rec = 0
    last_tau_achieved = np.zeros(3)
    last_estimated_error = np.zeros(3)

    for step in range(steps + 1):
        q_true = get_true_quat(data, body_id)
        omega_true = get_body_rate_eci_body(data, body_id)
        y_gyro_cal = calibrate_gyro(gyro_sim.measure(omega_true, dt=DT))
        if step > 0:
            mekf.predict(y_gyro_cal, DT)

        mag_truth = data.sensors.measure("mag", noisy=False).copy()
        sun_truth = data.sensors.measure("orbit_sun_body", noisy=False).copy()
        hor_truth = data.sensors.measure("orbit_horizon_body", noisy=False).copy()

        y_mag = calibrate_mag(measure_magnetometer_affine(mag_truth, rng))
        y_mag /= np.linalg.norm(y_mag)
        B_eci = data.env.mag_field_eci.copy()
        B_mag = np.linalg.norm(B_eci)
        if B_mag > 1e-12:
            R_mag_dir = (MAG_SIGMA / B_mag) ** 2 * np.eye(3)
            mekf.update_vector(y_mag, B_eci / B_mag, R_mag_dir)

        sun_eci = data.env.sun_vector_eci.copy()
        sun_eci /= np.linalg.norm(sun_eci)
        mekf.update_vector(
            calibrate_sun(measure_sun_affine(sun_truth, rng)),
            sun_eci,
            SUN_SIGMA_RAD**2 * np.eye(3),
        )

        nadir_eci = -data.orbit.R_eci / np.linalg.norm(data.orbit.R_eci)
        mekf.update_vector(
            calibrate_horizon(measure_horizon_affine(hor_truth, rng)),
            nadir_eci,
            HOR_SIGMA_RAD**2 * np.eye(3),
        )

        q_star = measure_star_tracker_quat(q_true, rng)
        mekf.update_quaternion(q_star, R_STAR_BODY)

        omega_est = y_gyro_cal - mekf.beta
        tau_cmd, estimated_error, _ = lqr_torque_from_estimate(
            design,
            data,
            mekf.q,
            omega_est,
        )
        _, tau_achieved = command_cmg_torque(data, tau_cmd)
        last_tau_achieved = tau_achieved
        last_estimated_error = estimated_error

        if step % stride == 0 or step == steps:
            attitude_error = attitude_error_body(data, body_id)
            cmg_momentum, _ = cmg_momentum_and_jacobian(data.actuators.cmg_gimbal_angle)
            time_s[rec] = min(step * DT, duration_s)
            attitude_error_deg[rec] = np.rad2deg(np.linalg.norm(attitude_error))
            earth_pointing_deg[rec] = earth_pointing_error_deg(data, body_id)
            estimator_attitude_error_deg[rec] = np.rad2deg(quat_error_angle(mekf.q, q_true))
            estimated_control_error_deg[rec] = np.rad2deg(np.linalg.norm(last_estimated_error))
            rate_norm_deg_s[rec] = np.rad2deg(np.linalg.norm(data.qvel[3:6]))
            torque_achieved_body[rec] = last_tau_achieved
            cmg_momentum_norm[rec] = np.linalg.norm(cmg_momentum)
            rec += 1

        if step < steps:
            mjo_step(model, data)

    try:
        xml_path.unlink()
    except OSError:
        pass

    time_s = time_s[:rec]
    attitude_error_deg = attitude_error_deg[:rec]
    earth_pointing_deg = earth_pointing_deg[:rec]
    estimator_attitude_error_deg = estimator_attitude_error_deg[:rec]
    estimated_control_error_deg = estimated_control_error_deg[:rec]
    rate_norm_deg_s = rate_norm_deg_s[:rec]
    torque_achieved_body = torque_achieved_body[:rec]
    cmg_momentum_norm = cmg_momentum_norm[:rec]

    tail_mask = time_s >= max(0.0, duration_s - orbit_period_s())
    return {
        "time_s": time_s,
        "attitude_error_deg": attitude_error_deg,
        "earth_pointing_deg": earth_pointing_deg,
        "estimator_attitude_error_deg": estimator_attitude_error_deg,
        "estimated_control_error_deg": estimated_control_error_deg,
        "rate_norm_deg_s": rate_norm_deg_s,
        "torque_achieved_body": torque_achieved_body,
        "cmg_momentum_norm": cmg_momentum_norm,
        "settling_time_s": settling_time_s(time_s, attitude_error_deg, rate_norm_deg_s),
        "final_attitude_error_deg": float(attitude_error_deg[-1]),
        "max_attitude_error_deg": float(np.max(attitude_error_deg)),
        "rms_pointing_error_deg": float(np.sqrt(np.mean(earth_pointing_deg**2))),
        "tail_rms_pointing_error_deg": float(np.sqrt(np.mean(earth_pointing_deg[tail_mask] ** 2))),
        "tail_rms_estimator_error_deg": float(
            np.sqrt(np.mean(estimator_attitude_error_deg[tail_mask] ** 2))
        ),
        "max_estimator_error_deg": float(np.max(estimator_attitude_error_deg)),
        "max_torque_achieved_nm": float(np.max(np.linalg.norm(torque_achieved_body, axis=1))),
        "max_cmg_momentum_norm_nms": float(np.max(cmg_momentum_norm)),
    }


def plot_random_cases(results: list[dict[str, np.ndarray | float | None]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(4.6, 4.8), sharex=True)
    for idx, result in enumerate(results, start=1):
        time_min = result["time_s"] / 60.0
        label = f"case {idx}"
        axes[0].plot(time_min, result["attitude_error_deg"], linewidth=1.0, label=label)
        torque = np.linalg.norm(result["torque_achieved_body"], axis=1)
        axes[1].plot(time_min, torque, linewidth=1.0)
    axes[0].set_ylabel("Attitude error [deg]")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=6, ncol=2)
    axes[1].set_ylabel("CMG torque [N m]")
    axes[1].set_xlabel("Time [min]")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Perfect-State LQR From Random Initial Attitudes", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "attitude_regulation_random_ic.png", dpi=180)
    plt.close(fig)


def plot_disturbed_case(result: dict[str, np.ndarray | float | None]) -> None:
    time_orbits = result["time_s"] / orbit_period_s()
    torque = np.linalg.norm(result["torque_achieved_body"], axis=1)
    fig, axes = plt.subplots(3, 1, figsize=(4.6, 5.2), sharex=True)
    axes[0].plot(time_orbits, result["attitude_error_deg"], label="Attitude error")
    axes[0].plot(time_orbits, result["earth_pointing_deg"], label="Nadir pointing")
    axes[0].set_ylabel("Error [deg]")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[1].plot(time_orbits, torque)
    axes[1].set_ylabel("CMG torque [N m]")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(time_orbits, result["cmg_momentum_norm"])
    axes[2].axhline(CMG_INSCRIBED_MOMENTUM, color="k", linestyle=":", linewidth=1.0)
    axes[2].set_ylabel("CMG |H| [N m s]")
    axes[2].set_xlabel("Orbit number")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle("Perfect-State LQR With Drag and Gravity Gradient", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "attitude_regulation_disturbed.png", dpi=180)
    plt.close(fig)


def plot_estimator_case(result: dict[str, np.ndarray | float | None]) -> None:
    time_orbits = result["time_s"] / orbit_period_s()
    fig, axes = plt.subplots(3, 1, figsize=(4.6, 5.2), sharex=True)
    axes[0].plot(time_orbits, result["earth_pointing_deg"], label="True nadir pointing")
    axes[0].plot(time_orbits, result["estimated_control_error_deg"], label="Estimated error")
    axes[0].set_ylabel("Error [deg]")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[1].plot(time_orbits, result["estimator_attitude_error_deg"])
    axes[1].set_ylabel("MEKF error [deg]")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(time_orbits, np.linalg.norm(result["torque_achieved_body"], axis=1))
    axes[2].set_ylabel("CMG torque [N m]")
    axes[2].set_xlabel("Orbit number")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle("LQR With HW3 MEKF State Estimate", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "attitude_regulation_estimator.png", dpi=180)
    plt.close(fig)


def serializable_case_summary(
    result: dict[str, np.ndarray | float | None],
) -> dict[str, float | None]:
    return {
        "settling_time_s": result["settling_time_s"],
        "final_attitude_error_deg": result["final_attitude_error_deg"],
        "max_attitude_error_deg": result["max_attitude_error_deg"],
        "rms_pointing_error_deg": result["rms_pointing_error_deg"],
        "tail_rms_pointing_error_deg": result["tail_rms_pointing_error_deg"],
        "max_torque_cmd_nm": result["max_torque_cmd_nm"],
        "max_torque_achieved_nm": result["max_torque_achieved_nm"],
        "max_gimbal_rate_rad_s": result["max_gimbal_rate_rad_s"],
        "max_cmg_momentum_norm_nms": result["max_cmg_momentum_norm_nms"],
    }


def serializable_estimator_summary(
    result: dict[str, np.ndarray | float | None],
) -> dict[str, float | None]:
    return {
        "settling_time_s": result["settling_time_s"],
        "final_attitude_error_deg": result["final_attitude_error_deg"],
        "max_attitude_error_deg": result["max_attitude_error_deg"],
        "rms_pointing_error_deg": result["rms_pointing_error_deg"],
        "tail_rms_pointing_error_deg": result["tail_rms_pointing_error_deg"],
        "tail_rms_estimator_error_deg": result["tail_rms_estimator_error_deg"],
        "max_estimator_error_deg": result["max_estimator_error_deg"],
        "max_torque_achieved_nm": result["max_torque_achieved_nm"],
        "max_cmg_momentum_norm_nms": result["max_cmg_momentum_norm_nms"],
    }


def main() -> None:
    design = design_lqr()
    period = orbit_period_s()
    rng = np.random.default_rng(7)

    initial_conditions = sample_random_initial_conditions(rng)
    random_results = []
    for attitude_error, rate in initial_conditions:
        random_results.append(
            run_case(
                design=design,
                initial_attitude_error_rad=attitude_error,
                initial_rate_body_rad_s=rate,
                duration_s=RANDOM_CASE_ORBITS * period,
                use_environment=False,
            )
        )

    disturbed_result = run_case(
        design=design,
        initial_attitude_error_rad=np.deg2rad(np.array([2.0, -1.0, 1.0])),
        initial_rate_body_rad_s=np.zeros(3),
        duration_s=DISTURBED_ORBITS * period,
        use_environment=True,
    )
    estimator_result = run_estimator_case(
        design=design,
        initial_attitude_error_rad=np.deg2rad(np.array([2.0, -1.0, 1.0])),
        initial_rate_body_rad_s=np.zeros(3),
        duration_s=DISTURBED_ORBITS * period,
    )

    plot_random_cases(random_results)
    plot_disturbed_case(disturbed_result)
    plot_estimator_case(estimator_result)

    random_summaries = [serializable_case_summary(result) for result in random_results]
    settled_times = [
        item["settling_time_s"] for item in random_summaries if item["settling_time_s"] is not None
    ]
    summary = {
        "orbit": {
            "altitude_km": ALT_KM,
            "inclination_deg": INC_DEG,
            "period_s": period,
            "period_min": period / 60.0,
        },
        "simulation": {
            "dt_s": DT,
            "random_case_orbits": RANDOM_CASE_ORBITS,
            "disturbed_orbits": DISTURBED_ORBITS,
            "random_cases": RANDOM_CASES,
        },
        "linearization": {
            "state": "[small attitude error rad, body rate rad/s]",
            "A": design.A.tolist(),
            "B": design.B.tolist(),
            "Q_diagonal": np.diag(design.Q).tolist(),
            "R_diagonal": np.diag(design.R).tolist(),
            "K": design.K.tolist(),
            "attitude_scale_deg": ATTITUDE_SCALE_DEG,
            "rate_scale_deg_s": RATE_SCALE_DEG_S,
            "torque_scale_nm": TORQUE_SCALE_NM,
        },
        "cmg": {
            "gimbal_rate_limit_rad_s": CMG_GIMBAL_RATE_MAX,
            "single_cmg_peak_torque_nm": CMG_OUTPUT_TORQUE_MAX,
            "inscribed_momentum_nms": CMG_INSCRIBED_MOMENTUM,
            "body_torque_limit_nm": TORQUE_LIMIT_BODY_NM.tolist(),
        },
        "random_initial_condition_results": {
            "cases": random_summaries,
            "max_initial_error_deg": float(
                max(np.rad2deg(np.linalg.norm(ic[0])) for ic in initial_conditions)
            ),
            "all_cases_settled": len(settled_times) == RANDOM_CASES,
            "max_settling_time_s": float(max(settled_times)) if settled_times else None,
            "max_final_attitude_error_deg": float(
                max(item["final_attitude_error_deg"] for item in random_summaries)
            ),
            "max_torque_achieved_nm": float(
                max(item["max_torque_achieved_nm"] for item in random_summaries)
            ),
        },
        "disturbed_perfect_state_result": serializable_case_summary(disturbed_result),
        "disturbed_mekf_result": serializable_estimator_summary(estimator_result),
    }

    out = HW4_DIR / "attitude_regulation_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")

    print("ISS HW4 Q3 perfect-state LQR summary")
    print(f"Orbit period: {period / 60.0:.2f} min")
    print("LQR K:")
    print(np.array2string(design.K, precision=3, suppress_small=False))
    print(
        "Random IC max final attitude error: "
        f"{summary['random_initial_condition_results']['max_final_attitude_error_deg']:.4f} deg"
    )
    print(
        "Random IC max settling time: "
        f"{summary['random_initial_condition_results']['max_settling_time_s']:.1f} s"
    )
    print(
        "Disturbed tail RMS nadir pointing error: "
        f"{summary['disturbed_perfect_state_result']['tail_rms_pointing_error_deg']:.5f} deg"
    )
    print(
        "MEKF tail RMS nadir pointing error: "
        f"{summary['disturbed_mekf_result']['tail_rms_pointing_error_deg']:.5f} deg"
    )
    print(
        "MEKF tail RMS estimator attitude error: "
        f"{summary['disturbed_mekf_result']['tail_rms_estimator_error_deg']:.5f} deg"
    )
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
