"""HW4 Q4 - eigen-axis slew with inverse dynamics and MEKF feedback.

This script plans a rest-to-rest 180 degree eigen-axis slew using the
versine profile from Lecture 16, computes the nominal inverse-dynamics
torque, maps the torque to the modeled Zvezda attitude-thruster couples, and tracks the
trajectory with the HW3 MEKF estimate in the full disturbed ISS model.

Usage:
    uv run python ISS/hw4/eigen_axis_slew.py
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from actuator_specs import build_thruster_specs, thruster_wrench_jacobian
from attitude_regulation import (
    DT,
    ESTIMATOR_ATTITUDE_PROCESS_FLOOR_DEG,
    RECORD_STRIDE_S,
    LqrDesign,
    design_lqr,
    ensure_hw3_import_path,
    matrix_to_quat,
    quat_to_matrix,
    rotvec_to_matrix,
)
from environmental_torques import (
    ISS_INERTIA_BODY,
    make_surfaces,
    nominal_lvlh_attitude_matrix,
    orbit_init,
    orbit_period_s,
)
from scipy.optimize import lsq_linear

from mujoco_orbit import MjoData, MjoModel, mjo_forward, mjo_step

ROOT = pathlib.Path(__file__).resolve().parents[2]
HW4_DIR = pathlib.Path(__file__).resolve().parent
PLOT_DIR = HW4_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

SLEW_DURATION_ORBITS = 1.0
HOLD_AFTER_SLEW_ORBITS = 2.0
REGULATOR_DURATION_ORBITS = SLEW_DURATION_ORBITS + HOLD_AFTER_SLEW_ORBITS
SLEW_AXIS_BODY = np.array([0.0, 1.0, 0.0])
SLEW_ANGLE_RAD = np.pi
BODY_TORQUE_LIMIT_NM = np.array([250.0, 200.0, 250.0])
FEEDBACK_GAIN_SCALE = 2.0
THRUSTERS, THRUSTER_LABELS_ALL = build_thruster_specs()
ATTITUDE_THRUSTERS = THRUSTERS[:8]
ATTITUDE_THRUSTER_LABELS = THRUSTER_LABELS_ALL[:8]
THRUSTER_FORCE_LIMITS = np.array([thr.force_limit for thr in ATTITUDE_THRUSTERS])
THRUSTER_WRENCH_B = thruster_wrench_jacobian(ATTITUDE_THRUSTERS)
THRUSTER_ALLOC_WEIGHTS = np.diag([25.0, 25.0, 25.0, 1.0, 1.0, 1.0])

SETTLE_ATTITUDE_DEG = 0.25
SETTLE_RATE_DEG_S = 0.01


@dataclass(frozen=True)
class TrajectorySample:
    R_wb: np.ndarray
    omega_body: np.ndarray
    alpha_body: np.ndarray
    angle_rad: float
    angle_rate_rad_s: float
    angle_accel_rad_s2: float


@dataclass(frozen=True)
class EigenAxisTrajectory:
    R_initial_wb: np.ndarray
    R_final_wb: np.ndarray
    axis_initial_body: np.ndarray
    angle_rad: float
    duration_s: float

    @classmethod
    def from_attitudes(
        cls,
        R_initial_wb: np.ndarray,
        R_final_wb: np.ndarray,
        duration_s: float,
    ) -> EigenAxisTrajectory:
        rotvec = matrix_to_rotvec_robust(R_initial_wb.T @ R_final_wb)
        angle = float(np.linalg.norm(rotvec))
        if angle < 1e-12:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = rotvec / angle
        return cls(
            R_initial_wb=R_initial_wb,
            R_final_wb=R_final_wb,
            axis_initial_body=axis,
            angle_rad=angle,
            duration_s=duration_s,
        )

    def sample(self, time_s: float) -> TrajectorySample:
        if time_s >= self.duration_s:
            return TrajectorySample(
                R_wb=self.R_final_wb,
                omega_body=np.zeros(3),
                alpha_body=np.zeros(3),
                angle_rad=self.angle_rad,
                angle_rate_rad_s=0.0,
                angle_accel_rad_s2=0.0,
            )

        u = max(0.0, time_s) / self.duration_s
        phase = np.pi * u
        angle = 0.5 * self.angle_rad * (1.0 - np.cos(phase))
        angle_rate = 0.5 * self.angle_rad * np.pi / self.duration_s * np.sin(phase)
        angle_accel = (
            0.5 * self.angle_rad * (np.pi / self.duration_s) ** 2 * np.cos(phase)
        )
        R_wb = self.R_initial_wb @ rotvec_to_matrix(angle * self.axis_initial_body)
        omega = angle_rate * self.axis_initial_body
        alpha = angle_accel * self.axis_initial_body
        return TrajectorySample(
            R_wb=R_wb,
            omega_body=omega,
            alpha_body=alpha,
            angle_rad=float(angle),
            angle_rate_rad_s=float(angle_rate),
            angle_accel_rad_s2=float(angle_accel),
        )


def matrix_to_rotvec_robust(R: np.ndarray) -> np.ndarray:
    """Return the principal rotation vector for R, including 180 deg cases."""
    q = matrix_to_quat(R)
    if q[0] < 0.0:
        q = -q
    vector_norm = float(np.linalg.norm(q[1:]))
    if vector_norm < 1e-12:
        return 2.0 * q[1:]
    angle = 2.0 * np.arctan2(vector_norm, float(q[0]))
    return angle * q[1:] / vector_norm


def set_state_from_attitude(
    model,
    data: MjoData,
    R_wb: np.ndarray,
    omega_body: np.ndarray,
) -> None:
    data.qpos[:3] = 0.0
    data.qpos[3:7] = matrix_to_quat(R_wb)
    data.qvel[:] = 0.0
    data.qvel[3:6] = omega_body
    data.actuators.cmg_gimbal_angle[:] = 0.0
    data.actuators.cmg_gimbal_rate_cmd[:] = 0.0
    data.actuators.thr_force_cmd[:] = 0.0
    mjo_forward(model, data)


def inverse_dynamics_torque_body(omega_body: np.ndarray, alpha_body: np.ndarray) -> np.ndarray:
    return ISS_INERTIA_BODY @ alpha_body + np.cross(omega_body, ISS_INERTIA_BODY @ omega_body)


def build_slew_sensor_model() -> tuple[MjoModel, pathlib.Path]:
    ensure_hw3_import_path()
    from sensors_revisited import make_iss_sensor_xml

    xml_path = pathlib.Path(make_iss_sensor_xml(ISS_INERTIA_BODY))
    model = MjoModel.from_xml_path(
        str(xml_path),
        surfaces=make_surfaces(),
        thrusters=ATTITUDE_THRUSTERS,
        mj_timestep=DT,
        orbit_dt=DT,
        use_j2=True,
        use_drag=True,
        use_srp=False,
        use_magnetic=True,
        use_gravity_gradient=True,
    )
    return model, xml_path


def command_thruster_body_torque(
    data: MjoData,
    tau_cmd_body: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Allocate body torque to the eight Zvezda attitude thrusters.

    The least-squares allocation strongly penalizes net force, so the solver
    prefers force-canceling thruster couples and leaves torque error rather than
    using a large translational impulse.
    """
    desired_wrench = np.concatenate([np.zeros(3), tau_cmd_body])
    result = lsq_linear(
        THRUSTER_ALLOC_WEIGHTS @ THRUSTER_WRENCH_B,
        THRUSTER_ALLOC_WEIGHTS @ desired_wrench,
        bounds=(np.zeros_like(THRUSTER_FORCE_LIMITS), THRUSTER_FORCE_LIMITS),
        lsmr_tol="auto",
    )
    forces = result.x
    data.actuators.thr_force_cmd[:] = forces
    achieved_wrench = THRUSTER_WRENCH_B @ forces
    return forces, achieved_wrench[:3], achieved_wrench[3:]


def make_trajectory() -> EigenAxisTrajectory:
    period = orbit_period_s()
    R_final = nominal_lvlh_attitude_matrix()
    R_initial = R_final @ rotvec_to_matrix(SLEW_ANGLE_RAD * SLEW_AXIS_BODY)
    return EigenAxisTrajectory.from_attitudes(
        R_initial,
        R_final,
        duration_s=SLEW_DURATION_ORBITS * period,
    )


def tracking_control_torque(
    design: LqrDesign,
    trajectory: EigenAxisTrajectory,
    time_s: float,
    data: MjoData,
    q_eci_body_est: np.ndarray,
    omega_eci_body_est: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, TrajectorySample]:
    sample = trajectory.sample(time_s)
    R_eci_body_est = quat_to_matrix(q_eci_body_est)
    R_est_wb = data.frame.C_LI @ R_eci_body_est
    omega_est_body = omega_eci_body_est - R_est_wb.T @ data.frame.omega_lvlh

    attitude_error = matrix_to_rotvec_robust(sample.R_wb.T @ R_est_wb)
    omega_ref_world = sample.R_wb @ sample.omega_body
    omega_ref_est_body = R_est_wb.T @ omega_ref_world
    rate_error = omega_est_body - omega_ref_est_body

    tau_ff_ref_body = inverse_dynamics_torque_body(sample.omega_body, sample.alpha_body)
    tau_ff_est_body = R_est_wb.T @ (sample.R_wb @ tau_ff_ref_body)
    tau_fb_body = -FEEDBACK_GAIN_SCALE * design.K @ np.concatenate([attitude_error, rate_error])
    tau_cmd = np.clip(tau_ff_est_body + tau_fb_body, -BODY_TORQUE_LIMIT_NM, BODY_TORQUE_LIMIT_NM)
    return tau_cmd, tau_ff_est_body, tau_fb_body, attitude_error, omega_est_body, sample


def regulator_control_torque(
    design: LqrDesign,
    R_final_wb: np.ndarray,
    data: MjoData,
    q_eci_body_est: np.ndarray,
    omega_eci_body_est: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    R_eci_body_est = quat_to_matrix(q_eci_body_est)
    R_est_wb = data.frame.C_LI @ R_eci_body_est
    omega_est_body = omega_eci_body_est - R_est_wb.T @ data.frame.omega_lvlh
    attitude_error = matrix_to_rotvec_robust(R_final_wb.T @ R_est_wb)
    tau_fb_body = -FEEDBACK_GAIN_SCALE * design.K @ np.concatenate([attitude_error, omega_est_body])
    tau_cmd = np.clip(tau_fb_body, -BODY_TORQUE_LIMIT_NM, BODY_TORQUE_LIMIT_NM)
    return tau_cmd, tau_fb_body, attitude_error, omega_est_body


def settle_time(
    time_s: np.ndarray,
    final_error_deg: np.ndarray,
    rate_norm_deg_s: np.ndarray,
) -> float | None:
    settled = (final_error_deg < SETTLE_ATTITUDE_DEG) & (rate_norm_deg_s < SETTLE_RATE_DEG_S)
    for idx in np.flatnonzero(settled):
        if np.all(settled[idx:]):
            return float(time_s[idx])
    return None


def run_closed_loop_case(
    *,
    design: LqrDesign,
    trajectory: EigenAxisTrajectory,
    use_trajectory: bool,
    duration_s: float,
    rng_seed: int,
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

    rng = np.random.default_rng(rng_seed)
    model, xml_path = build_slew_sensor_model()
    data = MjoData(model, orbit=orbit_init())
    body_id = model.body_id("iss")
    set_state_from_attitude(model, data, trajectory.R_initial_wb, np.zeros(3))

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
    reference_angle_deg = np.zeros(records)
    actual_slew_angle_deg = np.zeros(records)
    tracking_error_deg = np.zeros(records)
    final_attitude_error_deg = np.zeros(records)
    rate_ref_norm_deg_s = np.zeros(records)
    rate_true_norm_deg_s = np.zeros(records)
    estimator_attitude_error_deg = np.zeros(records)
    estimated_tracking_error_deg = np.zeros(records)
    torque_ff_body = np.zeros((records, 3))
    torque_fb_body = np.zeros((records, 3))
    torque_cmd_body = np.zeros((records, 3))
    torque_achieved_body = np.zeros((records, 3))
    thruster_force_cmd = np.zeros((records, len(ATTITUDE_THRUSTERS)))
    net_force_body = np.zeros((records, 3))

    rec = 0
    last_sample = trajectory.sample(0.0)
    last_tau_ff = np.zeros(3)
    last_tau_fb = np.zeros(3)
    last_tau_cmd = np.zeros(3)
    last_tau_achieved = np.zeros(3)
    last_thruster_force = np.zeros(len(ATTITUDE_THRUSTERS))
    last_net_force = np.zeros(3)
    last_estimated_error = np.zeros(3)

    for step in range(steps + 1):
        t = min(step * DT, duration_s)
        q_true = get_true_quat(data, body_id)
        omega_true_eci_body = get_body_rate_eci_body(data, body_id)
        y_gyro_cal = calibrate_gyro(gyro_sim.measure(omega_true_eci_body, dt=DT))
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

        omega_est_eci_body = y_gyro_cal - mekf.beta
        if use_trajectory:
            (
                tau_cmd,
                tau_ff,
                tau_fb,
                estimated_error,
                omega_est_body,
                sample,
            ) = tracking_control_torque(
                design,
                trajectory,
                t,
                data,
                mekf.q,
                omega_est_eci_body,
            )
        else:
            tau_cmd, tau_fb, estimated_error, omega_est_body = regulator_control_torque(
                design,
                trajectory.R_final_wb,
                data,
                mekf.q,
                omega_est_eci_body,
            )
            tau_ff = np.zeros(3)
            sample = TrajectorySample(
                R_wb=trajectory.R_final_wb,
                omega_body=np.zeros(3),
                alpha_body=np.zeros(3),
                angle_rad=trajectory.angle_rad,
                angle_rate_rad_s=0.0,
                angle_accel_rad_s2=0.0,
            )

        thruster_force, net_force, tau_achieved = command_thruster_body_torque(data, tau_cmd)
        last_sample = sample
        last_tau_ff = tau_ff
        last_tau_fb = tau_fb
        last_tau_cmd = tau_cmd
        last_tau_achieved = tau_achieved
        last_thruster_force = thruster_force
        last_net_force = net_force
        last_estimated_error = estimated_error

        if step % stride == 0 or step == steps:
            R_actual = data.xmat[body_id].reshape(3, 3)
            actual_slew_rotvec = matrix_to_rotvec_robust(trajectory.R_initial_wb.T @ R_actual)
            tracking_error = matrix_to_rotvec_robust(last_sample.R_wb.T @ R_actual)
            final_error = matrix_to_rotvec_robust(trajectory.R_final_wb.T @ R_actual)
            time_s[rec] = t
            reference_angle_deg[rec] = np.rad2deg(last_sample.angle_rad)
            actual_slew_angle_deg[rec] = np.rad2deg(
                float(np.dot(actual_slew_rotvec, trajectory.axis_initial_body))
            )
            tracking_error_deg[rec] = np.rad2deg(np.linalg.norm(tracking_error))
            final_attitude_error_deg[rec] = np.rad2deg(np.linalg.norm(final_error))
            rate_ref_norm_deg_s[rec] = np.rad2deg(np.linalg.norm(last_sample.omega_body))
            rate_true_norm_deg_s[rec] = np.rad2deg(np.linalg.norm(data.qvel[3:6]))
            estimator_attitude_error_deg[rec] = np.rad2deg(quat_error_angle(mekf.q, q_true))
            estimated_tracking_error_deg[rec] = np.rad2deg(np.linalg.norm(last_estimated_error))
            torque_ff_body[rec] = last_tau_ff
            torque_fb_body[rec] = last_tau_fb
            torque_cmd_body[rec] = last_tau_cmd
            torque_achieved_body[rec] = last_tau_achieved
            thruster_force_cmd[rec] = last_thruster_force
            net_force_body[rec] = last_net_force
            rec += 1

        if step < steps:
            mjo_step(model, data)

    try:
        xml_path.unlink()
    except OSError:
        pass

    arrays = {
        "time_s": time_s[:rec],
        "reference_angle_deg": reference_angle_deg[:rec],
        "actual_slew_angle_deg": actual_slew_angle_deg[:rec],
        "tracking_error_deg": tracking_error_deg[:rec],
        "final_attitude_error_deg": final_attitude_error_deg[:rec],
        "rate_ref_norm_deg_s": rate_ref_norm_deg_s[:rec],
        "rate_true_norm_deg_s": rate_true_norm_deg_s[:rec],
        "estimator_attitude_error_deg": estimator_attitude_error_deg[:rec],
        "estimated_tracking_error_deg": estimated_tracking_error_deg[:rec],
        "torque_ff_body": torque_ff_body[:rec],
        "torque_fb_body": torque_fb_body[:rec],
        "torque_cmd_body": torque_cmd_body[:rec],
        "torque_achieved_body": torque_achieved_body[:rec],
        "thruster_force_cmd": thruster_force_cmd[:rec],
        "net_force_body": net_force_body[:rec],
    }

    hold_mask = arrays["time_s"] >= trajectory.duration_s
    if not np.any(hold_mask):
        hold_mask = arrays["time_s"] >= arrays["time_s"][-1]

    torque_cmd_norm = np.linalg.norm(arrays["torque_cmd_body"], axis=1)
    torque_achieved_norm = np.linalg.norm(arrays["torque_achieved_body"], axis=1)
    net_force_norm = np.linalg.norm(arrays["net_force_body"], axis=1)
    total_impulse = float(np.trapz(np.sum(arrays["thruster_force_cmd"], axis=1), arrays["time_s"]))
    rate_error = np.abs(arrays["rate_true_norm_deg_s"] - arrays["rate_ref_norm_deg_s"])

    return {
        **arrays,
        "settling_time_s": settle_time(
            arrays["time_s"],
            arrays["final_attitude_error_deg"],
            arrays["rate_true_norm_deg_s"],
        ),
        "final_tracking_error_deg": float(arrays["tracking_error_deg"][-1]),
        "final_attitude_error_deg_scalar": float(arrays["final_attitude_error_deg"][-1]),
        "hold_rms_tracking_error_deg": float(
            np.sqrt(np.mean(arrays["tracking_error_deg"][hold_mask] ** 2))
        ),
        "hold_rms_final_error_deg": float(
            np.sqrt(np.mean(arrays["final_attitude_error_deg"][hold_mask] ** 2))
        ),
        "hold_rms_estimator_error_deg": float(
            np.sqrt(np.mean(arrays["estimator_attitude_error_deg"][hold_mask] ** 2))
        ),
        "max_tracking_error_deg": float(np.max(arrays["tracking_error_deg"])),
        "max_rate_error_deg_s": float(np.max(rate_error)),
        "max_torque_cmd_nm": float(np.max(torque_cmd_norm)),
        "max_torque_achieved_nm": float(np.max(torque_achieved_norm)),
        "max_thruster_force_n": float(np.max(arrays["thruster_force_cmd"])),
        "max_net_force_n": float(np.max(net_force_norm)),
        "total_abs_thruster_impulse_n_s": total_impulse,
        "max_estimator_error_deg": float(np.max(arrays["estimator_attitude_error_deg"])),
    }


def plot_tracking(result: dict[str, np.ndarray | float | None]) -> None:
    time_orbits = result["time_s"] / orbit_period_s()
    fig, axes = plt.subplots(4, 1, figsize=(4.7, 6.0), sharex=True)
    axes[0].plot(time_orbits, result["reference_angle_deg"], label="nominal")
    axes[0].plot(time_orbits, result["actual_slew_angle_deg"], "--", label="closed-loop")
    axes[0].set_ylabel("Slew angle [deg]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].plot(time_orbits, result["tracking_error_deg"], label="tracking")
    axes[1].plot(time_orbits, result["estimated_tracking_error_deg"], label="estimated")
    axes[1].set_ylabel("Error [deg]")
    axes[1].set_yscale("log")
    error_max = max(
        float(np.max(result["tracking_error_deg"])),
        float(np.max(result["estimated_tracking_error_deg"])),
        1.0,
    )
    axes[1].set_ylim(1e-3, 1.5 * error_max)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7)

    axes[2].plot(time_orbits, result["rate_ref_norm_deg_s"], label="nominal")
    axes[2].plot(time_orbits, result["rate_true_norm_deg_s"], "--", label="closed-loop")
    axes[2].set_ylabel("Rate [deg/s]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=7)

    axes[3].plot(time_orbits, result["estimator_attitude_error_deg"])
    axes[3].set_ylabel("MEKF error [deg]")
    axes[3].set_xlabel("Orbit number")
    axes[3].set_yscale("log")
    axes[3].grid(True, alpha=0.3)

    fig.suptitle("MEKF-Tracked 180 deg Eigen-Axis Slew", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "eigen_axis_slew_tracking.png", dpi=180)
    plt.close(fig)


def plot_inputs(result: dict[str, np.ndarray | float | None]) -> None:
    time_orbits = result["time_s"] / orbit_period_s()
    torque_cmd = result["torque_cmd_body"]
    torque_achieved_norm = np.linalg.norm(result["torque_achieved_body"], axis=1)
    thruster_force_max = np.max(result["thruster_force_cmd"], axis=1)
    net_force_norm = np.linalg.norm(result["net_force_body"], axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(4.7, 5.5), sharex=True)
    for i, label in enumerate(["x", "y", "z"]):
        axes[0].plot(time_orbits, torque_cmd[:, i], label=label)
    axes[0].plot(time_orbits, torque_achieved_norm, "k:", linewidth=1.0, label="|achieved|")
    axes[0].set_ylabel("Torque [N m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=4)

    axes[1].plot(time_orbits, thruster_force_max)
    axes[1].axhline(float(np.max(THRUSTER_FORCE_LIMITS)), color="k", linestyle=":", linewidth=1.0)
    axes[1].set_ylabel("Max thruster [N]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_orbits, net_force_norm)
    axes[2].set_ylabel("Net force [N]")
    axes[2].set_xlabel("Orbit number")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Eigen-Axis Slew RCS Inputs", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "eigen_axis_slew_inputs.png", dpi=180)
    plt.close(fig)


def plot_regulator_comparison(
    slew_result: dict[str, np.ndarray | float | None],
    regulator_result: dict[str, np.ndarray | float | None],
) -> None:
    time_orbits_slew = slew_result["time_s"] / orbit_period_s()
    time_orbits_reg = regulator_result["time_s"] / orbit_period_s()
    fig, axes = plt.subplots(3, 1, figsize=(4.7, 5.6), sharex=True)

    axes[0].plot(time_orbits_slew, slew_result["final_attitude_error_deg"], label="slew tracking")
    axes[0].plot(time_orbits_reg, regulator_result["final_attitude_error_deg"], label="direct LQR")
    axes[0].set_ylabel("Final attitude error [deg]")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].plot(
        time_orbits_slew,
        np.linalg.norm(slew_result["torque_cmd_body"], axis=1),
        label="slew tracking",
    )
    axes[1].plot(
        time_orbits_reg,
        np.linalg.norm(regulator_result["torque_cmd_body"], axis=1),
        label="direct LQR",
    )
    axes[1].set_ylabel("|Torque cmd| [N m]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        time_orbits_slew,
        np.max(slew_result["thruster_force_cmd"], axis=1),
        label="slew tracking",
    )
    axes[2].plot(
        time_orbits_reg,
        np.max(regulator_result["thruster_force_cmd"], axis=1),
        label="direct LQR",
    )
    axes[2].axhline(float(np.max(THRUSTER_FORCE_LIMITS)), color="k", linestyle=":", linewidth=1.0)
    axes[2].set_ylabel("Max thruster [N]")
    axes[2].set_xlabel("Orbit number")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Planned Slew vs Direct Regulator From 180 deg", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "eigen_axis_slew_regulator_compare.png", dpi=180)
    plt.close(fig)


def serializable_summary(result: dict[str, np.ndarray | float | None]) -> dict[str, float | None]:
    return {
        "settling_time_s": result["settling_time_s"],
        "final_tracking_error_deg": result["final_tracking_error_deg"],
        "final_attitude_error_deg": result["final_attitude_error_deg_scalar"],
        "hold_rms_tracking_error_deg": result["hold_rms_tracking_error_deg"],
        "hold_rms_final_error_deg": result["hold_rms_final_error_deg"],
        "hold_rms_estimator_error_deg": result["hold_rms_estimator_error_deg"],
        "max_tracking_error_deg": result["max_tracking_error_deg"],
        "max_rate_error_deg_s": result["max_rate_error_deg_s"],
        "max_torque_cmd_nm": result["max_torque_cmd_nm"],
        "max_torque_achieved_nm": result["max_torque_achieved_nm"],
        "max_thruster_force_n": result["max_thruster_force_n"],
        "max_net_force_n": result["max_net_force_n"],
        "total_abs_thruster_impulse_n_s": result["total_abs_thruster_impulse_n_s"],
        "max_estimator_error_deg": result["max_estimator_error_deg"],
    }


def main() -> None:
    design = design_lqr()
    trajectory = make_trajectory()
    duration = REGULATOR_DURATION_ORBITS * orbit_period_s()

    slew_result = run_closed_loop_case(
        design=design,
        trajectory=trajectory,
        use_trajectory=True,
        duration_s=duration,
        rng_seed=19,
    )
    regulator_result = run_closed_loop_case(
        design=design,
        trajectory=trajectory,
        use_trajectory=False,
        duration_s=duration,
        rng_seed=19,
    )

    plot_tracking(slew_result)
    plot_inputs(slew_result)
    plot_regulator_comparison(slew_result, regulator_result)

    nominal_samples = [trajectory.sample(t) for t in np.linspace(0.0, trajectory.duration_s, 1001)]
    nominal_torque = np.array(
        [
            inverse_dynamics_torque_body(sample.omega_body, sample.alpha_body)
            for sample in nominal_samples
        ]
    )
    nominal_rate = np.array([np.linalg.norm(sample.omega_body) for sample in nominal_samples])
    nominal_bus_momentum = np.array(
        [np.linalg.norm(ISS_INERTIA_BODY @ sample.omega_body) for sample in nominal_samples]
    )

    summary = {
        "trajectory": {
            "method": "eigen-axis versine",
            "initial_error_angle_deg": 180.0,
            "axis_initial_body": trajectory.axis_initial_body.tolist(),
            "duration_s": trajectory.duration_s,
            "duration_orbits": SLEW_DURATION_ORBITS,
            "hold_after_slew_orbits": HOLD_AFTER_SLEW_ORBITS,
            "max_nominal_rate_deg_s": float(np.rad2deg(np.max(nominal_rate))),
            "max_inverse_dynamics_torque_nm": float(
                np.max(np.linalg.norm(nominal_torque, axis=1))
            ),
            "max_inverse_dynamics_torque_body_nm": np.max(np.abs(nominal_torque), axis=0).tolist(),
            "max_nominal_bus_momentum_nms": float(np.max(nominal_bus_momentum)),
            "one_orbit_cmg_momentum_note": (
                "This exceeds the modeled CMG envelope, so Q4 uses RCS torque couples."
            ),
        },
        "rcs": {
            "body_torque_limit_nm": BODY_TORQUE_LIMIT_NM.tolist(),
            "attitude_thruster_labels": ATTITUDE_THRUSTER_LABELS,
            "thruster_force_limits_n": THRUSTER_FORCE_LIMITS.tolist(),
            "allocator": (
                "bounded weighted least squares on [force; torque], "
                "with force residual weighted 25x"
            ),
        },
        "tracked_slew": serializable_summary(slew_result),
        "direct_regulator": serializable_summary(regulator_result),
    }

    out = HW4_DIR / "eigen_axis_slew_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")

    print("ISS HW4 Q4 eigen-axis slew summary")
    print(f"Slew duration: {trajectory.duration_s / 60.0:.2f} min")
    print(
        "Nominal max torque: "
        f"{summary['trajectory']['max_inverse_dynamics_torque_nm']:.2f} N m"
    )
    print(
        "Tracked final attitude error: "
        f"{summary['tracked_slew']['final_attitude_error_deg']:.4f} deg"
    )
    print(
        "Tracked max thruster force: "
        f"{summary['tracked_slew']['max_thruster_force_n']:.2f} N"
    )
    print(
        "Direct regulator final attitude error: "
        f"{summary['direct_regulator']['final_attitude_error_deg']:.4f} deg"
    )
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
