"""HW3 Section 2 — Recursive Attitude Estimation (MEKF).

Implements a Multiplicative Extended Kalman Filter (MEKF) for attitude
estimation using the refined sensor models from Section 1.

The MEKF state is the 6D error state [δθ (3), δβ (3)]:
  - δθ: attitude error rotation vector (body frame)
  - δβ: gyro bias error

Reference state (q_ref, β_hat) is propagated separately.

Prediction uses bias-corrected gyro measurements.  Measurement updates
fuse vector observations (magnetometer, sun, horizon) and star tracker
quaternion measurements.

The script addresses four investigation topics:
  1. Filter sample rate selection
  2. Comparison to static (Wahba) estimates
  3. Estimator consistency (NEES)
  4. Convergence behavior with different initial conditions

Usage:
    uv run python ISS/hw3/recursive_estimation.py
"""

from __future__ import annotations

import pathlib
import sys
import time as pytime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hw2"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import (
    J_NOMINAL,
    perturb_inertia,
    quat_to_rotmat,
    skew,
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
    make_iss_sensor_xml,
    measure_horizon_affine,
    measure_magnetometer_affine,
    measure_star_tracker_quat,
    measure_sun_affine,
    quat_mult,
    rotvec_to_quat,
)

from mujoco_orbit import MjoData, MjoModel, OrbitInit, mjo_forward, mjo_step
from mujoco_orbit.constants import R_EARTH
from mujoco_orbit.orbit.elements import keplerian_to_cartesian

plt.rcParams.update({
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 16,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

DEFAULT_ATTITUDE_PROCESS_FLOOR_DEG = 0.01
ATTITUDE_NORM_BOUND_PROB = 0.9973002039367398
# Same probability mass as a two-sided 1D 3-sigma interval, applied to ||delta theta||.
VALID_SENSOR_UPDATES = frozenset({"mag", "sun", "horizon", "star"})

# =====================================================================
# MEKF IMPLEMENTATION
# =====================================================================

class MEKF:
    """Multiplicative Extended Kalman Filter for attitude + gyro bias.

    Error-state formulation:
      q_true ≈ q_ref ⊗ dq(δθ)    (body-frame error parameterization)
      β_true ≈ β_hat + δβ

    Refs:  Markley & Crassidis (2014), Ch. 6;
           Lefferts, Markley & Shuster (1982)
    """

    def __init__(
        self,
        q0: np.ndarray,
        beta0: np.ndarray,
        P0: np.ndarray,
        sigma_v: float,
        sigma_u: float,
        attitude_process_floor: float = np.deg2rad(DEFAULT_ATTITUDE_PROCESS_FLOOR_DEG),
    ):
        self.q = q0.copy()
        self.q /= np.linalg.norm(self.q)
        self.beta = beta0.copy()
        self.P = P0.copy()
        self.sigma_v = sigma_v
        self.sigma_u = sigma_u
        self.attitude_process_floor = attitude_process_floor

    def predict(self, omega_meas: np.ndarray, dt: float) -> None:
        """Propagate reference state and covariance using gyro measurement."""
        omega_c = omega_meas - self.beta

        # Propagate reference quaternion: q_ref ← q_ref ⊗ dq(ω_c dt)
        theta = omega_c * dt
        dq = rotvec_to_quat(theta)
        self.q = quat_mult(self.q, dq)
        self.q /= np.linalg.norm(self.q)

        # State transition matrix Φ (6×6)
        F = np.eye(6)
        F[:3, :3] -= skew(omega_c) * dt
        F[:3, 3:6] = -np.eye(3) * dt

        # Process noise Q. The floor covers zero-order-hold gyro propagation
        # and attitude-model errors that are much larger than sensor ARW at 1 Hz.
        Q = np.zeros((6, 6))
        q_att = (self.sigma_v**2 + self.attitude_process_floor**2) * dt
        Q[:3, :3] = q_att * np.eye(3)
        Q[3:6, 3:6] = self.sigma_u**2 * dt * np.eye(3)

        self.P = F @ self.P @ F.T + Q

    def update_vector(
        self,
        b_meas: np.ndarray,
        r_ref: np.ndarray,
        R_meas: np.ndarray,
    ) -> None:
        """Update with a body-frame vector observation.

        b_meas: measured direction in body frame
        r_ref:  reference direction in ECI
        R_meas: 3×3 measurement noise covariance
        """
        C = quat_to_rotmat(self.q)  # body-to-ECI
        b_pred = C.T @ r_ref
        norm = np.linalg.norm(b_pred)
        if norm > 1e-10:
            b_pred /= norm

        # H = [[b_pred×], 0_{3×3}]  (body-frame error convention)
        H = np.zeros((3, 6))
        H[:3, :3] = skew(b_pred)

        z = b_meas - b_pred
        self._kalman_update(H, z, R_meas)

    def update_quaternion(
        self,
        q_meas: np.ndarray,
        R_meas: np.ndarray,
    ) -> None:
        """Update with a star-tracker quaternion measurement.

        q_meas: measured quaternion [w,x,y,z]
        R_meas: 3×3 attitude error covariance
        """
        # Error quaternion: δq = q_ref^{-1} ⊗ q_meas
        q_inv = np.array([self.q[0], -self.q[1], -self.q[2], -self.q[3]])
        dq = quat_mult(q_inv, q_meas)
        if dq[0] < 0:
            dq = -dq

        # Small-angle rotation vector
        if abs(dq[0]) > 1e-6:
            z = 2.0 * dq[1:4] / dq[0]
        else:
            z = 2.0 * dq[1:4]

        # H = [I_3, 0_{3×3}]
        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)

        self._kalman_update(H, z, R_meas)

    def _kalman_update(
        self, H: np.ndarray, z: np.ndarray, R: np.ndarray,
    ) -> None:
        """Standard Kalman measurement update with Joseph form."""
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.solve(S.T, np.eye(S.shape[0])).T

        dx = K @ z
        delta_theta = dx[:3]
        delta_beta = dx[3:6]

        # Correct reference quaternion: q_ref ← q_ref ⊗ dq(δθ)
        dq = rotvec_to_quat(delta_theta)
        self.q = quat_mult(self.q, dq)
        self.q /= np.linalg.norm(self.q)

        self.beta += delta_beta

        # Joseph form for numerical stability
        IKH = np.eye(6) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T


# =====================================================================
# TRUTH SIMULATION
# =====================================================================

def quat_error_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """Total rotation angle between two quaternions (rad)."""
    q2_inv = np.array([q2[0], -q2[1], -q2[2], -q2[3]])
    dq = quat_mult(q1, q2_inv)
    return 2.0 * np.arccos(np.clip(abs(dq[0]), 0.0, 1.0))


def initialize_truth_model_data(
    J_truth: np.ndarray,
    q0_true: np.ndarray,
    omega0: np.ndarray,
) -> tuple[MjoModel, MjoData, int]:
    """Build the MuJoCo truth model/data pair for a fixed inertia draw."""
    xml_path = make_iss_sensor_xml(J_truth)
    a_km = R_EARTH + 410.0
    R_eci, V_eci = keplerian_to_cartesian(
        a=a_km,
        e=0.0,
        inc=np.deg2rad(51.6),
        raan=0.0,
        argp=0.0,
        nu=0.0,
    )
    model = MjoModel.from_xml_path(xml_path, use_j2=True, use_magnetic=True)
    data = MjoData(model, orbit=OrbitInit(R_eci=R_eci, V_eci=V_eci))
    bid = model.body_id("iss")

    mjo_forward(model, data)
    C_IL = data.frame.C_IL
    R_target = quat_to_rotmat(q0_true)
    R_lvlh = C_IL.T @ R_target
    q_lvlh = np.zeros(4)
    mujoco.mju_mat2Quat(q_lvlh, R_lvlh.flatten())
    data.qpos[3:7] = q_lvlh
    data.qvel[3:6] = omega0
    mjo_forward(model, data)
    return model, data, bid


def get_true_quat(data: MjoData, bid: int) -> np.ndarray:
    """Return the current body attitude as an ECI quaternion."""
    R_wb = data.xmat[bid].reshape(3, 3).copy()
    C_now = data.frame.C_IL
    R_eci_body = C_now @ R_wb
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R_eci_body.flatten())
    if q[0] < 0:
        q = -q
    return q


def get_body_rate_eci_body(data: MjoData, bid: int) -> np.ndarray:
    """Return body angular velocity relative to ECI, expressed in the body frame."""
    R_wb = data.xmat[bid].reshape(3, 3).copy()
    R_bw = R_wb.T
    omega_lvlh_body = data.qvel[3:6].copy()
    omega_frame_body = R_bw @ data.frame.omega_lvlh
    return omega_lvlh_body + omega_frame_body


def build_event_schedule(
    t_total: float,
    dt_filter: float,
    star_tracker_period: float = 1.0,
) -> tuple[np.ndarray, set[float], set[float]]:
    """Return sorted event times plus lookup tables for filter and star updates."""
    filter_times = np.arange(0.0, t_total + 0.5 * dt_filter, dt_filter)
    star_times = np.arange(0.0, t_total + 0.5 * star_tracker_period, star_tracker_period)
    event_times = np.unique(np.round(np.concatenate([filter_times, star_times]), 9))
    filter_lookup = {round(float(t), 9) for t in filter_times[1:]}
    star_lookup = {round(float(t), 9) for t in star_times[1:]}
    return event_times, filter_lookup, star_lookup


def mean_over_tail_window(times: np.ndarray, values: np.ndarray, window_s: float) -> float:
    """Average over a fixed-duration tail window rather than a fixed sample count."""
    t_start = max(float(times[0]), float(times[-1]) - window_s)
    return float(np.nanmean(values[times >= t_start]))


def mean_after_time(times: np.ndarray, values: np.ndarray, t_start: float) -> float:
    """Average after a specified physical time."""
    mask = times >= t_start
    return float(np.nanmean(values[mask]))


def run_mekf_simulation(
    dt_filter: float,
    t_total: float,
    omega0: np.ndarray,
    q0_true: np.ndarray,
    q0_est: np.ndarray | None = None,
    beta0_est: np.ndarray | None = None,
    P0_diag: tuple[float, float] | None = None,
    rng_seed: int = 42,
    label: str = "",
    J_truth: np.ndarray | None = None,
    star_tracker_period: float = 1.0,
    enabled_sensors: tuple[str, ...] | None = None,
    attitude_process_floor_deg: float = DEFAULT_ATTITUDE_PROCESS_FLOOR_DEG,
) -> dict:
    """Run truth simulation + MEKF and return histories."""
    rng = np.random.default_rng(rng_seed)
    sensor_updates = (
        VALID_SENSOR_UPDATES
        if enabled_sensors is None
        else frozenset(enabled_sensors)
    )
    unknown_sensors = sensor_updates - VALID_SENSOR_UPDATES
    if unknown_sensors:
        unknown = ", ".join(sorted(unknown_sensors))
        valid = ", ".join(sorted(VALID_SENSOR_UPDATES))
        raise ValueError(f"Unknown MEKF sensor update(s): {unknown}. Valid updates: {valid}.")

    # Build ISS model with sensors
    if J_truth is None:
        J_truth = perturb_inertia(J_NOMINAL, rng=np.random.default_rng(rng_seed))
    model, data, bid = initialize_truth_model_data(J_truth, q0_true, omega0)

    # Initialize gyro simulator with random walk bias
    gyro_sim = GyroSimulator(rng, dt_filter)

    # Initialize MEKF
    if P0_diag is None:
        P0_diag = (np.deg2rad(10.0)**2, (1e-6)**2)
    P0 = np.diag([P0_diag[0]] * 3 + [P0_diag[1]] * 3)

    if q0_est is None:
        # Perturb true attitude by ~10 deg for initial estimate
        init_err = rng.normal(0, np.deg2rad(5), size=3)
        dq_init = rotvec_to_quat(init_err)
        q0_est = quat_mult(q0_true, dq_init)
        q0_est /= np.linalg.norm(q0_est)

    if beta0_est is None:
        beta0_est = np.zeros(3)

    mekf = MEKF(
        q0_est,
        beta0_est,
        P0,
        GYRO_ARW,
        GYRO_SIGMA_U,
        attitude_process_floor=np.deg2rad(attitude_process_floor_deg),
    )

    # Simulation loop
    dt_mj = float(model.mj_model.opt.timestep)
    event_times, filter_update_times, star_update_times = build_event_schedule(
        t_total,
        dt_filter,
        star_tracker_period=star_tracker_period,
    )
    n_events = event_times.size

    times = np.zeros(n_events)
    att_error = np.zeros(n_events)
    bias_error = np.zeros((n_events, 3))
    P_att_trace = np.zeros(n_events)
    P_att_sigma_max = np.zeros(n_events)
    P_att_norm_bound = np.zeros(n_events)
    P_bias_trace = np.zeros(n_events)
    P_bias_sigma_axes = np.zeros((n_events, 3))
    nees = np.zeros(n_events)

    def record(idx, t):
        q_true = get_true_quat(data, bid)
        times[idx] = t
        att_error[idx] = np.rad2deg(quat_error_angle(mekf.q, q_true))
        bias_error[idx] = gyro_sim.bias - mekf.beta
        P_att_trace[idx] = np.rad2deg(np.sqrt(np.trace(mekf.P[:3, :3]) / 3))
        p_att_max_eig = np.max(np.linalg.eigvalsh(mekf.P[:3, :3]))
        P_att_sigma_max[idx] = np.rad2deg(np.sqrt(p_att_max_eig))
        P_att_norm_bound[idx] = np.rad2deg(
            np.sqrt(chi2.ppf(ATTITUDE_NORM_BOUND_PROB, 3) * p_att_max_eig)
        )
        P_bias_trace[idx] = np.sqrt(np.trace(mekf.P[3:6, 3:6]) / 3)
        P_bias_sigma_axes[idx] = np.sqrt(np.diag(mekf.P[3:6, 3:6]))

        # NEES: (x_true - x_est)^T P^{-1} (x_true - x_est)
        q_inv = np.array([mekf.q[0], -mekf.q[1], -mekf.q[2], -mekf.q[3]])
        dq = quat_mult(q_inv, q_true)
        if dq[0] < 0:
            dq = -dq
        delta_theta = 2.0 * dq[1:4] / max(abs(dq[0]), 1e-8)
        delta_beta = gyro_sim.bias - mekf.beta
        dx = np.concatenate([delta_theta, delta_beta])
        try:
            nees[idx] = dx @ np.linalg.solve(mekf.P, dx)
        except np.linalg.LinAlgError:
            nees[idx] = np.nan

    record(0, 0.0)
    prev_t = 0.0
    for i, t in enumerate(event_times[1:], start=1):
        dt_event = float(t - prev_t)
        steps_this_event = int(round(dt_event / dt_mj))
        for _ in range(steps_this_event):
            mjo_step(model, data)
        mjo_forward(model, data)

        # Get truth data from framework
        omega_true = get_body_rate_eci_body(data, bid)

        # Generate noisy measurements using HW3 affine models
        y_gyro = gyro_sim.measure(omega_true, dt=dt_event)
        y_gyro_cal = calibrate_gyro(y_gyro)  # remove known M

        # MEKF prediction
        mekf.predict(y_gyro_cal, dt_event)

        event_key = round(float(t), 9)
        if event_key in filter_update_times and sensor_updates & {"mag", "sun", "horizon"}:
            mag_truth = data.sensors.measure("mag", noisy=False).copy()
            sun_truth = data.sensors.measure("orbit_sun_body", noisy=False).copy()
            hor_truth = data.sensors.measure("orbit_horizon_body", noisy=False).copy()

            y_mag = measure_magnetometer_affine(mag_truth, rng)
            y_mag_cal = calibrate_mag(y_mag)
            y_mag_dir = y_mag_cal / np.linalg.norm(y_mag_cal)
            y_sun = calibrate_sun(measure_sun_affine(sun_truth, rng))
            y_hor = calibrate_horizon(measure_horizon_affine(hor_truth, rng))

            if "mag" in sensor_updates:
                B_eci = data.env.mag_field_eci.copy()
                B_mag = np.linalg.norm(B_eci)
                B_eci_hat = B_eci / B_mag
                mag_dir_sigma = MAG_SIGMA / B_mag
                R_mag_dir = mag_dir_sigma**2 * np.eye(3)
                mekf.update_vector(y_mag_dir, B_eci_hat, R_mag_dir)

            if "sun" in sensor_updates:
                sun_eci = data.env.sun_vector_eci.copy()
                sun_eci /= np.linalg.norm(sun_eci)
                R_sun = SUN_SIGMA_RAD**2 * np.eye(3)
                mekf.update_vector(y_sun, sun_eci, R_sun)

            if "horizon" in sensor_updates:
                nadir_eci = -data.orbit.R_eci / np.linalg.norm(data.orbit.R_eci)
                R_hor = HOR_SIGMA_RAD**2 * np.eye(3)
                mekf.update_vector(y_hor, nadir_eci, R_hor)

        # Keep the star tracker on its own 1 Hz cadence even for low-rate vector updates.
        if "star" in sensor_updates and event_key in star_update_times:
            q_true_now = get_true_quat(data, bid)
            q_star = measure_star_tracker_quat(q_true_now, rng)
            mekf.update_quaternion(q_star, R_STAR_BODY)

        record(i, t)
        prev_t = float(t)

    return {
        "times": times,
        "att_error": att_error,
        "bias_error": bias_error,
        "P_att_trace": P_att_trace,
        "P_att_sigma_max": P_att_sigma_max,
        "P_att_norm_bound": P_att_norm_bound,
        "P_bias_trace": P_bias_trace,
        "P_bias_sigma_axes": P_bias_sigma_axes,
        "nees": nees,
        "label": label,
        "dt": dt_filter,
        "sensors": tuple(sorted(sensor_updates)),
        "attitude_process_floor_deg": attitude_process_floor_deg,
    }


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    print("HW3 Section 2 — Recursive Attitude Estimation (MEKF)")
    print("=" * 65)

    plot_dir = pathlib.Path(__file__).parent / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Common parameters
    omega_tumble = np.array([0.02, 0.03, -0.015])  # ~2 deg/s slow tumble
    q0_true = np.array([1.0, 0.0, 0.0, 0.0])       # identity
    steady_state_window_s = 50.0
    J_truth = perturb_inertia(J_NOMINAL, rng=np.random.default_rng(42))

    print(f"Initial tumble rate: {np.linalg.norm(omega_tumble)*180/np.pi:.2f} deg/s")
    print(f"  omega = [{omega_tumble[0]:.4f}, {omega_tumble[1]:.4f}, "
          f"{omega_tumble[2]:.4f}] rad/s")
    print(f"Attitude process floor: {DEFAULT_ATTITUDE_PROCESS_FLOOR_DEG:.3f} deg/sqrt(s)")

    # ==================================================================
    # 1. Baseline MEKF run (1 Hz, 500 s)
    # ==================================================================
    print("\n--- Baseline MEKF (1 Hz, 500 s) ---")
    t0 = pytime.perf_counter()
    baseline = run_mekf_simulation(
        dt_filter=1.0, t_total=500.0,
        omega0=omega_tumble, q0_true=q0_true,
        rng_seed=42, label="Baseline (1 Hz)",
        J_truth=J_truth,
    )
    print(f"  Completed in {pytime.perf_counter() - t0:.1f}s")
    print(f"  Final attitude error:  {baseline['att_error'][-1]:.4f} deg")
    print(f"  Final 1σ (filter):     {baseline['P_att_trace'][-1]:.4f} deg")
    print(f"  Final 99.7% norm bd:   {baseline['P_att_norm_bound'][-1]:.4f} deg")
    baseline_ss_error = mean_over_tail_window(
        baseline["times"],
        baseline["att_error"],
        steady_state_window_s,
    )
    print(
        f"  Steady-state error:    {baseline_ss_error:.4f} deg "
        f"(last {steady_state_window_s:.0f} s)"
    )

    # ==================================================================
    # 2. Sample rate study
    # ==================================================================
    print("\n--- Sample Rate Study ---")
    rates = [0.2, 0.5, 1.0, 2.0, 5.0]
    rate_results = []
    for rate in rates:
        dt = 1.0 / rate
        t0 = pytime.perf_counter()
        result = run_mekf_simulation(
            dt_filter=dt, t_total=300.0,
            omega0=omega_tumble, q0_true=q0_true,
            rng_seed=42, label=f"{rate} Hz",
            J_truth=J_truth,
        )
        elapsed = pytime.perf_counter() - t0
        ss_err = mean_over_tail_window(result["times"], result["att_error"], steady_state_window_s)
        print(f"  {rate:5.1f} Hz: SS error = {ss_err:.4f} deg, "
              f"wall time = {elapsed:.1f}s")
        rate_results.append(result)

    # ==================================================================
    # 3. Sensor isolation study
    # ==================================================================
    print("\n--- Sensor Isolation Study ---")
    isolation_cases: list[tuple[str, tuple[str, ...] | None]] = [
        ("All sensors", None),
        ("Star only", ("star",)),
        ("Sun only", ("sun",)),
        ("Horizon only", ("horizon",)),
        ("Mag only", ("mag",)),
    ]
    isolation_results = []
    for case_label, sensors in isolation_cases:
        result = run_mekf_simulation(
            dt_filter=1.0,
            t_total=120.0,
            omega0=omega_tumble,
            q0_true=q0_true,
            rng_seed=42,
            label=case_label,
            J_truth=J_truth,
            enabled_sensors=sensors,
        )
        ss_err = mean_over_tail_window(result["times"], result["att_error"], 30.0)
        print(f"  {case_label:12s}: tail error = {ss_err:.4f} deg")
        isolation_results.append(result)

    # ==================================================================
    # 4. Comparison to static estimates (Wahba q-method)
    # ==================================================================
    print("\n--- Comparison to Static Estimates ---")
    # Run Wahba q-method at same measurement epochs
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hw2"))
    from attitude_estimation import q_method  # noqa: E402

    # Run separate static estimation
    rng_w = np.random.default_rng(99)
    model_w, data_w, bid_w = initialize_truth_model_data(J_truth, q0_true, omega_tumble)

    gyro_w = GyroSimulator(rng_w, 1.0)
    dt_mj = float(model_w.mj_model.opt.timestep)
    steps_per = int(1.0 / dt_mj)

    n_wahba = 300
    t_w = np.arange(1, n_wahba + 1, dtype=float)
    wahba_errors = np.zeros(n_wahba)
    mekf_w = MEKF(
        q0_true.copy(), np.zeros(3),
        np.diag([np.deg2rad(10)**2]*3 + [1e-12]*3),
        GYRO_ARW, GYRO_SIGMA_U,
    )
    # Give MEKF a bad initial guess too
    dq_init = rotvec_to_quat(rng_w.normal(0, np.deg2rad(5), size=3))
    mekf_w.q = quat_mult(q0_true, dq_init)
    mekf_w.q /= np.linalg.norm(mekf_w.q)
    mekf_errors = np.zeros(n_wahba)

    for i in range(n_wahba):
        for _ in range(steps_per):
            mjo_step(model_w, data_w)
        mjo_forward(model_w, data_w)

        omega_true = get_body_rate_eci_body(data_w, bid_w)
        mag_truth = data_w.sensors.measure("mag", noisy=False).copy()
        sun_truth = data_w.sensors.measure("orbit_sun_body", noisy=False).copy()
        hor_truth = data_w.sensors.measure("orbit_horizon_body", noisy=False).copy()

        R_wb = data_w.xmat[bid_w].reshape(3, 3)
        C_IL_now = data_w.frame.C_IL
        R_eci_body = C_IL_now @ R_wb
        q_true_now = np.zeros(4)
        mujoco.mju_mat2Quat(q_true_now, R_eci_body.flatten())
        if q_true_now[0] < 0:
            q_true_now = -q_true_now

        # Generate measurements
        y_gyro = gyro_w.measure(omega_true)
        y_mag = measure_magnetometer_affine(mag_truth, rng_w)
        y_sun = measure_sun_affine(sun_truth, rng_w)
        y_hor = measure_horizon_affine(hor_truth, rng_w)
        q_star = measure_star_tracker_quat(q_true_now, rng_w)

        # Wahba (static) — use star direction + sun + nadir
        B_eci = data_w.env.mag_field_eci
        sun_eci = data_w.env.sun_vector_eci
        nadir_eci = -data_w.orbit.R_eci / np.linalg.norm(data_w.orbit.R_eci)

        y_sun_cal = calibrate_sun(y_sun)
        y_hor_cal = calibrate_horizon(y_hor)

        body_vecs = [y_sun_cal, y_hor_cal,
                     calibrate_mag(y_mag) / np.linalg.norm(calibrate_mag(y_mag))]
        ref_vecs = [sun_eci, nadir_eci, B_eci / np.linalg.norm(B_eci)]
        weights = [1.0 / SUN_SIGMA_RAD**2, 1.0 / HOR_SIGMA_RAD**2,
                   1.0 / (MAG_SIGMA / np.linalg.norm(B_eci))**2]

        q_wahba = q_method(body_vecs, ref_vecs, weights)
        wahba_errors[i] = np.rad2deg(quat_error_angle(q_wahba, q_true_now))

        # MEKF
        y_gyro_cal = calibrate_gyro(y_gyro)
        mekf_w.predict(y_gyro_cal, 1.0)

        B_hat = B_eci / np.linalg.norm(B_eci)
        R_mag_dir = (MAG_SIGMA / np.linalg.norm(B_eci))**2 * np.eye(3)
        mekf_w.update_vector(calibrate_mag(y_mag) / np.linalg.norm(calibrate_mag(y_mag)),
                             B_hat, R_mag_dir)
        mekf_w.update_vector(y_sun_cal, sun_eci, SUN_SIGMA_RAD**2 * np.eye(3))
        mekf_w.update_vector(y_hor_cal, nadir_eci, HOR_SIGMA_RAD**2 * np.eye(3))
        mekf_w.update_quaternion(q_star, R_STAR_BODY)

        mekf_errors[i] = np.rad2deg(quat_error_angle(mekf_w.q, q_true_now))

    print(f"  Wahba (static) mean error:  {np.mean(wahba_errors):.4f} deg")
    print(
        f"  MEKF (recursive) mean error: {mean_after_time(t_w, mekf_errors, 50.0):.4f} deg "
        f"(after 50 s)"
    )
    print("  MEKF converged by ~50 s")

    # ==================================================================
    # 5. Convergence study (different initial conditions)
    # ==================================================================
    print("\n--- Convergence Study ---")
    init_errors_deg = [1.0, 5.0, 15.0, 30.0, 60.0]
    conv_results = []
    for err_deg in init_errors_deg:
        err_rad = np.deg2rad(err_deg)
        # Fixed direction perturbation for reproducibility
        rng_conv = np.random.default_rng(77)
        axis = rng_conv.normal(size=3)
        axis /= np.linalg.norm(axis)
        dq_init = rotvec_to_quat(err_rad * axis)
        q0_est = quat_mult(q0_true, dq_init)
        q0_est /= np.linalg.norm(q0_est)

        result = run_mekf_simulation(
            dt_filter=1.0, t_total=300.0,
            omega0=omega_tumble, q0_true=q0_true,
            q0_est=q0_est, beta0_est=np.zeros(3),
            P0_diag=(err_rad**2, (1e-6)**2),
            rng_seed=42, label=f"Init error {err_deg} deg",
            J_truth=J_truth,
        )
        conv_results.append(result)
        ss = mean_over_tail_window(result["times"], result["att_error"], steady_state_window_s)
        print(f"  Init error {err_deg:5.1f} deg: "
              f"converged to {ss:.4f} deg, "
              f"final NEES = {result['nees'][-1]:.2f}")

    # ==================================================================
    # PLOTS
    # ==================================================================

    # Figure 1: Baseline MEKF performance
    fig1, axes1 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes1[0]
    t = baseline["times"]
    ax.plot(t, baseline["att_error"], "b-", lw=0.8, label="Attitude error")
    ax.plot(
        t,
        baseline["P_att_norm_bound"],
        "r--",
        lw=0.8,
        label="99.7% attitude-norm bound",
    )
    ax.set_ylabel("Attitude error (deg)")
    ax.set_title("MEKF Attitude Estimation — Baseline (1 Hz, slow tumble)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    ax = axes1[1]
    bias_labels = ["$\\beta_x$", "$\\beta_y$", "$\\beta_z$"]
    for k, lab in enumerate(bias_labels):
        ax.plot(t, baseline["bias_error"][:, k] * 1e6, lw=0.8, label=lab)
        ax.plot(
            t,
            3 * baseline["P_bias_sigma_axes"][:, k] * 1e6,
            ls="--",
            lw=0.8,
            color=ax.lines[-1].get_color(),
            alpha=0.7,
        )
        ax.plot(
            t,
            -3 * baseline["P_bias_sigma_axes"][:, k] * 1e6,
            ls="--",
            lw=0.8,
            color=ax.lines[-1].get_color(),
            alpha=0.7,
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Bias error ($\\mu$rad/s)")
    ax.set_title("Gyro Bias Estimation Error with Per-Axis 3$\\sigma$ Bounds", fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1.savefig(plot_dir / "mekf_baseline.png", dpi=200, bbox_inches="tight")
    print(f"\nPlot saved: {plot_dir / 'mekf_baseline.png'}")

    # Figure 2: Sample rate study
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes2[0]
    for result in rate_results:
        ax.plot(result["times"], result["att_error"], lw=0.8,
                label=result["label"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Attitude error (deg)")
    ax.set_title("Sample Rate Comparison", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    ax = axes2[1]
    ss_errors = [
        mean_over_tail_window(r["times"], r["att_error"], steady_state_window_s)
        for r in rate_results
    ]
    ax.bar([f"{r} Hz" for r in rates], ss_errors, color="C0", alpha=0.7,
           edgecolor="black")
    ax.set_ylabel("Steady-state error (deg)")
    ax.set_title(
        f"Steady-State Error vs Sample Rate (last {steady_state_window_s:.0f} s)",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(ss_errors):
        ax.text(i, v * 1.05, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    fig2.suptitle("MEKF Sample Rate Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig(plot_dir / "mekf_sample_rate.png", dpi=200, bbox_inches="tight")
    print(f"Plot saved: {plot_dir / 'mekf_sample_rate.png'}")

    # Figure 3: Sensor isolation study
    fig_iso, ax_iso = plt.subplots(figsize=(12, 5))
    for result in isolation_results:
        ax_iso.plot(result["times"], result["att_error"], lw=0.8, label=result["label"])
    ax_iso.set_xlabel("Time (s)")
    ax_iso.set_ylabel("Attitude error (deg)")
    ax_iso.set_title("MEKF Sensor Isolation at 1 Hz", fontsize=14)
    ax_iso.legend(fontsize=9)
    ax_iso.grid(True, alpha=0.3)
    ax_iso.set_yscale("log")
    plt.tight_layout()
    fig_iso.savefig(plot_dir / "mekf_sensor_isolation.png", dpi=200, bbox_inches="tight")
    print(f"Plot saved: {plot_dir / 'mekf_sensor_isolation.png'}")

    # Figure 4: Comparison to Wahba
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3.plot(t_w, wahba_errors, "C1-", lw=0.8, alpha=0.7, label="Wahba (static)")
    ax3.plot(t_w, mekf_errors, "C0-", lw=0.8, alpha=0.7, label="MEKF (recursive)")
    ax3.axhline(np.mean(wahba_errors), color="C1", ls="--", lw=1,
                label=f"Wahba mean = {np.mean(wahba_errors):.4f} deg")
    mekf_mean_after_convergence = mean_after_time(t_w, mekf_errors, 50.0)
    ax3.axhline(mekf_mean_after_convergence, color="C0", ls="--", lw=1,
                label=f"MEKF mean after 50 s = {mekf_mean_after_convergence:.4f} deg")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Attitude error (deg)")
    ax3.set_title("MEKF vs Static Wahba Estimation", fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale("log")
    plt.tight_layout()
    fig3.savefig(plot_dir / "mekf_vs_wahba.png", dpi=200, bbox_inches="tight")
    print(f"Plot saved: {plot_dir / 'mekf_vs_wahba.png'}")

    # Figure 5: NEES consistency
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    nees_data = baseline["nees"]
    ax4.plot(baseline["times"], nees_data, "b-", lw=0.6, alpha=0.7,
             label="NEES")
    n_dof = 6
    lower = chi2.ppf(0.025, n_dof)
    upper = chi2.ppf(0.975, n_dof)
    ax4.axhline(n_dof, color="k", ls="-", lw=1, label=f"Expected ({n_dof})")
    ax4.axhline(lower, color="r", ls="--", lw=1, label=f"95% bounds [{lower:.1f}, {upper:.1f}]")
    ax4.axhline(upper, color="r", ls="--", lw=1)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("NEES")
    ax4.set_title("Normalized Estimation Error Squared (NEES) — Consistency Check", fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    finite_nees = nees_data[np.isfinite(nees_data) & (nees_data > 0.0)]
    if finite_nees.size:
        ax4.set_yscale("log")
        ymin = max(min(float(np.min(finite_nees)), lower) * 0.8, 1e-2)
        ymax = max(float(np.max(finite_nees)), upper) * 1.2
        ax4.set_ylim(ymin, ymax)
    plt.tight_layout()
    fig4.savefig(plot_dir / "mekf_nees.png", dpi=200, bbox_inches="tight")
    print(f"Plot saved: {plot_dir / 'mekf_nees.png'}")

    # Figure 6: Convergence study
    fig5, ax5 = plt.subplots(figsize=(12, 5))
    for result in conv_results:
        ax5.plot(result["times"], result["att_error"], lw=0.8,
                 label=result["label"])
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Attitude error (deg)")
    ax5.set_title("MEKF Convergence from Different Initial Conditions", fontsize=14)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale("log")
    plt.tight_layout()
    fig5.savefig(plot_dir / "mekf_convergence.png", dpi=200, bbox_inches="tight")
    print(f"Plot saved: {plot_dir / 'mekf_convergence.png'}")

    plt.close("all")
    print("\nDone.")


if __name__ == "__main__":
    main()
