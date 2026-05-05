"""HW3 Section 1 — Attitude Sensors Revisited.

Builds on the mujoco_orbit sensor framework (``data.sensors.measure()``) to
implement three categories of realistic error models as specified by the
assignment:

  1. Vector/bearing sensors (magnetometer, sun sensor, horizon sensor):
       y = M * y_ideal + b + w,   w ~ N(0, W)
     M contains scale-factor and cross-axis misalignment errors.
     b is a constant bias vector.  These extend the base framework's
     additive/rotation noise and constant-bias model.

  2. Star tracker — returns a full attitude quaternion:
       q_meas = q_true * dq(delta_theta),   delta_theta ~ N(0, R_star)
     Errors sampled from an *anisotropic* Gaussian in axis-angle space
     (tighter cross-boresight, looser about the roll/boresight axis).

  3. Gyroscope — affine model with time-varying (random-walk) bias:
       y_gyro = M * omega_true + b(t) + w,   w ~ N(0, W_gyro)
       b(t+dt) = b(t) + eta,   eta ~ N(0, Q_b * dt)
     Extends the framework's constant gyro bias to a random walk process.

All parameters are justified from published sensor datasheets (cited below).

Monte Carlo validation (N = 10 000) demonstrates correct error statistics.

Usage:
    pixi run python ISS/hw3/sensors_revisited.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

# Import helpers from hw2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hw2"))
from common import (
    ISS_MASS,
    J_NOMINAL,
    OMEGA_RAD_S,
    SOLAR_NORMAL,
    perturb_inertia,
    set_sun_pointing_attitude,
)

from mujoco_orbit import MjoData, MjoModel, mjo_forward, mjo_step
from mujoco_orbit.constants import R_EARTH
from tests.mujoco_orbit.reference.orbit.elements import keplerian_to_cartesian

np.random.seed(42)
plt.rcParams.update({
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 16,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


# =====================================================================
# 1. VECTOR / BEARING SENSORS  —  y = M * y_ideal + b + w
# =====================================================================
#
# These build on mujoco_orbit's sensor framework.  Truth vectors come
# from the orbit-aware sensor callback (orbit_sun_, orbit_horizon_,
# orbit_star_ prefixes) or from MuJoCo's built-in magnetometer.  The
# affine error model adds scale-factor/misalignment (M), constant bias
# (b), and additive white noise (w) on top of the truth.
#
# After calibration, measurements can be corrected:
#   y_cal = M^{-1}(y - b) = y_ideal + M^{-1} w  ≈  y_ideal + w
# =====================================================================

# -------------------------------------------------------------------
# 1a. Magnetometer — Honeywell HMC2003-class
# -------------------------------------------------------------------
# Ref: Honeywell HMC2003 datasheet; Wertz & Larson SMAD Ch. 11.7
#
# Scale-factor error:   ±500 ppm per axis (residual after cal)
# Cross-axis coupling:  0.29 deg (0.5% sin-equivalent)
# Hard-iron bias:       50 nT per axis (post-cal residual)
# Noise density:        30 nT/√Hz per axis
# At 10 Hz bandwidth:   σ ≈ 95 nT per axis
# -------------------------------------------------------------------

MAG_SCALE = np.array([5e-4, -3e-4, 4e-4])
MAG_MISALIGN_RAD = np.deg2rad(0.29)
MAG_M = (np.eye(3) + np.diag(MAG_SCALE)
         + np.array([[0, MAG_MISALIGN_RAD, -MAG_MISALIGN_RAD / 2],
                      [-MAG_MISALIGN_RAD / 2, 0, MAG_MISALIGN_RAD],
                      [MAG_MISALIGN_RAD, -MAG_MISALIGN_RAD / 2, 0]]))
MAG_BIAS = np.array([50e-9, -30e-9, 40e-9])  # T
MAG_NOISE_DENSITY = 30e-9   # T/√Hz
MAG_BW_HZ = 10.0
MAG_SIGMA = MAG_NOISE_DENSITY * np.sqrt(MAG_BW_HZ)
MAG_W = MAG_SIGMA**2 * np.eye(3)
MAG_M_INV = np.linalg.inv(MAG_M)

# -------------------------------------------------------------------
# 1b. Fine Sun Sensor — Adcole two-axis digital
# -------------------------------------------------------------------
# Ref: Adcole Corp specs; ISS ADCS heritage documentation
#
# Scale-factor error:  ±200 ppm
# Misalignment:        0.01 deg
# Null offset (bias):  0.005 deg equivalent
# Noise (1σ):          0.05 deg per axis
# -------------------------------------------------------------------

SUN_SCALE = np.array([2e-4, -1e-4, 1.5e-4])
SUN_MISALIGN_RAD = np.deg2rad(0.01)
SUN_M = (np.eye(3) + np.diag(SUN_SCALE)
         + np.array([[0, SUN_MISALIGN_RAD, 0],
                      [-SUN_MISALIGN_RAD, 0, SUN_MISALIGN_RAD / 2],
                      [0, -SUN_MISALIGN_RAD / 2, 0]]))
SUN_BIAS = np.deg2rad(0.005) * np.array([1.0, -0.5, 0.3])
SUN_SIGMA_RAD = np.deg2rad(0.05)
SUN_W = SUN_SIGMA_RAD**2 * np.eye(3)
SUN_M_INV = np.linalg.inv(SUN_M)

# -------------------------------------------------------------------
# 1c. Earth Horizon Sensor — Barnes 13-230 (infrared)
# -------------------------------------------------------------------
# Ref: NASA ISS ADCS reference; Barnes Engineering specs
#
# Scale-factor error:  ±500 ppm
# Misalignment:        0.05 deg
# Bias:                0.02 deg equivalent
# Noise (1σ):          0.1 deg per axis
# -------------------------------------------------------------------

HOR_SCALE = np.array([5e-4, -2e-4, 3e-4])
HOR_MISALIGN_RAD = np.deg2rad(0.05)
HOR_M = (np.eye(3) + np.diag(HOR_SCALE)
         + np.array([[0, HOR_MISALIGN_RAD, 0],
                      [-HOR_MISALIGN_RAD, 0, HOR_MISALIGN_RAD / 2],
                      [0, -HOR_MISALIGN_RAD / 2, 0]]))
HOR_BIAS = np.deg2rad(0.02) * np.array([0.7, -0.3, 0.5])
HOR_SIGMA_RAD = np.deg2rad(0.1)
HOR_W = HOR_SIGMA_RAD**2 * np.eye(3)
HOR_M_INV = np.linalg.inv(HOR_M)


# =====================================================================
# 2. STAR TRACKER  —  quaternion with anisotropic rotation error
# =====================================================================
#
# Ref: Sodern SED36 datasheet; ESA AOCS literature
#
# q_meas = dq(δθ) ⊗ q_true,   δθ ~ N(0, R_star)
#
# The mujoco_orbit framework supports isotropic rotation noise for
# quaternion sensors; HW3 requires *anisotropic* noise (different
# σ for cross-boresight vs roll), so we implement it directly.
#
# σ_cross = 5 arcsec = 2.424e-5 rad  (per cross-boresight axis)
# σ_roll  = 25 arcsec = 1.212e-4 rad (about boresight)
#
# Boresight is body +X (ISS truss-mounted tracker).
# =====================================================================

STAR_CROSS_ARCSEC = 5.0
STAR_ROLL_ARCSEC = 25.0
STAR_CROSS_RAD = STAR_CROSS_ARCSEC * np.pi / 648000.0
STAR_ROLL_RAD = STAR_ROLL_ARCSEC * np.pi / 648000.0

# Sensor frame (boresight=z) → body frame (boresight=x) rotation
C_SENSOR_BODY = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
R_STAR_SENSOR = np.diag([STAR_CROSS_RAD**2, STAR_CROSS_RAD**2, STAR_ROLL_RAD**2])
R_STAR_BODY = C_SENSOR_BODY @ R_STAR_SENSOR @ C_SENSOR_BODY.T
L_STAR = np.linalg.cholesky(R_STAR_BODY)


# =====================================================================
# 3. GYROSCOPE  —  affine model with random-walk bias
# =====================================================================
#
# Ref: Honeywell GG1320AN RLG datasheet; Titterton & Weston Ch. 4
#
# y_gyro = M ω_true + b(t) + w,    w ~ N(0, σ_v²/dt · I)
# b(t+dt) = b(t) + η,              η ~ N(0, σ_u² dt · I)
#
# The mujoco_orbit gyro sensor has constant bias (drawn once at init).
# HW3 extends this to a random-walk process on the bias.
#
# GG1320AN parameters:
#   ARW  = 0.002 deg/√hr = 5.818e-7 rad/√s  (white noise PSD)
#   BI   = 0.003 deg/hr  = 1.454e-8 rad/s    (bias instability)
#   σ_u  ≈ 1e-9 rad/s/√s                     (bias random walk)
#   Scale factor: 5 ppm per axis
#   Misalignment: 10 μrad
# =====================================================================

GYRO_ARW_DEG_SQRT_HR = 0.002
GYRO_ARW = GYRO_ARW_DEG_SQRT_HR * np.pi / 180.0 / 60.0
GYRO_BIAS_INST = 0.003 * np.pi / 180.0 / 3600.0
GYRO_SIGMA_U = 1.0e-9   # rad/s/√s  (bias random walk PSD)

GYRO_SCALE = np.array([5e-6, -3e-6, 4e-6])
GYRO_MISALIGN_RAD = 10e-6
GYRO_M = (np.eye(3) + np.diag(GYRO_SCALE)
          + np.array([[0, GYRO_MISALIGN_RAD, 0],
                       [-GYRO_MISALIGN_RAD, 0, GYRO_MISALIGN_RAD],
                       [0, -GYRO_MISALIGN_RAD, 0]]))
GYRO_M_INV = np.linalg.inv(GYRO_M)


# =====================================================================
# MEASUREMENT FUNCTIONS
# =====================================================================

def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton quaternion product [w,x,y,z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def rotvec_to_quat(v: np.ndarray) -> np.ndarray:
    """Rotation vector → quaternion [w,x,y,z]."""
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = v / angle
    return np.array([np.cos(angle / 2), *(np.sin(angle / 2) * axis)])


def measure_magnetometer_affine(
    truth_body: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Apply affine error model y = M B_body + b + w to magnetometer truth."""
    w = rng.normal(0.0, MAG_SIGMA, size=3)
    return MAG_M @ truth_body + MAG_BIAS + w


def measure_sun_affine(
    truth_body: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Apply affine error model to sun direction truth, re-normalize."""
    w = rng.normal(0.0, SUN_SIGMA_RAD, size=3)
    y = SUN_M @ truth_body + SUN_BIAS + w
    return y / np.linalg.norm(y)


def measure_horizon_affine(
    truth_body: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Apply affine error model to nadir direction truth, re-normalize."""
    w = rng.normal(0.0, HOR_SIGMA_RAD, size=3)
    y = HOR_M @ truth_body + HOR_BIAS + w
    return y / np.linalg.norm(y)


def measure_star_tracker_quat(
    q_true: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Star tracker quaternion with anisotropic rotation noise.

    q_meas = q_true ⊗ dq(δθ),  δθ ~ N(0, R_STAR_BODY) (body-frame error).
    """
    delta_theta = L_STAR @ rng.standard_normal(3)
    dq = rotvec_to_quat(delta_theta)
    # RIGHT multiplication: body-frame error convention
    q_meas = quat_mult(q_true, dq)
    q_meas /= np.linalg.norm(q_meas)
    if q_meas[0] < 0:
        q_meas = -q_meas
    return q_meas


class GyroSimulator:
    """Gyroscope with affine error model and random-walk bias.

    Wraps the mujoco_orbit gyro truth with y = M ω + b(t) + w,
    where b evolves as a discrete random walk each time measure() is called.
    """

    def __init__(self, rng: np.random.Generator, dt: float):
        self.rng = rng
        self.dt = dt
        self.bias = rng.normal(0.0, GYRO_BIAS_INST, size=3)

    def measure(self, omega_true: np.ndarray, dt: float | None = None) -> np.ndarray:
        """Return y_gyro = M ω + b(t) + w and advance bias random walk."""
        dt_step = self.dt if dt is None else dt
        sigma_w = GYRO_ARW / np.sqrt(dt_step)
        w = self.rng.normal(0.0, sigma_w, size=3)
        y = GYRO_M @ omega_true + self.bias + w

        # Advance bias random walk
        self.bias += self.rng.normal(0.0, GYRO_SIGMA_U * np.sqrt(dt_step), size=3)
        return y


# =====================================================================
# CALIBRATION (remove known M, b to isolate residual noise)
# =====================================================================

def calibrate_mag(y: np.ndarray) -> np.ndarray:
    return MAG_M_INV @ (y - MAG_BIAS)


def calibrate_sun(y: np.ndarray) -> np.ndarray:
    corrected = SUN_M_INV @ (y - SUN_BIAS)
    return corrected / np.linalg.norm(corrected)


def calibrate_horizon(y: np.ndarray) -> np.ndarray:
    corrected = HOR_M_INV @ (y - HOR_BIAS)
    return corrected / np.linalg.norm(corrected)


def calibrate_gyro(y: np.ndarray) -> np.ndarray:
    """Remove known M from gyro. Bias is estimated by the MEKF."""
    return GYRO_M_INV @ y


# =====================================================================
# ISS MODEL WITH SENSORS
# =====================================================================

def make_iss_sensor_xml(J: np.ndarray) -> str:
    """Generate ISS XML with full sensor suite for mujoco_orbit."""
    fi = f"{J[0,0]} {J[1,1]} {J[2,2]} {J[0,1]} {J[0,2]} {J[1,2]}"
    # Canopus: RA=96°, Dec=-53°
    ra, dec = np.deg2rad(96.0), np.deg2rad(-53.0)
    star_x = np.cos(dec) * np.cos(ra)
    star_y = np.cos(dec) * np.sin(ra)
    star_z = np.sin(dec)

    xml = f"""<mujoco model="iss_hw3">
  <compiler angle="radian"/>
  <size nuser_sensor="7"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="iss" pos="0 0 0">
      <freejoint name="base"/>
      <inertial pos="0 0 0" mass="{ISS_MASS}" fullinertia="{fi}"/>
      <geom name="modules" type="box" size="36.5 2.1 2.1"
            rgba="0.8 0.8 0.8 1" mass="0"/>
      <geom name="truss" type="box" size="2.3 54.25 2.3"
            rgba="0.6 0.6 0.6 1" mass="0"/>
      <site name="imu" pos="0 0 0"/>
      <site name="star_tracker" pos="10 0 0"/>
    </body>
  </worldbody>
  <sensor>
    <gyro name="gyro" site="imu" noise="0"/>
    <magnetometer name="mag" site="imu" noise="0"/>
    <framequat name="body_quat" objtype="body" objname="iss"/>
    <user name="orbit_sun_body" objtype="site" objname="imu"
          datatype="axis" needstage="pos" dim="3" noise="0"/>
    <user name="orbit_horizon_body" objtype="site" objname="imu"
          datatype="axis" needstage="pos" dim="3" noise="0"/>
    <user name="orbit_star_canopus" objtype="site" objname="star_tracker"
          datatype="axis" needstage="pos" dim="3" noise="0"
          user="{star_x:.15f} {star_y:.15f} {star_z:.15f} 0 0 0 0"/>
  </sensor>
</mujoco>"""
    tmpfile = tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False)
    tmpfile.write(xml)
    tmpfile.flush()
    return tmpfile.name


# =====================================================================
# MAIN — Monte Carlo Validation
# =====================================================================

def main() -> None:
    print("HW3 Section 1 — Attitude Sensors Revisited")
    print("=" * 60)

    # ---- Print all sensor parameters ----
    print("\n--- 1. Vector/Bearing Sensors (y = M y_ideal + b + w) ---")
    print("\nMagnetometer (HMC2003-class):")
    print(f"  Scale perturbation:  {MAG_SCALE * 1e6} ppm")
    print(f"  Misalignment:        {np.rad2deg(MAG_MISALIGN_RAD):.3f} deg")
    print(f"  Hard-iron bias:      {MAG_BIAS * 1e9} nT")
    print(f"  Noise sigma:         {MAG_SIGMA * 1e9:.1f} nT/axis")
    print(f"  M matrix:\n{MAG_M}")
    print(f"  W = {MAG_SIGMA**2:.3e} T^2 * I_3")

    print("\nSun Sensor (Adcole-class):")
    print(f"  Scale perturbation:  {SUN_SCALE * 1e6} ppm")
    print(f"  Misalignment:        {np.rad2deg(SUN_MISALIGN_RAD):.4f} deg")
    print(f"  Bias:                {np.rad2deg(np.linalg.norm(SUN_BIAS)):.4f} deg equiv")
    print(f"  Noise sigma:         {np.rad2deg(SUN_SIGMA_RAD):.3f} deg/axis")

    print("\nHorizon Sensor (Barnes 13-230-class):")
    print(f"  Scale perturbation:  {HOR_SCALE * 1e6} ppm")
    print(f"  Misalignment:        {np.rad2deg(HOR_MISALIGN_RAD):.3f} deg")
    print(f"  Bias:                {np.rad2deg(np.linalg.norm(HOR_BIAS)):.4f} deg equiv")
    print(f"  Noise sigma:         {np.rad2deg(HOR_SIGMA_RAD):.2f} deg/axis")

    print("\n--- 2. Star Tracker (quaternion error model) ---")
    print(f"  sigma_cross: {STAR_CROSS_ARCSEC} arcsec = {STAR_CROSS_RAD:.4e} rad")
    print(f"  sigma_roll:  {STAR_ROLL_ARCSEC} arcsec = {STAR_ROLL_RAD:.4e} rad")
    print(f"  R_star (body frame):\n{R_STAR_BODY}")

    print("\n--- 3. Gyroscope (affine + random-walk bias) ---")
    print(f"  ARW:         {GYRO_ARW_DEG_SQRT_HR} deg/sqrt(hr) = {GYRO_ARW:.4e} rad/sqrt(s)")
    print(f"  Bias inst:   0.003 deg/hr = {GYRO_BIAS_INST:.4e} rad/s")
    print(f"  sigma_u:     {GYRO_SIGMA_U:.2e} rad/s/sqrt(s) (bias random walk)")
    print(f"  Scale error: {GYRO_SCALE * 1e6} ppm")
    print(f"  Misalignment: {GYRO_MISALIGN_RAD * 1e6:.1f} urad")

    # ---- Build scenario using mujoco_orbit ----
    J = perturb_inertia(J_NOMINAL)
    xml_path = make_iss_sensor_xml(J)

    from mujoco_orbit import OrbitInit

    a_km = R_EARTH + 410.0
    R_eci, V_eci = keplerian_to_cartesian(a=a_km, e=0.0, inc=np.deg2rad(51.6),
                                           raan=0.0, argp=0.0, nu=0.0)
    model = MjoModel.from_xml_path(xml_path, use_j2=True, use_magnetic=True)
    data = MjoData(model, orbit=OrbitInit(R_eci=R_eci, V_eci=V_eci))

    # Set attitude: +Z body → sun
    sun_eci = data.env.sun_vector_eci.copy()
    set_sun_pointing_attitude(model, data, sun_eci)
    omega0 = OMEGA_RAD_S * SOLAR_NORMAL
    data.qvel[3:6] = omega0
    mjo_forward(model, data)
    for _ in range(100):
        mjo_step(model, data)
    mjo_forward(model, data)

    bid = model.body_id("iss")

    # ---- Get truth from mujoco_orbit sensors ----
    omega_true = data.qvel[3:6].copy()
    mag_truth = data.sensors.measure("mag", noisy=False).copy()
    sun_truth = data.sensors.measure("orbit_sun_body", noisy=False).copy()
    hor_truth = data.sensors.measure("orbit_horizon_body", noisy=False).copy()

    # True quaternion (ECI-from-body) for star tracker
    R_wb = data.xmat[bid].reshape(3, 3).copy()
    C_IL = data.frame.C_IL
    R_eci_body = C_IL @ R_wb
    q_true = np.zeros(4)
    mujoco.mju_mat2Quat(q_true, R_eci_body.flatten())
    if q_true[0] < 0:
        q_true = -q_true

    print("\nTruth state at measurement epoch:")
    print(f"  omega  = [{omega_true[0]:+.6f}, {omega_true[1]:+.6f}, {omega_true[2]:+.6f}] rad/s")
    print(f"  |B|    = {np.linalg.norm(mag_truth)*1e6:.2f} uT")
    print(f"  q_true = [{q_true[0]:.6f}, {q_true[1]:.6f}, {q_true[2]:.6f}, {q_true[3]:.6f}]")

    # ---- Monte Carlo ----
    N_MC = 10_000
    rng = np.random.default_rng(seed=123)
    print(f"\n--- Monte Carlo Validation (N = {N_MC}) ---")

    mag_errors_raw = np.zeros((N_MC, 3))
    sun_angle_errors = np.zeros(N_MC)
    hor_angle_errors = np.zeros(N_MC)
    star_rot_errors = np.zeros(N_MC)
    star_axis_errors = np.zeros((N_MC, 3))
    gyro_errors = np.zeros((N_MC, 3))

    dt_gyro = 1.0
    gyro_sigma_design = GYRO_ARW / np.sqrt(dt_gyro)

    mag_systematic = (MAG_M - np.eye(3)) @ mag_truth + MAG_BIAS

    for i in range(N_MC):
        # Magnetometer: truth from framework, HW3 affine error model on top
        y_mag = measure_magnetometer_affine(mag_truth, rng)
        mag_errors_raw[i] = y_mag - mag_truth

        # Sun sensor
        y_sun = measure_sun_affine(sun_truth, rng)
        sun_angle_errors[i] = np.arccos(np.clip(np.dot(y_sun, sun_truth), -1, 1))

        # Horizon sensor
        y_hor = measure_horizon_affine(hor_truth, rng)
        hor_angle_errors[i] = np.arccos(np.clip(np.dot(y_hor, hor_truth), -1, 1))

        # Star tracker quaternion (RIGHT convention: q_meas = q_true ⊗ dq_err)
        q_meas = measure_star_tracker_quat(q_true, rng)
        q_true_inv = np.array([q_true[0], -q_true[1], -q_true[2], -q_true[3]])
        dq = quat_mult(q_true_inv, q_meas)  # extract body-frame error
        if dq[0] < 0:
            dq = -dq
        star_rot_errors[i] = 2 * np.arccos(np.clip(dq[0], 0, 1))
        star_axis_errors[i] = 2 * dq[1:4]

        # Gyro: isolate white noise (reset bias each trial)
        gyro = GyroSimulator(rng, dt_gyro)
        y_gyro = gyro.measure(omega_true)
        # Error relative to M*omega + bias = white noise only
        gyro_errors[i] = y_gyro - GYRO_M @ omega_true - gyro.bias

    # ---- Print statistics ----
    print("\n--- Error Statistics (empirical vs design) ---")

    mag_noise_only = mag_errors_raw - mag_systematic
    print("\nMagnetometer:")
    print(f"  Systematic error: [{mag_systematic[0]*1e9:.1f}, "
          f"{mag_systematic[1]*1e9:.1f}, {mag_systematic[2]*1e9:.1f}] nT")
    for k, ax in enumerate("xyz"):
        emp = np.std(mag_noise_only[:, k])
        print(f"  {ax}: sigma={emp*1e9:.1f}/{MAG_SIGMA*1e9:.1f} nT "
              f"(ratio {emp/MAG_SIGMA:.3f})")

    sun_rms = np.sqrt(np.mean(sun_angle_errors**2))
    print(f"\nSun sensor RMS error:     {np.rad2deg(sun_rms):.4f} deg")
    hor_rms = np.sqrt(np.mean(hor_angle_errors**2))
    print(f"Horizon sensor RMS error: {np.rad2deg(hor_rms):.4f} deg")

    star_std = np.std(star_axis_errors, axis=0)
    design_std = np.sqrt(np.diag(R_STAR_BODY))
    print("\nStar tracker per-axis sigma (arcsec):")
    for k, ax in enumerate("xyz"):
        print(f"  {ax}: {np.rad2deg(star_std[k])*3600:.2f} / "
              f"{np.rad2deg(design_std[k])*3600:.2f} "
              f"(ratio {star_std[k]/design_std[k]:.3f})")

    gyro_std = np.std(gyro_errors, axis=0)
    print("\nGyro white noise sigma (urad/s):")
    for k, ax in enumerate("xyz"):
        print(f"  {ax}: {gyro_std[k]*1e6:.3f} / {gyro_sigma_design*1e6:.3f} "
              f"(ratio {gyro_std[k]/gyro_sigma_design:.3f})")

    # ---- Gyro bias random walk (1-hour trajectory) ----
    print("\n--- Gyro Bias Random Walk (1-hour trajectory) ---")
    dt_rw = 1.0
    n_rw = 3600
    gyro_rw = GyroSimulator(np.random.default_rng(42), dt_rw)
    bias_hist = np.zeros((n_rw + 1, 3))
    bias_hist[0] = gyro_rw.bias.copy()
    for k in range(n_rw):
        gyro_rw.measure(np.zeros(3))
        bias_hist[k + 1] = gyro_rw.bias.copy()
    t_rw = np.arange(n_rw + 1) * dt_rw
    print(f"  Initial bias: [{bias_hist[0, 0]*1e6:.4f}, {bias_hist[0, 1]*1e6:.4f}, "
          f"{bias_hist[0, 2]*1e6:.4f}] urad/s")
    print(f"  Final bias:   [{bias_hist[-1, 0]*1e6:.4f}, {bias_hist[-1, 1]*1e6:.4f}, "
          f"{bias_hist[-1, 2]*1e6:.4f}] urad/s")

    # ---- Plotting ----
    plot_dir = pathlib.Path(__file__).parent / "plots"
    plot_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 1. Magnetometer noise (systematic removed)
    ax = axes[0, 0]
    for k, label in enumerate(["$B_x$", "$B_y$", "$B_z$"]):
        ax.hist(mag_noise_only[:, k] * 1e9, bins=60, alpha=0.5, density=True, label=label)
    x_plot = np.linspace(-4 * MAG_SIGMA * 1e9, 4 * MAG_SIGMA * 1e9, 200)
    gauss = np.exp(-0.5 * (x_plot / (MAG_SIGMA * 1e9))**2) / (MAG_SIGMA * 1e9 * np.sqrt(2 * np.pi))
    ax.plot(x_plot, gauss, "k--", lw=1.5, label=f"Design $\\sigma$={MAG_SIGMA*1e9:.1f} nT")
    ax.set_xlabel("Field error (nT)")
    ax.set_ylabel("Density")
    ax.set_title("Magnetometer Noise\n(systematic $Mb+b$ removed)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Sun sensor angular error
    ax = axes[0, 1]
    ax.hist(np.rad2deg(sun_angle_errors), bins=60, density=True, alpha=0.7, color="C2")
    th = np.linspace(0, np.rad2deg(SUN_SIGMA_RAD) * 5, 200)
    s = np.rad2deg(SUN_SIGMA_RAD)
    ax.plot(th, (th / s**2) * np.exp(-th**2 / (2 * s**2)), "k--", lw=1.5,
            label=f"Rayleigh($\\sigma$={s:.3f}$^\\circ$)")
    ax.set_xlabel("Angular error (deg)")
    ax.set_ylabel("Density")
    ax.set_title("Sun Sensor Error (affine model)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. Horizon sensor angular error
    ax = axes[0, 2]
    ax.hist(np.rad2deg(hor_angle_errors), bins=60, density=True, alpha=0.7, color="C4")
    th = np.linspace(0, np.rad2deg(HOR_SIGMA_RAD) * 5, 200)
    s = np.rad2deg(HOR_SIGMA_RAD)
    ax.plot(th, (th / s**2) * np.exp(-th**2 / (2 * s**2)), "k--", lw=1.5,
            label=f"Rayleigh($\\sigma$={s:.2f}$^\\circ$)")
    ax.set_xlabel("Angular error (deg)")
    ax.set_ylabel("Density")
    ax.set_title("Horizon Sensor Error (affine model)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. Star tracker per-axis errors
    ax = axes[1, 0]
    for k, lab in enumerate(["$\\delta\\theta_x$", "$\\delta\\theta_y$", "$\\delta\\theta_z$"]):
        ax.hist(np.rad2deg(star_axis_errors[:, k]) * 3600, bins=60,
                alpha=0.5, density=True, label=lab)
    ax.set_xlabel("Rotation error (arcsec)")
    ax.set_ylabel("Density")
    ax.set_title(f"Star Tracker Quaternion Error\n"
                 f"($\\sigma_{{cross}}$={STAR_CROSS_ARCSEC}\", "
                 f"$\\sigma_{{roll}}$={STAR_ROLL_ARCSEC}\")", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 5. Gyro white noise
    ax = axes[1, 1]
    for k, lab in enumerate(["$\\omega_x$", "$\\omega_y$", "$\\omega_z$"]):
        ax.hist(gyro_errors[:, k] * 1e6, bins=60, alpha=0.5, density=True, label=lab)
    x_plot = np.linspace(-4 * gyro_sigma_design * 1e6, 4 * gyro_sigma_design * 1e6, 200)
    gauss = np.exp(-0.5 * (x_plot / (gyro_sigma_design * 1e6))**2) / \
            (gyro_sigma_design * 1e6 * np.sqrt(2 * np.pi))
    ax.plot(x_plot, gauss, "k--", lw=1.5,
            label=f"Design $\\sigma$={gyro_sigma_design*1e6:.3f} $\\mu$rad/s")
    ax.set_xlabel("Rate error ($\\mu$rad/s)")
    ax.set_ylabel("Density")
    ax.set_title("Gyro White Noise (affine model)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 6. Gyro bias random walk
    ax = axes[1, 2]
    for k, lab in enumerate(["$b_x$", "$b_y$", "$b_z$"]):
        ax.plot(t_rw / 60, bias_hist[:, k] * 1e6, lw=0.8, label=lab)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Bias ($\\mu$rad/s)")
    ax.set_title(f"Gyro Bias Random Walk\n"
                 f"$\\sigma_u$ = {GYRO_SIGMA_U:.1e} rad/s/$\\sqrt{{s}}$", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle("HW3: Refined Sensor Error Models — Monte Carlo Validation (N=10,000)",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(plot_dir / "sensors_revisited.png", dpi=200, bbox_inches="tight")
    print(f"\nPlot saved: {plot_dir / 'sensors_revisited.png'}")
    plt.close("all")
    print("Done.")


if __name__ == "__main__":
    main()
