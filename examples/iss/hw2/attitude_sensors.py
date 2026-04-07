"""HW2 Section 3 — Attitude Sensors for the ISS.

Simulates five attitude sensors onboard the ISS with realistic noise models
derived from published specifications. Uses MuJoCo sensors (gyro, framequat)
for truth where possible, then adds sensor-specific noise.

Sensors modeled:
  1. Rate Gyroscope (Honeywell GG1320AN-equivalent)
  2. Star Tracker (Sodern SED36-equivalent)
  3. Fine Sun Sensor (Adcole two-axis digital)
  4. Three-axis Magnetometer (Honeywell HMC2003-equivalent)
  5. Earth Horizon Sensor (infrared scanning)

Each sensor section documents:
  - Published specifications and source
  - Covariance matrix derivation
  - Noise model implementation

The script validates error statistics via Monte Carlo (N=10000 samples) and
plots histograms comparing empirical distributions to the design covariances.

Usage:
    uv run python examples/iss/hw2/attitude_sensors.py
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from mujoco_orbit import step

from common import (
    J_NOMINAL,
    SOLAR_NORMAL,
    OMEGA_RAD_S,
    perturb_inertia,
    compute_rotor_momentum,
    build_scenario,
    set_sun_pointing_attitude,
)

np.random.seed(42)


# ===================================================================
# SENSOR SPECIFICATIONS AND COVARIANCE DERIVATIONS
# ===================================================================
#
# All noise values are 1-sigma unless stated otherwise.
#
# -------------------------------------------------------------------
# 1. RATE GYROSCOPE — Honeywell GG1320AN ring laser gyro
# -------------------------------------------------------------------
# Published specs (Honeywell GG1320AN datasheet):
#   - Angle Random Walk (ARW):  0.002 deg/sqrt(hr)
#   - Bias instability:         0.003 deg/hr
#   - Scale factor error:       5 ppm (neglected here)
#
# Covariance derivation:
#   ARW gives the white-noise spectral density of the rate measurement.
#     ARW = 0.002 deg/sqrt(hr)
#         = 0.002 * (pi/180) / 60  rad/sqrt(s)
#         = 5.818e-7 rad/sqrt(s)
#
#   For discrete samples at interval dt:
#     sigma_rate = ARW / sqrt(dt)  [rad/s per axis]
#
#   The gyro measurement covariance (per sample) is:
#     R_gyro = (ARW^2 / dt) * I_3
#
#   Bias is modeled as a constant offset drawn once per "power cycle":
#     sigma_bias = 0.003 deg/hr = 0.003 * (pi/180) / 3600 = 1.454e-8 rad/s
#
#   Total gyro covariance (white noise only, bias separate):
#     R_gyro = diag(sigma_rate^2, sigma_rate^2, sigma_rate^2)
# -------------------------------------------------------------------

GYRO_ARW_DEG_SQRT_HR = 0.002  # deg/sqrt(hr)
GYRO_ARW = GYRO_ARW_DEG_SQRT_HR * (np.pi / 180.0) / 60.0  # rad/sqrt(s)

GYRO_BIAS_DEG_HR = 0.003  # deg/hr
GYRO_BIAS_SIGMA = GYRO_BIAS_DEG_HR * (np.pi / 180.0) / 3600.0  # rad/s


def gyro_covariance(dt: float) -> np.ndarray:
    """3x3 measurement covariance for the rate gyro (white noise only)."""
    return (GYRO_ARW**2 / dt) * np.eye(3)


# -------------------------------------------------------------------
# 2. STAR TRACKER — Sodern SED36 (representative of ISS trackers)
# -------------------------------------------------------------------
# Published specs (Sodern SED36 datasheet / ESA literature):
#   - Cross-boresight accuracy:  5 arcsec (1-sigma)  per axis
#   - Roll (about boresight):   25 arcsec (1-sigma)
#   - Update rate:              4 Hz
#   - FOV:                      20 deg x 20 deg
#
# Covariance derivation (vector observation model):
#   A star tracker measures the direction to a known star in body frame.
#   The measurement is a unit vector b_meas = R_true^T * r_eci + noise.
#
#   The angular error is anisotropic: tighter cross-boresight, looser
#   about the boresight (roll) axis.  In the sensor frame (boresight = z):
#     sigma_x = sigma_y = 5 arcsec = 5 * (pi / 648000) = 2.424e-5 rad
#     sigma_z (roll)    = 25 arcsec                     = 1.212e-4 rad
#
#   For a unit-vector observation b with true value b0, the measurement
#   noise covariance on the angular error is:
#     R_star = diag(sigma_x^2, sigma_y^2, sigma_z^2)  in sensor frame
#
#   Rotated to body frame: R_body = C_bs * R_star * C_bs^T
#   where C_bs is sensor-to-body rotation.  Here we assume boresight
#   aligned with body +X (typical ISS S5/S6 truss-mounted tracker).
#
#   For the vector observation covariance used in Wahba's problem:
#     sigma_v^2 = sigma_cross^2 per axis of the projected vector error.
# -------------------------------------------------------------------

STAR_CROSS_ARCSEC = 5.0   # arcsec, 1-sigma per cross-boresight axis
STAR_ROLL_ARCSEC = 25.0   # arcsec, 1-sigma about boresight

STAR_CROSS_RAD = STAR_CROSS_ARCSEC * (np.pi / 648000.0)  # rad
STAR_ROLL_RAD = STAR_ROLL_ARCSEC * (np.pi / 648000.0)    # rad

STAR_BORESIGHT_BODY = np.array([1.0, 0.0, 0.0])


def star_tracker_covariance_sensor() -> np.ndarray:
    """3x3 angular error covariance in sensor frame [rad^2]."""
    return np.diag([STAR_CROSS_RAD**2, STAR_CROSS_RAD**2, STAR_ROLL_RAD**2])


def star_tracker_covariance_body() -> np.ndarray:
    """3x3 angular error covariance in body frame [rad^2]."""
    C_bs = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    return C_bs @ star_tracker_covariance_sensor() @ C_bs.T


# -------------------------------------------------------------------
# 3. FINE SUN SENSOR — Adcole two-axis digital sun sensor
# -------------------------------------------------------------------
# Published specs (Adcole Corp / ISS ADCS heritage):
#   - Accuracy:  0.05 deg (1-sigma) per axis
#   - FOV:       +-64 deg (half angle)
#
# Covariance derivation:
#   sigma = 0.05 deg = 8.727e-4 rad per cross-axis.
#   R_sun = sigma^2 * (I_3 - s*s^T)   (rank-2 matrix)
#   Scalar weight for Wahba: sigma_v^2 = sigma^2
# -------------------------------------------------------------------

SUN_SENSOR_DEG = 0.05
SUN_SENSOR_RAD = np.deg2rad(SUN_SENSOR_DEG)


def sun_sensor_covariance() -> float:
    """Scalar variance for sun-vector angular error [rad^2]."""
    return SUN_SENSOR_RAD**2


# -------------------------------------------------------------------
# 4. THREE-AXIS MAGNETOMETER — Honeywell HMC2003 (representative)
# -------------------------------------------------------------------
# Published specs (Honeywell HMC2003 datasheet):
#   - Noise density:    30 nT/sqrt(Hz) per axis
#   - At 10 Hz BW:      sigma_B = 30 * sqrt(10) = 94.9 nT ~ 100 nT
#   - Direction accuracy (at |B|=35 uT): ~ 0.16 deg
# -------------------------------------------------------------------

MAG_NOISE_DENSITY = 30e-9   # T/sqrt(Hz) per axis
MAG_BANDWIDTH_HZ = 10.0
MAG_SIGMA = MAG_NOISE_DENSITY * np.sqrt(MAG_BANDWIDTH_HZ)  # T per axis


def magnetometer_covariance() -> np.ndarray:
    """3x3 measurement covariance for the magnetometer [T^2]."""
    return MAG_SIGMA**2 * np.eye(3)


def magnetometer_direction_variance(B_magnitude: float) -> float:
    """Angular variance of the magnetometer direction estimate [rad^2]."""
    return (MAG_SIGMA / B_magnitude)**2


# -------------------------------------------------------------------
# 5. EARTH HORIZON SENSOR — Barnes 13-230 (ISS heritage, infrared)
# -------------------------------------------------------------------
# Published specs (NASA ISS ADCS reference / Barnes Engineering):
#   - Nadir accuracy:  0.1 deg (1-sigma) per axis
# -------------------------------------------------------------------

HORIZON_SENSOR_DEG = 0.1
HORIZON_SENSOR_RAD = np.deg2rad(HORIZON_SENSOR_DEG)


def horizon_sensor_covariance() -> float:
    """Scalar variance for nadir-vector angular error [rad^2]."""
    return HORIZON_SENSOR_RAD**2


# ===================================================================
# SENSOR MEASUREMENT FUNCTIONS
# ===================================================================


def measure_gyro(
    sensordata_gyro: np.ndarray, dt: float,
    bias: np.ndarray, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate rate gyroscope measurement (MuJoCo gyro truth + noise)."""
    R = gyro_covariance(dt)
    sigma = GYRO_ARW / np.sqrt(dt)
    noise = rng.normal(0.0, sigma, size=3)
    return sensordata_gyro + bias + noise, R


def measure_star_tracker(
    R_world_body: np.ndarray, C_IL: np.ndarray,
    star_eci: np.ndarray, rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Simulate star tracker unit-vector observation.

    Cross-boresight noise only — roll about the boresight does not affect
    individual star direction measurements (roll accuracy comes from the
    pattern of multiple stars, not individual vector observations).
    """
    R_eci_body = C_IL @ R_world_body
    b_true = R_eci_body.T @ star_eci
    b_true /= np.linalg.norm(b_true)

    boresight = STAR_BORESIGHT_BODY
    if abs(np.dot(b_true, boresight)) > 0.99:
        e1 = np.cross(boresight, np.array([0.0, 1.0, 0.0]))
    else:
        e1 = np.cross(boresight, b_true)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(boresight, e1)
    e2 /= np.linalg.norm(e2)

    delta = rng.normal(0.0, STAR_CROSS_RAD) * e1 \
          + rng.normal(0.0, STAR_CROSS_RAD) * e2
    b_meas = b_true + np.cross(delta, b_true)
    b_meas /= np.linalg.norm(b_meas)
    return b_meas, STAR_CROSS_RAD**2


def measure_sun_sensor(
    R_world_body: np.ndarray, C_IL: np.ndarray,
    sun_eci: np.ndarray, rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Simulate fine sun sensor measurement (isotropic angular noise)."""
    R_eci_body = C_IL @ R_world_body
    s_true = R_eci_body.T @ sun_eci
    s_true /= np.linalg.norm(s_true)
    noise_angle = rng.normal(0.0, SUN_SENSOR_RAD, size=3)
    s_meas = s_true + np.cross(noise_angle, s_true)
    s_meas /= np.linalg.norm(s_meas)
    return s_meas, SUN_SENSOR_RAD**2


def measure_magnetometer(
    R_world_body: np.ndarray, C_IL: np.ndarray,
    B_eci: np.ndarray, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate three-axis magnetometer measurement (additive Gaussian)."""
    R_eci_body = C_IL @ R_world_body
    B_true = R_eci_body.T @ B_eci
    noise = rng.normal(0.0, MAG_SIGMA, size=3)
    return B_true + noise, magnetometer_covariance()


def measure_horizon_sensor(
    R_world_body: np.ndarray, C_IL: np.ndarray,
    nadir_eci: np.ndarray, rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Simulate Earth horizon sensor measurement (isotropic angular noise)."""
    R_eci_body = C_IL @ R_world_body
    n_true = R_eci_body.T @ nadir_eci
    n_true /= np.linalg.norm(n_true)
    noise_angle = rng.normal(0.0, HORIZON_SENSOR_RAD, size=3)
    n_meas = n_true + np.cross(noise_angle, n_true)
    n_meas /= np.linalg.norm(n_meas)
    return n_meas, HORIZON_SENSOR_RAD**2


# ===================================================================
# MAIN — Build scenario, run sensors, Monte Carlo validation
# ===================================================================

# Extra XML for MuJoCo sensors
_SENSOR_XML = """\
      <!-- Sensor sites -->
      <site name="imu" pos="0 0 0"/>
      <site name="star_tracker" pos="10 0 0"/>
    </body>
  </worldbody>

  <sensor>
    <gyro name="gyro" site="imu"/>
    <framequat name="body_quat" objtype="body" objname="iss"/>
  </sensor>
</mujoco>"""


def main() -> None:
    print("HW2 Section 3 — Attitude Sensors for the ISS")
    print("=" * 60)

    # ---- Print sensor specifications ----
    print("\n--- Sensor Specifications ---")
    print(f"1. Rate Gyroscope (Honeywell GG1320AN-class):")
    print(f"   ARW       = {GYRO_ARW_DEG_SQRT_HR} deg/sqrt(hr) = {GYRO_ARW:.3e} rad/sqrt(s)")
    print(f"   Bias inst = {GYRO_BIAS_DEG_HR} deg/hr = {GYRO_BIAS_SIGMA:.3e} rad/s")
    print(f"\n2. Star Tracker (Sodern SED36-class):")
    print(f"   Cross-boresight = {STAR_CROSS_ARCSEC} arcsec = {STAR_CROSS_RAD:.3e} rad")
    print(f"   Roll            = {STAR_ROLL_ARCSEC} arcsec = {STAR_ROLL_RAD:.3e} rad")
    print(f"\n3. Fine Sun Sensor (Adcole-class):")
    print(f"   Accuracy = {SUN_SENSOR_DEG} deg = {SUN_SENSOR_RAD:.3e} rad")
    print(f"\n4. Three-Axis Magnetometer (HMC2003-class):")
    print(f"   Noise density = {MAG_NOISE_DENSITY*1e9:.0f} nT/sqrt(Hz)")
    print(f"   At {MAG_BANDWIDTH_HZ} Hz BW: sigma = {MAG_SIGMA*1e9:.1f} nT = {MAG_SIGMA:.3e} T")
    B_typical = 35e-6
    print(f"   Direction accuracy (at |B|={B_typical*1e6:.0f} uT): "
          f"{np.rad2deg(MAG_SIGMA / B_typical):.3f} deg")
    print(f"\n5. Earth Horizon Sensor (Barnes 13-230-class):")
    print(f"   Accuracy = {HORIZON_SENSOR_DEG} deg = {HORIZON_SENSOR_RAD:.3e} rad")

    # ---- Print covariance matrices ----
    dt_sample = 0.1
    print("\n--- Covariance Matrices ---")
    R_gyro = gyro_covariance(dt_sample)
    print(f"\nGyro R (dt={dt_sample}s):")
    print(f"  diag = [{R_gyro[0,0]:.3e}, {R_gyro[1,1]:.3e}, {R_gyro[2,2]:.3e}] rad^2/s^2")
    print(f"  sigma_rate = {np.sqrt(R_gyro[0,0]):.3e} rad/s = "
          f"{np.rad2deg(np.sqrt(R_gyro[0,0]))*3600:.4f} deg/hr")

    R_star_body = star_tracker_covariance_body()
    print(f"\nStar tracker R (body frame):")
    print(f"  diag = [{R_star_body[0,0]:.3e}, {R_star_body[1,1]:.3e}, {R_star_body[2,2]:.3e}] rad^2")

    R_mag = magnetometer_covariance()
    print(f"\nMagnetometer R: sigma = {np.sqrt(R_mag[0,0])*1e9:.1f} nT per axis")
    print(f"Sun sensor sigma^2 = {sun_sensor_covariance():.3e} rad^2")
    print(f"Horizon sensor sigma^2 = {horizon_sensor_covariance():.3e} rad^2")

    # ---- Build scenario with MuJoCo sensors ----
    J = perturb_inertia(J_NOMINAL)
    omega_desired = OMEGA_RAD_S * SOLAR_NORMAL
    h, lam, I_trans_max, _ = compute_rotor_momentum(J, omega_desired, 1.2)

    # The extra_xml must close the body/worldbody and add sensor block
    # We need to handle the XML structure properly — build_scenario's make_iss_xml
    # inserts extra_xml before closing </body>, so we add sites + close + sensors
    extra_xml = _SENSOR_XML
    # build_scenario closes </body></worldbody></mujoco> after extra_xml,
    # but we need to override that. Instead, build scenario normally and
    # just use the body rotation matrix for truth.
    dt = 0.002
    scenario, rw_speeds = build_scenario(J, h, dt=dt,
        extra_xml="      <site name=\"imu\" pos=\"0 0 0\"/>",
        use_magnetic=True)
    bid = scenario.body_id("iss")

    # Set initial attitude: body +Z -> sun
    sun_eci = scenario.env_cache.sun_vector_eci.copy()
    set_sun_pointing_attitude(scenario, sun_eci)
    scenario.mjd.qvel[3:6] = omega_desired
    mujoco.mj_forward(scenario.mjm, scenario.mjd)

    # Let simulation settle
    print("\nRunning 100 steps to settle dynamics...")
    for _ in range(100):
        step(scenario)
    mujoco.mj_forward(scenario.mjm, scenario.mjd)

    # ---- Reference vectors ----
    R_eci_sc = scenario.orbit.R_eci
    nadir_eci = -R_eci_sc / np.linalg.norm(R_eci_sc)
    B_eci = scenario.env_cache.mag_field_eci
    sun_eci = scenario.env_cache.sun_vector_eci

    # Catalog star (Canopus: RA=96 deg, Dec=-53 deg)
    ra_star, dec_star = np.deg2rad(96.0), np.deg2rad(-53.0)
    star_eci = np.array([
        np.cos(dec_star) * np.cos(ra_star),
        np.cos(dec_star) * np.sin(ra_star),
        np.sin(dec_star)])

    R_wb = scenario.body_com_rotmat(bid)
    C_IL = scenario.frame_cache.C_IL

    print(f"\nTruth state at measurement epoch:")
    omega_true = scenario.mjd.qvel[3:6].copy()
    print(f"  omega (body frame)  = [{omega_true[0]:+.6f}, {omega_true[1]:+.6f}, "
          f"{omega_true[2]:+.6f}] rad/s")
    print(f"  |B_eci|             = {np.linalg.norm(B_eci)*1e6:.2f} uT")

    # ==================================================================
    # MONTE CARLO VALIDATION
    # ==================================================================
    N_MC = 10_000
    rng = np.random.default_rng(seed=12345)
    print(f"\n--- Monte Carlo Validation (N = {N_MC}) ---")

    gyro_bias = rng.normal(0.0, GYRO_BIAS_SIGMA, size=3)
    print(f"Gyro bias draw: [{gyro_bias[0]:.3e}, {gyro_bias[1]:.3e}, {gyro_bias[2]:.3e}] rad/s")

    gyro_errors = np.zeros((N_MC, 3))
    star_angle_errors = np.zeros(N_MC)
    sun_angle_errors = np.zeros(N_MC)
    mag_errors = np.zeros((N_MC, 3))
    horizon_angle_errors = np.zeros(N_MC)

    omega_true_sensor = omega_true.copy()

    for i in range(N_MC):
        omega_meas, _ = measure_gyro(omega_true_sensor, dt_sample, gyro_bias, rng)
        gyro_errors[i] = omega_meas - omega_true_sensor - gyro_bias

        b_meas, _ = measure_star_tracker(R_wb, C_IL, star_eci, rng)
        R_eci_body = C_IL @ R_wb
        b_true = R_eci_body.T @ star_eci; b_true /= np.linalg.norm(b_true)
        star_angle_errors[i] = np.arccos(np.clip(np.dot(b_meas, b_true), -1, 1))

        s_meas, _ = measure_sun_sensor(R_wb, C_IL, sun_eci, rng)
        s_true = R_eci_body.T @ sun_eci; s_true /= np.linalg.norm(s_true)
        sun_angle_errors[i] = np.arccos(np.clip(np.dot(s_meas, s_true), -1, 1))

        B_meas, _ = measure_magnetometer(R_wb, C_IL, B_eci, rng)
        B_true = R_eci_body.T @ B_eci
        mag_errors[i] = B_meas - B_true

        n_meas, _ = measure_horizon_sensor(R_wb, C_IL, nadir_eci, rng)
        n_true = R_eci_body.T @ nadir_eci; n_true /= np.linalg.norm(n_true)
        horizon_angle_errors[i] = np.arccos(np.clip(np.dot(n_meas, n_true), -1, 1))

    # ---- Print statistics ----
    print("\n--- Error Statistics (empirical vs design) ---")
    gyro_std = np.std(gyro_errors, axis=0)
    gyro_design = GYRO_ARW / np.sqrt(dt_sample)
    print(f"\nGyro white noise (rad/s):")
    for ax, label in enumerate(["x", "y", "z"]):
        print(f"  {label}: empirical sigma = {gyro_std[ax]:.4e}, "
              f"design = {gyro_design:.4e}, ratio = {gyro_std[ax]/gyro_design:.4f}")

    star_rms = np.sqrt(np.mean(star_angle_errors**2))
    star_design_rms = STAR_CROSS_RAD * np.sqrt(2)
    print(f"\nStar tracker angular error (arcsec):")
    print(f"  empirical RMS = {np.rad2deg(star_rms)*3600:.2f}, "
          f"design RMS = {np.rad2deg(star_design_rms)*3600:.2f}")

    sun_rms = np.sqrt(np.mean(sun_angle_errors**2))
    sun_design_rms = SUN_SENSOR_RAD * np.sqrt(2)
    print(f"\nSun sensor angular error (deg):")
    print(f"  empirical RMS = {np.rad2deg(sun_rms):.5f}, design RMS = {np.rad2deg(sun_design_rms):.5f}")

    mag_std = np.std(mag_errors, axis=0)
    print(f"\nMagnetometer noise (nT):")
    for ax, label in enumerate(["x", "y", "z"]):
        print(f"  {label}: empirical sigma = {mag_std[ax]*1e9:.2f}, "
              f"design = {MAG_SIGMA*1e9:.2f}, ratio = {mag_std[ax]/MAG_SIGMA:.4f}")

    hor_rms = np.sqrt(np.mean(horizon_angle_errors**2))
    hor_design_rms = HORIZON_SENSOR_RAD * np.sqrt(2)
    print(f"\nHorizon sensor angular error (deg):")
    print(f"  empirical RMS = {np.rad2deg(hor_rms):.5f}, design RMS = {np.rad2deg(hor_design_rms):.5f}")

    # ==================================================================
    # PLOTS
    # ==================================================================
    plot_dir = pathlib.Path(__file__).parent / "plots"
    plot_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 1. Gyro noise histogram
    ax = axes[0, 0]
    for k, label in enumerate(["x", "y", "z"]):
        ax.hist(gyro_errors[:, k] * 1e6, bins=60, alpha=0.5, density=True,
                label=f"$\\omega_{label}$")
    x_plot = np.linspace(-4*gyro_design*1e6, 4*gyro_design*1e6, 200)
    ax.plot(x_plot, 1/(gyro_design*1e6*np.sqrt(2*np.pi))
            * np.exp(-0.5*(x_plot/(gyro_design*1e6))**2),
            'k--', lw=1.5, label=f"Design $\\sigma$={gyro_design*1e6:.2f} $\\mu$rad/s")
    ax.set_xlabel("Rate error ($\\mu$rad/s)"); ax.set_ylabel("Density")
    ax.set_title("Gyroscope White Noise"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 2. Star tracker angular error
    ax = axes[0, 1]
    ax.hist(np.rad2deg(star_angle_errors) * 3600, bins=60, density=True, alpha=0.7, color="C1", label="Empirical")
    theta_plot = np.linspace(0, np.rad2deg(STAR_CROSS_RAD*5)*3600, 200)
    sigma_as = np.rad2deg(STAR_CROSS_RAD) * 3600
    rayleigh_pdf = (theta_plot / sigma_as**2) * np.exp(-theta_plot**2 / (2*sigma_as**2))
    ax.plot(theta_plot, rayleigh_pdf, 'k--', lw=1.5, label=f"Rayleigh($\\sigma$={sigma_as:.1f}\")")
    ax.set_xlabel("Angular error (arcsec)"); ax.set_ylabel("Density")
    ax.set_title(f"Star Tracker Error ($\\sigma_{{cross}}$={STAR_CROSS_ARCSEC}\")"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 3. Sun sensor angular error
    ax = axes[0, 2]
    ax.hist(np.rad2deg(sun_angle_errors), bins=60, density=True, alpha=0.7, color="C2", label="Empirical")
    theta_plot = np.linspace(0, SUN_SENSOR_DEG * 5, 200)
    rayleigh_pdf = (theta_plot / SUN_SENSOR_DEG**2) * np.exp(-theta_plot**2 / (2*SUN_SENSOR_DEG**2))
    ax.plot(theta_plot, rayleigh_pdf, 'k--', lw=1.5, label=f"Rayleigh($\\sigma$={SUN_SENSOR_DEG:.3f}$^\\circ$)")
    ax.set_xlabel("Angular error (deg)"); ax.set_ylabel("Density")
    ax.set_title("Fine Sun Sensor Error"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 4. Magnetometer noise
    ax = axes[1, 0]
    for k, label in enumerate(["$B_x$", "$B_y$", "$B_z$"]):
        ax.hist(mag_errors[:, k] * 1e9, bins=60, alpha=0.5, density=True, label=label)
    x_plot = np.linspace(-4*MAG_SIGMA*1e9, 4*MAG_SIGMA*1e9, 200)
    ax.plot(x_plot, 1/(MAG_SIGMA*1e9*np.sqrt(2*np.pi))
            * np.exp(-0.5*(x_plot/(MAG_SIGMA*1e9))**2),
            'k--', lw=1.5, label=f"Design $\\sigma$={MAG_SIGMA*1e9:.1f} nT")
    ax.set_xlabel("Field error (nT)"); ax.set_ylabel("Density")
    ax.set_title("Magnetometer Noise"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 5. Horizon sensor angular error
    ax = axes[1, 1]
    ax.hist(np.rad2deg(horizon_angle_errors), bins=60, density=True, alpha=0.7, color="C4", label="Empirical")
    theta_plot = np.linspace(0, HORIZON_SENSOR_DEG * 5, 200)
    rayleigh_pdf = (theta_plot / HORIZON_SENSOR_DEG**2) * np.exp(-theta_plot**2 / (2*HORIZON_SENSOR_DEG**2))
    ax.plot(theta_plot, rayleigh_pdf, 'k--', lw=1.5, label=f"Rayleigh($\\sigma$={HORIZON_SENSOR_DEG:.2f}$^\\circ$)")
    ax.set_xlabel("Angular error (deg)"); ax.set_ylabel("Density")
    ax.set_title("Earth Horizon Sensor Error"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 6. Summary table
    ax = axes[1, 2]; ax.axis("off")
    table_data = [
        ["Sensor", "Design $\\sigma$", "Empirical $\\sigma$", "Ratio"],
        ["Gyro (x)", f"{gyro_design*1e6:.2f} $\\mu$rad/s",
         f"{gyro_std[0]*1e6:.2f} $\\mu$rad/s", f"{gyro_std[0]/gyro_design:.3f}"],
        ["Star tracker", f"{np.rad2deg(star_design_rms)*3600:.1f}\" RMS",
         f"{np.rad2deg(star_rms)*3600:.1f}\" RMS", f"{star_rms/star_design_rms:.3f}"],
        ["Sun sensor", f"{SUN_SENSOR_DEG:.3f} deg",
         f"{np.rad2deg(sun_rms/np.sqrt(2)):.4f} deg", f"{sun_rms/sun_design_rms:.3f}"],
        ["Magnetometer", f"{MAG_SIGMA*1e9:.1f} nT",
         f"{np.mean(mag_std)*1e9:.1f} nT", f"{np.mean(mag_std)/MAG_SIGMA:.3f}"],
        ["Horizon sensor", f"{HORIZON_SENSOR_DEG:.2f} deg",
         f"{np.rad2deg(hor_rms/np.sqrt(2)):.4f} deg", f"{hor_rms/hor_design_rms:.3f}"],
    ]
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0], loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.1, 1.4)
    ax.set_title("Error Statistics Summary", fontsize=10, pad=10)

    fig.suptitle("ISS Attitude Sensor Noise Validation (Monte Carlo, N=10000)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(plot_dir / "attitude_sensors.png", dpi=200, bbox_inches="tight")
    print(f"\nPlot saved: {plot_dir / 'attitude_sensors.png'}")

    # ==================================================================
    # TIME-SERIES DEMO
    # ==================================================================
    print("\n--- Time-series demo: 5 minutes of sensor readings ---")
    t_demo = 300.0
    n_demo_steps = int(t_demo / dt)
    record_interval = int(1.0 / dt)
    n_records = n_demo_steps // record_interval + 1

    times = np.zeros(n_records)
    gyro_meas_hist = np.zeros((n_records, 3))
    gyro_true_hist = np.zeros((n_records, 3))
    sun_error_hist = np.zeros(n_records)
    mag_error_hist = np.zeros(n_records)
    nadir_error_hist = np.zeros(n_records)

    gyro_bias_ts = rng.normal(0.0, GYRO_BIAS_SIGMA, size=3)

    def record_sensors(idx, t):
        times[idx] = t
        C_IL_now = scenario.frame_cache.C_IL
        R_wb_now = scenario.body_com_rotmat(bid)
        R_eci_body = C_IL_now @ R_wb_now

        omega_t = scenario.mjd.qvel[3:6].copy()
        omega_m, _ = measure_gyro(omega_t, 1.0, gyro_bias_ts, rng)
        gyro_true_hist[idx] = omega_t; gyro_meas_hist[idx] = omega_m

        sun_now = scenario.env_cache.sun_vector_eci
        s_m, _ = measure_sun_sensor(R_wb_now, C_IL_now, sun_now, rng)
        s_true = R_eci_body.T @ sun_now; s_true /= np.linalg.norm(s_true)
        sun_error_hist[idx] = np.rad2deg(np.arccos(np.clip(np.dot(s_m, s_true), -1, 1)))

        B_now = scenario.env_cache.mag_field_eci
        B_m, _ = measure_magnetometer(R_wb_now, C_IL_now, B_now, rng)
        B_true = R_eci_body.T @ B_now
        b_hat_true = B_true / np.linalg.norm(B_true)
        b_hat_meas = B_m / np.linalg.norm(B_m)
        mag_error_hist[idx] = np.rad2deg(np.arccos(np.clip(np.dot(b_hat_meas, b_hat_true), -1, 1)))

        R_eci_now = scenario.orbit.R_eci
        nadir_now = -R_eci_now / np.linalg.norm(R_eci_now)
        n_m, _ = measure_horizon_sensor(R_wb_now, C_IL_now, nadir_now, rng)
        n_true = R_eci_body.T @ nadir_now; n_true /= np.linalg.norm(n_true)
        nadir_error_hist[idx] = np.rad2deg(np.arccos(np.clip(np.dot(n_m, n_true), -1, 1)))

    record_sensors(0, 0.0)
    rec_idx = 1
    for i in range(n_demo_steps):
        step(scenario)
        if (i + 1) % record_interval == 0 and rec_idx < n_records:
            record_sensors(rec_idx, (i + 1) * dt)
            rec_idx += 1

    n_rec = rec_idx
    t_plot = times[:n_rec] / 60.0

    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
    ax = axes2[0, 0]
    for k, label in enumerate(["x", "y", "z"]):
        ax.plot(t_plot, gyro_true_hist[:n_rec, k], '-', lw=0.8, alpha=0.5, label=f"true $\\omega_{label}$")
        ax.plot(t_plot, gyro_meas_hist[:n_rec, k], '.', ms=1, alpha=0.3)
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Angular rate (rad/s)")
    ax.set_title("Gyroscope: True vs Measured"); ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    ax = axes2[0, 1]
    ax.plot(t_plot, sun_error_hist[:n_rec], '.', ms=2, alpha=0.5, color="C2")
    ax.axhline(SUN_SENSOR_DEG, color='k', ls='--', lw=1, label=f"1$\\sigma$ = {SUN_SENSOR_DEG} deg")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Angular error (deg)")
    ax.set_title("Fine Sun Sensor Error"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes2[1, 0]
    ax.plot(t_plot, mag_error_hist[:n_rec], '.', ms=2, alpha=0.5, color="C3")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Direction error (deg)")
    ax.set_title("Magnetometer Direction Error"); ax.grid(True, alpha=0.3)

    ax = axes2[1, 1]
    ax.plot(t_plot, nadir_error_hist[:n_rec], '.', ms=2, alpha=0.5, color="C4")
    ax.axhline(HORIZON_SENSOR_DEG, color='k', ls='--', lw=1, label=f"1$\\sigma$ = {HORIZON_SENSOR_DEG} deg")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Angular error (deg)")
    ax.set_title("Earth Horizon Sensor Error"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig2.suptitle("Sensor Measurements Over 5 Minutes", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig(plot_dir / "attitude_sensors_timeseries.png", dpi=200, bbox_inches="tight")
    print(f"Plot saved: {plot_dir / 'attitude_sensors_timeseries.png'}")
    plt.close("all")
    print("\nDone.")


if __name__ == "__main__":
    main()
