"""Shared helpers for HW2 examples.

Contains ISS parameters, inertia perturbation, rotor momentum computation,
quaternion utilities, surface model, and MuJoCo scenario setup used across
all HW2 scripts.
"""

from __future__ import annotations

import tempfile
from typing import Optional

import mujoco
import numpy as np
from scipy.linalg import expm

from mujoco_orbit import compile
from mujoco_orbit.constants import R_EARTH
from mujoco_orbit.core.config import (
    MuJoCoCfg,
    OrbitCfg,
    ReactionWheelCfg,
    ScenarioCfg,
    SurfaceCfg,
)
from mujoco_orbit.core.scenario import Scenario
from mujoco_orbit.orbit.elements import keplerian_to_cartesian

# ---------------------------------------------------------------------------
# ISS parameters
# ---------------------------------------------------------------------------

ISS_MASS = 420_000.0     # kg
ISS_IXX = 128e6          # kg*m^2
ISS_IYY = 107e6
ISS_IZZ = 201e6
J_NOMINAL = np.diag([ISS_IXX, ISS_IYY, ISS_IZZ])

ALT_KM = 410.0           # km
INC_DEG = 51.6            # deg

SOLAR_NORMAL = np.array([0.0, 0.0, 1.0])  # body +Z

OMEGA_RPM = 10.0
OMEGA_RAD_S = OMEGA_RPM * 2.0 * np.pi / 60.0  # rad/s

INERTIA_RATIO_MIN = 1.2


# ---------------------------------------------------------------------------
# Linear algebra helpers
# ---------------------------------------------------------------------------

def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from 3-vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


# ---------------------------------------------------------------------------
# Inertia perturbation
# ---------------------------------------------------------------------------

def perturb_inertia(
    J: np.ndarray,
    eigenvalue_sigma: float = 0.03,
    axis_sigma_deg: float = 3.0,
) -> np.ndarray:
    """Perturb inertia via eigendecomposition.

    J = V D V^T  ->  J_tilde = V_tilde D_tilde V_tilde^T

    where D_tilde = D(I + diag(d)), d ~ N(0, sigma^2)
    and   V_tilde = V exp(skew(v)), v ~ N(0, sigma_axis^2)
    """
    D_vals, V = np.linalg.eigh(J)
    d = np.random.randn(3) * eigenvalue_sigma
    D_tilde = np.diag(D_vals * (1.0 + d))
    v = np.random.randn(3) * np.deg2rad(axis_sigma_deg)
    V_tilde = V @ expm(skew(v))
    return V_tilde @ D_tilde @ V_tilde.T


# ---------------------------------------------------------------------------
# Rotor momentum (superspin + dynamic balance)
# ---------------------------------------------------------------------------

def compute_rotor_momentum(
    J: np.ndarray,
    omega: np.ndarray,
    inertia_ratio: float = 1.2,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Compute rotor angular momentum h for dynamic balance + superspin.

    Returns:
        h: rotor momentum vector, shape (3,).
        lam: effective spin-axis inertia.
        I_trans_max: maximum transverse effective inertia.
        I_trans: transverse eigenvalues, shape (2,).
    """
    omega_hat = omega / np.linalg.norm(omega)
    omega_mag = np.linalg.norm(omega)
    J_omega = J @ omega

    # Transverse effective inertias
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

    lam = max(inertia_ratio * I_trans_max,
              np.dot(J_omega, omega_hat) / omega_mag)
    h = lam * omega - J_omega
    return h, lam, I_trans_max, I_trans


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------

def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Quaternion [w,x,y,z] -> 3x3 rotation matrix (world-from-body)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)],
    ])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (scalar-first [w,x,y,z])."""
    return quat_to_rotmat(q) @ v


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def quat_error_angle(q_est: np.ndarray, q_true: np.ndarray) -> float:
    """Total rotation angle between two quaternions, in radians."""
    # delta_q = q_est * q_true^{-1}
    # For unit quaternions, q^{-1} = [w, -x, -y, -z]
    q_true_inv = np.array([q_true[0], -q_true[1], -q_true[2], -q_true[3]])
    # Quaternion product
    w1, x1, y1, z1 = q_est
    w2, x2, y2, z2 = q_true_inv
    dw = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return 2.0 * np.arccos(np.clip(abs(dw), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Surface model
# ---------------------------------------------------------------------------

def make_surfaces() -> list[SurfaceCfg]:
    """ISS flat-plate surface model for drag/SRP."""
    surfaces: list[SurfaceCfg] = []
    sa_area = 375.0
    for y_off in [-45.0, -35.0, 35.0, 45.0]:
        surfaces.append(SurfaceCfg(
            body_name="iss",
            center_of_pressure_body=np.array([0.0, y_off, 0.5]),
            normal_body=np.array([0.0, 0.0, 1.0]),
            area=sa_area, drag_coeff=2.2, srp_coeff=1.8))
        surfaces.append(SurfaceCfg(
            body_name="iss",
            center_of_pressure_body=np.array([0.0, y_off, -0.5]),
            normal_body=np.array([0.0, 0.0, -1.0]),
            area=sa_area, drag_coeff=2.2, srp_coeff=1.8))
    for y_off in [-25.0, -15.0, 15.0, 20.0, 25.0, 30.0]:
        surfaces.append(SurfaceCfg(
            body_name="iss",
            center_of_pressure_body=np.array([0.0, y_off, 3.0]),
            normal_body=np.array([0.0, 0.0, 1.0]),
            area=75.0, drag_coeff=2.2, srp_coeff=1.5))
    surfaces.append(SurfaceCfg(
        body_name="iss",
        center_of_pressure_body=np.array([36.5, 0.0, 0.0]),
        normal_body=np.array([1.0, 0.0, 0.0]),
        area=180.0, drag_coeff=2.2, srp_coeff=1.8))
    surfaces.append(SurfaceCfg(
        body_name="iss",
        center_of_pressure_body=np.array([-36.5, 0.0, 0.0]),
        normal_body=np.array([-1.0, 0.0, 0.0]),
        area=180.0, drag_coeff=2.2, srp_coeff=1.8))
    surfaces.append(SurfaceCfg(
        body_name="iss",
        center_of_pressure_body=np.array([0.0, 54.25, 0.0]),
        normal_body=np.array([0.0, 1.0, 0.0]),
        area=250.0, drag_coeff=2.2, srp_coeff=1.5))
    surfaces.append(SurfaceCfg(
        body_name="iss",
        center_of_pressure_body=np.array([0.0, -54.25, 0.0]),
        normal_body=np.array([0.0, -1.0, 0.0]),
        area=250.0, drag_coeff=2.2, srp_coeff=1.5))
    return surfaces


# ---------------------------------------------------------------------------
# MuJoCo scenario setup
# ---------------------------------------------------------------------------

def make_iss_xml(J: np.ndarray, extra_xml: str = "") -> str:
    """Generate MuJoCo XML string for the ISS with given inertia.

    Args:
        J: 3x3 inertia matrix (may have off-diagonal terms).
        extra_xml: additional XML to insert inside the <body> element
                   (e.g. sensor sites).

    Returns:
        Path to a temporary XML file.
    """
    fi = f"{J[0,0]} {J[1,1]} {J[2,2]} {J[0,1]} {J[0,2]} {J[1,2]}"
    xml_str = f"""<mujoco model="iss_hw2">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="iss" pos="0 0 0">
      <freejoint name="base"/>
      <inertial pos="0 0 0" mass="{ISS_MASS}" fullinertia="{fi}"/>
      <geom name="modules" type="box" size="36.5 2.1 2.1"
            rgba="0.8 0.8 0.8 1" mass="0"/>
      <geom name="truss" type="box" size="2.3 54.25 2.3"
            rgba="0.6 0.6 0.6 1" mass="0"/>
      <geom name="sa_port" type="box" size="6.0 17.0 0.05"
            pos="0 -37.25 0" rgba="0.2 0.2 0.5 0.7" mass="0"/>
      <geom name="sa_starboard" type="box" size="6.0 17.0 0.05"
            pos="0 37.25 0" rgba="0.2 0.2 0.5 0.7" mass="0"/>
      <geom name="rad_port" type="box" size="1.7 11.5 0.03"
            pos="0 -20.0 3.0" rgba="0.9 0.9 0.9 0.6" mass="0"/>
      <geom name="rad_starboard" type="box" size="1.7 11.5 0.03"
            pos="0 20.0 3.0" rgba="0.9 0.9 0.9 0.6" mass="0"/>
{extra_xml}
    </body>
  </worldbody>
</mujoco>"""
    tmpfile = tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False)
    tmpfile.write(xml_str)
    tmpfile.flush()
    return tmpfile.name


def build_scenario(
    J: np.ndarray,
    h: np.ndarray,
    dt: float = 0.002,
    extra_xml: str = "",
    use_magnetic: bool = False,
) -> tuple[Scenario, list[float]]:
    """Build a compiled Scenario with reaction wheels set for rotor momentum h.

    Returns:
        scenario: compiled Scenario ready to step.
        rw_speeds: initial wheel speeds (already set on scenario).
    """
    xml_path = make_iss_xml(J, extra_xml)
    a_km = R_EARTH + ALT_KM
    R_eci, V_eci = keplerian_to_cartesian(
        a=a_km, e=0.0, inc=np.deg2rad(INC_DEG), raan=0.0, argp=0.0, nu=0.0)

    RW_INERTIA = 50.0
    rw_configs = []
    rw_speeds = []
    axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    for i, axis in enumerate(axes):
        speed = h[i] / RW_INERTIA
        rw_configs.append(ReactionWheelCfg(
            body_name="iss", axis_body=axis.astype(float), inertia=RW_INERTIA))
        rw_speeds.append(speed)

    cfg = ScenarioCfg(
        orbit=OrbitCfg(R_eci=R_eci, V_eci=V_eci),
        mujoco=MuJoCoCfg(xml_path=xml_path, dt=dt),
        surfaces=make_surfaces(),
        reaction_wheels=rw_configs,
        use_j2=True, use_drag=True, use_srp=True,
        use_magnetic=use_magnetic,
    )
    scenario = compile(cfg)

    for i, speed in enumerate(rw_speeds):
        scenario.actuator_state.rw_speed[i] = speed
    scenario.actuator_state.update_rw_momentum()

    return scenario, rw_speeds


def set_sun_pointing_attitude(scenario: Scenario, sun_eci: np.ndarray) -> None:
    """Set body attitude so +Z body points toward sun_eci."""
    C_LI = scenario.frame_cache.C_LI
    body_z_eci = sun_eci / np.linalg.norm(sun_eci)
    body_x_eci = np.array([0.0, 0.0, 1.0])
    body_y_eci = np.cross(body_z_eci, body_x_eci)
    body_y_eci /= np.linalg.norm(body_y_eci)
    body_x_eci = np.cross(body_y_eci, body_z_eci)
    R_body_eci = np.column_stack([body_x_eci, body_y_eci, body_z_eci])
    R_body_lvlh = C_LI @ R_body_eci
    q_init = np.zeros(4)
    mujoco.mju_mat2Quat(q_init, R_body_lvlh.flatten())
    scenario.mjd.qpos[3:7] = q_init
