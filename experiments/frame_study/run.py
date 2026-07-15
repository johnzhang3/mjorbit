"""Self-contained Newton-Euler frame study (no MuJoCo).

Compares three choices for the local-simulator world coordinates:

- ``eci``: absolute Earth-centered inertial position/velocity.
- ``chief_inertial``: chief-centered position/velocity with inertially fixed axes.
- ``lvlh``: chief-centered rotating LVLH coordinates.

The local simulator is a 6-DOF rigid-body Newton-Euler propagator implemented
in this module. Single-precision runs use ``np.float32`` for every state array
and intermediate computation in the body propagator end-to-end; the chief and
truth-body reference orbits stay in float64 (they are the truth signal).

Usage:
    pixi run python experiments/frame_study/run.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable, Literal

import numpy as np

MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH = 6_378_137.0  # m
MASS = 100.0  # kg
BOX_HALF_EXTENTS = np.array([0.5, 0.3, 0.2])  # m, matches the prior MuJoCo XML
OUT_DIR = Path(__file__).with_name("out")
StudyPrecision = Literal["float64", "float32"]
Integrator = Literal["euler", "rk4", "implicit"]

ORBIT_TIMESTEP = 0.5  # s
SIM_TIMESTEP = 0.1  # s, body propagator step
N_ORBITS = 3.0
DEFAULT_INTEGRATOR: Integrator = "rk4"
TRUTH_SUBSTEPS = 10  # truth-body RK4 substeps per body propagator step

INITIAL_REL_POS_LVLH = np.array([1.0, 0.0, 5.0])  # m
INITIAL_ATTITUDE_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
INITIAL_OMEGA_BODY = np.array([0.012, -0.018, 0.009])  # rad/s


class FrameMode(StrEnum):
    ECI = "eci"
    CHIEF_INERTIAL = "chief_inertial"
    LVLH = "lvlh"


def box_inertia_diag(mass: float, half_extents: np.ndarray) -> np.ndarray:
    """Principal-axes inertia of a uniform-density box about its center."""
    full = 2.0 * np.asarray(half_extents, dtype=np.float64)
    a, b, c = full
    return mass / 12.0 * np.array(
        [b * b + c * c, a * a + c * c, a * a + b * b],
        dtype=np.float64,
    )


INERTIA_DIAG = box_inertia_diag(MASS, BOX_HALF_EXTENTS)


@dataclass
class OrbitState:
    r: np.ndarray
    v: np.ndarray


@dataclass
class FrameCache:
    c_li: np.ndarray
    c_il: np.ndarray
    omega_lvlh: np.ndarray
    omega_dot_lvlh: np.ndarray


@dataclass(frozen=True)
class Scenario:
    name: str
    semi_major_axis: float
    eccentricity: float
    inclination: float
    raan: float = 0.0
    arg_periapsis: float = 0.0
    true_anomaly: float = 0.0


@dataclass
class Result:
    scenario: Scenario
    mode: FrameMode
    max_position_error: float
    final_position_error: float
    max_relative_radius: float
    final_relative_position: np.ndarray
    max_orbital_energy_drift_rel: float
    final_orbital_energy_drift_rel: float
    max_orbital_angular_momentum_drift_rel: float
    final_orbital_angular_momentum_drift_rel: float
    energy_drift_rel: float
    angular_momentum_drift_rel: float
    max_quat_norm_error: float
    finite: bool


@dataclass
class TimeHistory:
    label: str
    scenario: Scenario
    mode: FrameMode
    integrator: Integrator
    precision: StudyPrecision
    length_unit_m: float
    times_s: np.ndarray
    position_error_m: np.ndarray
    energy_error_rel: np.ndarray
    relative_radius_m: np.ndarray
    finite: bool


def format_float_for_filename(value: float) -> str:
    """Format a float for stable, path-safe output names."""
    text = f"{value:g}".replace("-", "m").replace("+", "")
    return text.replace(".", "p")


SCENARIOS = [
    Scenario(
        name="circular_equatorial",
        semi_major_axis=R_EARTH + 400_000.0,
        eccentricity=0.0,
        inclination=0.0,
    ),
    Scenario(
        name="circular_inclined",
        semi_major_axis=R_EARTH + 400_000.0,
        eccentricity=0.0,
        inclination=np.deg2rad(51.6),
        raan=np.deg2rad(20.0),
    ),
    Scenario(
        name="elliptical_equatorial",
        semi_major_axis=R_EARTH + 2_000_000.0,
        eccentricity=0.2,
        inclination=0.0,
        true_anomaly=np.deg2rad(20.0),
    ),
    Scenario(
        name="elliptical_inclined",
        semi_major_axis=R_EARTH + 2_000_000.0,
        eccentricity=0.2,
        inclination=np.deg2rad(63.4),
        raan=np.deg2rad(30.0),
        arg_periapsis=np.deg2rad(45.0),
        true_anomaly=np.deg2rad(20.0),
    ),
]


def gravity_in_units(r: np.ndarray, length_unit_m: float) -> np.ndarray:
    """Two-body acceleration in the requested length unit per second squared."""
    radius = np.linalg.norm(r)
    mu = MU_EARTH / length_unit_m**3
    dtype = r.dtype if isinstance(r, np.ndarray) else np.float64
    mu = np.asarray(mu, dtype=dtype)
    return -mu / radius**3 * r


def rk4_orbit_step(
    state: OrbitState,
    dt: float,
    length_unit_m: float = 1.0,
) -> OrbitState:
    """RK4 step for an inertial two-body point mass (always float64)."""

    def deriv(r: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return v, gravity_in_units(r, length_unit_m)

    k1_r, k1_v = deriv(state.r, state.v)
    k2_r, k2_v = deriv(state.r + 0.5 * dt * k1_r, state.v + 0.5 * dt * k1_v)
    k3_r, k3_v = deriv(state.r + 0.5 * dt * k2_r, state.v + 0.5 * dt * k2_v)
    k4_r, k4_v = deriv(state.r + dt * k3_r, state.v + dt * k3_v)

    r_next = state.r + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
    v_next = state.v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return OrbitState(r_next, v_next)


def frame_cache(chief: OrbitState, length_unit_m: float = 1.0) -> FrameCache:
    """Build an RSW/LVLH frame from the chief ECI state."""
    r_norm = np.linalg.norm(chief.r)
    x_hat = chief.r / r_norm
    h = np.cross(chief.r, chief.v)
    z_hat = h / np.linalg.norm(h)
    y_hat = np.cross(z_hat, x_hat)

    c_li = np.stack([x_hat, y_hat, z_hat])
    c_il = c_li.T

    omega_eci = h / r_norm**2
    omega_lvlh = c_li @ omega_eci

    chief_accel = gravity_in_units(chief.r, length_unit_m)
    dh_dt = np.cross(chief.r, chief_accel)
    dr_dt = np.dot(chief.r, chief.v) / r_norm
    omega_dot_eci = dh_dt / r_norm**2 - 2.0 * h * dr_dt / r_norm**3
    omega_dot_lvlh = c_li @ omega_dot_eci

    return FrameCache(c_li, c_il, omega_lvlh, omega_dot_lvlh)


def propagate_truth_body(
    state: OrbitState,
    dt: float,
    length_unit_m: float,
    substeps: int = TRUTH_SUBSTEPS,
) -> OrbitState:
    """Advance the truth-body reference by ``dt`` with finer RK4 substeps."""
    sub_dt = dt / substeps
    for _ in range(substeps):
        state = rk4_orbit_step(state, sub_dt, length_unit_m)
    return state


def scale_orbit_state(state: OrbitState, factor: float) -> OrbitState:
    """Multiply an OrbitState's r and v by ``factor`` (e.g., orbit-unit -> meters)."""
    return OrbitState(state.r * factor, state.v * factor)


def rot_x(angle: float) -> np.ndarray:
    """Rotation matrix about x."""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_z(angle: float) -> np.ndarray:
    """Rotation matrix about z."""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def keplerian_to_cartesian(scenario: Scenario) -> OrbitState:
    """Convert one Keplerian scenario to ECI Cartesian state in SI units."""
    a = scenario.semi_major_axis
    e = scenario.eccentricity
    nu = scenario.true_anomaly
    p = a * (1.0 - e**2)
    r_pf = np.array(
        [
            p * np.cos(nu) / (1.0 + e * np.cos(nu)),
            p * np.sin(nu) / (1.0 + e * np.cos(nu)),
            0.0,
        ]
    )
    v_pf = np.sqrt(MU_EARTH / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    c_ip = (
        rot_z(scenario.raan)
        @ rot_x(scenario.inclination)
        @ rot_z(scenario.arg_periapsis)
    )
    return OrbitState(c_ip @ r_pf, c_ip @ v_pf)


def make_initial_chief(
    scenario: Scenario,
    length_unit_m: float = 1.0,
) -> tuple[OrbitState, float, float]:
    """Return a chief orbit state, mean motion, and period for a scenario."""
    mean_motion = np.sqrt(MU_EARTH / scenario.semi_major_axis**3)
    orbit_period = 2.0 * np.pi / mean_motion
    chief_si = keplerian_to_cartesian(scenario)
    chief = OrbitState(chief_si.r / length_unit_m, chief_si.v / length_unit_m)
    return chief, mean_motion, orbit_period


def initial_body_from_lvlh(
    chief: OrbitState,
    rho_lvlh: np.ndarray,
    rhod_lvlh: np.ndarray,
    length_unit_m: float = 1.0,
) -> OrbitState:
    """Convert initial LVLH relative state to absolute ECI."""
    fc = frame_cache(chief, length_unit_m)
    r = chief.r + fc.c_il @ rho_lvlh
    v = chief.v + fc.c_il @ (rhod_lvlh + np.cross(fc.omega_lvlh, rho_lvlh))
    return OrbitState(r, v)


def initial_states(
    scenario: Scenario,
    rel_vel_bias_lvlh: np.ndarray | None = None,
    length_unit_m: float = 1.0,
) -> tuple[OrbitState, OrbitState, float, float]:
    """Return initial chief/body states plus mean motion and orbit period."""
    chief0, mean_motion, orbit_period = make_initial_chief(scenario, length_unit_m)
    rho0 = INITIAL_REL_POS_LVLH / length_unit_m
    rhod0_si = np.array([0.0, -2.0 * mean_motion * INITIAL_REL_POS_LVLH[0], 0.0])
    if rel_vel_bias_lvlh is not None:
        rhod0_si = rhod0_si + np.asarray(rel_vel_bias_lvlh, dtype=np.float64).reshape(3)
    rhod0 = rhod0_si / length_unit_m
    body0 = initial_body_from_lvlh(chief0, rho0, rhod0, length_unit_m)
    return chief0, body0, mean_motion, orbit_period


def initial_local_state(
    mode: FrameMode,
    chief: OrbitState,
    body: OrbitState,
    length_unit_m: float,
    dtype: np.dtype,
) -> np.ndarray:
    """Pack initial 13-vector body state in the active frame and dtype."""
    fc = frame_cache(chief, length_unit_m)
    rho_eci = body.r - chief.r
    rel_v_eci = body.v - chief.v

    if mode == FrameMode.ECI:
        r_local = body.r
        v_local = body.v
    elif mode == FrameMode.CHIEF_INERTIAL:
        r_local = rho_eci
        v_local = rel_v_eci
    elif mode == FrameMode.LVLH:
        rho_lvlh = fc.c_li @ rho_eci
        rhod_lvlh = fc.c_li @ rel_v_eci - np.cross(fc.omega_lvlh, rho_lvlh)
        r_local = rho_lvlh
        v_local = rhod_lvlh
    else:  # pragma: no cover - exhaustive for type checkers
        raise ValueError(mode)

    y = np.empty(13, dtype=dtype)
    y[0:3] = np.asarray(r_local, dtype=dtype)
    y[3:6] = np.asarray(v_local, dtype=dtype)
    y[6:10] = np.asarray(INITIAL_ATTITUDE_WXYZ, dtype=dtype)
    y[10:13] = np.asarray(INITIAL_OMEGA_BODY, dtype=dtype)
    return y


def reconstruct_eci_state(
    y: np.ndarray,
    mode: FrameMode,
    chief: OrbitState,
    length_unit_m: float = 1.0,
) -> OrbitState:
    """Convert a body 13-vector back to absolute ECI state."""
    fc = frame_cache(chief, length_unit_m)
    qpos = np.asarray(y[0:3], dtype=np.float64)
    qvel = np.asarray(y[3:6], dtype=np.float64)

    if mode == FrameMode.ECI:
        return OrbitState(qpos, qvel)
    if mode == FrameMode.CHIEF_INERTIAL:
        return OrbitState(chief.r + qpos, chief.v + qvel)
    if mode == FrameMode.LVLH:
        r = chief.r + fc.c_il @ qpos
        v = chief.v + fc.c_il @ (qvel + np.cross(fc.omega_lvlh, qpos))
        return OrbitState(r, v)
    raise ValueError(mode)  # pragma: no cover


def encke_diff_gravity(
    chief_r: np.ndarray,
    r_local_eci: np.ndarray,
    length_unit_m: float,
) -> np.ndarray:
    """Exact differential gravity g(chief + r) - g(chief) (Encke).

    Uses the identity  g_diff = -mu/|chief|^3 * (r - f(q) * (chief + r))  with
    q = r . (2 chief + r) / |chief|^2  and
    f(q) = 1 - 1/(1+q)^{3/2}  computed cancellation-free as
    f(q) = [q(3+3q+q^2) / (1+(1+q)^{3/2})] / (1+q)^{3/2}.

    No Taylor truncation, no catastrophic cancellation. ``r_local_eci`` and
    ``chief_r`` must be in the same axes (ECI). Output dtype follows
    ``r_local_eci``.
    """
    dtype = r_local_eci.dtype
    c = np.asarray(chief_r, dtype=np.float64)
    r = np.asarray(r_local_eci, dtype=np.float64)
    c_sq = float(np.dot(c, c))
    q = float(np.dot(r, 2.0 * c + r)) / c_sq
    one_plus_q_pow_three_half = (1.0 + q) ** 1.5
    big_f = q * (3.0 + 3.0 * q + q * q) / (1.0 + one_plus_q_pow_three_half)
    f = big_f / one_plus_q_pow_three_half
    coeff = -MU_EARTH / length_unit_m**3 / (c_sq * np.sqrt(c_sq))
    diff = coeff * (r - f * (c + r))
    return np.asarray(diff, dtype=dtype)


def frame_acceleration(
    r_local: np.ndarray,
    v_local: np.ndarray,
    mode: FrameMode,
    chief: OrbitState,
    length_unit_m: float,
) -> np.ndarray:
    """Translational acceleration in the active frame, in coord units / s^2.

    Output dtype follows ``r_local`` so float32 propagation stays in float32.
    """
    dtype = r_local.dtype

    def cast(value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=dtype)

    if mode == FrameMode.ECI:
        return cast(gravity_in_units(r_local, length_unit_m))

    if mode == FrameMode.CHIEF_INERTIAL:
        return encke_diff_gravity(chief.r, r_local, length_unit_m)

    if mode == FrameMode.LVLH:
        fc = frame_cache(chief, length_unit_m)
        c_il = cast(fc.c_il)
        c_li = cast(fc.c_li)
        omega = cast(fc.omega_lvlh)
        omega_dot = cast(fc.omega_dot_lvlh)
        r_local_eci = c_il @ r_local
        gravity_diff = c_li @ encke_diff_gravity(chief.r, r_local_eci, length_unit_m)
        coriolis = cast(-2.0) * np.cross(omega, v_local)
        euler = -np.cross(omega_dot, r_local)
        centrifugal = -np.cross(omega, np.cross(omega, r_local))
        return gravity_diff + coriolis + euler + centrifugal
    raise ValueError(mode)  # pragma: no cover


def quat_kinematic_rhs(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """Body-to-frame quaternion derivative (w-first) for body-frame omega.

    qdot = 0.5 * q (Hamilton-product) (0, omega_body)
    """
    qw, qx, qy, qz = q
    wx, wy, wz = omega_body
    half = np.asarray(0.5, dtype=q.dtype)
    return half * np.array(
        [
            -qx * wx - qy * wy - qz * wz,
            qw * wx + qy * wz - qz * wy,
            qw * wy + qz * wx - qx * wz,
            qw * wz + qx * wy - qy * wx,
        ],
        dtype=q.dtype,
    )


def euler_rotational_rhs(
    omega_body: np.ndarray,
    inertia_diag: np.ndarray,
) -> np.ndarray:
    """Torque-free Euler equation: omega_dot = -I^{-1} (omega x I omega)."""
    Iw = inertia_diag * omega_body
    return -np.cross(omega_body, Iw) / inertia_diag


def state_dot(
    y: np.ndarray,
    t: float,
    mode: FrameMode,
    chief_at: Callable[[float], OrbitState],
    length_unit_m: float,
    inertia_diag: np.ndarray,
) -> np.ndarray:
    """RHS for the packed 13-vector body state."""
    r = y[0:3]
    v = y[3:6]
    q = y[6:10]
    omega_body = y[10:13]

    chief = chief_at(t)
    rdot = v
    vdot = frame_acceleration(r, v, mode, chief, length_unit_m)
    qdot = quat_kinematic_rhs(q, omega_body)
    omegadot = euler_rotational_rhs(omega_body, inertia_diag)

    out = np.empty(13, dtype=y.dtype)
    out[0:3] = rdot
    out[3:6] = vdot
    out[6:10] = qdot
    out[10:13] = omegadot
    return out


def _renormalize_quat(y: np.ndarray) -> tuple[np.ndarray, float]:
    """Renormalize quaternion in-place; return (new_y, |q|-1) at pre-renorm."""
    q = y[6:10]
    norm = float(np.linalg.norm(q))
    drift = abs(norm - 1.0)
    if norm > 0.0:
        y = y.copy()
        y[6:10] = (q / norm).astype(y.dtype)
    return y, drift


def euler_step(
    y: np.ndarray,
    t: float,
    dt: float,
    mode: FrameMode,
    chief_at: Callable[[float], OrbitState],
    length_unit_m: float,
    inertia_diag: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Semi-implicit (symplectic) Euler — matches MuJoCo's mjINT_EULER default.

    Velocities are advanced using the current acceleration; positions and
    orientation are then advanced using the *new* velocities. On Hamiltonian
    systems this preserves energy on average, giving bounded periodic error
    instead of the secular blow-up of forward Euler.
    """
    dy = state_dot(y, t, mode, chief_at, length_unit_m, inertia_diag)
    dt_t = np.asarray(dt, dtype=y.dtype)

    y_next = np.empty_like(y)
    y_next[3:6] = y[3:6] + dt_t * dy[3:6]
    y_next[10:13] = y[10:13] + dt_t * dy[10:13]
    y_next[0:3] = y[0:3] + dt_t * y_next[3:6]
    qdot_new = quat_kinematic_rhs(y[6:10], y_next[10:13])
    y_next[6:10] = y[6:10] + dt_t * qdot_new
    return _renormalize_quat(y_next)


def rk4_step(
    y: np.ndarray,
    t: float,
    dt: float,
    mode: FrameMode,
    chief_at: Callable[[float], OrbitState],
    length_unit_m: float,
    inertia_diag: np.ndarray,
) -> tuple[np.ndarray, float]:
    dt_t = np.asarray(dt, dtype=y.dtype)
    half_dt = np.asarray(0.5 * dt, dtype=y.dtype)
    k1 = state_dot(y, t, mode, chief_at, length_unit_m, inertia_diag)
    k2 = state_dot(y + half_dt * k1, t + 0.5 * dt, mode, chief_at, length_unit_m, inertia_diag)
    k3 = state_dot(y + half_dt * k2, t + 0.5 * dt, mode, chief_at, length_unit_m, inertia_diag)
    k4 = state_dot(y + dt_t * k3, t + dt, mode, chief_at, length_unit_m, inertia_diag)
    sixth = np.asarray(dt / 6.0, dtype=y.dtype)
    two = np.asarray(2.0, dtype=y.dtype)
    y_next = y + sixth * (k1 + two * k2 + two * k3 + k4)
    return _renormalize_quat(y_next)


def _skew(a: np.ndarray) -> np.ndarray:
    """3x3 cross-product matrix [a]_x (float64)."""
    a = np.asarray(a, dtype=np.float64)
    return np.array(
        [[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]],
        dtype=np.float64,
    )


def implicit_step(
    y: np.ndarray,
    t: float,
    dt: float,
    mode: FrameMode,
    chief_at: Callable[[float], OrbitState],
    length_unit_m: float,
    inertia_diag: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Linearly-implicit (implicit-in-velocity) Euler — matches MuJoCo's
    ``mjINT_IMPLICIT``.

    The velocity DOFs (translational ``v`` and body ``omega``) are advanced with
    an implicit treatment of the *velocity-dependent* forces, using the analytic
    Jacobian ``J = d(accel)/d(vel)``::

        (I - dt J) dvel = dt accel,   vel = [v, omega].

    Two velocity-dependent blocks appear: the LVLH Coriolis acceleration
    ``-2 omega_L x v`` (translational) with ``dvdot/dv = -2 [omega_L]_x``, and the
    rotational gyroscopic term ``omegadot = -I^{-1}(omega x I omega)`` with
    ``domegadot/domega = I^{-1}([I omega]_x - [omega]_x I)``. Positions and the
    quaternion are then advanced with the new velocities (semi-implicit), exactly
    as in :func:`euler_step`. For position-only forces (ECI, chief-inertial) the
    translational block is zero, so the translational update reduces *exactly* to
    semi-implicit Euler; only the rotational integration changes there.
    """
    dtype = y.dtype
    dt_t = np.asarray(dt, dtype=dtype)
    dy = state_dot(y, t, mode, chief_at, length_unit_m, inertia_diag)
    accel = np.concatenate([dy[3:6], dy[10:13]]).astype(np.float64)  # [vdot, omegadot]

    # Block-diagonal Jacobian J = d(accel)/d(vel), vel = [v, omega].
    inertia = np.asarray(inertia_diag, dtype=np.float64)
    omega = np.asarray(y[10:13], dtype=np.float64)
    jac = np.zeros((6, 6), dtype=np.float64)
    # Rotational gyroscopic block: I^{-1} ([I w]_x - [w]_x I).
    jac[3:6, 3:6] = np.diag(1.0 / inertia) @ (
        _skew(inertia * omega) - _skew(omega) @ np.diag(inertia)
    )
    # LVLH translational Coriolis block: -2 [omega_L]_x.
    if mode == FrameMode.LVLH:
        omega_lvlh = frame_cache(chief_at(t), length_unit_m).omega_lvlh
        jac[0:3, 0:3] = -2.0 * _skew(omega_lvlh)

    amat = np.eye(6, dtype=np.float64) - float(dt) * jac
    dvel = np.linalg.solve(amat, float(dt) * accel).astype(dtype)

    y_next = np.empty_like(y)
    y_next[3:6] = y[3:6] + dvel[0:3]
    y_next[10:13] = y[10:13] + dvel[3:6]
    y_next[0:3] = y[0:3] + dt_t * y_next[3:6]
    qdot_new = quat_kinematic_rhs(y[6:10], y_next[10:13])
    y_next[6:10] = y[6:10] + dt_t * qdot_new
    return _renormalize_quat(y_next)


def step_function(integrator: Integrator) -> Callable[..., tuple[np.ndarray, float]]:
    if integrator == "euler":
        return euler_step
    if integrator == "rk4":
        return rk4_step
    if integrator == "implicit":
        return implicit_step
    raise ValueError(f"Unknown integrator: {integrator}")


def make_chief_at(
    chief_step_start: OrbitState,
    step_start_time: float,
    length_unit_m: float,
) -> Callable[[float], OrbitState]:
    """Closure returning the chief state at any time within the current step."""

    def chief_at(stage_time: float) -> OrbitState:
        elapsed = stage_time - step_start_time
        if abs(elapsed) < 1.0e-15:
            return chief_step_start
        return rk4_orbit_step(chief_step_start, elapsed, length_unit_m)

    return chief_at


def attitude_invariants_from_y(
    y: np.ndarray,
    inertia_diag: np.ndarray,
) -> tuple[float, float, float]:
    """Return rotational energy, angular momentum norm, and quaternion norm error.

    Uses post-renormalization quaternion, so quat_norm_error is reported
    separately (per-step pre-renorm drift) by the simulation loop.
    """
    omega = np.asarray(y[10:13], dtype=np.float64)
    inertia = np.asarray(inertia_diag, dtype=np.float64)
    energy = 0.5 * float(np.dot(inertia * omega, omega))
    angular_momentum_norm = float(np.linalg.norm(inertia * omega))
    quat_norm_error = abs(float(np.linalg.norm(y[6:10])) - 1.0)
    return energy, angular_momentum_norm, quat_norm_error


def orbital_invariants(
    state: OrbitState,
    length_unit_m: float = 1.0,
) -> tuple[float, float]:
    """Return specific two-body energy and angular momentum norm."""
    radius = float(np.linalg.norm(state.r))
    mu = MU_EARTH / length_unit_m**3
    energy = 0.5 * float(np.dot(state.v, state.v)) - mu / radius
    angular_momentum_norm = float(np.linalg.norm(np.cross(state.r, state.v)))
    return energy, angular_momentum_norm


def simulate(
    scenario: Scenario,
    mode: FrameMode,
    n_orbits: float = N_ORBITS,
    integrator: Integrator = DEFAULT_INTEGRATOR,
    precision: StudyPrecision = "float64",
    dt: float = SIM_TIMESTEP,
    orbit_unit_m: float = 1.0,
) -> Result:
    """Run one frame mode and return numerical stability metrics.

    The body propagator runs in meters; chief and truth-body propagator run in
    ``orbit_unit_m`` (e.g. 1000 for km, 100000 for 100 km).
    """
    chief0_m, mean_motion, orbit_period = make_initial_chief(scenario, length_unit_m=1.0)
    rho0 = INITIAL_REL_POS_LVLH
    rhod0 = np.array([0.0, -2.0 * mean_motion * INITIAL_REL_POS_LVLH[0], 0.0])
    body0_m = initial_body_from_lvlh(chief0_m, rho0, rhod0, length_unit_m=1.0)

    dtype = np.dtype(precision)
    chief = OrbitState(chief0_m.r / orbit_unit_m, chief0_m.v / orbit_unit_m)
    truth_body = OrbitState(body0_m.r / orbit_unit_m, body0_m.v / orbit_unit_m)

    y = initial_local_state(mode, chief0_m, body0_m, length_unit_m=1.0, dtype=dtype)
    inertia_diag = np.asarray(INERTIA_DIAG, dtype=dtype)

    n_steps = int(round(n_orbits * orbit_period / dt))
    if n_steps < 1:
        raise ValueError("n_orbits and dt must produce at least one step")
    step_fn = step_function(integrator)

    energy0, h0, _ = attitude_invariants_from_y(y, inertia_diag)
    orbital_energy0, orbital_h0 = orbital_invariants(body0_m, length_unit_m=1.0)

    max_energy_drift = 0.0
    max_h_drift = 0.0
    max_quat_norm_error = 0.0
    max_position_error = 0.0
    max_relative_radius = 0.0
    max_orbital_energy_drift = 0.0
    max_orbital_h_drift = 0.0
    final_orbital_energy_drift = 0.0
    final_orbital_h_drift = 0.0
    finite = True

    t = 0.0
    for _ in range(n_steps):
        chief_step_start = OrbitState(chief.r.copy(), chief.v.copy())
        chief_at_orbit = make_chief_at(chief_step_start, t, orbit_unit_m)

        def chief_at_m(stage_time: float, _f=chief_at_orbit, _u=orbit_unit_m) -> OrbitState:
            return scale_orbit_state(_f(stage_time), _u)

        y, quat_drift = step_fn(y, t, dt, mode, chief_at_m, 1.0, inertia_diag)
        t += dt
        chief = rk4_orbit_step(chief, dt, orbit_unit_m)
        truth_body = propagate_truth_body(truth_body, dt, orbit_unit_m)

        chief_m = scale_orbit_state(chief, orbit_unit_m)
        truth_body_m = scale_orbit_state(truth_body, orbit_unit_m)
        sim_body_m = reconstruct_eci_state(y, mode, chief_m, length_unit_m=1.0)
        relative_m = sim_body_m.r - chief_m.r
        position_error = float(np.linalg.norm(sim_body_m.r - truth_body_m.r))

        orbital_energy, orbital_h = orbital_invariants(sim_body_m, length_unit_m=1.0)
        final_orbital_energy_drift = abs(orbital_energy - orbital_energy0) / abs(orbital_energy0)
        final_orbital_h_drift = abs(orbital_h - orbital_h0) / orbital_h0
        max_orbital_energy_drift = max(max_orbital_energy_drift, final_orbital_energy_drift)
        max_orbital_h_drift = max(max_orbital_h_drift, final_orbital_h_drift)

        energy, h_norm, _ = attitude_invariants_from_y(y, inertia_diag)
        max_energy_drift = max(max_energy_drift, abs(energy - energy0) / energy0)
        max_h_drift = max(max_h_drift, abs(h_norm - h0) / h0)
        max_quat_norm_error = max(max_quat_norm_error, quat_drift)
        max_position_error = max(max_position_error, position_error)
        max_relative_radius = max(max_relative_radius, float(np.linalg.norm(relative_m)))
        finite = finite and bool(np.all(np.isfinite(y)))

    chief_m = scale_orbit_state(chief, orbit_unit_m)
    truth_body_m = scale_orbit_state(truth_body, orbit_unit_m)
    final_body_m = reconstruct_eci_state(y, mode, chief_m, length_unit_m=1.0)
    final_error = float(np.linalg.norm(final_body_m.r - truth_body_m.r))
    final_relative_m = final_body_m.r - chief_m.r
    final_relative_lvlh = frame_cache(chief_m, length_unit_m=1.0).c_li @ final_relative_m
    return Result(
        scenario=scenario,
        mode=mode,
        max_position_error=max_position_error,
        final_position_error=final_error,
        max_relative_radius=max_relative_radius,
        final_relative_position=final_relative_lvlh,
        max_orbital_energy_drift_rel=max_orbital_energy_drift,
        final_orbital_energy_drift_rel=final_orbital_energy_drift,
        max_orbital_angular_momentum_drift_rel=max_orbital_h_drift,
        final_orbital_angular_momentum_drift_rel=final_orbital_h_drift,
        energy_drift_rel=max_energy_drift,
        angular_momentum_drift_rel=max_h_drift,
        max_quat_norm_error=max_quat_norm_error,
        finite=finite,
    )


def simulate_time_history(
    scenario: Scenario,
    mode: FrameMode,
    integrator: Integrator,
    dt: float,
    duration: float,
    label: str,
    precision: StudyPrecision,
    rel_vel_bias_lvlh: np.ndarray,
    orbit_unit_m: float,
) -> TimeHistory:
    """Body in meters; chief and truth-body in ``orbit_unit_m``."""
    chief0_m, body0_m, _, _ = initial_states(scenario, rel_vel_bias_lvlh, length_unit_m=1.0)
    chief = OrbitState(chief0_m.r / orbit_unit_m, chief0_m.v / orbit_unit_m)
    truth_body = OrbitState(body0_m.r / orbit_unit_m, body0_m.v / orbit_unit_m)

    dtype = np.dtype(precision)
    y = initial_local_state(mode, chief0_m, body0_m, length_unit_m=1.0, dtype=dtype)
    inertia_diag = np.asarray(INERTIA_DIAG, dtype=dtype)

    n_steps = int(round(duration / dt))
    if n_steps < 1:
        raise ValueError("duration must cover at least one step")
    duration = n_steps * dt
    step_fn = step_function(integrator)

    times_s = np.empty(n_steps + 1, dtype=np.float64)
    position_error_m = np.empty(n_steps + 1, dtype=np.float64)
    energy_error_rel = np.empty(n_steps + 1, dtype=np.float64)
    relative_radius_m = np.empty(n_steps + 1, dtype=np.float64)
    finite = True
    energy0, _ = orbital_invariants(body0_m, length_unit_m=1.0)

    times_s[0] = 0.0
    sim_body_m = reconstruct_eci_state(y, mode, chief0_m, length_unit_m=1.0)
    position_error_m[0] = float(np.linalg.norm(sim_body_m.r - body0_m.r))
    energy, _ = orbital_invariants(sim_body_m, length_unit_m=1.0)
    energy_error_rel[0] = abs(energy - energy0) / abs(energy0)
    relative_radius_m[0] = float(np.linalg.norm(sim_body_m.r - chief0_m.r))

    t = 0.0
    for idx in range(1, n_steps + 1):
        chief_step_start = OrbitState(chief.r.copy(), chief.v.copy())
        chief_at_orbit = make_chief_at(chief_step_start, t, orbit_unit_m)

        def chief_at_m(stage_time: float, _f=chief_at_orbit, _u=orbit_unit_m) -> OrbitState:
            return scale_orbit_state(_f(stage_time), _u)

        y, _ = step_fn(y, t, dt, mode, chief_at_m, 1.0, inertia_diag)
        t += dt

        chief = rk4_orbit_step(chief, dt, orbit_unit_m)
        truth_body = propagate_truth_body(truth_body, dt, orbit_unit_m)

        chief_m = scale_orbit_state(chief, orbit_unit_m)
        truth_body_m = scale_orbit_state(truth_body, orbit_unit_m)
        sim_body_m = reconstruct_eci_state(y, mode, chief_m, length_unit_m=1.0)

        times_s[idx] = min(t, duration)
        position_error_m[idx] = float(np.linalg.norm(sim_body_m.r - truth_body_m.r))
        energy, _ = orbital_invariants(sim_body_m, length_unit_m=1.0)
        energy_error_rel[idx] = abs(energy - energy0) / abs(energy0)
        relative_radius_m[idx] = float(np.linalg.norm(sim_body_m.r - chief_m.r))
        finite = finite and bool(np.all(np.isfinite(y)))

    return TimeHistory(
        label=label,
        scenario=scenario,
        mode=mode,
        integrator=integrator,
        precision=precision,
        length_unit_m=orbit_unit_m,
        times_s=times_s,
        position_error_m=position_error_m,
        energy_error_rel=energy_error_rel,
        relative_radius_m=relative_radius_m,
        finite=finite,
    )


def run_integrator_study(
    scenario: Scenario,
    dt: float,
    n_orbits: float,
    precision: StudyPrecision,
    rel_vel_bias_lvlh: np.ndarray,
    orbit_unit_m: float,
) -> list[TimeHistory]:
    """Run the ECI/local-chief/LVLH x Euler/implicit/RK4 time-history cases."""
    _, _, _, orbit_period = initial_states(scenario, rel_vel_bias_lvlh, length_unit_m=1.0)
    cases: list[tuple[str, FrameMode, Integrator]] = [
        ("ECI + Euler", FrameMode.ECI, "euler"),
        ("ECI + implicit", FrameMode.ECI, "implicit"),
        ("ECI + RK4", FrameMode.ECI, "rk4"),
        ("local chief + Euler", FrameMode.CHIEF_INERTIAL, "euler"),
        ("local chief + implicit", FrameMode.CHIEF_INERTIAL, "implicit"),
        ("local chief + RK4", FrameMode.CHIEF_INERTIAL, "rk4"),
        ("LVLH + Euler", FrameMode.LVLH, "euler"),
        ("LVLH + implicit", FrameMode.LVLH, "implicit"),
        ("LVLH + RK4", FrameMode.LVLH, "rk4"),
    ]
    return [
        simulate_time_history(
            scenario=scenario,
            mode=mode,
            integrator=integrator,
            dt=dt,
            duration=n_orbits * orbit_period,
            label=label,
            precision=precision,
            rel_vel_bias_lvlh=rel_vel_bias_lvlh,
            orbit_unit_m=orbit_unit_m,
        )
        for label, mode, integrator in cases
    ]


def write_integrator_study(
    histories: list[TimeHistory],
    out_dir: Path,
    n_orbits: float,
    dt: float,
    precision: StudyPrecision,
    rel_vel_bias_lvlh: np.ndarray,
    orbit_unit_m: float,
) -> tuple[Path, Path, Path]:
    """Save the integrator-study error plots and raw samples."""
    if not histories:
        raise ValueError("histories must not be empty")

    out_dir.mkdir(parents=True, exist_ok=True)
    scenario = histories[0].scenario
    orbit_suffix = "1_orbit" if np.isclose(n_orbits, 1.0) else f"{n_orbits:g}_orbits"
    orbit_suffix = orbit_suffix.replace(".", "p")
    dt_suffix = f"dt_{dt:g}s".replace(".", "p")
    velocity_suffix = ""
    if not np.allclose(rel_vel_bias_lvlh, 0.0):
        velocity_suffix = "_dv_" + "_".join(
            format_float_for_filename(float(value)) for value in rel_vel_bias_lvlh
        )
    unit_suffix = f"_unit_{format_float_for_filename(orbit_unit_m)}m"
    stem = (
        f"{scenario.name}_{orbit_suffix}_{dt_suffix}_{precision}"
        f"{velocity_suffix}{unit_suffix}_eci_vs_local_integrators"
    )
    plot_path = out_dir / f"{stem}.png"
    energy_plot_path = out_dir / f"{stem}_energy.png"
    samples_path = out_dir / f"{stem}.npz"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"euler": "tab:blue", "rk4": "tab:orange", "implicit": "tab:green"}

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for history in histories:
        elapsed_orbits = history.times_s / histories[0].times_s[-1] * n_orbits
        error = np.maximum(history.position_error_m, 1.0e-12)
        ax.semilogy(
            elapsed_orbits,
            error,
            color=colors[history.integrator],
            linestyle="-" if history.mode == FrameMode.ECI else "--",
            linewidth=2.0,
            label=history.label,
        )

    ax.set_title(
        f"ECI vs Local Chief Frame: {scenario.name}, {n_orbits:g} orbits, "
        f"{orbit_unit_m:g} m/unit"
    )
    ax.set_xlabel("time [orbits]")
    ax.set_ylabel("ECI position error vs RK4 reference [m]")
    ax.grid(True, which="both", linewidth=0.6, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for history in histories:
        elapsed_orbits = history.times_s / histories[0].times_s[-1] * n_orbits
        error = np.maximum(history.energy_error_rel, 1.0e-18)
        ax.semilogy(
            elapsed_orbits,
            error,
            color=colors[history.integrator],
            linestyle="-" if history.mode == FrameMode.ECI else "--",
            linewidth=2.0,
            label=history.label,
        )

    ax.set_title(
        f"Specific Energy Error: {scenario.name}, {n_orbits:g} orbits, "
        f"{orbit_unit_m:g} m/unit"
    )
    ax.set_xlabel("time [orbits]")
    ax.set_ylabel("relative specific orbital energy error")
    ax.grid(True, which="both", linewidth=0.6, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(energy_plot_path, dpi=200)
    plt.close(fig)

    np.savez(
        samples_path,
        labels=np.asarray([history.label for history in histories]),
        precision=np.asarray(precision),
        length_unit_m=np.asarray(orbit_unit_m),
        rel_vel_bias_lvlh=np.asarray(rel_vel_bias_lvlh, dtype=np.float64),
        times_s=histories[0].times_s,
        position_error_m=np.vstack([history.position_error_m for history in histories]),
        energy_error_rel=np.vstack([history.energy_error_rel for history in histories]),
        relative_radius_m=np.vstack([history.relative_radius_m for history in histories]),
        finite=np.asarray([history.finite for history in histories]),
    )
    return plot_path, energy_plot_path, samples_path


def parse_args() -> argparse.Namespace:
    """Parse frame-study command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-orbits",
        type=float,
        default=N_ORBITS,
        help="Number of chief orbits per scenario.",
    )
    parser.add_argument(
        "--orbit-timestep",
        type=float,
        default=ORBIT_TIMESTEP,
        help="Chief/reference RK4 timestep in seconds.",
    )
    parser.add_argument(
        "--sim-timestep",
        type=float,
        default=SIM_TIMESTEP,
        help="Body propagator timestep in seconds for the default sweep.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(scenario.name for scenario in SCENARIOS),
        help="Scenario name to run. Repeat to run multiple; omitted runs all.",
    )
    parser.add_argument(
        "--integrator-study",
        action="store_true",
        help=(
            "Run an ECI/local-chief/LVLH comparison with Euler, implicit, and RK4, "
            "then save error plots."
        ),
    )
    parser.add_argument(
        "--study-orbits",
        type=float,
        default=1.0,
        help="Number of orbits to simulate for --integrator-study.",
    )
    parser.add_argument(
        "--study-dt",
        type=float,
        default=SIM_TIMESTEP,
        help="Shared body, chief, and reference timestep for --integrator-study.",
    )
    parser.add_argument(
        "--study-precision",
        choices=("float64", "float32"),
        default="float64",
        help=(
            "Precision used end-to-end inside the body propagator for "
            "--integrator-study. The chief/reference orbit stays in float64."
        ),
    )
    parser.add_argument(
        "--study-rel-vel-lvlh",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "VZ"),
        default=(0.0, 0.0, 0.0),
        help=(
            "Additional initial LVLH relative velocity in m/s for --integrator-study. "
            "This is added to the default bounded-drift initialization."
        ),
    )
    parser.add_argument(
        "--orbit-length-unit-m",
        "--study-length-unit-m",
        dest="orbit_length_unit_m",
        type=float,
        default=1.0,
        help=(
            "Meters per length unit used by the chief and truth-body orbit propagators "
            "for --integrator-study (e.g. 1000 for km, 100000 for 100 km). The body "
            "propagator always runs in meters; plotted errors are reported in meters."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for integrator-study plot and sample outputs.",
    )
    return parser.parse_args()


def main() -> None:
    global SIM_TIMESTEP, N_ORBITS, ORBIT_TIMESTEP

    args = parse_args()
    ORBIT_TIMESTEP = args.orbit_timestep
    SIM_TIMESTEP = args.sim_timestep
    N_ORBITS = args.n_orbits
    scenarios = [
        scenario
        for scenario in SCENARIOS
        if args.scenario is None or scenario.name in args.scenario
    ]
    if args.integrator_study:
        study_scenarios = scenarios if args.scenario is not None else [SCENARIOS[0]]
        if len(study_scenarios) != 1:
            raise SystemExit("--integrator-study expects exactly one --scenario")
        if args.orbit_length_unit_m <= 0.0:
            raise SystemExit("--orbit-length-unit-m must be positive")
        study = run_integrator_study(
            study_scenarios[0],
            args.study_dt,
            args.study_orbits,
            args.study_precision,
            np.asarray(args.study_rel_vel_lvlh, dtype=np.float64),
            args.orbit_length_unit_m,
        )
        plot_path, energy_plot_path, samples_path = write_integrator_study(
            study,
            args.out_dir,
            args.study_orbits,
            args.study_dt,
            args.study_precision,
            np.asarray(args.study_rel_vel_lvlh, dtype=np.float64),
            args.orbit_length_unit_m,
        )
        print("Frame study integrator comparison")
        print(f"Scenario: {study_scenarios[0].name}")
        print(f"Shared dt: {args.study_dt:g} s")
        print(f"Study precision: {args.study_precision}")
        print(f"Body unit: 1 m   Orbit length unit: {args.orbit_length_unit_m:g} m")
        print(f"Initial LVLH velocity bias: {np.asarray(args.study_rel_vel_lvlh)} m/s")
        print(f"Requested duration: {args.study_orbits:g} orbits")
        print(f"Duration: {study[0].times_s[-1]:.6f} s")
        print(f"Position plot: {plot_path}")
        print(f"Energy plot: {energy_plot_path}")
        print(f"Samples: {samples_path}")
        print()
        print(
            "case                         max pos [m]  final pos [m]  "
            "max dE/E     final dE/E   finite"
        )
        print("-----------------------------------------------------------------------------------------")
        for history in study:
            print(
                f"{history.label:28s} "
                f"{np.max(history.position_error_m):11.6e} "
                f"{history.position_error_m[-1]:13.6e} "
                f"{np.max(history.energy_error_rel):11.6e} "
                f"{history.energy_error_rel[-1]:11.6e} "
                f"{str(history.finite):6s}"
            )
        return

    print("Frame study: Newton-Euler 6-DOF propagator (no MuJoCo)")
    print(f"Body integrator: {DEFAULT_INTEGRATOR}")
    print(
        f"orbit dt: {ORBIT_TIMESTEP:g} s, body dt: {SIM_TIMESTEP:g} s, "
        f"duration: {N_ORBITS:g} orbits per scenario"
    )

    all_results: list[Result] = []
    for scenario in scenarios:
        chief, _, orbit_period = make_initial_chief(scenario)
        perigee_alt = scenario.semi_major_axis * (1.0 - scenario.eccentricity) - R_EARTH
        apogee_alt = scenario.semi_major_axis * (1.0 + scenario.eccentricity) - R_EARTH
        print()
        print("=" * 100)
        print(
            f"Scenario: {scenario.name} | "
            f"a={scenario.semi_major_axis / 1000.0:.1f} km, "
            f"e={scenario.eccentricity:.3f}, "
            f"inc={np.rad2deg(scenario.inclination):.1f} deg, "
            f"period={orbit_period:.1f} s"
        )
        print(
            f"initial |R|={np.linalg.norm(chief.r) / 1000.0:.1f} km, "
            f"perigee alt={perigee_alt / 1000.0:.1f} km, "
            f"apogee alt={apogee_alt / 1000.0:.1f} km"
        )
        print()

        results = [
            simulate(
                scenario,
                mode,
                n_orbits=N_ORBITS,
                dt=SIM_TIMESTEP,
            )
            for mode in FrameMode
        ]
        all_results.extend(results)
        header = (
            "frame             finite  max pos err [m]  final err [m]  "
            "max |rho| [m]  orbit E drift  orbit |H|  rot E drift  |H| drift  "
            "max |q|-1"
        )
        print(header)
        print("-" * len(header))
        for result in results:
            print(
                f"{result.mode.value:17s} "
                f"{str(result.finite):6s} "
                f"{result.max_position_error:15.6e} "
                f"{result.final_position_error:13.6e} "
                f"{result.max_relative_radius:13.6e} "
                f"{result.max_orbital_energy_drift_rel:14.6e} "
                f"{result.max_orbital_angular_momentum_drift_rel:10.6e} "
                f"{result.energy_drift_rel:12.6e} "
                f"{result.angular_momentum_drift_rel:10.6e} "
                f"{result.max_quat_norm_error:10.6e}"
            )

        print()
        print("Final relative positions in chief LVLH frame [m]:")
        for result in results:
            rel = result.final_relative_position
            print(
                f"  {result.mode.value:17s} "
                f"[{rel[0]: .6f}, {rel[1]: .6f}, {rel[2]: .6f}]"
            )

    if not all(result.finite for result in all_results):
        raise SystemExit("At least one frame produced a non-finite body state.")


if __name__ == "__main__":
    main()
