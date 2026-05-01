"""Minimal MuJoCo frame study for orbital relative motion and attitude.

This experiment intentionally does not import ``mujoco_orbit``. It uses raw
MuJoCo plus a tiny two-body propagator to compare three choices for the MuJoCo
world coordinates:

- ``eci``: absolute Earth-centered inertial position/velocity.
- ``chief_inertial``: chief-centered position/velocity with inertially fixed axes.
- ``lvlh``: chief-centered rotating LVLH coordinates.

Each scenario uses a small chief-relative LVLH initial condition plus
torque-free rigid-body attitude. It runs for a few orbits and compares each
MuJoCo result against an independent ECI two-body RK4 reference.

Usage:
    pixi run python experiments/frame_study/run.py
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterator, Literal

import mujoco
import numpy as np

MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH = 6_378_137.0  # m
MASS = 100.0  # kg
OUT_DIR = Path(__file__).with_name("out")
StudyPrecision = Literal["float64", "float32"]

ORBIT_TIMESTEP = 0.5  # s
MUJOCO_TIMESTEP = 0.1  # s
N_ORBITS = 3.0
MUJOCO_INTEGRATOR = mujoco.mjtIntegrator.mjINT_RK4

INITIAL_REL_POS_LVLH = np.array([1.0, 0.0, 5.0])  # m
INITIAL_ATTITUDE_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
INITIAL_OMEGA_BODY = np.array([0.012, -0.018, 0.009])  # rad/s

XML = f"""
<mujoco model="frame_study_body">
  <compiler angle="radian"/>
  <option timestep="{MUJOCO_TIMESTEP}" gravity="0 0 0"/>
  <worldbody>
    <body name="body" pos="0 0 0">
      <freejoint/>
      <geom type="box" size="0.5 0.3 0.2" mass="{MASS}" rgba="0.3 0.5 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
"""


class FrameMode(StrEnum):
    ECI = "eci"
    CHIEF_INERTIAL = "chief_inertial"
    LVLH = "lvlh"


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


@dataclass
class ChiefContext:
    step_start: OrbitState
    step_start_time: float
    precision: StudyPrecision
    length_unit_m: float

    def state_at(self, time_s: float) -> OrbitState:
        elapsed = time_s - self.step_start_time
        if abs(elapsed) < 1.0e-15:
            return self.step_start
        return rk4_orbit_step(self.step_start, elapsed, self.length_unit_m)


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
    integrator: mujoco.mjtIntegrator
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


def gravity(r: np.ndarray) -> np.ndarray:
    """Two-body acceleration in SI units."""
    return gravity_in_units(r, length_unit_m=1.0)


def gravity_in_units(r: np.ndarray, length_unit_m: float) -> np.ndarray:
    """Two-body acceleration in the requested length unit per second squared."""
    radius = np.linalg.norm(r)
    mu = MU_EARTH / length_unit_m**3
    return -mu / radius**3 * r


def precision_view(values: np.ndarray, precision: StudyPrecision) -> np.ndarray:
    """Return values rounded to the requested study precision."""
    if precision == "float32":
        return np.asarray(values, dtype=np.float32).astype(np.float64)
    return np.asarray(values, dtype=np.float64)


def quantize_mujoco_state(data: mujoco.MjData, precision: StudyPrecision) -> None:
    """Round MuJoCo state storage for reduced-precision study runs."""
    if precision != "float32":
        return
    data.qpos[:] = np.asarray(data.qpos, dtype=np.float32).astype(np.float64)
    data.qvel[:] = np.asarray(data.qvel, dtype=np.float32).astype(np.float64)


def rk4_orbit_step(
    state: OrbitState,
    dt: float,
    length_unit_m: float = 1.0,
) -> OrbitState:
    """RK4 step for an inertial two-body point mass."""

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


def make_model_data(
    integrator: mujoco.mjtIntegrator = MUJOCO_INTEGRATOR,
    timestep: float = MUJOCO_TIMESTEP,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile the one-body MuJoCo model."""
    model = mujoco.MjModel.from_xml_string(XML)
    model.opt.timestep = timestep
    model.opt.integrator = integrator
    data = mujoco.MjData(model)
    return model, data


def interpolate_orbit(start: OrbitState, end: OrbitState, alpha: float) -> OrbitState:
    """Linear interpolation used only to update frame terms during substeps."""
    return OrbitState(
        (1.0 - alpha) * start.r + alpha * end.r,
        (1.0 - alpha) * start.v + alpha * end.v,
    )


def set_initial_mujoco_state(
    data: mujoco.MjData,
    mode: FrameMode,
    chief: OrbitState,
    body: OrbitState,
    length_unit_m: float = 1.0,
) -> None:
    """Set free-joint position, attitude, and velocity for one frame mode."""
    fc = frame_cache(chief, length_unit_m)
    rho_eci = body.r - chief.r
    rel_v_eci = body.v - chief.v

    if mode == FrameMode.ECI:
        data.qpos[:3] = body.r
        data.qvel[:3] = body.v
    elif mode == FrameMode.CHIEF_INERTIAL:
        data.qpos[:3] = rho_eci
        data.qvel[:3] = rel_v_eci
    elif mode == FrameMode.LVLH:
        rho_lvlh = fc.c_li @ rho_eci
        rhod_lvlh = fc.c_li @ rel_v_eci - np.cross(fc.omega_lvlh, rho_lvlh)
        data.qpos[:3] = rho_lvlh
        data.qvel[:3] = rhod_lvlh
    else:  # pragma: no cover - exhaustive for type checkers
        raise ValueError(mode)

    data.qpos[3:7] = INITIAL_ATTITUDE_WXYZ
    data.qvel[3:6] = INITIAL_OMEGA_BODY


def reconstruct_eci_state(
    data: mujoco.MjData,
    mode: FrameMode,
    chief: OrbitState,
    precision: StudyPrecision = "float64",
    length_unit_m: float = 1.0,
) -> OrbitState:
    """Convert the MuJoCo free-joint state back to absolute ECI."""
    fc = frame_cache(chief, length_unit_m)
    qpos = precision_view(data.qpos[:3], precision)
    qvel = precision_view(data.qvel[:3], precision)

    if mode == FrameMode.ECI:
        return OrbitState(qpos, qvel)
    if mode == FrameMode.CHIEF_INERTIAL:
        return OrbitState(chief.r + qpos, chief.v + qvel)
    if mode == FrameMode.LVLH:
        r = chief.r + fc.c_il @ qpos
        v = chief.v + fc.c_il @ (qvel + np.cross(fc.omega_lvlh, qpos))
        return OrbitState(r, v)
    raise ValueError(mode)  # pragma: no cover


def frame_acceleration(
    data: mujoco.MjData,
    mode: FrameMode,
    chief: OrbitState,
    precision: StudyPrecision = "float64",
    length_unit_m: float = 1.0,
) -> np.ndarray:
    """Return the translational acceleration to apply in the active frame."""
    qpos = precision_view(data.qpos[:3], precision)
    qvel = precision_view(data.qvel[:3], precision)

    if mode == FrameMode.ECI:
        return gravity_in_units(qpos, length_unit_m)

    chief_gravity = gravity_in_units(chief.r, length_unit_m)
    if mode == FrameMode.CHIEF_INERTIAL:
        return gravity_in_units(chief.r + qpos, length_unit_m) - chief_gravity
    if mode == FrameMode.LVLH:
        fc = frame_cache(chief, length_unit_m)
        body_r = chief.r + fc.c_il @ qpos
        gravity_diff = fc.c_li @ (gravity_in_units(body_r, length_unit_m) - chief_gravity)
        coriolis = -2.0 * np.cross(fc.omega_lvlh, qvel)
        euler = -np.cross(fc.omega_dot_lvlh, qpos)
        centrifugal = -np.cross(fc.omega_lvlh, np.cross(fc.omega_lvlh, qpos))
        return gravity_diff + coriolis + euler + centrifugal
    raise ValueError(mode)  # pragma: no cover


def attitude_invariants(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float, float]:
    """Return rotational energy, angular momentum norm, and quaternion norm error."""
    inertia = model.body_inertia[1]
    omega_body = data.qvel[3:6]
    energy = 0.5 * float(np.dot(inertia * omega_body, omega_body))
    angular_momentum_norm = float(np.linalg.norm(inertia * omega_body))
    quat_norm_error = abs(float(np.linalg.norm(data.qpos[3:7])) - 1.0)
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


@contextmanager
def frame_force_callback(
    model: mujoco.MjModel,
    mode: FrameMode,
    chief_context: ChiefContext,
) -> Iterator[None]:
    """Apply frame acceleration inside MuJoCo dynamics evaluations."""
    previous_callback = mujoco.get_mjcb_passive()
    zero_torque = np.zeros(3, dtype=np.float64)
    body_id = 1
    body_mass = float(model.body_mass[body_id])

    def passive_callback(cb_model: mujoco.MjModel, cb_data: mujoco.MjData) -> None:
        point = np.asarray(cb_data.xipos[body_id], dtype=np.float64)
        force = body_mass * frame_acceleration(
            cb_data,
            mode,
            chief_context.state_at(float(cb_data.time)),
            chief_context.precision,
            chief_context.length_unit_m,
        )
        mujoco.mj_applyFT(
            cb_model,
            cb_data,
            force,
            zero_torque,
            point,
            body_id,
            cb_data.qfrc_passive,
        )

    mujoco.set_mjcb_passive(passive_callback)
    try:
        yield
    finally:
        mujoco.set_mjcb_passive(previous_callback)


def simulate(
    scenario: Scenario,
    mode: FrameMode,
    n_orbits: float = N_ORBITS,
) -> Result:
    """Run one frame mode and return numerical stability metrics."""
    chief0, mean_motion, orbit_period = make_initial_chief(scenario)
    rho0 = INITIAL_REL_POS_LVLH
    rhod0 = np.array([0.0, -2.0 * mean_motion * rho0[0], 0.0])
    body0 = initial_body_from_lvlh(chief0, rho0, rhod0)

    chief = OrbitState(chief0.r.copy(), chief0.v.copy())
    truth_body = OrbitState(body0.r.copy(), body0.v.copy())

    model, data = make_model_data()
    set_initial_mujoco_state(data, mode, chief, body0)
    chief_context = ChiefContext(chief, float(data.time), "float64", 1.0)

    n_steps = int(round(n_orbits * orbit_period / ORBIT_TIMESTEP))
    substeps = int(round(ORBIT_TIMESTEP / MUJOCO_TIMESTEP))
    if not np.isclose(substeps * MUJOCO_TIMESTEP, ORBIT_TIMESTEP):
        raise ValueError("MUJOCO_TIMESTEP must divide ORBIT_TIMESTEP for this experiment.")

    with frame_force_callback(model, mode, chief_context):
        mujoco.mj_forward(model, data)

        energy0, h0, _ = attitude_invariants(model, data)
        max_energy_drift = 0.0
        max_h_drift = 0.0
        max_quat_norm_error = 0.0
        max_position_error = 0.0
        max_relative_radius = 0.0
        orbital_energy0, orbital_h0 = orbital_invariants(body0)
        max_orbital_energy_drift = 0.0
        max_orbital_h_drift = 0.0
        final_orbital_energy_drift = 0.0
        final_orbital_h_drift = 0.0
        finite = True

        for _ in range(n_steps):
            for _ in range(substeps):
                chief_context.step_start = chief
                chief_context.step_start_time = float(data.time)
                data.xfrc_applied[:] = 0.0
                mujoco.mj_step(model, data)
                chief = rk4_orbit_step(chief, MUJOCO_TIMESTEP)

            for _ in range(substeps):
                truth_body = rk4_orbit_step(truth_body, MUJOCO_TIMESTEP)

            sim_body = reconstruct_eci_state(data, mode, chief)
            relative = sim_body.r - chief.r
            position_error = float(np.linalg.norm(sim_body.r - truth_body.r))

            orbital_energy, orbital_h = orbital_invariants(sim_body)
            final_orbital_energy_drift = abs(orbital_energy - orbital_energy0) / abs(
                orbital_energy0
            )
            final_orbital_h_drift = abs(orbital_h - orbital_h0) / orbital_h0
            max_orbital_energy_drift = max(
                max_orbital_energy_drift,
                final_orbital_energy_drift,
            )
            max_orbital_h_drift = max(max_orbital_h_drift, final_orbital_h_drift)

            energy, h_norm, quat_norm_error = attitude_invariants(model, data)
            max_energy_drift = max(max_energy_drift, abs(energy - energy0) / energy0)
            max_h_drift = max(max_h_drift, abs(h_norm - h0) / h0)
            max_quat_norm_error = max(max_quat_norm_error, quat_norm_error)
            max_position_error = max(max_position_error, position_error)
            max_relative_radius = max(max_relative_radius, float(np.linalg.norm(relative)))
            finite = finite and np.all(np.isfinite(data.qpos)) and np.all(
                np.isfinite(data.qvel)
            )

    final_body = reconstruct_eci_state(data, mode, chief)
    final_error = float(np.linalg.norm(final_body.r - truth_body.r))
    final_relative_lvlh = frame_cache(chief).c_li @ (final_body.r - chief.r)
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


def simulate_time_history(
    scenario: Scenario,
    mode: FrameMode,
    integrator: mujoco.mjtIntegrator,
    dt: float,
    duration: float,
    label: str,
    precision: StudyPrecision,
    rel_vel_bias_lvlh: np.ndarray,
    length_unit_m: float,
) -> TimeHistory:
    """Run one frame/integrator pair and record position error at each step."""
    chief0, body0, _, _ = initial_states(scenario, rel_vel_bias_lvlh, length_unit_m)
    chief = OrbitState(chief0.r.copy(), chief0.v.copy())
    truth_body = OrbitState(body0.r.copy(), body0.v.copy())

    model, data = make_model_data(integrator=integrator, timestep=dt)
    set_initial_mujoco_state(data, mode, chief, body0, length_unit_m)
    quantize_mujoco_state(data, precision)
    chief_context = ChiefContext(chief, float(data.time), precision, length_unit_m)

    n_steps = int(round(duration / dt))
    if n_steps < 1:
        raise ValueError("duration must cover at least one step")
    duration = n_steps * dt

    times_s = np.empty(n_steps + 1, dtype=np.float64)
    position_error_m = np.empty(n_steps + 1, dtype=np.float64)
    energy_error_rel = np.empty(n_steps + 1, dtype=np.float64)
    relative_radius_m = np.empty(n_steps + 1, dtype=np.float64)
    finite = True
    energy0, _ = orbital_invariants(body0, length_unit_m)

    with frame_force_callback(model, mode, chief_context):
        mujoco.mj_forward(model, data)

        times_s[0] = float(data.time)
        sim_body = reconstruct_eci_state(data, mode, chief, precision, length_unit_m)
        position_error_m[0] = float(
            np.linalg.norm(sim_body.r - truth_body.r) * length_unit_m
        )
        energy, _ = orbital_invariants(sim_body, length_unit_m)
        energy_error_rel[0] = abs(energy - energy0) / abs(energy0)
        relative_radius_m[0] = float(np.linalg.norm(sim_body.r - chief.r) * length_unit_m)

        for idx in range(1, n_steps + 1):
            chief_context.step_start = chief
            chief_context.step_start_time = float(data.time)
            data.xfrc_applied[:] = 0.0
            mujoco.mj_step(model, data)
            quantize_mujoco_state(data, precision)

            chief = rk4_orbit_step(chief, dt, length_unit_m)
            truth_body = rk4_orbit_step(truth_body, dt, length_unit_m)
            sim_body = reconstruct_eci_state(data, mode, chief, precision, length_unit_m)

            times_s[idx] = min(float(data.time), duration)
            position_error_m[idx] = float(
                np.linalg.norm(sim_body.r - truth_body.r) * length_unit_m
            )
            energy, _ = orbital_invariants(sim_body, length_unit_m)
            energy_error_rel[idx] = abs(energy - energy0) / abs(energy0)
            relative_radius_m[idx] = float(
                np.linalg.norm(sim_body.r - chief.r) * length_unit_m
            )
            finite = finite and np.all(np.isfinite(data.qpos)) and np.all(
                np.isfinite(data.qvel)
            )

    return TimeHistory(
        label=label,
        scenario=scenario,
        mode=mode,
        integrator=integrator,
        precision=precision,
        length_unit_m=length_unit_m,
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
    length_unit_m: float,
) -> list[TimeHistory]:
    """Run the ECI/local-chief and MuJoCo integrator time-history cases."""
    _, _, _, orbit_period = initial_states(scenario, rel_vel_bias_lvlh, length_unit_m)
    cases = [
        ("ECI + Euler", FrameMode.ECI, mujoco.mjtIntegrator.mjINT_EULER),
        ("ECI + RK4", FrameMode.ECI, mujoco.mjtIntegrator.mjINT_RK4),
        ("ECI + implicit", FrameMode.ECI, mujoco.mjtIntegrator.mjINT_IMPLICIT),
        (
            "ECI + implicitfast",
            FrameMode.ECI,
            mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        ),
        (
            "local chief + Euler",
            FrameMode.CHIEF_INERTIAL,
            mujoco.mjtIntegrator.mjINT_EULER,
        ),
        (
            "local chief + RK4",
            FrameMode.CHIEF_INERTIAL,
            mujoco.mjtIntegrator.mjINT_RK4,
        ),
        (
            "local chief + implicit",
            FrameMode.CHIEF_INERTIAL,
            mujoco.mjtIntegrator.mjINT_IMPLICIT,
        ),
        (
            "local chief + implicitfast",
            FrameMode.CHIEF_INERTIAL,
            mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        ),
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
            length_unit_m=length_unit_m,
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
    length_unit_m: float,
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
    unit_suffix = f"_unit_{format_float_for_filename(length_unit_m)}m"
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

    colors = {
        mujoco.mjtIntegrator.mjINT_EULER: "tab:blue",
        mujoco.mjtIntegrator.mjINT_RK4: "tab:orange",
        mujoco.mjtIntegrator.mjINT_IMPLICIT: "tab:green",
        mujoco.mjtIntegrator.mjINT_IMPLICITFAST: "tab:red",
    }

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
        f"{length_unit_m:g} m/unit"
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
        f"{length_unit_m:g} m/unit"
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
        length_unit_m=np.asarray(length_unit_m),
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
        "--mujoco-timestep",
        type=float,
        default=MUJOCO_TIMESTEP,
        help="MuJoCo timestep in seconds. Must divide --orbit-timestep.",
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
            "Run an ECI/local-chief comparison with Euler, RK4, implicit, and implicitfast, "
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
        default=MUJOCO_TIMESTEP,
        help="Shared MuJoCo, chief-propagation, and reference timestep for --integrator-study.",
    )
    parser.add_argument(
        "--study-precision",
        choices=("float64", "float32"),
        default="float64",
        help=(
            "Precision used for MuJoCo state rounding and callback force inputs in "
            "--integrator-study. MuJoCo itself remains the wheel's compiled precision."
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
        "--study-length-unit-m",
        type=float,
        default=1.0,
        help=(
            "Meters per coordinate length unit for --integrator-study. Use 1000 for km "
            "or 100000 for 100 km units; plotted errors are still reported in meters."
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
    global MUJOCO_TIMESTEP, N_ORBITS, ORBIT_TIMESTEP

    args = parse_args()
    ORBIT_TIMESTEP = args.orbit_timestep
    MUJOCO_TIMESTEP = args.mujoco_timestep
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
        if args.study_length_unit_m <= 0.0:
            raise SystemExit("--study-length-unit-m must be positive")
        study = run_integrator_study(
            study_scenarios[0],
            args.study_dt,
            args.study_orbits,
            args.study_precision,
            np.asarray(args.study_rel_vel_lvlh, dtype=np.float64),
            args.study_length_unit_m,
        )
        plot_path, energy_plot_path, samples_path = write_integrator_study(
            study,
            args.out_dir,
            args.study_orbits,
            args.study_dt,
            args.study_precision,
            np.asarray(args.study_rel_vel_lvlh, dtype=np.float64),
            args.study_length_unit_m,
        )
        print("Frame study integrator comparison")
        print(f"Scenario: {study_scenarios[0].name}")
        print(f"Shared dt: {args.study_dt:g} s")
        print(f"Study precision: {args.study_precision}")
        print(f"Coordinate length unit: {args.study_length_unit_m:g} m")
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

    print("Frame study: raw MuJoCo + tiny two-body propagator")
    print(f"MuJoCo integrator: {MUJOCO_INTEGRATOR.name}")
    print("Frame forces: passive callback, re-evaluated inside RK4 stages")
    print(
        f"orbit dt: {ORBIT_TIMESTEP:g} s, MuJoCo dt: {MUJOCO_TIMESTEP:g} s, "
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
        raise SystemExit("At least one frame produced a non-finite MuJoCo state.")


if __name__ == "__main__":
    main()
