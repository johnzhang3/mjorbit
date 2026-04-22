"""Minimal MuJoCo frame study for orbital relative motion and attitude.

This experiment intentionally does not import ``mujoco_orbit``. It uses raw
MuJoCo plus a tiny two-body propagator to compare three choices for the MuJoCo
world coordinates:

- ``eci``: absolute Earth-centered inertial position/velocity, in meters.
- ``chief_inertial``: chief-centered position/velocity with inertially fixed axes.
- ``lvlh``: chief-centered rotating LVLH coordinates.

Each scenario uses a small chief-relative LVLH initial condition plus
torque-free rigid-body attitude. It runs for a few orbits and compares each
MuJoCo result against an independent ECI two-body RK4 reference.

Usage:
    uv run python experiments/frame_study/run.py
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import mujoco
import numpy as np

MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH = 6_378_137.0  # m
MASS = 100.0  # kg

ORBIT_TIMESTEP = 0.5  # s
MUJOCO_TIMESTEP = 0.25  # s
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
    r: np.ndarray  # m
    v: np.ndarray  # m/s


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
    energy_drift_rel: float
    angular_momentum_drift_rel: float
    max_quat_norm_error: float
    finite: bool


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
    radius = np.linalg.norm(r)
    return -MU_EARTH / radius**3 * r


def rk4_orbit_step(state: OrbitState, dt: float) -> OrbitState:
    """RK4 step for an inertial two-body point mass."""

    def deriv(r: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return v, gravity(r)

    k1_r, k1_v = deriv(state.r, state.v)
    k2_r, k2_v = deriv(state.r + 0.5 * dt * k1_r, state.v + 0.5 * dt * k1_v)
    k3_r, k3_v = deriv(state.r + 0.5 * dt * k2_r, state.v + 0.5 * dt * k2_v)
    k4_r, k4_v = deriv(state.r + dt * k3_r, state.v + dt * k3_v)

    r_next = state.r + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
    v_next = state.v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return OrbitState(r_next, v_next)


def frame_cache(chief: OrbitState) -> FrameCache:
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

    chief_accel = gravity(chief.r)
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


def make_initial_chief(scenario: Scenario) -> tuple[OrbitState, float, float]:
    """Return a chief orbit state, mean motion, and period for a scenario."""
    mean_motion = np.sqrt(MU_EARTH / scenario.semi_major_axis**3)
    orbit_period = 2.0 * np.pi / mean_motion
    return keplerian_to_cartesian(scenario), mean_motion, orbit_period


def initial_body_from_lvlh(
    chief: OrbitState,
    rho_lvlh: np.ndarray,
    rhod_lvlh: np.ndarray,
) -> OrbitState:
    """Convert initial LVLH relative state to absolute ECI."""
    fc = frame_cache(chief)
    r = chief.r + fc.c_il @ rho_lvlh
    v = chief.v + fc.c_il @ (rhod_lvlh + np.cross(fc.omega_lvlh, rho_lvlh))
    return OrbitState(r, v)


def make_model_data() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile the one-body MuJoCo model."""
    model = mujoco.MjModel.from_xml_string(XML)
    model.opt.integrator = MUJOCO_INTEGRATOR
    data = mujoco.MjData(model)
    return model, data


def interpolate_orbit(start: OrbitState, end: OrbitState, alpha: float) -> OrbitState:
    """Linear interpolation used only to refresh frame forces during substeps."""
    return OrbitState(
        (1.0 - alpha) * start.r + alpha * end.r,
        (1.0 - alpha) * start.v + alpha * end.v,
    )


def set_initial_mujoco_state(
    data: mujoco.MjData,
    mode: FrameMode,
    chief: OrbitState,
    body: OrbitState,
) -> None:
    """Set free-joint position, attitude, and velocity for one frame mode."""
    fc = frame_cache(chief)
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
) -> OrbitState:
    """Convert the MuJoCo free-joint state back to absolute ECI."""
    fc = frame_cache(chief)
    qpos = data.qpos[:3].copy()
    qvel = data.qvel[:3].copy()

    if mode == FrameMode.ECI:
        return OrbitState(qpos, qvel)
    if mode == FrameMode.CHIEF_INERTIAL:
        return OrbitState(chief.r + qpos, chief.v + qvel)
    if mode == FrameMode.LVLH:
        r = chief.r + fc.c_il @ qpos
        v = chief.v + fc.c_il @ (qvel + np.cross(fc.omega_lvlh, qpos))
        return OrbitState(r, v)
    raise ValueError(mode)  # pragma: no cover


def frame_acceleration(data: mujoco.MjData, mode: FrameMode, chief: OrbitState) -> np.ndarray:
    """Return the translational acceleration to apply in the active frame."""
    fc = frame_cache(chief)
    qpos = data.qpos[:3].copy()
    qvel = data.qvel[:3].copy()
    chief_gravity = gravity(chief.r)

    if mode == FrameMode.ECI:
        return gravity(qpos)
    if mode == FrameMode.CHIEF_INERTIAL:
        return gravity(chief.r + qpos) - chief_gravity
    if mode == FrameMode.LVLH:
        body_r = chief.r + fc.c_il @ qpos
        gravity_diff = fc.c_li @ (gravity(body_r) - chief_gravity)
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


def simulate(scenario: Scenario, mode: FrameMode, n_orbits: float = N_ORBITS) -> Result:
    """Run one frame mode and return numerical stability metrics."""
    chief0, mean_motion, orbit_period = make_initial_chief(scenario)
    rho0 = INITIAL_REL_POS_LVLH
    rhod0 = np.array([0.0, -2.0 * mean_motion * rho0[0], 0.0])
    body0 = initial_body_from_lvlh(chief0, rho0, rhod0)

    chief = OrbitState(chief0.r.copy(), chief0.v.copy())
    truth_body = OrbitState(body0.r.copy(), body0.v.copy())

    model, data = make_model_data()
    set_initial_mujoco_state(data, mode, chief, body0)
    mujoco.mj_forward(model, data)

    energy0, h0, _ = attitude_invariants(model, data)
    max_energy_drift = 0.0
    max_h_drift = 0.0
    max_quat_norm_error = 0.0
    max_position_error = 0.0
    max_relative_radius = 0.0
    finite = True

    n_steps = int(round(n_orbits * orbit_period / ORBIT_TIMESTEP))
    substeps = int(round(ORBIT_TIMESTEP / MUJOCO_TIMESTEP))
    if not np.isclose(substeps * MUJOCO_TIMESTEP, ORBIT_TIMESTEP):
        raise ValueError("MUJOCO_TIMESTEP must divide ORBIT_TIMESTEP for this experiment.")

    for _ in range(n_steps):
        chief_next = rk4_orbit_step(chief, ORBIT_TIMESTEP)
        for substep_idx in range(substeps):
            alpha = substep_idx / substeps
            substep_chief = interpolate_orbit(chief, chief_next, alpha)
            data.xfrc_applied[:] = 0.0
            data.xfrc_applied[1, :3] = MASS * frame_acceleration(data, mode, substep_chief)
            mujoco.mj_step(model, data)

        chief = chief_next
        truth_body = rk4_orbit_step(truth_body, ORBIT_TIMESTEP)

        sim_body = reconstruct_eci_state(data, mode, chief)
        relative = sim_body.r - chief.r
        position_error = float(np.linalg.norm(sim_body.r - truth_body.r))

        energy, h_norm, quat_norm_error = attitude_invariants(model, data)
        max_energy_drift = max(max_energy_drift, abs(energy - energy0) / energy0)
        max_h_drift = max(max_h_drift, abs(h_norm - h0) / h0)
        max_quat_norm_error = max(max_quat_norm_error, quat_norm_error)
        max_position_error = max(max_position_error, position_error)
        max_relative_radius = max(max_relative_radius, float(np.linalg.norm(relative)))
        finite = finite and np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))

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
        energy_drift_rel=max_energy_drift,
        angular_momentum_drift_rel=max_h_drift,
        max_quat_norm_error=max_quat_norm_error,
        finite=finite,
    )


def main() -> None:
    print("Frame study: raw MuJoCo + tiny two-body propagator")
    print(f"MuJoCo integrator: {MUJOCO_INTEGRATOR.name}")
    print(
        f"orbit dt: {ORBIT_TIMESTEP:g} s, MuJoCo dt: {MUJOCO_TIMESTEP:g} s, "
        f"duration: {N_ORBITS:g} orbits per scenario"
    )

    all_results: list[Result] = []
    for scenario in SCENARIOS:
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

        results = [simulate(scenario, mode) for mode in FrameMode]
        all_results.extend(results)
        header = (
            "frame             finite  max pos err [m]  final err [m]  "
            "max |rho| [m]  rot E drift  |H| drift  max |q|-1"
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
