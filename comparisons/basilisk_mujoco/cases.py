"""Runnable comparison cases for mujoco_orbit and Basilisk-MuJoCo style setups."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mujoco_orbit import MjoData, OrbitInit, mjo_forward, mjo_step

from .common import (
    ASSET_DIR,
    GM_EARTH,
    basilisk_mujoco_import_status,
    body_eci_state,
    circular_orbit_state_at,
    compile_mjorbit_model,
    make_circular_orbit,
    orbital_energy_km2_s2,
    sample_steps,
    system_com_world_m,
)

SINGLE_BODY_XML = ASSET_DIR / "single_body_orbit.xml"
HINGED_SATELLITE_XML = ASSET_DIR / "hinged_satellite.xml"


@dataclass
class SingleBodyRun:
    summary: dict[str, Any]
    times_s: np.ndarray
    r_eci_km: np.ndarray
    v_eci_km_s: np.ndarray
    quat_world_body: np.ndarray
    r_ref_eci_km: np.ndarray
    v_ref_eci_km_s: np.ndarray
    basilisk_times_s: np.ndarray | None = None
    basilisk_r_eci_km: np.ndarray | None = None
    basilisk_v_eci_km_s: np.ndarray | None = None
    basilisk_quat_world_body: np.ndarray | None = None


@dataclass
class ArticulatedRun:
    summary: dict[str, Any]
    times_s: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    hinge_targets_rad: np.ndarray
    hub_r_eci_km: np.ndarray
    hub_v_eci_km_s: np.ndarray
    system_com_world_m: np.ndarray
    basilisk_times_s: np.ndarray | None = None
    basilisk_hinge_angles_rad: np.ndarray | None = None
    basilisk_hinge_rates_rad_s: np.ndarray | None = None
    basilisk_hub_r_eci_km: np.ndarray | None = None
    basilisk_hub_v_eci_km_s: np.ndarray | None = None


def run_single_body_mujoco_orbit(
    *,
    alt_km: float = 400.0,
    inc_deg: float = 51.6,
    n_steps: int = 8000,
    dt_s: float | None = None,
    duration_s: float | None = None,
    orbit_dt: float = 0.1,
    max_samples: int = 512,
    basilisk_integrator: str = "rkf45",
    initial_quat_world_body: np.ndarray | None = None,
    initial_omega_body_rad_s: np.ndarray | None = None,
    xml_path: Path = SINGLE_BODY_XML,
) -> SingleBodyRun:
    """Run a one-period single-body orbit in mujoco_orbit and compare to exact circular motion."""
    orbit = make_circular_orbit(alt_km=alt_km, inc_rad=np.deg2rad(inc_deg))
    if dt_s is None:
        dt = orbit.period_s / float(n_steps)
        duration = orbit.period_s if duration_s is None else float(duration_s)
    else:
        dt = float(dt_s)
        duration = orbit.period_s if duration_s is None else float(duration_s)
        n_steps = int(round(duration / dt))
    if initial_quat_world_body is None:
        initial_quat_world_body = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if initial_omega_body_rad_s is None:
        initial_omega_body_rad_s = np.zeros(3, dtype=np.float64)
    model = compile_mjorbit_model(
        xml_path,
        plugin_body="spacecraft",
        mj_timestep=dt,
        orbit_dt=orbit_dt,
        use_gravity_gradient=False,
    )
    data = MjoData(model, orbit=OrbitInit(orbit.r_eci_km, orbit.v_eci_km_s))
    body_id = model.body_id("spacecraft")
    data.qpos[3:7] = _normalized_quat(initial_quat_world_body)
    data.qvel[3:6] = np.asarray(initial_omega_body_rad_s, dtype=np.float64)
    mjo_forward(model, data)

    steps_to_sample = set(int(step) for step in sample_steps(n_steps, max_samples))
    times: list[float] = []
    r_samples: list[np.ndarray] = []
    v_samples: list[np.ndarray] = []
    q_samples: list[np.ndarray] = []

    def record() -> None:
        r_eci_km, v_eci_km_s = body_eci_state(data, body_id=body_id)
        times.append(float(data.orbit.t))
        r_samples.append(r_eci_km)
        v_samples.append(v_eci_km_s)
        q_samples.append(np.asarray(data.qpos[3:7], dtype=np.float64).copy())

    record()
    for step in range(1, n_steps + 1):
        mjo_step(model, data)
        if step in steps_to_sample:
            record()

    times_s = np.asarray(times, dtype=np.float64)
    r_eci_km = np.vstack(r_samples)
    v_eci_km_s = np.vstack(v_samples)
    quat_world_body = np.vstack(q_samples)
    r_ref_eci_km, v_ref_eci_km_s = circular_orbit_state_at(
        times_s,
        alt_km=alt_km,
        inc_rad=np.deg2rad(inc_deg),
    )

    position_errors_m = np.linalg.norm(r_eci_km - r_ref_eci_km, axis=1) * 1.0e3
    velocity_errors_m_s = np.linalg.norm(v_eci_km_s - v_ref_eci_km_s, axis=1) * 1.0e3

    basilisk = _run_basilisk_single_body(
        orbit=orbit,
        dt_s=dt,
        n_steps=n_steps,
        xml_path=xml_path,
        body_name="spacecraft",
        integrator_name=basilisk_integrator,
        initial_quat_world_body=_normalized_quat(initial_quat_world_body),
        initial_omega_body_rad_s=np.asarray(initial_omega_body_rad_s, dtype=np.float64),
    )
    basilisk_payload: dict[str, Any] = basilisk["summary"]
    basilisk_times_s = basilisk.get("times_s")
    basilisk_r_eci_km = basilisk.get("r_eci_km")
    basilisk_v_eci_km_s = basilisk.get("v_eci_km_s")
    basilisk_quat_world_body = basilisk.get("quat_world_body")
    if (
        isinstance(basilisk_times_s, np.ndarray)
        and isinstance(basilisk_r_eci_km, np.ndarray)
        and isinstance(basilisk_v_eci_km_s, np.ndarray)
        and isinstance(basilisk_quat_world_body, np.ndarray)
        and basilisk_times_s.size >= 2
    ):
        r_basilisk_at_ours = _interp_columns(times_s, basilisk_times_s, basilisk_r_eci_km)
        v_basilisk_at_ours = _interp_columns(times_s, basilisk_times_s, basilisk_v_eci_km_s)
        q_basilisk_at_ours = _interp_quat(times_s, basilisk_times_s, basilisk_quat_world_body)
        attitude_errors_rad = _quat_angle_errors(quat_world_body, q_basilisk_at_ours)
        basilisk_payload.update(
            {
                "ours_vs_basilisk_max_position_error_m": float(
                    np.max(np.linalg.norm(r_eci_km - r_basilisk_at_ours, axis=1)) * 1.0e3
                ),
                "ours_vs_basilisk_max_velocity_error_m_s": float(
                    np.max(np.linalg.norm(v_eci_km_s - v_basilisk_at_ours, axis=1)) * 1.0e3
                ),
                "ours_vs_basilisk_final_attitude_error_rad": float(attitude_errors_rad[-1]),
                "ours_vs_basilisk_max_attitude_error_rad": float(np.max(attitude_errors_rad)),
            }
        )

    summary = {
        "case": "single_body_orbit",
        "backend": "mujoco_orbit",
        "alt_km": alt_km,
        "inc_deg": inc_deg,
        "period_s": orbit.period_s,
        "duration_s": n_steps * dt,
        "dt_s": dt,
        "n_steps": n_steps,
        "orbit_dt_s": orbit_dt,
        "initial_omega_body_rad_s": np.asarray(initial_omega_body_rad_s, dtype=np.float64),
        "samples": int(times_s.size),
        "final_time_s": float(times_s[-1]),
        "period_error_s": float(times_s[-1] - orbit.period_s),
        "final_position_error_m": float(position_errors_m[-1]),
        "max_position_error_m": float(np.max(position_errors_m)),
        "final_velocity_error_m_s": float(velocity_errors_m_s[-1]),
        "max_velocity_error_m_s": float(np.max(velocity_errors_m_s)),
        "basilisk_direct": basilisk_payload,
    }
    return SingleBodyRun(
        summary,
        times_s,
        r_eci_km,
        v_eci_km_s,
        quat_world_body,
        r_ref_eci_km,
        v_ref_eci_km_s,
        basilisk_times_s if isinstance(basilisk_times_s, np.ndarray) else None,
        basilisk_r_eci_km if isinstance(basilisk_r_eci_km, np.ndarray) else None,
        basilisk_v_eci_km_s if isinstance(basilisk_v_eci_km_s, np.ndarray) else None,
        basilisk_quat_world_body if isinstance(basilisk_quat_world_body, np.ndarray) else None,
    )


def run_articulated_hinges_mujoco_orbit(
    *,
    alt_km: float = 400.0,
    inc_deg: float = 51.6,
    duration_s: float = 120.0,
    dt_s: float = 0.01,
    orbit_dt: float = 0.1,
    max_samples: int = 512,
    basilisk_integrator: str = "rkf45",
    xml_path: Path = HINGED_SATELLITE_XML,
) -> ArticulatedRun:
    """Run a two-hinge articulated satellite with deterministic position targets."""
    orbit = make_circular_orbit(alt_km=alt_km, inc_rad=np.deg2rad(inc_deg))
    n_steps = int(round(duration_s / dt_s))
    model = compile_mjorbit_model(
        xml_path,
        plugin_body="hub",
        mj_timestep=dt_s,
        orbit_dt=orbit_dt,
        use_gravity_gradient=False,
    )
    data = MjoData(model, orbit=OrbitInit(orbit.r_eci_km, orbit.v_eci_km_s))
    passive = MjoData(model, orbit=OrbitInit(orbit.r_eci_km.copy(), orbit.v_eci_km_s.copy()))
    hub_id = model.body_id("hub")

    mjo_forward(model, data)
    mjo_forward(model, passive)
    initial_energy = orbital_energy_km2_s2(data.orbit.R_eci.copy(), data.orbit.V_eci.copy())

    steps_to_sample = set(int(step) for step in sample_steps(n_steps, max_samples))
    times: list[float] = []
    qpos: list[np.ndarray] = []
    qvel: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    hub_r: list[np.ndarray] = []
    hub_v: list[np.ndarray] = []
    com_world: list[np.ndarray] = []

    def record(target: np.ndarray) -> None:
        r_eci_km, v_eci_km_s = body_eci_state(data, body_id=hub_id)
        times.append(float(data.orbit.t))
        qpos.append(np.asarray(data.qpos, dtype=np.float64).copy())
        qvel.append(np.asarray(data.qvel, dtype=np.float64).copy())
        targets.append(target.copy())
        hub_r.append(r_eci_km)
        hub_v.append(v_eci_km_s)
        com_world.append(system_com_world_m(model, data))

    target = _hinge_targets(0.0, duration_s)
    data.ctrl[:] = target
    record(target)
    for step in range(1, n_steps + 1):
        t = step * dt_s
        target = _hinge_targets(t, duration_s)
        data.ctrl[:] = target
        mjo_step(model, data)
        mjo_step(model, passive)
        if step in steps_to_sample:
            record(target)

    times_arr = np.asarray(times, dtype=np.float64)
    qpos_arr = np.vstack(qpos)
    qvel_arr = np.vstack(qvel)
    target_arr = np.vstack(targets)
    hub_r_arr = np.vstack(hub_r)
    hub_v_arr = np.vstack(hub_v)
    com_world_arr = np.vstack(com_world)
    final_energy = orbital_energy_km2_s2(data.orbit.R_eci.copy(), data.orbit.V_eci.copy())
    final_joint_error = qpos_arr[-1, 7:9] - target_arr[-1]
    chief_delta_m = np.linalg.norm(data.orbit.R_eci - passive.orbit.R_eci) * 1.0e3
    chief_delta_m_s = np.linalg.norm(data.orbit.V_eci - passive.orbit.V_eci) * 1.0e3
    basilisk = _run_basilisk_articulated_hinges(
        orbit=orbit,
        duration_s=duration_s,
        dt_s=dt_s,
        xml_path=xml_path,
        integrator_name=basilisk_integrator,
    )
    basilisk_payload: dict[str, Any] = basilisk["summary"]
    basilisk_times_s = basilisk.get("times_s")
    basilisk_hinge_angles_rad = basilisk.get("hinge_angles_rad")
    basilisk_hinge_rates_rad_s = basilisk.get("hinge_rates_rad_s")
    basilisk_hub_r_eci_km = basilisk.get("hub_r_eci_km")
    basilisk_hub_v_eci_km_s = basilisk.get("hub_v_eci_km_s")
    if (
        isinstance(basilisk_times_s, np.ndarray)
        and isinstance(basilisk_hinge_angles_rad, np.ndarray)
        and isinstance(basilisk_hinge_rates_rad_s, np.ndarray)
        and isinstance(basilisk_hub_r_eci_km, np.ndarray)
        and isinstance(basilisk_hub_v_eci_km_s, np.ndarray)
        and basilisk_times_s.size >= 2
    ):
        b_angles = _interp_columns(times_arr, basilisk_times_s, basilisk_hinge_angles_rad)
        b_rates = _interp_columns(times_arr, basilisk_times_s, basilisk_hinge_rates_rad_s)
        b_hub_r = _interp_columns(times_arr, basilisk_times_s, basilisk_hub_r_eci_km)
        b_hub_v = _interp_columns(times_arr, basilisk_times_s, basilisk_hub_v_eci_km_s)
        basilisk_payload.update(
            {
                "ours_vs_basilisk_max_hinge_angle_error_rad": float(
                    np.max(np.linalg.norm(qpos_arr[:, 7:9] - b_angles, axis=1))
                ),
                "ours_vs_basilisk_max_hinge_rate_error_rad_s": float(
                    np.max(np.linalg.norm(qvel_arr[:, 6:8] - b_rates, axis=1))
                ),
                "ours_vs_basilisk_max_hub_position_error_m": float(
                    np.max(np.linalg.norm(hub_r_arr - b_hub_r, axis=1)) * 1.0e3
                ),
                "ours_vs_basilisk_max_hub_velocity_error_m_s": float(
                    np.max(np.linalg.norm(hub_v_arr - b_hub_v, axis=1)) * 1.0e3
                ),
            }
        )

    summary = {
        "case": "articulated_hinges",
        "backend": "mujoco_orbit",
        "alt_km": alt_km,
        "inc_deg": inc_deg,
        "duration_s": duration_s,
        "dt_s": dt_s,
        "n_steps": n_steps,
        "orbit_dt_s": orbit_dt,
        "samples": int(len(times)),
        "all_finite": bool(np.all(np.isfinite(qpos_arr)) and np.all(np.isfinite(qvel_arr))),
        "final_hinge_1_rad": float(qpos_arr[-1, 7]),
        "final_hinge_2_rad": float(qpos_arr[-1, 8]),
        "final_target_hinge_1_rad": float(target_arr[-1, 0]),
        "final_target_hinge_2_rad": float(target_arr[-1, 1]),
        "final_joint_error_norm_rad": float(np.linalg.norm(final_joint_error)),
        "max_joint_speed_rad_s": float(np.max(np.abs(qvel_arr[:, 6:8]))),
        "chief_delta_vs_passive_m": float(chief_delta_m),
        "chief_delta_vs_passive_m_s": float(chief_delta_m_s),
        "orbit_energy_rel_change": float(abs(final_energy - initial_energy) / abs(initial_energy)),
        "basilisk_direct": basilisk_payload,
    }
    return ArticulatedRun(
        summary=summary,
        times_s=times_arr,
        qpos=qpos_arr,
        qvel=qvel_arr,
        hinge_targets_rad=target_arr,
        hub_r_eci_km=hub_r_arr,
        hub_v_eci_km_s=hub_v_arr,
        system_com_world_m=com_world_arr,
        basilisk_times_s=basilisk_times_s if isinstance(basilisk_times_s, np.ndarray) else None,
        basilisk_hinge_angles_rad=(
            basilisk_hinge_angles_rad if isinstance(basilisk_hinge_angles_rad, np.ndarray) else None
        ),
        basilisk_hinge_rates_rad_s=(
            basilisk_hinge_rates_rad_s
            if isinstance(basilisk_hinge_rates_rad_s, np.ndarray)
            else None
        ),
        basilisk_hub_r_eci_km=(
            basilisk_hub_r_eci_km if isinstance(basilisk_hub_r_eci_km, np.ndarray) else None
        ),
        basilisk_hub_v_eci_km_s=(
            basilisk_hub_v_eci_km_s if isinstance(basilisk_hub_v_eci_km_s, np.ndarray) else None
        ),
    )


def _hinge_targets(t: float, duration_s: float) -> np.ndarray:
    slew_time = 0.5 * duration_s
    alpha = min(max(t / slew_time, 0.0), 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return np.array([0.9 * alpha, -0.65 * alpha], dtype=np.float64)


def _basilisk_status_payload() -> dict[str, Any]:
    available, message = basilisk_mujoco_import_status()
    if not available:
        return {
            "available": False,
            "ran": False,
            "message": message,
        }
    return {
        "available": True,
        "ran": False,
        "message": (
            "Basilisk MuJoCo is importable. Direct execution is intentionally optional; "
            "this harness currently exports comparable trajectories for side-by-side checks."
        ),
    }


def _run_basilisk_single_body(
    *,
    orbit: Any,
    dt_s: float,
    n_steps: int,
    xml_path: Path,
    body_name: str,
    integrator_name: str,
    initial_quat_world_body: np.ndarray,
    initial_omega_body_rad_s: np.ndarray,
) -> dict[str, Any]:
    available, message = basilisk_mujoco_import_status()
    if not available:
        return {
            "summary": {
                "available": False,
                "ran": False,
                "message": message,
            }
        }

    try:  # pragma: no cover - depends on optional external Basilisk install
        from Basilisk.architecture import messaging
        from Basilisk.simulation import NBodyGravity, mujoco, pointMassGravityModel, svIntegrators
        from Basilisk.utilities import SimulationBaseClass
    except Exception as exc:  # pragma: no cover - depends on optional external install
        return {
            "summary": {
                "available": False,
                "ran": False,
                "message": f"{type(exc).__name__}: {exc}",
            }
        }

    try:  # pragma: no cover - depends on optional external Basilisk install
        task_name = "basilisk_mujoco_single"
        task_dt_ns = max(1, int(round(dt_s * 1.0e9)))
        sim = SimulationBaseClass.SimBaseClass()
        process = sim.CreateNewProcess("basilisk_mujoco_compare")
        process.addTask(sim.CreateNewTask(task_name, task_dt_ns))

        scene = mujoco.MJScene.fromFile(str(xml_path))
        scene.ModelTag = "mujocoScene"
        sim.AddModelToTask(task_name, scene)
        integrator = _make_basilisk_integrator(svIntegrators, scene, integrator_name)
        scene.setIntegrator(integrator)

        body = scene.getBody(body_name)

        gravity = NBodyGravity.NBodyGravity()
        gravity.ModelTag = "gravity"
        scene.AddModelToDynamicsTask(gravity)
        earth = pointMassGravityModel.PointMassGravityModel()
        earth.muBody = GM_EARTH * 1.0e9
        source = gravity.addGravitySource("earth", earth, isCentralBody=True)
        earth_msg = _earth_state_msg(messaging)
        source.stateInMsg.subscribeTo(earth_msg)
        gravity.addGravityTarget(body_name, body)

        recorder = body.getCenterOfMass().stateOutMsg.recorder()
        sim.AddModelToTask(task_name, recorder)

        sim.InitializeSimulation()
        body.setPosition((orbit.r_eci_km * 1.0e3).tolist())
        body.setVelocity((orbit.v_eci_km_s * 1.0e3).tolist())
        body.setAttitude(_quat_world_body_to_basilisk_mrp(initial_quat_world_body).tolist())
        body.setAttitudeRate(np.asarray(initial_omega_body_rad_s, dtype=np.float64).tolist())

        sim.ConfigureStopTime((n_steps + 1) * task_dt_ns)
        sim.ExecuteSimulation()

        times_s = np.asarray(recorder.times(), dtype=np.float64) * 1.0e-9
        r_eci_km = np.asarray(recorder.r_BN_N, dtype=np.float64) * 1.0e-3
        v_eci_km_s = np.asarray(recorder.v_BN_N, dtype=np.float64) * 1.0e-3
        quat_world_body = _basilisk_mrp_to_quat_world_body(
            np.asarray(recorder.sigma_BN, dtype=np.float64)
        )
        if times_s.size == 0 or times_s[0] > 1.0e-12:
            times_s = np.concatenate(([0.0], times_s))
            r_eci_km = np.vstack((orbit.r_eci_km, r_eci_km))
            v_eci_km_s = np.vstack((orbit.v_eci_km_s, v_eci_km_s))
            quat_world_body = np.vstack((initial_quat_world_body, quat_world_body))
        final_time_s = n_steps * task_dt_ns * 1.0e-9
        keep = times_s <= final_time_s + 1.0e-12
        times_s = times_s[keep]
        r_eci_km = r_eci_km[keep]
        v_eci_km_s = v_eci_km_s[keep]
        quat_world_body = quat_world_body[keep]
        r_ref, v_ref = circular_orbit_state_at(
            times_s,
            alt_km=orbit.alt_km,
            inc_rad=orbit.inc_rad,
        )
        pos_err_m = np.linalg.norm(r_eci_km - r_ref, axis=1) * 1.0e3
        vel_err_m_s = np.linalg.norm(v_eci_km_s - v_ref, axis=1) * 1.0e3
        return {
            "summary": {
                "available": True,
                "ran": True,
                "message": "Basilisk.simulation.mujoco direct single-body run completed",
                "integrator": _normalize_basilisk_integrator_name(integrator_name),
                "samples": int(times_s.size),
                "final_position_error_m": float(pos_err_m[-1]),
                "max_position_error_m": float(np.max(pos_err_m)),
                "final_velocity_error_m_s": float(vel_err_m_s[-1]),
                "max_velocity_error_m_s": float(np.max(vel_err_m_s)),
            },
            "times_s": times_s,
            "r_eci_km": r_eci_km,
            "v_eci_km_s": v_eci_km_s,
            "quat_world_body": quat_world_body,
        }
    except Exception as exc:  # pragma: no cover - depends on optional external install
        return {
            "summary": {
                "available": True,
                "ran": False,
                "message": f"Direct Basilisk single-body run failed: {type(exc).__name__}: {exc}",
            }
        }


def _run_basilisk_articulated_hinges(
    *,
    orbit: Any,
    duration_s: float,
    dt_s: float,
    xml_path: Path,
    integrator_name: str,
) -> dict[str, Any]:
    available, message = basilisk_mujoco_import_status()
    if not available:
        return {
            "summary": {
                "available": False,
                "ran": False,
                "message": message,
            }
        }

    try:  # pragma: no cover - depends on optional external Basilisk install
        from Basilisk.architecture import messaging
        from Basilisk.simulation import NBodyGravity, mujoco, pointMassGravityModel, svIntegrators
        from Basilisk.utilities import SimulationBaseClass
    except Exception as exc:  # pragma: no cover - depends on optional external install
        return {
            "summary": {
                "available": False,
                "ran": False,
                "message": f"{type(exc).__name__}: {exc}",
            }
        }

    try:  # pragma: no cover - depends on optional external Basilisk install
        n_steps = int(round(duration_s / dt_s))
        task_dt_ns = max(1, int(round(dt_s * 1.0e9)))
        task_name = "basilisk_mujoco_hinges"

        sim = SimulationBaseClass.SimBaseClass()
        process = sim.CreateNewProcess("basilisk_mujoco_compare")
        process.addTask(sim.CreateNewTask(task_name, task_dt_ns))

        scene = mujoco.MJScene.fromFile(str(xml_path))
        scene.ModelTag = "mujocoScene"
        sim.AddModelToTask(task_name, scene)
        integrator = _make_basilisk_integrator(svIntegrators, scene, integrator_name)
        scene.setIntegrator(integrator)

        hub = scene.getBody("hub")
        panel_1 = scene.getBody("panel_1")
        panel_2 = scene.getBody("panel_2")
        hinge_1 = panel_1.getScalarJoint("hinge_1")
        hinge_2 = panel_2.getScalarJoint("hinge_2")
        actuator_1 = scene.getSingleActuator("hinge_1_pos")
        actuator_2 = scene.getSingleActuator("hinge_2_pos")

        profile_times_ns = np.arange(n_steps + 2, dtype=np.float64) * float(task_dt_ns)
        profile_targets = np.vstack(
            [_hinge_targets(step * dt_s, duration_s) for step in range(n_steps + 2)]
        )
        interpolators = []
        for idx, actuator in enumerate((actuator_1, actuator_2)):
            interpolator = mujoco.SingleActuatorInterpolator()
            interpolator.ModelTag = f"hinge_{idx + 1}_target"
            interpolator.setDataPoints(
                np.column_stack((profile_times_ns, profile_targets[:, idx])),
                1,
            )
            scene.AddModelToDynamicsTask(interpolator)
            actuator.actuatorInMsg.subscribeTo(interpolator.interpolatedOutMsg)
            interpolators.append(interpolator)

        gravity = NBodyGravity.NBodyGravity()
        gravity.ModelTag = "gravity"
        scene.AddModelToDynamicsTask(gravity)
        earth = pointMassGravityModel.PointMassGravityModel()
        earth.muBody = GM_EARTH * 1.0e9
        source = gravity.addGravitySource("earth", earth, isCentralBody=True)
        earth_msg = _earth_state_msg(messaging)
        source.stateInMsg.subscribeTo(earth_msg)
        for name, body in (("hub", hub), ("panel_1", panel_1), ("panel_2", panel_2)):
            gravity.addGravityTarget(name, body)

        scene_recorder = scene.stateOutMsg.recorder()
        hub_recorder = hub.getCenterOfMass().stateOutMsg.recorder()
        hinge_1_recorder = hinge_1.stateOutMsg.recorder()
        hinge_2_recorder = hinge_2.stateOutMsg.recorder()
        hinge_1_rate_recorder = hinge_1.stateDotOutMsg.recorder()
        hinge_2_rate_recorder = hinge_2.stateDotOutMsg.recorder()
        recorders = [
            scene_recorder,
            hub_recorder,
            hinge_1_recorder,
            hinge_2_recorder,
            hinge_1_rate_recorder,
            hinge_2_rate_recorder,
        ]
        for recorder in recorders:
            sim.AddModelToTask(task_name, recorder)

        sim.InitializeSimulation()
        hub.setPosition((orbit.r_eci_km * 1.0e3).tolist())
        hub.setVelocity((orbit.v_eci_km_s * 1.0e3).tolist())
        hub.setAttitude([0.0, 0.0, 0.0])
        hub.setAttitudeRate([0.0, 0.0, 0.0])

        sim.ConfigureStopTime((n_steps + 1) * task_dt_ns)
        sim.ExecuteSimulation()

        times_s = np.asarray(scene_recorder.times(), dtype=np.float64) * 1.0e-9
        hinge_angles = np.column_stack(
            (
                _squeeze_state_column(hinge_1_recorder.state),
                _squeeze_state_column(hinge_2_recorder.state),
            )
        )
        hinge_rates = np.column_stack(
            (
                _squeeze_state_column(hinge_1_rate_recorder.state),
                _squeeze_state_column(hinge_2_rate_recorder.state),
            )
        )
        hub_r_eci_km = np.asarray(hub_recorder.r_BN_N, dtype=np.float64) * 1.0e-3
        hub_v_eci_km_s = np.asarray(hub_recorder.v_BN_N, dtype=np.float64) * 1.0e-3

        final_time_s = n_steps * task_dt_ns * 1.0e-9
        keep = times_s <= final_time_s + 1.0e-12
        times_s = times_s[keep]
        hinge_angles = hinge_angles[keep]
        hinge_rates = hinge_rates[keep]
        hub_r_eci_km = hub_r_eci_km[keep]
        hub_v_eci_km_s = hub_v_eci_km_s[keep]

        final_target = _hinge_targets(times_s[-1], duration_s)
        joint_error = hinge_angles[-1] - final_target
        return {
            "summary": {
                "available": True,
                "ran": True,
                "message": "Basilisk.simulation.mujoco direct articulated-hinge run completed",
                "integrator": _normalize_basilisk_integrator_name(integrator_name),
                "samples": int(times_s.size),
                "all_finite": bool(
                    np.all(np.isfinite(hinge_angles))
                    and np.all(np.isfinite(hinge_rates))
                    and np.all(np.isfinite(hub_r_eci_km))
                    and np.all(np.isfinite(hub_v_eci_km_s))
                ),
                "final_hinge_1_rad": float(hinge_angles[-1, 0]),
                "final_hinge_2_rad": float(hinge_angles[-1, 1]),
                "final_joint_error_norm_rad": float(np.linalg.norm(joint_error)),
            },
            "times_s": times_s,
            "hinge_angles_rad": hinge_angles,
            "hinge_rates_rad_s": hinge_rates,
            "hub_r_eci_km": hub_r_eci_km,
            "hub_v_eci_km_s": hub_v_eci_km_s,
        }
    except Exception as exc:  # pragma: no cover - depends on optional external install
        return {
            "summary": {
                "available": True,
                "ran": False,
                "message": (
                    "Direct Basilisk articulated-hinge run failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        }


def _earth_state_msg(messaging: Any) -> Any:
    payload = messaging.SpicePlanetStateMsgPayload()
    payload.PositionVector = [0.0, 0.0, 0.0]
    payload.VelocityVector = [0.0, 0.0, 0.0]
    payload.J20002Pfix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    payload.J20002Pfix_dot = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    payload.PlanetName = "earth"
    return messaging.SpicePlanetStateMsg().write(payload)


def _normalize_basilisk_integrator_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "euler": "euler",
        "rk1": "euler",
        "rk2": "rk2",
        "rk4": "rk4",
        "rkf45": "rkf45",
        "rk45": "rkf45",
        "rkf78": "rkf78",
        "rk78": "rkf78",
    }
    if normalized not in aliases:
        supported = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"unsupported Basilisk integrator {name!r}; choose one of {supported}")
    return aliases[normalized]


def _make_basilisk_integrator(sv_integrators: Any, scene: Any, name: str) -> Any:
    constructors = {
        "euler": sv_integrators.svIntegratorEuler,
        "rk2": sv_integrators.svIntegratorRK2,
        "rk4": sv_integrators.svIntegratorRK4,
        "rkf45": sv_integrators.svIntegratorRKF45,
        "rkf78": sv_integrators.svIntegratorRKF78,
    }
    return constructors[_normalize_basilisk_integrator_name(name)](scene)


def _quat_world_body_to_basilisk_mrp(quat_world_body: np.ndarray) -> np.ndarray:
    quat = _normalized_quat(quat_world_body)
    if quat[0] < 0.0:
        quat = -quat
    denom = 1.0 + quat[0]
    if denom <= 1.0e-14:
        return -quat[1:4]
    return quat[1:4] / denom


def _basilisk_mrp_to_quat_world_body(sigma_bn: np.ndarray) -> np.ndarray:
    sigma = np.asarray(sigma_bn, dtype=np.float64)
    if sigma.ndim == 1:
        sigma = sigma.reshape(1, 3)
    s2 = np.sum(sigma * sigma, axis=1)
    quat_world_body = np.column_stack(
        (
            (1.0 - s2) / (1.0 + s2),
            2.0 * sigma[:, 0] / (1.0 + s2),
            2.0 * sigma[:, 1] / (1.0 + s2),
            2.0 * sigma[:, 2] / (1.0 + s2),
        )
    )
    return _normalized_quat_rows(quat_world_body)


def _interp_columns(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(x, xp, fp[:, i]) for i in range(fp.shape[1])])


def _interp_quat(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    fp = _normalized_quat_rows(fp)
    out = np.empty((x.size, 4), dtype=np.float64)
    for idx, value in enumerate(x):
        right = int(np.searchsorted(xp, value, side="left"))
        if right <= 0:
            out[idx] = fp[0]
            continue
        if right >= xp.size:
            out[idx] = fp[-1]
            continue
        left = right - 1
        denom = xp[right] - xp[left]
        alpha = 0.0 if abs(denom) <= 1.0e-15 else (value - xp[left]) / denom
        q0 = fp[left]
        q1 = fp[right]
        if float(np.dot(q0, q1)) < 0.0:
            q1 = -q1
        out[idx] = _normalized_quat((1.0 - alpha) * q0 + alpha * q1)
    return out


def _quat_angle_errors(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    qa = _normalized_quat_rows(q_a)
    qb = _normalized_quat_rows(q_b)
    dots = np.abs(np.sum(qa * qb, axis=1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def _normalized_quat(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        raise ValueError("quaternion must be non-zero")
    return q / norm


def _normalized_quat_rows(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(q, axis=1)
    if np.any(norm <= 0.0):
        raise ValueError("quaternion rows must be non-zero")
    return q / norm[:, None]


def _quat_conj(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).copy()
    q[1:4] *= -1.0
    return q


def _squeeze_state_column(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)
