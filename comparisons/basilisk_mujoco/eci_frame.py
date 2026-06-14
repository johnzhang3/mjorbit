"""ECI-frame raw MuJoCo checks for the two-arm comparison case."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from experiments.frame_study.run import gravity_in_units

from .cases import (
    TWO_ARM_BODY_NAMES,
    TWO_ARM_FREE_DRIFT_XML,
    _normalized_quat,
    _quat_angle_errors,
    run_two_arm_free_drift_mjorbit,
)
from .common import make_circular_orbit, sample_steps


def frame_study_gravity(r: np.ndarray) -> np.ndarray:
    """Point-mass gravity in SI units, matching the frame-study helper."""
    return gravity_in_units(np.asarray(r), length_unit_m=1.0)


@dataclass
class TwoArmEciFrameCheck:
    summary: dict[str, Any]
    times_s: np.ndarray
    local_hinge_angles_rad: np.ndarray
    eci_hinge_angles_rad: np.ndarray
    local_hinge_rates_rad_s: np.ndarray
    eci_hinge_rates_rad_s: np.ndarray
    local_hub_quat_world_body: np.ndarray
    eci_hub_quat_world_body: np.ndarray
    local_body_r_eci_km: np.ndarray
    eci_body_r_eci_km: np.ndarray
    local_body_v_eci_km_s: np.ndarray
    eci_body_v_eci_km_s: np.ndarray
    local_system_com_eci_km: np.ndarray
    eci_system_com_eci_km: np.ndarray


def run_two_arm_eci_frame_check(
    *,
    alt_km: float = 400.0,
    inc_deg: float = 51.6,
    duration_s: float | None = None,
    dt_s: float = 0.1,
    mj_integrator: str = "RK4",
    max_samples: int = 2048,
    initial_hinge_angles_rad: tuple[float, float] | np.ndarray = (0.0, 0.0),
    initial_hinge_rates_rad_s: tuple[float, float] | np.ndarray = (0.0, 0.0),
    initial_quat_world_body: np.ndarray | None = None,
    initial_omega_body_rad_s: np.ndarray | None = None,
    gravity_application: str = "callback",
    xml_path: Path = TWO_ARM_FREE_DRIFT_XML,
) -> TwoArmEciFrameCheck:
    """Compare chief-centered ``mjorbit`` against raw absolute-ECI MuJoCo."""
    orbit = make_circular_orbit(alt_km=alt_km, inc_rad=np.deg2rad(inc_deg))
    if duration_s is None:
        duration_s = orbit.period_s
    n_steps = int(round(float(duration_s) / dt_s))
    duration = n_steps * dt_s
    if n_steps < 1:
        raise ValueError("duration_s must cover at least one step")

    if initial_quat_world_body is None:
        initial_quat_world_body = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if initial_omega_body_rad_s is None:
        initial_omega_body_rad_s = np.zeros(3, dtype=np.float64)
    initial_quat = _normalized_quat(initial_quat_world_body)
    initial_omega = np.asarray(initial_omega_body_rad_s, dtype=np.float64)
    hinge_angles0 = np.asarray(initial_hinge_angles_rad, dtype=np.float64).reshape(2)
    hinge_rates0 = np.asarray(initial_hinge_rates_rad_s, dtype=np.float64).reshape(2)

    local = run_two_arm_free_drift_mjorbit(
        alt_km=alt_km,
        inc_deg=inc_deg,
        duration_s=duration,
        dt_s=dt_s,
        orbit_dt=dt_s,
        mj_integrator=mj_integrator,
        max_samples=max_samples,
        initial_hinge_angles_rad=hinge_angles0,
        initial_hinge_rates_rad_s=hinge_rates0,
        initial_quat_world_body=initial_quat,
        initial_omega_body_rad_s=initial_omega,
        run_basilisk_direct=False,
    )
    eci = _run_raw_mujoco_eci_two_arm(
        orbit_r_eci_km=orbit.r_eci_km,
        orbit_v_eci_km_s=orbit.v_eci_km_s,
        duration_s=duration,
        dt_s=dt_s,
        mj_integrator=mj_integrator,
        max_samples=max_samples,
        initial_hinge_angles_rad=hinge_angles0,
        initial_hinge_rates_rad_s=hinge_rates0,
        initial_quat_world_body=initial_quat,
        initial_omega_body_rad_s=initial_omega,
        gravity_application=gravity_application,
        xml_path=xml_path,
    )

    attitude_errors = _quat_angle_errors(
        local.hub_quat_world_body,
        eci["hub_quat_world_body"],
    )
    body_position_errors_m = np.linalg.norm(local.body_r_eci_km - eci["body_r_eci_km"], axis=2)
    body_velocity_errors_m_s = (
        np.linalg.norm(local.body_v_eci_km_s - eci["body_v_eci_km_s"], axis=2) * 1.0e3
    )
    local_body_rel_m = (local.body_r_eci_km - local.body_r_eci_km[:, 0:1, :]) * 1.0e3
    eci_body_rel_m = (eci["body_r_eci_km"] - eci["body_r_eci_km"][:, 0:1, :]) * 1.0e3
    body_relative_position_errors_m = np.linalg.norm(local_body_rel_m - eci_body_rel_m, axis=2)
    local_body_rel_v_m_s = (
        local.body_v_eci_km_s - local.body_v_eci_km_s[:, 0:1, :]
    ) * 1.0e3
    eci_body_rel_v_m_s = (eci["body_v_eci_km_s"] - eci["body_v_eci_km_s"][:, 0:1, :]) * 1.0e3
    body_relative_velocity_errors_m_s = np.linalg.norm(
        local_body_rel_v_m_s - eci_body_rel_v_m_s,
        axis=2,
    )
    hinge_angle_errors = np.linalg.norm(
        local.hinge_angles_rad - eci["hinge_angles_rad"],
        axis=1,
    )
    hinge_rate_errors = np.linalg.norm(
        local.hinge_rates_rad_s - eci["hinge_rates_rad_s"],
        axis=1,
    )
    com_position_errors_m = (
        np.linalg.norm(local.system_com_eci_km - eci["system_com_eci_km"], axis=1) * 1.0e3
    )
    summary = {
        "case": "two_arm_eci_frame_check",
        "duration_s": duration,
        "dt_s": dt_s,
        "n_steps": n_steps,
        "mj_integrator": mj_integrator,
        "samples": int(local.times_s.size),
        "max_hinge_angle_error_rad": float(np.max(hinge_angle_errors)),
        "final_hinge_angle_error_rad": float(hinge_angle_errors[-1]),
        "max_hinge_rate_error_rad_s": float(np.max(hinge_rate_errors)),
        "max_hub_attitude_error_rad": float(np.max(attitude_errors)),
        "final_hub_attitude_error_rad": float(attitude_errors[-1]),
        "max_hub_position_error_m": float(np.max(body_position_errors_m[:, 0]) * 1.0e3),
        "final_hub_position_error_m": float(body_position_errors_m[-1, 0] * 1.0e3),
        "max_body_position_error_m": float(np.max(body_position_errors_m) * 1.0e3),
        "final_body_position_error_m": float(np.max(body_position_errors_m[-1]) * 1.0e3),
        "max_body_relative_position_error_m": float(np.max(body_relative_position_errors_m)),
        "final_body_relative_position_error_m": float(
            np.max(body_relative_position_errors_m[-1])
        ),
        "max_hub_velocity_error_m_s": float(np.max(body_velocity_errors_m_s[:, 0])),
        "max_body_velocity_error_m_s": float(np.max(body_velocity_errors_m_s)),
        "max_body_relative_velocity_error_m_s": float(np.max(body_relative_velocity_errors_m_s)),
        "max_system_com_position_error_m": float(np.max(com_position_errors_m)),
        "final_system_com_position_error_m": float(com_position_errors_m[-1]),
        "local_final_hinges_rad": local.hinge_angles_rad[-1],
        "eci_final_hinges_rad": eci["hinge_angles_rad"][-1],
        "body_names": TWO_ARM_BODY_NAMES,
        "raw_eci_style": (
            "experiments.frame_study absolute ECI world with point-mass body forces"
        ),
        "gravity_application": gravity_application,
    }
    return TwoArmEciFrameCheck(
        summary=summary,
        times_s=local.times_s,
        local_hinge_angles_rad=local.hinge_angles_rad,
        eci_hinge_angles_rad=eci["hinge_angles_rad"],
        local_hinge_rates_rad_s=local.hinge_rates_rad_s,
        eci_hinge_rates_rad_s=eci["hinge_rates_rad_s"],
        local_hub_quat_world_body=local.hub_quat_world_body,
        eci_hub_quat_world_body=eci["hub_quat_world_body"],
        local_body_r_eci_km=local.body_r_eci_km,
        eci_body_r_eci_km=eci["body_r_eci_km"],
        local_body_v_eci_km_s=local.body_v_eci_km_s,
        eci_body_v_eci_km_s=eci["body_v_eci_km_s"],
        local_system_com_eci_km=local.system_com_eci_km,
        eci_system_com_eci_km=eci["system_com_eci_km"],
    )


def _run_raw_mujoco_eci_two_arm(
    *,
    orbit_r_eci_km: np.ndarray,
    orbit_v_eci_km_s: np.ndarray,
    duration_s: float,
    dt_s: float,
    mj_integrator: str,
    max_samples: int,
    initial_hinge_angles_rad: np.ndarray,
    initial_hinge_rates_rad_s: np.ndarray,
    initial_quat_world_body: np.ndarray,
    initial_omega_body_rad_s: np.ndarray,
    gravity_application: str,
    xml_path: Path,
) -> dict[str, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = dt_s
    model.opt.integrator = _mujoco_integrator_value(mj_integrator)
    data = mujoco.MjData(model)
    body_ids = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in TWO_ARM_BODY_NAMES
    )
    body_masses = np.asarray([model.body_mass[body_id] for body_id in body_ids], dtype=np.float64)

    data.qpos[:] = model.qpos0
    data.qpos[0:3] = orbit_r_eci_km * 1.0e3
    data.qpos[3:7] = _normalized_quat(initial_quat_world_body)
    data.qpos[7:9] = initial_hinge_angles_rad
    data.qvel[:] = 0.0
    data.qvel[0:3] = orbit_v_eci_km_s * 1.0e3
    data.qvel[3:6] = initial_omega_body_rad_s
    data.qvel[6:8] = initial_hinge_rates_rad_s
    mujoco.mj_forward(model, data)

    n_steps = int(round(duration_s / dt_s))
    steps_to_sample = set(int(step) for step in sample_steps(n_steps, max_samples))
    times: list[float] = []
    hinge_angles: list[np.ndarray] = []
    hinge_rates: list[np.ndarray] = []
    hub_quat: list[np.ndarray] = []
    body_r: list[np.ndarray] = []
    body_v: list[np.ndarray] = []
    system_com: list[np.ndarray] = []

    def record() -> None:
        body_r_eci_km, body_v_eci_km_s = _raw_body_origin_eci_states(data, body_ids)
        times.append(float(data.time))
        hinge_angles.append(np.asarray(data.qpos[7:9], dtype=np.float64).copy())
        hinge_rates.append(np.asarray(data.qvel[6:8], dtype=np.float64).copy())
        hub_quat.append(np.asarray(data.qpos[3:7], dtype=np.float64).copy())
        body_r.append(body_r_eci_km)
        body_v.append(body_v_eci_km_s)
        system_com.append(_raw_system_com_eci_km(data, body_ids, body_masses))

    gravity_mode = gravity_application.strip().lower().replace("-", "_")
    if gravity_mode not in {"callback", "xfrc"}:
        raise ValueError("gravity_application must be either 'callback' or 'xfrc'")

    previous_passive_callback = mujoco.get_mjcb_passive()
    if gravity_mode == "callback":
        body_ids_for_callback = body_ids

        def passive_callback(cb_model: mujoco.MjModel, cb_data: mujoco.MjData) -> None:
            _apply_point_mass_gravity_to_passive(cb_model, cb_data, body_ids_for_callback)

        mujoco.set_mjcb_passive(passive_callback)

    try:
        record()
        for step in range(1, n_steps + 1):
            if gravity_mode == "xfrc":
                _apply_point_mass_gravity_to_xfrc(model, data, body_ids)
            else:
                data.xfrc_applied[:] = 0.0
            mujoco.mj_step(model, data)
            if step in steps_to_sample:
                record()
    finally:
        if gravity_mode == "callback":
            mujoco.set_mjcb_passive(previous_passive_callback)

    return {
        "times_s": np.asarray(times, dtype=np.float64),
        "hinge_angles_rad": np.vstack(hinge_angles),
        "hinge_rates_rad_s": np.vstack(hinge_rates),
        "hub_quat_world_body": np.vstack(hub_quat),
        "body_r_eci_km": np.stack(body_r, axis=0),
        "body_v_eci_km_s": np.stack(body_v, axis=0),
        "system_com_eci_km": np.vstack(system_com),
    }


def _apply_point_mass_gravity_to_xfrc(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_ids: tuple[int, ...],
) -> None:
    data.xfrc_applied[:] = 0.0
    for body_id in body_ids:
        mass = float(model.body_mass[body_id])
        if mass <= 0.0:
            continue
        data.xfrc_applied[body_id, 0:3] = mass * frame_study_gravity(data.xipos[body_id])


def _apply_point_mass_gravity_to_passive(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_ids: tuple[int, ...],
) -> None:
    zero_torque = np.zeros(3, dtype=np.float64)
    for body_id in body_ids:
        mass = float(model.body_mass[body_id])
        if mass <= 0.0:
            continue
        point = np.asarray(data.xipos[body_id], dtype=np.float64)
        force = mass * frame_study_gravity(point)
        mujoco.mj_applyFT(model, data, force, zero_torque, point, body_id, data.qfrc_passive)


def _raw_body_origin_eci_states(
    data: mujoco.MjData,
    body_ids: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    positions = []
    velocities = []
    for body_id in body_ids:
        origin_m = np.asarray(data.xpos[body_id], dtype=np.float64)
        com_m = np.asarray(data.xipos[body_id], dtype=np.float64)
        com_velocity_m_s = np.asarray(data.cvel[body_id, 3:6], dtype=np.float64)
        omega_world_rad_s = np.asarray(data.cvel[body_id, 0:3], dtype=np.float64)
        origin_velocity_m_s = com_velocity_m_s - np.cross(omega_world_rad_s, com_m - origin_m)
        positions.append(origin_m * 1.0e-3)
        velocities.append(origin_velocity_m_s * 1.0e-3)
    return np.vstack(positions), np.vstack(velocities)


def _raw_system_com_eci_km(
    data: mujoco.MjData,
    body_ids: tuple[int, ...],
    body_masses: np.ndarray,
) -> np.ndarray:
    positions_m = np.asarray(data.xipos[list(body_ids)], dtype=np.float64)
    total_mass = float(np.sum(body_masses))
    if total_mass <= 0.0:
        return np.zeros(3, dtype=np.float64)
    return np.average(positions_m, weights=body_masses, axis=0) * 1.0e-3


def _mujoco_integrator_value(name: str) -> int:
    normalized = name.strip().lower().replace("_", "").replace("-", "")
    if normalized == "euler":
        return int(mujoco.mjtIntegrator.mjINT_EULER)
    if normalized == "rk4":
        return int(mujoco.mjtIntegrator.mjINT_RK4)
    if normalized == "implicit":
        return int(mujoco.mjtIntegrator.mjINT_IMPLICIT)
    if normalized == "implicitfast":
        return int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    raise ValueError(f"unsupported MuJoCo integrator {name!r}")
