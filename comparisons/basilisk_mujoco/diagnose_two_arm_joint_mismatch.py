"""Diagnose passive two-arm joint mismatch between MuJoCo, mjorbit, and Basilisk."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from mjorbit import MjoData, OrbitInit, mjo_forward

from .cases import (
    TWO_ARM_BODY_NAMES,
    TWO_ARM_FREE_DRIFT_XML,
    _basilisk_mrp_to_quat_world_body,
    _earth_state_msg,
    _interp_body_vectors,
    _interp_columns,
    _interp_quat,
    _normalize_basilisk_integrator_name,
    _normalized_quat,
    _quat_angle_errors,
    _quat_world_body_to_basilisk_mrp,
    run_two_arm_free_drift_mjorbit,
)
from .common import (
    GM_EARTH,
    basilisk_mujoco_import_status,
    compile_mjorbit_model,
    ensure_out_dir,
    make_circular_orbit,
    sample_steps,
    write_json,
)


def main() -> None:
    if "--_basilisk-local-worker" in sys.argv:
        _basilisk_local_worker_main()
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--local-duration", type=float, default=20.0)
    parser.add_argument("--orbit-duration", type=float, default=600.0)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument(
        "--mj-integrators",
        nargs="+",
        choices=("Euler", "RK4"),
        default=("Euler", "RK4"),
        help="MuJoCo integrators to test on the mjorbit/plain-MuJoCo legs.",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Basilisk-MuJoCo diagnostic: passive two-arm joint mismatch")
    print("=" * 72)
    print("Basilisk integrator: rkf45")
    print(f"dt={args.dt:.6g} s")

    local_summary = _run_local_bridge_diagnostic(
        xml_path=TWO_ARM_FREE_DRIFT_XML,
        dt_s=args.dt,
        duration_s=args.local_duration,
        max_samples=args.max_samples,
        mj_integrators=tuple(args.mj_integrators),
    )
    _print_local_summary(local_summary)

    orbit_summaries = _run_mjorbit_integrator_diagnostic(
        duration_s=args.orbit_duration,
        dt_s=args.dt,
        max_samples=args.max_samples,
        mj_integrators=tuple(args.mj_integrators),
    )
    _print_orbit_summary(orbit_summaries)

    wrench_summary = _run_tidal_wrench_snapshot(dt_s=args.dt)
    _print_wrench_summary(wrench_summary)

    payload = {
        "case": "two_arm_joint_mismatch_diagnostics",
        "dt_s": args.dt,
        "local_duration_s": args.local_duration,
        "orbit_duration_s": args.orbit_duration,
        "basilisk_integrator": "rkf45",
        "local_bridge": local_summary,
        "mjorbit_integrator_sweep": orbit_summaries,
        "tidal_wrench_snapshot": wrench_summary,
    }
    write_json(ensure_out_dir() / "two_arm_joint_diagnostics_summary.json", payload)


def _run_local_bridge_diagnostic(
    *,
    xml_path: Path,
    dt_s: float,
    duration_s: float,
    max_samples: int,
    mj_integrators: tuple[str, ...],
) -> dict[str, Any]:
    initial_hinge_angles = np.array([0.35, -0.2], dtype=np.float64)
    initial_hinge_rates = np.array([0.01, -0.015], dtype=np.float64)
    initial_omega = np.array([0.005, -0.003, 0.007], dtype=np.float64)
    initial_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    basilisk = _run_basilisk_local_no_gravity(
        xml_path=xml_path,
        duration_s=duration_s,
        dt_s=dt_s,
        initial_quat_world_body=initial_quat,
        initial_omega_body_rad_s=initial_omega,
        initial_hinge_angles_rad=initial_hinge_angles,
        initial_hinge_rates_rad_s=initial_hinge_rates,
    )
    runs: list[dict[str, Any]] = []
    for mj_integrator in mj_integrators:
        plain = _run_plain_mujoco_local(
            xml_path=xml_path,
            duration_s=duration_s,
            dt_s=dt_s,
            mj_integrator=mj_integrator,
            max_samples=max_samples,
            initial_quat_world_body=initial_quat,
            initial_omega_body_rad_s=initial_omega,
            initial_hinge_angles_rad=initial_hinge_angles,
            initial_hinge_rates_rad_s=initial_hinge_rates,
        )
        summary = {
            "plain_mujoco_integrator": mj_integrator,
            "plain_mujoco": plain["summary"],
            "basilisk": basilisk["summary"],
        }
        if basilisk["summary"].get("ran"):
            summary.update(_compare_plain_mujoco_to_basilisk(plain, basilisk))
        runs.append(summary)

    return {
        "description": "plain MuJoCo local inertial model versus Basilisk MJScene, no gravity",
        "initial_hinge_angles_rad": initial_hinge_angles,
        "initial_hinge_rates_rad_s": initial_hinge_rates,
        "initial_omega_body_rad_s": initial_omega,
        "runs": runs,
    }


def _run_mjorbit_integrator_diagnostic(
    *,
    duration_s: float,
    dt_s: float,
    max_samples: int,
    mj_integrators: tuple[str, ...],
) -> list[dict[str, Any]]:
    summaries = []
    for mj_integrator in mj_integrators:
        result = run_two_arm_free_drift_mjorbit(
            duration_s=duration_s,
            dt_s=dt_s,
            orbit_dt=dt_s,
            mj_integrator=mj_integrator,
            max_samples=max_samples,
            basilisk_integrator="rkf45",
        )
        basilisk = result.summary["basilisk_direct"]
        summaries.append(
            {
                "mj_integrator": mj_integrator,
                "mjorbit": {
                    "final_hinge_1_rad": result.summary["final_hinge_1_rad"],
                    "final_hinge_2_rad": result.summary["final_hinge_2_rad"],
                    "max_abs_hinge_angle_rad": result.summary["max_abs_hinge_angle_rad"],
                    "max_abs_hinge_rate_rad_s": result.summary["max_abs_hinge_rate_rad_s"],
                    "chief_final_reference_position_error_m": result.summary[
                        "chief_final_reference_position_error_m"
                    ],
                    "system_com_world_drift_m": result.summary["system_com_world_drift_m"],
                },
                "basilisk": basilisk,
            }
        )
    return summaries


def _run_tidal_wrench_snapshot(*, dt_s: float) -> dict[str, Any]:
    orbit = make_circular_orbit()
    model = compile_mjorbit_model(
        TWO_ARM_FREE_DRIFT_XML,
        plugin_body="hub",
        mj_timestep=dt_s,
        orbit_dt=dt_s,
        use_gravity_gradient=False,
    )
    data = MjoData(model, orbit=OrbitInit(orbit.r_eci_km, orbit.v_eci_km_s))
    mjo_forward(model, data)

    chief_accel = _point_mass_accel_km_s2(data.orbit.R_eci)
    body_payloads = []
    force_errors = []
    for name in TWO_ARM_BODY_NAMES:
        body_id = model.body_id(name)
        mass = float(model.body_mass[body_id])
        r_body_eci = data.orbit.R_eci + np.asarray(data.xipos[body_id]) * 1.0e-3
        expected_force = mass * (_point_mass_accel_km_s2(r_body_eci) - chief_accel) * 1.0e3
        actual_force = np.asarray(data.xfrc_applied[body_id, 0:3], dtype=np.float64)
        force_error = actual_force - expected_force
        force_errors.append(np.linalg.norm(force_error))
        body_payloads.append(
            {
                "body": name,
                "mass_kg": mass,
                "expected_force_world_n": expected_force,
                "actual_force_world_n": actual_force,
                "force_error_norm_n": float(np.linalg.norm(force_error)),
            }
        )
    return {
        "description": (
            "mjorbit initial differential-gravity wrench versus "
            "point-mass analytic value"
        ),
        "max_force_error_norm_n": float(np.max(force_errors)),
        "bodies": body_payloads,
    }


def _run_plain_mujoco_local(
    *,
    xml_path: Path,
    duration_s: float,
    dt_s: float,
    mj_integrator: str,
    max_samples: int,
    initial_quat_world_body: np.ndarray,
    initial_omega_body_rad_s: np.ndarray,
    initial_hinge_angles_rad: np.ndarray,
    initial_hinge_rates_rad_s: np.ndarray,
) -> dict[str, Any]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = dt_s
    model.opt.integrator = _mujoco_integrator_value(mujoco, mj_integrator)
    data = mujoco.MjData(model)
    body_ids = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in TWO_ARM_BODY_NAMES
    )
    root_position_m = make_circular_orbit().r_eci_km * 1.0e3

    data.qpos[:] = model.qpos0
    data.qpos[0:3] = root_position_m
    data.qpos[3:7] = _normalized_quat(initial_quat_world_body)
    data.qpos[7:9] = initial_hinge_angles_rad
    data.qvel[:] = 0.0
    data.qvel[3:6] = initial_omega_body_rad_s
    data.qvel[6:8] = initial_hinge_rates_rad_s
    mujoco.mj_forward(model, data)

    n_steps = int(round(duration_s / dt_s))
    steps_to_sample = set(int(step) for step in sample_steps(n_steps, max_samples))
    times: list[float] = []
    qpos: list[np.ndarray] = []
    qvel: list[np.ndarray] = []
    body_origin_m: list[np.ndarray] = []

    def record() -> None:
        times.append(float(data.time))
        qpos.append(np.asarray(data.qpos, dtype=np.float64).copy())
        qvel.append(np.asarray(data.qvel, dtype=np.float64).copy())
        body_origin_m.append(np.asarray(data.xpos[list(body_ids)], dtype=np.float64).copy())

    record()
    for step in range(1, n_steps + 1):
        mujoco.mj_step(model, data)
        if step in steps_to_sample:
            record()

    times_s = np.asarray(times, dtype=np.float64)
    qpos_arr = np.vstack(qpos)
    qvel_arr = np.vstack(qvel)
    body_origin_arr = np.stack(body_origin_m, axis=0)
    return {
        "summary": {
            "ran": True,
            "mj_integrator": mj_integrator,
            "duration_s": n_steps * dt_s,
            "samples": int(times_s.size),
            "all_finite": bool(np.all(np.isfinite(qpos_arr)) and np.all(np.isfinite(qvel_arr))),
            "final_hinge_1_rad": float(qpos_arr[-1, 7]),
            "final_hinge_2_rad": float(qpos_arr[-1, 8]),
            "max_abs_hinge_rate_rad_s": float(np.max(np.abs(qvel_arr[:, 6:8]))),
        },
        "times_s": times_s,
        "hinge_angles_rad": qpos_arr[:, 7:9],
        "hinge_rates_rad_s": qvel_arr[:, 6:8],
        "hub_quat_world_body": qpos_arr[:, 3:7],
        "body_origin_m": body_origin_arr,
    }


def _run_basilisk_local_no_gravity(
    *,
    xml_path: Path,
    duration_s: float,
    dt_s: float,
    initial_quat_world_body: np.ndarray,
    initial_omega_body_rad_s: np.ndarray,
    initial_hinge_angles_rad: np.ndarray,
    initial_hinge_rates_rad_s: np.ndarray,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        summary_path = tmp_dir / "summary.json"
        samples_path = tmp_dir / "samples.npz"
        cmd = [
            sys.executable,
            "-m",
            "comparisons.basilisk_mujoco.diagnose_two_arm_joint_mismatch",
            "--_basilisk-local-worker",
            "--xml-path",
            str(xml_path),
            "--duration",
            f"{duration_s:.17g}",
            "--dt",
            f"{dt_s:.17g}",
            "--summary",
            str(summary_path),
            "--samples",
            str(samples_path),
            "--quat",
            *[f"{value:.17g}" for value in initial_quat_world_body],
            "--omega",
            *[f"{value:.17g}" for value in initial_omega_body_rad_s],
            "--hinge-angles",
            *[f"{value:.17g}" for value in initial_hinge_angles_rad],
            "--hinge-rates",
            *[f"{value:.17g}" for value in initial_hinge_rates_rad_s],
        ]
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(30.0, duration_s * 2.0 + 30.0),
        )
        if completed.returncode != 0 or not summary_path.exists() or not samples_path.exists():
            stderr = completed.stderr.strip()
            if len(stderr) > 1200:
                stderr = stderr[-1200:]
            return {
                "summary": {
                    "available": True,
                    "ran": False,
                    "integrator": "rkf45",
                    "message": (
                        "Basilisk local no-gravity subprocess failed "
                        f"(returncode={completed.returncode}): {stderr}"
                    ),
                }
            }
        summary = json.loads(summary_path.read_text())
        samples = np.load(samples_path)
        return {
            "summary": summary,
            "times_s": samples["times_s"],
            "hinge_angles_rad": samples["hinge_angles_rad"],
            "hinge_rates_rad_s": samples["hinge_rates_rad_s"],
            "hub_quat_world_body": samples["hub_quat_world_body"],
            "body_origin_m": samples["body_origin_m"],
        }


def _run_basilisk_local_no_gravity_in_process(
    *,
    xml_path: Path,
    duration_s: float,
    dt_s: float,
    initial_quat_world_body: np.ndarray,
    initial_omega_body_rad_s: np.ndarray,
    initial_hinge_angles_rad: np.ndarray,
    initial_hinge_rates_rad_s: np.ndarray,
) -> dict[str, Any]:
    available, message = basilisk_mujoco_import_status()
    integrator_label = _normalize_basilisk_integrator_name("rkf45")
    if not available:
        return {
            "summary": {
                "available": False,
                "ran": False,
                "integrator": integrator_label,
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
                "integrator": integrator_label,
                "message": f"{type(exc).__name__}: {exc}",
            }
        }

    try:  # pragma: no cover - depends on optional external Basilisk install
        n_steps = int(round(duration_s / dt_s))
        task_dt_ns = max(1, int(round(dt_s * 1.0e9)))
        task_name = "basilisk_mujoco_two_arm_local_no_gravity"
        sim = SimulationBaseClass.SimBaseClass()
        process = sim.CreateNewProcess("basilisk_mujoco_diagnostic")
        process.addTask(sim.CreateNewTask(task_name, task_dt_ns))

        scene = mujoco.MJScene.fromFile(str(xml_path))
        scene.ModelTag = "mujocoScene"
        sim.AddModelToTask(task_name, scene)
        scene.setIntegrator(svIntegrators.svIntegratorRKF45(scene))

        hub = scene.getBody("hub")
        arm_1 = scene.getBody("arm_1")
        arm_2 = scene.getBody("arm_2")
        bodies = (hub, arm_1, arm_2)
        hinge_1 = arm_1.getScalarJoint("hinge_1")
        hinge_2 = arm_2.getScalarJoint("hinge_2")

        gravity = NBodyGravity.NBodyGravity()
        gravity.ModelTag = "zeroGravity"
        scene.AddModelToDynamicsTask(gravity)
        earth = pointMassGravityModel.PointMassGravityModel()
        earth.muBody = 0.0
        source = gravity.addGravitySource("zero_mu_earth", earth, isCentralBody=True)
        source.stateInMsg.subscribeTo(_earth_state_msg(messaging))
        for name, body in zip(TWO_ARM_BODY_NAMES, bodies, strict=True):
            gravity.addGravityTarget(name, body)

        body_recorders = [body.getCenterOfMass().stateOutMsg.recorder() for body in bodies]
        hinge_1_recorder = hinge_1.stateOutMsg.recorder()
        hinge_2_recorder = hinge_2.stateOutMsg.recorder()
        hinge_1_rate_recorder = hinge_1.stateDotOutMsg.recorder()
        hinge_2_rate_recorder = hinge_2.stateDotOutMsg.recorder()
        for recorder in (
            *body_recorders,
            hinge_1_recorder,
            hinge_2_recorder,
            hinge_1_rate_recorder,
            hinge_2_rate_recorder,
        ):
            sim.AddModelToTask(task_name, recorder)

        sim.InitializeSimulation()
        hub.setPosition((make_circular_orbit().r_eci_km * 1.0e3).tolist())
        hub.setVelocity([0.0, 0.0, 0.0])
        hub.setAttitude(_quat_world_body_to_basilisk_mrp(initial_quat_world_body).tolist())
        hub.setAttitudeRate(np.asarray(initial_omega_body_rad_s, dtype=np.float64).tolist())
        hinge_1.setPosition(float(initial_hinge_angles_rad[0]))
        hinge_2.setPosition(float(initial_hinge_angles_rad[1]))
        hinge_1.setVelocity(float(initial_hinge_rates_rad_s[0]))
        hinge_2.setVelocity(float(initial_hinge_rates_rad_s[1]))

        sim.ConfigureStopTime((n_steps + 1) * task_dt_ns)
        sim.ExecuteSimulation()

        times_s = np.asarray(body_recorders[0].times(), dtype=np.float64) * 1.0e-9
        body_origin_m = np.stack(
            [np.asarray(recorder.r_BN_N, dtype=np.float64) for recorder in body_recorders],
            axis=1,
        )
        hub_quat = _basilisk_mrp_to_quat_world_body(
            np.asarray(body_recorders[0].sigma_BN, dtype=np.float64)
        )
        hinge_angles = np.column_stack(
            (
                np.asarray(hinge_1_recorder.state, dtype=np.float64).reshape(-1),
                np.asarray(hinge_2_recorder.state, dtype=np.float64).reshape(-1),
            )
        )
        hinge_rates = np.column_stack(
            (
                np.asarray(hinge_1_rate_recorder.state, dtype=np.float64).reshape(-1),
                np.asarray(hinge_2_rate_recorder.state, dtype=np.float64).reshape(-1),
            )
        )
        final_time_s = n_steps * task_dt_ns * 1.0e-9
        keep = times_s <= final_time_s + 1.0e-12
        times_s = times_s[keep]
        body_origin_m = body_origin_m[keep]
        hub_quat = hub_quat[keep]
        hinge_angles = hinge_angles[keep]
        hinge_rates = hinge_rates[keep]
        return {
            "summary": {
                "available": True,
                "ran": True,
                "integrator": integrator_label,
                "duration_s": n_steps * dt_s,
                "samples": int(times_s.size),
                "message": "Basilisk local no-gravity run completed",
                "all_finite": bool(
                    np.all(np.isfinite(hinge_angles))
                    and np.all(np.isfinite(hinge_rates))
                    and np.all(np.isfinite(body_origin_m))
                    and np.all(np.isfinite(hub_quat))
                ),
                "final_hinge_1_rad": float(hinge_angles[-1, 0]),
                "final_hinge_2_rad": float(hinge_angles[-1, 1]),
                "max_abs_hinge_rate_rad_s": float(np.max(np.abs(hinge_rates))),
            },
            "times_s": times_s,
            "hinge_angles_rad": hinge_angles,
            "hinge_rates_rad_s": hinge_rates,
            "hub_quat_world_body": hub_quat,
            "body_origin_m": body_origin_m,
        }
    except Exception as exc:  # pragma: no cover - depends on optional external install
        return {
            "summary": {
                "available": True,
                "ran": False,
                "integrator": integrator_label,
                "message": (
                    "Direct Basilisk local no-gravity run failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        }


def _basilisk_local_worker_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--_basilisk-local-worker", action="store_true")
    parser.add_argument("--xml-path", type=Path, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--quat", type=float, nargs=4, required=True)
    parser.add_argument("--omega", type=float, nargs=3, required=True)
    parser.add_argument("--hinge-angles", type=float, nargs=2, required=True)
    parser.add_argument("--hinge-rates", type=float, nargs=2, required=True)
    args = parser.parse_args()

    result = _run_basilisk_local_no_gravity_in_process(
        xml_path=args.xml_path,
        duration_s=args.duration,
        dt_s=args.dt,
        initial_quat_world_body=np.asarray(args.quat, dtype=np.float64),
        initial_omega_body_rad_s=np.asarray(args.omega, dtype=np.float64),
        initial_hinge_angles_rad=np.asarray(args.hinge_angles, dtype=np.float64),
        initial_hinge_rates_rad_s=np.asarray(args.hinge_rates, dtype=np.float64),
    )
    args.summary.write_text(json.dumps(result["summary"]) + "\n")
    if result["summary"].get("ran"):
        np.savez(
            args.samples,
            times_s=result["times_s"],
            hinge_angles_rad=result["hinge_angles_rad"],
            hinge_rates_rad_s=result["hinge_rates_rad_s"],
            hub_quat_world_body=result["hub_quat_world_body"],
            body_origin_m=result["body_origin_m"],
        )


def _compare_plain_mujoco_to_basilisk(
    plain: dict[str, Any],
    basilisk: dict[str, Any],
) -> dict[str, float]:
    times = plain["times_s"]
    basilisk_times = basilisk["times_s"]
    b_angles = _interp_columns(times, basilisk_times, basilisk["hinge_angles_rad"])
    b_rates = _interp_columns(times, basilisk_times, basilisk["hinge_rates_rad_s"])
    b_quat = _interp_quat(times, basilisk_times, basilisk["hub_quat_world_body"])
    b_body_origin = _interp_body_vectors(times, basilisk_times, basilisk["body_origin_m"])
    attitude_errors = _quat_angle_errors(plain["hub_quat_world_body"], b_quat)
    body_position_errors = np.linalg.norm(plain["body_origin_m"] - b_body_origin, axis=2)
    return {
        "plain_vs_basilisk_max_hinge_angle_error_rad": float(
            np.max(np.linalg.norm(plain["hinge_angles_rad"] - b_angles, axis=1))
        ),
        "plain_vs_basilisk_max_hinge_rate_error_rad_s": float(
            np.max(np.linalg.norm(plain["hinge_rates_rad_s"] - b_rates, axis=1))
        ),
        "plain_vs_basilisk_max_hub_attitude_error_rad": float(np.max(attitude_errors)),
        "plain_vs_basilisk_max_body_origin_position_error_m": float(
            np.max(body_position_errors)
        ),
    }


def _mujoco_integrator_value(mujoco_module: Any, name: str) -> int:
    normalized = name.strip().lower()
    if normalized == "euler":
        return int(mujoco_module.mjtIntegrator.mjINT_EULER)
    if normalized == "rk4":
        return int(mujoco_module.mjtIntegrator.mjINT_RK4)
    raise ValueError(f"unsupported local MuJoCo integrator {name!r}")


def _point_mass_accel_km_s2(r_eci_km: np.ndarray) -> np.ndarray:
    r = np.asarray(r_eci_km, dtype=np.float64)
    r_norm = float(np.linalg.norm(r))
    return -GM_EARTH * r / (r_norm**3)


def _print_local_summary(summary: dict[str, Any]) -> None:
    print()
    print("Local no-gravity bridge")
    for run in summary["runs"]:
        print(f"  plain MuJoCo integrator: {run['plain_mujoco_integrator']}")
        print(f"    Basilisk ran: {run['basilisk'].get('ran')}")
        if not run["basilisk"].get("ran"):
            print(f"    message: {run['basilisk'].get('message')}")
        else:
            print(
                "    max hinge angle error: "
                f"{run['plain_vs_basilisk_max_hinge_angle_error_rad']:.6e} rad"
            )
            print(
                "    max hinge rate error:  "
                f"{run['plain_vs_basilisk_max_hinge_rate_error_rad_s']:.6e} rad/s"
            )
            print(
                "    max hub attitude error: "
                f"{run['plain_vs_basilisk_max_hub_attitude_error_rad']:.6e} rad"
            )


def _print_orbit_summary(summaries: list[dict[str, Any]]) -> None:
    print()
    print("mjorbit integrator sweep against Basilisk RKF45")
    for item in summaries:
        basilisk = item["basilisk"]
        print(f"  mjorbit integrator: {item['mj_integrator']}")
        print(f"    Basilisk ran: {basilisk.get('ran')}")
        if basilisk.get("ran"):
            print(
                "    max hub position error: "
                f"{basilisk['ours_vs_basilisk_max_hub_position_error_m']:.6e} m"
            )
            print(
                "    max hub attitude error: "
                f"{basilisk['ours_vs_basilisk_max_hub_attitude_error_rad']:.6e} rad"
            )
            print(
                "    max hinge angle error: "
                f"{basilisk['ours_vs_basilisk_max_hinge_angle_error_rad']:.6e} rad"
            )


def _print_wrench_summary(summary: dict[str, Any]) -> None:
    print()
    print("mjorbit tidal-wrench snapshot")
    print(f"  max force error: {summary['max_force_error_norm_n']:.6e} N")


if __name__ == "__main__":
    main()
