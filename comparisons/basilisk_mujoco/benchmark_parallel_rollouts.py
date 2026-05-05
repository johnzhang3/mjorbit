"""Benchmark mujoco_orbit batch rollouts against Basilisk independent runs."""

from __future__ import annotations

import argparse
import concurrent.futures
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from mujoco_orbit import MjoData, OrbitInit
from mujoco_orbit.rollout import mjo_get_state, rollout

from .cases import _earth_state_msg, _make_basilisk_integrator
from .common import (
    ASSET_DIR,
    GM_EARTH,
    basilisk_mujoco_import_status,
    compile_mjorbit_model,
    ensure_out_dir,
    make_circular_orbit,
    write_json,
)

SINGLE_BODY_XML = ASSET_DIR / "single_body_orbit.xml"


@dataclass(frozen=True)
class BenchmarkConfig:
    batch: int
    n_steps: int
    dt_s: float
    orbit_dt_s: float
    threads: tuple[int, ...]
    basilisk_processes: tuple[int, ...]
    basilisk_integrator: str
    basilisk_record: bool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--orbit-dt", type=float, default=0.1)
    parser.add_argument("--threads", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--basilisk-processes", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument(
        "--basilisk-integrator",
        choices=("euler", "rk2", "rk4", "rkf45", "rkf78"),
        default="euler",
    )
    parser.add_argument(
        "--basilisk-record",
        action="store_true",
        help="Record Basilisk COM state every step; disabled by default to favor throughput.",
    )
    args = parser.parse_args()

    config = BenchmarkConfig(
        batch=args.batch,
        n_steps=args.n_steps,
        dt_s=args.dt,
        orbit_dt_s=args.orbit_dt,
        threads=tuple(args.threads),
        basilisk_processes=tuple(args.basilisk_processes),
        basilisk_integrator=args.basilisk_integrator,
        basilisk_record=args.basilisk_record,
    )

    summary = run_benchmark(config)
    out_dir = ensure_out_dir()
    write_json(out_dir / "parallel_rollout_benchmark_summary.json", summary)
    _print_summary(summary)


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    ours = _benchmark_mujoco_orbit(config)
    basilisk_available, basilisk_message = basilisk_mujoco_import_status()
    if basilisk_available:
        basilisk = _benchmark_basilisk_process_pool(config)
    else:
        basilisk = {
            "available": False,
            "message": basilisk_message,
            "runs": [],
        }
    return {
        "case": "parallel_rollout_benchmark",
        "notes": [
            "mujoco_orbit uses one compiled model, batched state arrays, and per-thread MjoData.",
            "Basilisk uses independent MJScene simulations in a Python process pool.",
            "Basilisk timings include per-run SimulationBase/MJScene construction.",
            "Basilisk per-step COM recording is controlled by basilisk_record.",
        ],
        "config": {
            "batch": config.batch,
            "n_steps": config.n_steps,
            "dt_s": config.dt_s,
            "orbit_dt_s": config.orbit_dt_s,
            "threads": list(config.threads),
            "basilisk_processes": list(config.basilisk_processes),
            "basilisk_integrator": config.basilisk_integrator,
            "basilisk_record": config.basilisk_record,
        },
        "mujoco_orbit": ours,
        "basilisk": basilisk,
    }


def _benchmark_mujoco_orbit(config: BenchmarkConfig) -> dict[str, Any]:
    orbit = make_circular_orbit()
    model = compile_mjorbit_model(
        SINGLE_BODY_XML,
        plugin_body="spacecraft",
        mj_timestep=config.dt_s,
        orbit_dt=config.orbit_dt_s,
        use_gravity_gradient=False,
    )
    data = MjoData(model, orbit=OrbitInit(orbit.r_eci_km, orbit.v_eci_km_s))
    initial = mjo_get_state(model, data)
    initial_batch = np.tile(initial, (config.batch, 1))

    # Warm up native paths and thread worker construction outside measured runs.
    rollout(model, data, initial_batch[:1], nstep=min(config.n_steps, 2), nthread=1)

    runs = []
    for nthread in config.threads:
        nthread_eff = min(max(1, int(nthread)), config.batch)
        t0 = time.perf_counter()
        state, sensordata = rollout(
            model,
            data,
            initial_batch,
            nstep=config.n_steps,
            nthread=nthread_eff,
        )
        wall_s = time.perf_counter() - t0
        runs.append(
            {
                "threads": nthread_eff,
                "wall_s": wall_s,
                "state_shape": list(state.shape),
                "sensordata_shape": list(sensordata.shape),
                "sim_steps": config.batch * config.n_steps,
                "sim_steps_per_s": (config.batch * config.n_steps) / wall_s,
            }
        )
    return {
        "available": True,
        "backend": "mujoco_orbit.rollout",
        "runs": runs,
    }


def _benchmark_basilisk_process_pool(config: BenchmarkConfig) -> dict[str, Any]:
    runs = []
    xml_path = str(SINGLE_BODY_XML)
    for nproc in config.basilisk_processes:
        nproc_eff = min(max(1, int(nproc)), config.batch)
        t0 = time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(max_workers=nproc_eff) as pool:
            futures = [
                pool.submit(
                    _run_one_basilisk_single_body_for_benchmark,
                    config.n_steps,
                    config.dt_s,
                    xml_path,
                    config.basilisk_integrator,
                    config.basilisk_record,
                )
                for _ in range(config.batch)
            ]
            worker_times = [future.result() for future in futures]
        wall_s = time.perf_counter() - t0
        runs.append(
            {
                "processes": nproc_eff,
                "wall_s": wall_s,
                "worker_wall_s_mean": float(np.mean(worker_times)),
                "worker_wall_s_max": float(np.max(worker_times)),
                "sim_steps": config.batch * config.n_steps,
                "sim_steps_per_s": (config.batch * config.n_steps) / wall_s,
            }
        )
    return {
        "available": True,
        "backend": "Basilisk MJScene independent process pool",
        "integrator": config.basilisk_integrator,
        "recorded_each_step": config.basilisk_record,
        "runs": runs,
    }


def _run_one_basilisk_single_body_for_benchmark(
    n_steps: int,
    dt_s: float,
    xml_path: str,
    integrator_name: str,
    record: bool,
) -> float:
    t0 = time.perf_counter()

    from Basilisk.architecture import messaging
    from Basilisk.simulation import NBodyGravity, mujoco, pointMassGravityModel, svIntegrators
    from Basilisk.utilities import SimulationBaseClass

    orbit = make_circular_orbit()
    task_dt_ns = max(1, int(round(dt_s * 1.0e9)))
    task_name = "basilisk_mujoco_benchmark"

    sim = SimulationBaseClass.SimBaseClass()
    process = sim.CreateNewProcess("basilisk_mujoco_benchmark_process")
    process.addTask(sim.CreateNewTask(task_name, task_dt_ns))

    scene = mujoco.MJScene.fromFile(xml_path)
    scene.ModelTag = "mujocoScene"
    sim.AddModelToTask(task_name, scene)
    integrator = _make_basilisk_integrator(svIntegrators, scene, integrator_name)
    scene.setIntegrator(integrator)

    body = scene.getBody("spacecraft")
    gravity = NBodyGravity.NBodyGravity()
    gravity.ModelTag = "gravity"
    scene.AddModelToDynamicsTask(gravity)
    earth = pointMassGravityModel.PointMassGravityModel()
    earth.muBody = GM_EARTH * 1.0e9
    source = gravity.addGravitySource("earth", earth, isCentralBody=True)
    source.stateInMsg.subscribeTo(_earth_state_msg(messaging))
    gravity.addGravityTarget("spacecraft", body)

    if record:
        recorder = body.getCenterOfMass().stateOutMsg.recorder()
        sim.AddModelToTask(task_name, recorder)

    sim.InitializeSimulation()
    body.setPosition((orbit.r_eci_km * 1.0e3).tolist())
    body.setVelocity((orbit.v_eci_km_s * 1.0e3).tolist())
    body.setAttitude([0.0, 0.0, 0.0])
    body.setAttitudeRate([0.0, 0.0, 0.0])
    sim.ConfigureStopTime((n_steps + 1) * task_dt_ns)
    sim.ExecuteSimulation()

    return time.perf_counter() - t0


def _print_summary(summary: dict[str, Any]) -> None:
    print("=" * 72)
    print("Parallel rollout benchmark")
    print("=" * 72)
    config = summary["config"]
    print(
        f"Batch={config['batch']}, steps={config['n_steps']}, "
        f"dt={config['dt_s']:.6g} s"
    )
    print()
    print("mujoco_orbit:")
    for run in summary["mujoco_orbit"]["runs"]:
        print(
            f"  threads={run['threads']:>2}: {run['wall_s']:.6f} s, "
            f"{run['sim_steps_per_s']:.3e} sim-steps/s"
        )
    print()
    basilisk = summary["basilisk"]
    if not basilisk.get("available"):
        print(f"Basilisk: skipped ({basilisk['message']})")
        return
    print(
        "Basilisk process pool "
        f"(integrator={basilisk['integrator']}, record={basilisk['recorded_each_step']}):"
    )
    for run in basilisk["runs"]:
        print(
            f"  processes={run['processes']:>2}: {run['wall_s']:.6f} s, "
            f"{run['sim_steps_per_s']:.3e} sim-steps/s"
        )


if __name__ == "__main__":
    main()
