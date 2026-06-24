"""Paper experiment: bimanual multibody validation against Basilisk.

The experiment has two pieces intended to replace the current paper throughput figure if the
numbers are favorable:

1. Accuracy: run the bimanual spacecraft from paper example (a), with joint limits removed,
   under the same smooth random arm-position commands in Basilisk, mjorbit CPU, and
   optionally mjorbit-warp. Basilisk is treated as the external reference.
2. Throughput: run the same model under random controls and compare Basilisk independent
   MJScene simulations launched through Python threads with mjorbit CPU rollout threads and
   the batched GPU backend.

Basilisk and mjorbit-warp are optional. Missing backends are recorded as skipped rather than
making the script fail, so this can live in the repo without making Basilisk a hard dependency.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Basilisk and the native MuJoCo bindings can collide if mjorbit's bindings load first.
# Preload Basilisk's MuJoCo module when available; missing Basilisk remains optional.
try:
    from Basilisk.simulation import mujoco as _basilisk_mujoco_preload  # noqa: F401
except Exception:
    _basilisk_mujoco_preload = None

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
OUT_DIR = EXPERIMENT_DIR / "out"
SPACECRAFT_BIMANUAL_PANELS_XML = REPO_ROOT / "src/mjorbit/testdata/spacecraft_bimanual_panels.xml"

BIMANUAL_POSE = np.array([-1.1, -0.9, 1.1, 0.9], dtype=np.float64)
BODY_ROOT_NAME = "bus"
ACTUATED_JOINTS = ("shoulder_a", "elbow_a", "shoulder_b", "elbow_b")
GM_EARTH = 398600.4418
R_EARTH = 6378.137


@dataclass(frozen=True)
class CircularOrbit:
    alt_km: float
    inc_rad: float
    radius_km: float
    mean_motion_rad_s: float
    period_s: float
    r_eci_km: np.ndarray
    v_eci_km_s: np.ndarray


@dataclass(frozen=True)
class ExperimentAssets:
    """Generated XML assets and metadata shared by all backends."""

    mjorbit_xml: Path
    basilisk_xml: Path
    body_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    joint_body_names: tuple[str, ...]
    actuator_names: tuple[str, ...]


@dataclass(frozen=True)
class AccuracyConfig:
    alt_km: float
    inc_deg: float
    duration_s: float
    dt_s: float
    orbit_dt_s: float
    mj_integrator: str
    basilisk_integrator: str
    basilisk_gravity_mode: str
    seed: int
    max_samples: int
    run_warp: bool


@dataclass(frozen=True)
class ThroughputConfig:
    alt_km: float
    dt_s: float
    orbit_dt_s: float
    n_steps: int
    batch: int
    seed: int
    cpu_threads: tuple[int, ...]
    basilisk_threads: tuple[int, ...]
    gpu_worlds: tuple[int, ...]
    run_warp: bool
    basilisk_integrator: str
    basilisk_gravity_mode: str


@dataclass
class Trajectory:
    """Sampled trajectory from one backend."""

    backend: str
    precision: str
    times_s: np.ndarray
    body_r_eci_km: np.ndarray
    body_v_eci_km_s: np.ndarray
    hub_quat_world_body: np.ndarray
    joint_angles_rad: np.ndarray
    joint_rates_rad_s: np.ndarray
    summary: dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alt-km", type=float, default=400.0)
    parser.add_argument("--inc-deg", type=float, default=51.6)
    parser.add_argument("--accuracy-duration", type=float, default=30.0)
    parser.add_argument("--accuracy-dt", type=float, default=0.02)
    parser.add_argument("--accuracy-orbit-dt", type=float, default=0.02)
    parser.add_argument("--accuracy-integrator", default="RK4")
    parser.add_argument("--basilisk-integrator", default="default")
    parser.add_argument(
        "--basilisk-gravity-mode",
        choices=("relative", "absolute"),
        default="relative",
        help=(
            "relative runs Basilisk MJScene in mjorbit's chief-centered inertial frame "
            "with matched differential gravity; absolute uses Basilisk NBodyGravity in ECI"
        ),
    )
    parser.add_argument("--max-samples", type=int, default=800)
    parser.add_argument("--throughput-steps", type=int, default=500)
    parser.add_argument("--throughput-batch", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--basilisk-threads", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--gpu-worlds", type=int, nargs="+", default=(1, 64, 256, 1024, 4096))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-warp", action="store_true", help="skip mjorbit-warp runs")
    parser.add_argument("--no-figure", action="store_true", help="skip matplotlib figure output")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assets = write_bimanual_assets(
        OUT_DIR,
        dt_s=args.accuracy_dt,
        orbit_dt_s=args.accuracy_orbit_dt,
        mj_integrator=args.accuracy_integrator,
    )
    accuracy_config = AccuracyConfig(
        alt_km=args.alt_km,
        inc_deg=args.inc_deg,
        duration_s=args.accuracy_duration,
        dt_s=args.accuracy_dt,
        orbit_dt_s=args.accuracy_orbit_dt,
        mj_integrator=args.accuracy_integrator,
        basilisk_integrator=args.basilisk_integrator,
        basilisk_gravity_mode=args.basilisk_gravity_mode,
        seed=args.seed,
        max_samples=args.max_samples,
        run_warp=not args.no_warp,
    )
    throughput_config = ThroughputConfig(
        alt_km=args.alt_km,
        dt_s=args.accuracy_dt,
        orbit_dt_s=args.accuracy_orbit_dt,
        n_steps=args.throughput_steps,
        batch=args.throughput_batch,
        seed=args.seed + 1000,
        cpu_threads=tuple(args.cpu_threads),
        basilisk_threads=tuple(args.basilisk_threads),
        gpu_worlds=tuple(args.gpu_worlds),
        run_warp=not args.no_warp,
        basilisk_integrator=args.basilisk_integrator,
        basilisk_gravity_mode=args.basilisk_gravity_mode,
    )

    accuracy = run_accuracy_experiment(assets, accuracy_config)
    throughput = run_throughput_experiment(assets, throughput_config)
    summary = {
        "case": "basilisk_multibody_bimanual",
        "notes": [
            "Accuracy uses the bimanual spacecraft model with joint limits removed.",
            "mjorbit CPU is double precision; mjorbit-warp uses the MJWarp device precision.",
            "Basilisk uses independent MJScene runs under Python threads for throughput.",
            "Basilisk accuracy defaults to the same chief-centered local inertial frame "
            "as mjorbit, with equivalent differential point-mass gravity.",
            "The mjorbit gravity-gradient rigid-body torque is disabled to match Basilisk "
            "point-mass gravity applied to each articulated body.",
        ],
        "assets": {
            "mjorbit_xml": str(assets.mjorbit_xml),
            "basilisk_xml": str(assets.basilisk_xml),
            "body_names": assets.body_names,
            "joint_names": assets.joint_names,
            "actuator_names": assets.actuator_names,
        },
        "accuracy": accuracy["summary"],
        "throughput": throughput,
    }
    write_json(OUT_DIR / "basilisk_multibody_summary.json", summary)
    if not args.no_figure:
        figure = maybe_write_figure(OUT_DIR, accuracy, throughput)
        if figure is not None:
            summary["figure"] = str(figure)
            write_json(OUT_DIR / "basilisk_multibody_summary.json", summary)
    _print_summary(summary)


def write_bimanual_assets(
    out_dir: Path,
    *,
    dt_s: float,
    orbit_dt_s: float | None = None,
    mj_integrator: str,
) -> ExperimentAssets:
    """Write no-limit bimanual XML files for mjorbit and Basilisk."""
    if orbit_dt_s is None:
        orbit_dt_s = dt_s
    source = Path(SPACECRAFT_BIMANUAL_PANELS_XML).read_text()
    basilisk_root = _configured_bimanual_root(
        source,
        dt_s=dt_s,
        orbit_dt_s=orbit_dt_s,
        mj_integrator=mj_integrator,
        include_mjorbit=False,
    )
    mjorbit_root = _configured_bimanual_root(
        source,
        dt_s=dt_s,
        orbit_dt_s=orbit_dt_s,
        mj_integrator=mj_integrator,
        include_mjorbit=True,
    )
    metadata = _metadata_from_root(basilisk_root)
    basilisk_xml = out_dir / "bimanual_unlimited_basilisk.xml"
    mjorbit_xml = out_dir / "bimanual_unlimited_mjorbit.xml"
    _write_basilisk_xml(basilisk_xml, basilisk_root)
    _write_xml(mjorbit_xml, mjorbit_root)
    return ExperimentAssets(
        mjorbit_xml=mjorbit_xml,
        basilisk_xml=basilisk_xml,
        body_names=metadata["body_names"],
        joint_names=metadata["joint_names"],
        joint_body_names=metadata["joint_body_names"],
        actuator_names=metadata["actuator_names"],
    )


def make_circular_orbit(alt_km: float = 400.0, inc_rad: float | None = None) -> CircularOrbit:
    if inc_rad is None:
        inc_rad = np.deg2rad(51.6)
    radius_km = R_EARTH + alt_km
    mean_motion = float(np.sqrt(GM_EARTH / radius_km**3))
    period_s = float(2.0 * np.pi / mean_motion)
    speed_km_s = float(np.sqrt(GM_EARTH / radius_km))
    r_eci_km = np.array([radius_km, 0.0, 0.0], dtype=np.float64)
    v_eci_km_s = np.array(
        [0.0, speed_km_s * np.cos(inc_rad), speed_km_s * np.sin(inc_rad)],
        dtype=np.float64,
    )
    return CircularOrbit(
        alt_km=alt_km,
        inc_rad=inc_rad,
        radius_km=radius_km,
        mean_motion_rad_s=mean_motion,
        period_s=period_s,
        r_eci_km=r_eci_km,
        v_eci_km_s=v_eci_km_s,
    )


def sample_steps(n_steps: int, max_samples: int) -> np.ndarray:
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if max_samples < 2:
        raise ValueError("max_samples must be at least 2")
    return np.unique(np.linspace(0, n_steps, min(n_steps + 1, max_samples), dtype=int))


def basilisk_mujoco_import_status() -> tuple[bool, str]:
    try:
        from Basilisk.simulation import mujoco as _mujoco  # noqa: F401
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "Basilisk.simulation.mujoco import succeeded"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_accuracy_experiment(
    assets: ExperimentAssets,
    config: AccuracyConfig,
) -> dict[str, Any]:
    """Run Basilisk/mjorbit trajectories and compare each mjorbit leg to Basilisk."""
    n_steps = int(round(config.duration_s / config.dt_s))
    times_ctrl = np.arange(n_steps + 2, dtype=np.float64) * config.dt_s
    controls = make_smooth_random_controls(
        times_ctrl,
        n_control=len(assets.actuator_names),
        seed=config.seed,
    )
    sample_idx = sample_steps(n_steps, config.max_samples)

    cpu = run_mjorbit_cpu_trajectory(assets, config, controls, sample_idx)
    trajectories: dict[str, Trajectory] = {"mjorbit_cpu": cpu}

    basilisk = run_basilisk_trajectory(assets, config, controls)
    trajectories["basilisk"] = basilisk

    warp = None
    if config.run_warp:
        warp = run_mjorbit_warp_trajectory(assets, config, controls, sample_idx)
        if warp is not None:
            trajectories["mjorbit_warp"] = warp

    comparisons: dict[str, Any] = {}
    if basilisk.summary.get("ran"):
        comparisons["mjorbit_cpu_vs_basilisk"] = compare_trajectories(cpu, basilisk)
        if warp is not None and warp.summary.get("ran"):
            comparisons["mjorbit_warp_vs_basilisk"] = compare_trajectories(warp, basilisk)
    elif warp is not None and warp.summary.get("ran"):
        comparisons["mjorbit_warp_vs_cpu"] = compare_trajectories(warp, cpu)

    return {
        "trajectories": trajectories,
        "comparisons": comparisons,
        "summary": {
            "config": {
                "alt_km": config.alt_km,
                "inc_deg": config.inc_deg,
                "duration_s": config.duration_s,
                "dt_s": config.dt_s,
                "orbit_dt_s": config.orbit_dt_s,
                "mjorbit_integrator": config.mj_integrator,
                "basilisk_integrator": config.basilisk_integrator,
                "basilisk_gravity_mode": config.basilisk_gravity_mode,
                "seed": config.seed,
                "samples": int(sample_idx.size),
            },
            "basilisk_available": bool(basilisk.summary.get("ran")),
            "backends": {name: traj.summary for name, traj in trajectories.items()},
            "comparisons": {
                name: _comparison_summary(value)
                for name, value in comparisons.items()
            },
        },
    }


def run_mjorbit_cpu_trajectory(
    assets: ExperimentAssets,
    config: AccuracyConfig,
    controls: np.ndarray,
    sample_idx: np.ndarray,
) -> Trajectory:
    from mjorbit import MjoModel, OrbitInit, mjo_forward, mjo_step

    orbit = make_circular_orbit(config.alt_km, inc_rad=np.deg2rad(config.inc_deg))
    model = MjoModel.from_xml_path(str(assets.mjorbit_xml), mj_timestep=config.dt_s)
    data = model.make_data(orbit=OrbitInit(orbit.r_eci_km, orbit.v_eci_km_s))
    _set_initial_state(model, data)
    mjo_forward(model, data)

    body_ids = tuple(model.body_id(name) for name in assets.body_names)
    sample_set = set(int(idx) for idx in sample_idx)
    samples = _TrajectoryBuilder("mjorbit_cpu", "float64", assets)
    samples.record_from_mjorbit(model, data, body_ids)
    n_steps = controls.shape[0] - 2
    for step in range(1, n_steps + 1):
        data.ctrl[:] = controls[step - 1]
        mjo_step(model, data)
        if step in sample_set:
            samples.record_from_mjorbit(model, data, body_ids)
    return samples.build(
        {
            "available": True,
            "ran": True,
            "precision": "float64",
            "samples": int(len(samples.times_s)),
            "n_steps": n_steps,
        }
    )


def run_mjorbit_warp_trajectory(
    assets: ExperimentAssets,
    config: AccuracyConfig,
    controls: np.ndarray,
    sample_idx: np.ndarray,
) -> Trajectory:
    try:
        import mjorbit_warp as mjo_warp
    except Exception as exc:
        return _skipped_trajectory(
            "mjorbit_warp",
            "float32",
            f"{type(exc).__name__}: {exc}",
        )

    try:
        orbit = make_circular_orbit(config.alt_km, inc_rad=np.deg2rad(config.inc_deg))
        model = mjo_warp.MjoModel.from_xml_path(
            str(assets.mjorbit_xml),
            mj_timestep=config.dt_s,
            orbit_dt=config.orbit_dt_s,
            use_j2=False,
            use_drag=False,
            use_srp=False,
            use_magnetic=False,
            use_gravity_gradient=False,
        )
        data = model.make_data(orbit=mjo_warp.OrbitInit(orbit.r_eci_km, orbit.v_eci_km_s))
        _set_initial_state(model, data)
        mjo_warp.mjo_upload(model, data, fields=("state", "inputs", "core"))
        mjo_warp.mjo_forward(model, data)
        mjo_warp.mjo_pull(model, data)

        body_ids = tuple(model.body_id(name) for name in assets.body_names)
        sample_set = set(int(idx) for idx in sample_idx)
        samples = _TrajectoryBuilder("mjorbit_warp", "float32", assets)
        samples.record_from_mjorbit(model, data, body_ids)
        n_steps = controls.shape[0] - 2
        for step in range(1, n_steps + 1):
            data.ctrl[:] = controls[step - 1].astype(np.float32)
            mjo_warp.mjo_upload(model, data, fields=("ctrl",))
            mjo_warp.mjo_step(model, data)
            if step in sample_set:
                mjo_warp.mjo_pull(model, data)
                samples.record_from_mjorbit(model, data, body_ids)
        return samples.build(
            {
                "available": True,
                "ran": True,
                "precision": "float32",
                "samples": int(len(samples.times_s)),
                "n_steps": n_steps,
            }
        )
    except Exception as exc:
        return _skipped_trajectory(
            "mjorbit_warp",
            "float32",
            f"{type(exc).__name__}: {exc}",
        )


def run_basilisk_trajectory(
    assets: ExperimentAssets,
    config: AccuracyConfig,
    controls: np.ndarray,
) -> Trajectory:
    available, message = basilisk_mujoco_import_status()
    if not available:
        return _skipped_trajectory("basilisk", "float64", message)
    try:
        return _run_basilisk_bimanual(
            assets,
            alt_km=config.alt_km,
            inc_deg=config.inc_deg,
            dt_s=config.dt_s,
            controls=controls,
            integrator_name=config.basilisk_integrator,
            gravity_mode=config.basilisk_gravity_mode,
            record=True,
        )
    except Exception as exc:
        return _skipped_trajectory(
            "basilisk",
            "float64",
            f"{type(exc).__name__}: {exc}",
        )


def run_throughput_experiment(
    assets: ExperimentAssets,
    config: ThroughputConfig,
) -> dict[str, Any]:
    controls = make_random_control_batch(
        config.n_steps,
        config.batch,
        len(assets.actuator_names),
        seed=config.seed,
    )
    result = {
        "config": {
            "alt_km": config.alt_km,
            "dt_s": config.dt_s,
            "orbit_dt_s": config.orbit_dt_s,
            "n_steps": config.n_steps,
            "batch": config.batch,
            "basilisk_gravity_mode": config.basilisk_gravity_mode,
            "seed": config.seed,
        },
        "mjorbit_cpu": benchmark_mjorbit_cpu(assets, config, controls),
        "basilisk_threads": benchmark_basilisk_threads(assets, config),
        "mjorbit_warp": (
            benchmark_mjorbit_warp(assets, config)
            if config.run_warp
            else {"available": False, "message": "skipped by --no-warp", "runs": []}
        ),
    }
    return result


def benchmark_mjorbit_cpu(
    assets: ExperimentAssets,
    config: ThroughputConfig,
    controls: np.ndarray,
) -> dict[str, Any]:
    from mjorbit import MjoModel, OrbitInit, mjo_forward
    from mjorbit.rollout import mjo_get_state, rollout

    orbit = make_circular_orbit(config.alt_km)
    model = MjoModel.from_xml_path(str(assets.mjorbit_xml), mj_timestep=config.dt_s)
    data = model.make_data(orbit=OrbitInit(orbit.r_eci_km, orbit.v_eci_km_s))
    _set_initial_state(model, data)
    mjo_forward(model, data)
    initial = np.tile(mjo_get_state(model, data)[None, :], (config.batch, 1))

    # Warm up native rollout and worker construction.
    rollout(
        model,
        data,
        initial[:1],
        controls[:1, : min(4, config.n_steps), :],
        nstep=min(4, config.n_steps),
        nthread=1,
    )

    runs = []
    for nthread in config.cpu_threads:
        nthread_eff = max(1, min(int(nthread), config.batch))
        t0 = time.perf_counter()
        rollout(
            model,
            data,
            initial,
            controls,
            nstep=config.n_steps,
            nthread=nthread_eff,
        )
        wall_s = time.perf_counter() - t0
        runs.append(
            {
                "threads": nthread_eff,
                "wall_s": wall_s,
                "sim_steps": config.batch * config.n_steps,
                "sim_steps_per_s": (config.batch * config.n_steps) / wall_s,
            }
        )
    return {"available": True, "backend": "mjorbit.rollout", "runs": runs}


def benchmark_mjorbit_warp(
    assets: ExperimentAssets,
    config: ThroughputConfig,
) -> dict[str, Any]:
    try:
        import warp as wp

        import mjorbit_warp as mjo_warp
    except Exception as exc:
        return {"available": False, "message": f"{type(exc).__name__}: {exc}", "runs": []}

    runs = []
    for nworld in config.gpu_worlds:
        nworld_eff = max(1, int(nworld))
        try:
            orbit = make_circular_orbit(config.alt_km)
            model = mjo_warp.MjoModel.from_xml_path(
                str(assets.mjorbit_xml),
                mj_timestep=config.dt_s,
                orbit_dt=config.orbit_dt_s,
                use_j2=False,
                use_drag=False,
                use_srp=False,
                use_magnetic=False,
                use_gravity_gradient=False,
            )
            data = model.make_data(
                orbit=mjo_warp.OrbitInit(orbit.r_eci_km, orbit.v_eci_km_s),
                nworld=nworld_eff,
            )
            _set_initial_state(model, data, nworld=nworld_eff)
            mjo_warp.mjo_upload(model, data, fields=("state", "inputs", "core"))
            mjo_warp.mjo_forward(model, data)
            controls = make_random_control_batch(
                config.n_steps,
                nworld_eff,
                len(assets.actuator_names),
                seed=config.seed + nworld_eff,
            ).astype(np.float32)

            warm = min(8, config.n_steps)
            for step in range(warm):
                data.ctrl[:] = controls[:, step, :]
                mjo_warp.mjo_upload(model, data, fields=("ctrl",))
                mjo_warp.mjo_step(model, data)
            wp.synchronize()

            t0 = time.perf_counter()
            for step in range(config.n_steps):
                data.ctrl[:] = controls[:, step, :]
                mjo_warp.mjo_upload(model, data, fields=("ctrl",))
                mjo_warp.mjo_step(model, data)
            wp.synchronize()
            wall_s = time.perf_counter() - t0
            runs.append(
                {
                    "worlds": nworld_eff,
                    "wall_s": wall_s,
                    "sim_steps": nworld_eff * config.n_steps,
                    "sim_steps_per_s": (nworld_eff * config.n_steps) / wall_s,
                    "device": str(wp.get_device()),
                }
            )
        except Exception as exc:
            runs.append(
                {
                    "worlds": nworld_eff,
                    "available": False,
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
    return {"available": True, "backend": "mjorbit_warp", "runs": runs}


def benchmark_basilisk_threads(
    assets: ExperimentAssets,
    config: ThroughputConfig,
) -> dict[str, Any]:
    available, message = basilisk_mujoco_import_status()
    if not available:
        return {"available": False, "message": message, "runs": []}

    runs = []
    for nthread in config.basilisk_threads:
        nthread_eff = max(1, min(int(nthread), config.batch))
        t0 = time.perf_counter()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=nthread_eff) as pool:
                futures = [
                    pool.submit(
                        _run_one_basilisk_for_throughput,
                        assets,
                        config.alt_km,
                        config.dt_s,
                        config.n_steps,
                        config.basilisk_integrator,
                        config.basilisk_gravity_mode,
                        config.seed + world,
                    )
                    for world in range(config.batch)
                ]
                worker_times = [future.result() for future in futures]
            wall_s = time.perf_counter() - t0
            runs.append(
                {
                    "threads": nthread_eff,
                    "wall_s": wall_s,
                    "worker_wall_s_mean": float(np.mean(worker_times)),
                    "worker_wall_s_max": float(np.max(worker_times)),
                    "sim_steps": config.batch * config.n_steps,
                    "sim_steps_per_s": (config.batch * config.n_steps) / wall_s,
                }
            )
        except Exception as exc:
            runs.append(
                {
                    "threads": nthread_eff,
                    "available": False,
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "available": True,
        "backend": "Basilisk MJScene independent Python threads",
        "integrator": config.basilisk_integrator,
        "gravity_mode": config.basilisk_gravity_mode,
        "runs": runs,
    }


def _run_one_basilisk_for_throughput(
    assets: ExperimentAssets,
    alt_km: float,
    dt_s: float,
    n_steps: int,
    integrator_name: str,
    gravity_mode: str,
    seed: int,
) -> float:
    controls = make_random_control_profile(n_steps + 2, len(assets.actuator_names), seed=seed)
    t0 = time.perf_counter()
    _run_basilisk_bimanual(
        assets,
        alt_km=alt_km,
        inc_deg=51.6,
        dt_s=dt_s,
        controls=controls,
        integrator_name=integrator_name,
        gravity_mode=gravity_mode,
        record=False,
    )
    return time.perf_counter() - t0


def _write_basilisk_actuator_commands(
    messaging: Any,
    actuator_messages: Sequence[Any],
    controls: np.ndarray,
) -> None:
    for actuator_msg, value in zip(actuator_messages, controls, strict=True):
        actuator_msg.write(messaging.SingleActuatorMsgPayload(input=float(value)))


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
        "default": "default",
        "native": "default",
        "none": "default",
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


def _normalize_basilisk_gravity_mode(name: str) -> str:
    normalized = name.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "relative": "relative",
        "local": "relative",
        "differential": "relative",
        "encke": "relative",
        "absolute": "absolute",
        "eci": "absolute",
        "nbody": "absolute",
        "full": "absolute",
    }
    if normalized not in aliases:
        supported = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"unsupported Basilisk gravity mode {name!r}; choose one of {supported}")
    return aliases[normalized]


def _circular_orbit_state_at_times(
    times_s: np.ndarray,
    orbit: CircularOrbit,
) -> tuple[np.ndarray, np.ndarray]:
    phase = orbit.mean_motion_rad_s * np.asarray(times_s, dtype=np.float64)
    c = np.cos(phase)
    s = np.sin(phase)
    speed = orbit.mean_motion_rad_s * orbit.radius_km
    r_eci_km = np.column_stack(
        (
            orbit.radius_km * c,
            orbit.radius_km * s * np.cos(orbit.inc_rad),
            orbit.radius_km * s * np.sin(orbit.inc_rad),
        )
    )
    v_eci_km_s = np.column_stack(
        (
            -speed * s,
            speed * c * np.cos(orbit.inc_rad),
            speed * c * np.sin(orbit.inc_rad),
        )
    )
    return r_eci_km, v_eci_km_s


def _basilisk_body_masses(xml_path: Path, body_names: Sequence[str]) -> np.ndarray:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    masses = []
    for name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"unknown MuJoCo body {name!r}")
        masses.append(float(model.body_mass[body_id]))
    return np.asarray(masses, dtype=np.float64)


def _make_relative_gravity_model(
    *,
    sys_model: Any,
    messaging: Any,
    rbk: Any,
    body_names: Sequence[str],
    body_masses: np.ndarray,
    orbit: CircularOrbit,
) -> Any:
    class RelativePointGravity(sys_model.SysModel):
        def __init__(self) -> None:
            super().__init__()
            self.state_readers = [messaging.SCStatesMsgReader() for _ in body_names]
            self.force_out_msgs = [messaging.ForceAtSiteMsg() for _ in body_names]

        def UpdateState(self, CurrentSimNanos: int) -> None:  # noqa: N802
            t_s = float(CurrentSimNanos) * 1.0e-9
            r_ref_km, _ = _circular_orbit_state_at_times(np.asarray([t_s]), orbit)
            r_ref_m = r_ref_km[0] * 1.0e3
            ref_norm = float(np.linalg.norm(r_ref_m))
            g_ref = -GM_EARTH * 1.0e9 * r_ref_m / ref_norm**3
            for mass, reader, out_msg in zip(
                body_masses,
                self.state_readers,
                self.force_out_msgs,
                strict=True,
            ):
                state = reader()
                rho_m = np.asarray(state.r_BN_N, dtype=np.float64)
                r_body_m = r_ref_m + rho_m
                body_norm = float(np.linalg.norm(r_body_m))
                g_body = -GM_EARTH * 1.0e9 * r_body_m / body_norm**3
                force_world = mass * (g_body - g_ref)
                dcm_site_world = np.asarray(rbk.MRP2C(state.sigma_BN), dtype=np.float64)
                payload = messaging.ForceAtSiteMsgPayload(
                    force_S=(dcm_site_world @ force_world).tolist()
                )
                out_msg.write(payload, CurrentSimNanos, self.moduleID)

    return RelativePointGravity()


def _run_basilisk_bimanual(
    assets: ExperimentAssets,
    *,
    alt_km: float,
    inc_deg: float,
    dt_s: float,
    controls: np.ndarray,
    integrator_name: str,
    gravity_mode: str,
    record: bool,
) -> Trajectory:
    from Basilisk.architecture import messaging, sysModel
    from Basilisk.simulation import NBodyGravity, pointMassGravityModel, svIntegrators
    from Basilisk.simulation import mujoco as bsk_mujoco
    from Basilisk.utilities import RigidBodyKinematics as rbk
    from Basilisk.utilities import SimulationBaseClass

    n_steps = controls.shape[0] - 2
    orbit = make_circular_orbit(alt_km=alt_km, inc_rad=np.deg2rad(inc_deg))
    gravity_mode_normalized = _normalize_basilisk_gravity_mode(gravity_mode)
    task_dt_ns = max(1, int(round(dt_s * 1.0e9)))
    task_name = "basilisk_bimanual_multibody"

    sim = SimulationBaseClass.SimBaseClass()
    process = sim.CreateNewProcess("basilisk_bimanual_multibody_process")
    process.addTask(sim.CreateNewTask(task_name, task_dt_ns))

    scene = bsk_mujoco.MJScene.fromFile(str(assets.basilisk_xml))
    scene.ModelTag = "mujocoScene"
    sim.AddModelToTask(task_name, scene)
    if _normalize_basilisk_integrator_name(integrator_name) != "default":
        scene.setIntegrator(_make_basilisk_integrator(svIntegrators, scene, integrator_name))

    body_objects = [scene.getBody(name) for name in assets.body_names]
    root = scene.getBody(BODY_ROOT_NAME)
    joints = [
        scene.getBody(body_name).getScalarJoint(joint_name)
        for body_name, joint_name in zip(assets.joint_body_names, assets.joint_names, strict=True)
    ]

    actuator_messages = []
    for actuator_name in assets.actuator_names:
        actuator = scene.getSingleActuator(actuator_name)
        actuator_msg = messaging.SingleActuatorMsg()
        actuator.actuatorInMsg.subscribeTo(actuator_msg)
        actuator_messages.append(actuator_msg)

    gravity_models = []
    if gravity_mode_normalized == "absolute":
        gravity = NBodyGravity.NBodyGravity()
        gravity.ModelTag = "gravity"
        scene.AddModelToDynamicsTask(gravity)
        earth = pointMassGravityModel.PointMassGravityModel()
        earth.muBody = GM_EARTH * 1.0e9
        source = gravity.addGravitySource("earth", earth, isCentralBody=True)
        source.stateInMsg.subscribeTo(_earth_state_msg(messaging))
        for name, body in zip(assets.body_names, body_objects, strict=True):
            gravity.addGravityTarget(name, body)
        gravity_models.append(gravity)
    else:
        relative_gravity = _make_relative_gravity_model(
            sys_model=sysModel,
            messaging=messaging,
            rbk=rbk,
            body_names=assets.body_names,
            body_masses=_basilisk_body_masses(assets.basilisk_xml, assets.body_names),
            orbit=orbit,
        )
        relative_gravity.ModelTag = "relative_point_gravity"
        for idx, (name, body) in enumerate(zip(assets.body_names, body_objects, strict=True)):
            site = body.getCenterOfMass()
            relative_gravity.state_readers[idx].subscribeTo(site.stateOutMsg)
            actuator = scene.addForceActuator(f"relative_gravity_{name}", site)
            actuator.forceInMsg.subscribeTo(relative_gravity.force_out_msgs[idx])
        scene.AddModelToDynamicsTask(relative_gravity)
        gravity_models.append(relative_gravity)

    body_recorders = []
    body_com_recorders = []
    joint_recorders = []
    joint_rate_recorders = []
    if record:
        body_recorders = [body.getOrigin().stateOutMsg.recorder() for body in body_objects]
        body_com_recorders = [
            body.getCenterOfMass().stateOutMsg.recorder() for body in body_objects
        ]
        joint_recorders = [joint.stateOutMsg.recorder() for joint in joints]
        joint_rate_recorders = [joint.stateDotOutMsg.recorder() for joint in joints]
        for recorder in [
            *body_recorders,
            *body_com_recorders,
            *joint_recorders,
            *joint_rate_recorders,
        ]:
            sim.AddModelToTask(task_name, recorder)

    sim.InitializeSimulation()
    if gravity_mode_normalized == "absolute":
        root.setPosition((orbit.r_eci_km * 1.0e3).tolist())
        root.setVelocity((orbit.v_eci_km_s * 1.0e3).tolist())
    else:
        root.setPosition([0.0, 0.0, 0.0])
        root.setVelocity([0.0, 0.0, 0.0])
    root.setAttitude(_quat_world_body_to_basilisk_mrp(_initial_quat()).tolist())
    root.setAttitudeRate(_initial_omega().tolist())
    for joint, value in zip(joints, _initial_joint_positions(len(joints)), strict=True):
        joint.setPosition(float(value))
        joint.setVelocity(0.0)

    for step in range(n_steps):
        _write_basilisk_actuator_commands(messaging, actuator_messages, controls[step])
        sim.ConfigureStopTime((step + 1) * task_dt_ns)
        sim.ExecuteSimulation()

    if not record:
        return Trajectory(
            backend="basilisk",
            precision="float64",
            times_s=np.zeros(0),
            body_r_eci_km=np.zeros((0, len(assets.body_names), 3)),
            body_v_eci_km_s=np.zeros((0, len(assets.body_names), 3)),
            hub_quat_world_body=np.zeros((0, 4)),
            joint_angles_rad=np.zeros((0, len(assets.joint_names))),
            joint_rates_rad_s=np.zeros((0, len(assets.joint_names))),
            summary={
                "available": True,
                "ran": True,
                "record": False,
                "integrator": integrator_name,
                "gravity_mode": gravity_mode_normalized,
            },
        )

    times_s_full = np.asarray(body_recorders[0].times(), dtype=np.float64) * 1.0e-9
    final_time_s = n_steps * task_dt_ns * 1.0e-9
    keep = times_s_full <= final_time_s + 1.0e-12
    times_s = times_s_full[keep]
    body_origin_r_m = np.stack(
        [np.asarray(rec.r_BN_N, dtype=np.float64) for rec in body_recorders],
        axis=1,
    )[keep]
    body_com_r_m = np.stack(
        [np.asarray(rec.r_BN_N, dtype=np.float64) for rec in body_com_recorders],
        axis=1,
    )[keep]
    body_com_v_m_s = np.stack(
        [np.asarray(rec.v_BN_N, dtype=np.float64) for rec in body_com_recorders],
        axis=1,
    )[keep]
    body_omega_body_rad_s = np.stack(
        [np.asarray(rec.omega_BN_B, dtype=np.float64) for rec in body_com_recorders],
        axis=1,
    )[keep]
    body_sigma = np.stack(
        [np.asarray(rec.sigma_BN, dtype=np.float64) for rec in body_com_recorders],
        axis=1,
    )[keep]
    body_dcm_body_world = np.empty((*body_sigma.shape[:2], 3, 3), dtype=np.float64)
    for sample_idx in range(body_sigma.shape[0]):
        for body_idx in range(body_sigma.shape[1]):
            body_dcm_body_world[sample_idx, body_idx] = rbk.MRP2C(body_sigma[sample_idx, body_idx])
    body_omega_world_rad_s = np.einsum(
        "tbij,tbi->tbj",
        body_dcm_body_world,
        body_omega_body_rad_s,
    )
    body_v_m_s = body_com_v_m_s - np.cross(
        body_omega_world_rad_s,
        body_com_r_m - body_origin_r_m,
    )
    if gravity_mode_normalized == "absolute":
        body_r_eci_km = body_origin_r_m * 1.0e-3
        body_v_eci_km_s = body_v_m_s * 1.0e-3
    else:
        r_ref_km, v_ref_km_s = _circular_orbit_state_at_times(times_s, orbit)
        body_r_eci_km = r_ref_km[:, None, :] + body_origin_r_m * 1.0e-3
        body_v_eci_km_s = v_ref_km_s[:, None, :] + body_v_m_s * 1.0e-3
    hub_quat = _basilisk_mrp_to_quat_world_body(
        np.asarray(body_recorders[0].sigma_BN, dtype=np.float64)
    )[keep]
    joint_angles = np.column_stack(
        [_squeeze_state_column(rec.state) for rec in joint_recorders]
    )[keep]
    joint_rates = np.column_stack(
        [_squeeze_state_column(rec.state) for rec in joint_rate_recorders]
    )[keep]
    return Trajectory(
        backend="basilisk",
        precision="float64",
        times_s=times_s,
        body_r_eci_km=body_r_eci_km,
        body_v_eci_km_s=body_v_eci_km_s,
        hub_quat_world_body=hub_quat,
        joint_angles_rad=joint_angles,
        joint_rates_rad_s=joint_rates,
        summary={
            "available": True,
            "ran": True,
            "precision": "float64",
            "samples": int(times_s.size),
            "n_steps": n_steps,
            "integrator": integrator_name,
            "gravity_mode": gravity_mode_normalized,
        },
    )


class _TrajectoryBuilder:
    def __init__(self, backend: str, precision: str, assets: ExperimentAssets) -> None:
        self.backend = backend
        self.precision = precision
        self.assets = assets
        self._measurement_mujoco = None
        self._measurement_model = None
        self._measurement_data = None
        self._measurement_body_ids = None
        try:
            import mujoco

            measurement_model = mujoco.MjModel.from_xml_path(str(assets.basilisk_xml))
            self._measurement_mujoco = mujoco
            self._measurement_model = measurement_model
            self._measurement_data = mujoco.MjData(measurement_model)
            self._measurement_body_ids = tuple(
                mujoco.mj_name2id(measurement_model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in assets.body_names
            )
        except Exception:
            # Fall back to the wrapped arrays if the standalone MuJoCo binding is unavailable.
            pass
        self.times_s: list[float] = []
        self.body_r: list[np.ndarray] = []
        self.body_v: list[np.ndarray] = []
        self.hub_quat: list[np.ndarray] = []
        self.joint_angles: list[np.ndarray] = []
        self.joint_rates: list[np.ndarray] = []

    def record_from_mjorbit(self, model: Any, data: Any, body_ids: Sequence[int]) -> None:
        body_r, body_v = _mjorbit_body_origin_eci_states(
            data,
            body_ids,
            mujoco_module=self._measurement_mujoco,
            measurement_model=self._measurement_model,
            measurement_data=self._measurement_data,
            measurement_body_ids=self._measurement_body_ids,
        )
        self.times_s.append(float(np.asarray(data.orbit.t).reshape(-1)[0]))
        self.body_r.append(body_r)
        self.body_v.append(body_v)
        self.hub_quat.append(
            np.asarray(data.qpos[..., 3:7], dtype=np.float64).reshape(-1, 4)[0].copy()
        )
        joint_angles = np.asarray(data.qpos[..., 7:], dtype=np.float64)
        joint_rates = np.asarray(data.qvel[..., 6:], dtype=np.float64)
        self.joint_angles.append(joint_angles.reshape(-1, model.nq - 7)[0].copy())
        self.joint_rates.append(joint_rates.reshape(-1, model.nv - 6)[0].copy())

    def build(self, summary: dict[str, Any]) -> Trajectory:
        return Trajectory(
            backend=self.backend,
            precision=self.precision,
            times_s=np.asarray(self.times_s, dtype=np.float64),
            body_r_eci_km=np.stack(self.body_r, axis=0),
            body_v_eci_km_s=np.stack(self.body_v, axis=0),
            hub_quat_world_body=np.vstack(self.hub_quat),
            joint_angles_rad=np.vstack(self.joint_angles),
            joint_rates_rad_s=np.vstack(self.joint_rates),
            summary=summary,
        )


def compare_trajectories(candidate: Trajectory, reference: Trajectory) -> dict[str, np.ndarray]:
    ref_body_r = _interp_body_vectors(candidate.times_s, reference.times_s, reference.body_r_eci_km)
    ref_body_v = _interp_body_vectors(
        candidate.times_s,
        reference.times_s,
        reference.body_v_eci_km_s,
    )
    ref_joints = _interp_columns(candidate.times_s, reference.times_s, reference.joint_angles_rad)
    ref_rates = _interp_columns(candidate.times_s, reference.times_s, reference.joint_rates_rad_s)
    ref_quat = _interp_quat(
        candidate.times_s,
        reference.times_s,
        reference.hub_quat_world_body,
    )
    body_position_error_m = np.max(
        np.linalg.norm(candidate.body_r_eci_km - ref_body_r, axis=2) * 1.0e3,
        axis=1,
    )
    body_velocity_error_m_s = np.max(
        np.linalg.norm(candidate.body_v_eci_km_s - ref_body_v, axis=2) * 1.0e3,
        axis=1,
    )
    candidate_rel_r = candidate.body_r_eci_km - candidate.body_r_eci_km[:, :1, :]
    reference_rel_r = ref_body_r - ref_body_r[:, :1, :]
    body_relative_position_error_m = np.max(
        np.linalg.norm(candidate_rel_r - reference_rel_r, axis=2) * 1.0e3,
        axis=1,
    )
    candidate_rel_v = candidate.body_v_eci_km_s - candidate.body_v_eci_km_s[:, :1, :]
    reference_rel_v = ref_body_v - ref_body_v[:, :1, :]
    body_relative_velocity_error_m_s = np.max(
        np.linalg.norm(candidate_rel_v - reference_rel_v, axis=2) * 1.0e3,
        axis=1,
    )
    joint_angle_error_rad = np.max(np.abs(candidate.joint_angles_rad - ref_joints), axis=1)
    joint_rate_error_rad_s = np.max(np.abs(candidate.joint_rates_rad_s - ref_rates), axis=1)
    attitude_error_rad = _quat_angle_errors(candidate.hub_quat_world_body, ref_quat)
    return {
        "times_s": candidate.times_s,
        "body_position_error_m": body_position_error_m,
        "body_velocity_error_m_s": body_velocity_error_m_s,
        "body_relative_position_error_m": body_relative_position_error_m,
        "body_relative_velocity_error_m_s": body_relative_velocity_error_m_s,
        "joint_angle_error_rad": joint_angle_error_rad,
        "joint_rate_error_rad_s": joint_rate_error_rad_s,
        "hub_attitude_error_rad": attitude_error_rad,
    }


def _comparison_summary(value: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "max_body_position_error_m": float(np.max(value["body_position_error_m"])),
        "max_body_velocity_error_m_s": float(np.max(value["body_velocity_error_m_s"])),
        "max_body_relative_position_error_m": float(
            np.max(value["body_relative_position_error_m"])
        ),
        "max_body_relative_velocity_error_m_s": float(
            np.max(value["body_relative_velocity_error_m_s"])
        ),
        "max_joint_angle_error_rad": float(np.max(value["joint_angle_error_rad"])),
        "max_joint_rate_error_rad_s": float(np.max(value["joint_rate_error_rad_s"])),
        "max_hub_attitude_error_rad": float(np.max(value["hub_attitude_error_rad"])),
    }


def make_smooth_random_controls(
    times_s: np.ndarray,
    *,
    n_control: int,
    seed: int,
) -> np.ndarray:
    """Smooth random position targets used by the accuracy comparison."""
    rng = np.random.default_rng(seed)
    base = _control_base(n_control)
    controls = np.tile(base, (times_s.size, 1))
    for _ in range(3):
        periods = rng.uniform(4.0, 18.0, size=n_control)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=n_control)
        signs = rng.choice([-1.0, 1.0], size=n_control)
        amplitudes = signs * rng.uniform(0.04, 0.22, size=n_control)
        controls += amplitudes * np.sin(
            2.0 * np.pi * times_s[:, None] / periods[None, :] + phases[None, :]
        )
    return controls.astype(np.float64)


def make_random_control_batch(
    n_steps: int,
    batch: int,
    n_control: int,
    *,
    seed: int,
) -> np.ndarray:
    """Random but bounded controls for throughput runs, shaped for mjorbit rollout."""
    rng = np.random.default_rng(seed)
    base = _control_base(n_control)[None, None, :]
    controls = base + rng.uniform(-0.35, 0.35, size=(batch, n_steps, n_control))
    return controls.astype(np.float64)


def make_random_control_profile(
    n_samples: int,
    n_control: int,
    *,
    seed: int,
) -> np.ndarray:
    """Random bounded controls for one independent simulator run."""
    rng = np.random.default_rng(seed)
    base = _control_base(n_control)[None, :]
    controls = base + rng.uniform(-0.35, 0.35, size=(n_samples, n_control))
    return controls.astype(np.float64)


def _control_base(n_control: int) -> np.ndarray:
    base = np.zeros(n_control, dtype=np.float64)
    base[: min(n_control, BIMANUAL_POSE.size)] = BIMANUAL_POSE[: min(n_control, 4)]
    return base


def _configured_bimanual_root(
    xml_text: str,
    *,
    dt_s: float,
    orbit_dt_s: float,
    mj_integrator: str,
    include_mjorbit: bool,
) -> ET.Element:
    root = ET.fromstring(xml_text)
    for child in list(root):
        if child.tag == "mjorbit":
            root.remove(child)
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)
    option.set("timestep", f"{dt_s:.17g}")
    option.set("gravity", "0 0 0")
    option.set("integrator", _normalize_mujoco_integrator(mj_integrator))
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("contact", "disable")

    for joint in root.iter("joint"):
        joint.set("limited", "false")
        joint.attrib.pop("range", None)
    for actuator in root.iter("position"):
        actuator.set("ctrllimited", "false")
        actuator.attrib.pop("ctrlrange", None)

    if include_mjorbit:
        root.append(
            ET.Element(
                "mjorbit",
                {
                    "plugin_body": BODY_ROOT_NAME,
                    "orbit_dt": f"{orbit_dt_s:.17g}",
                    "use_j2": "false",
                    "use_drag": "false",
                    "use_srp": "false",
                    "use_magnetic": "false",
                    "use_gravity_gradient": "false",
                },
            )
        )
    return root


def _metadata_from_root(root: ET.Element) -> dict[str, tuple[str, ...]]:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("bimanual XML has no worldbody")

    body_names: list[str] = []
    joint_names: list[str] = []
    joint_body_names: list[str] = []

    def visit_body(body: ET.Element) -> None:
        name = body.get("name")
        if name:
            body_names.append(name)
        for joint in body.findall("joint"):
            joint_name = joint.get("name")
            if joint_name and name:
                joint_names.append(joint_name)
                joint_body_names.append(name)
        for child in body.findall("body"):
            visit_body(child)

    for body in worldbody.findall("body"):
        visit_body(body)

    actuator_names = []
    actuator_root = root.find("actuator")
    if actuator_root is not None:
        for actuator in actuator_root.findall("position"):
            name = actuator.get("name") or actuator.get("joint")
            if name:
                actuator_names.append(name)

    return {
        "body_names": tuple(body_names),
        "joint_names": tuple(joint_names),
        "joint_body_names": tuple(joint_body_names),
        "actuator_names": tuple(actuator_names),
    }


def _write_xml(path: Path, root: ET.Element) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="unicode", xml_declaration=False)


def _write_basilisk_xml(path: Path, root: ET.Element) -> None:
    _write_xml(path, root)
    try:
        import mujoco
    except Exception:
        return
    model = mujoco.MjModel.from_xml_path(str(path))
    mujoco.mj_saveLastXML(str(path), model)


def _normalize_mujoco_integrator(name: str) -> str:
    normalized = name.strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "euler": "Euler",
        "rk4": "RK4",
        "implicit": "implicit",
        "implicitfast": "implicitfast",
    }
    if normalized not in aliases:
        supported = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"unsupported MuJoCo integrator {name!r}; choose one of {supported}")
    return aliases[normalized]


def _set_initial_state(model: Any, data: Any, *, nworld: int = 1) -> None:
    quat = _initial_quat()
    joints = _initial_joint_positions(int(model.nq) - 7)
    qpos = np.zeros((nworld, int(model.nq)), dtype=np.float64)
    qvel = np.zeros((nworld, int(model.nv)), dtype=np.float64)
    qpos[:, 3:7] = quat
    qpos[:, 7:] = joints
    qvel[:, 3:6] = _initial_omega()
    if nworld == 1:
        data.qpos[:] = qpos[0]
        data.qvel[:] = qvel[0]
        data.ctrl[:] = _control_base(int(model.nu))
    else:
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.ctrl[:] = np.tile(_control_base(int(model.nu)), (nworld, 1))


def _initial_quat() -> np.ndarray:
    quat = np.array([0.985, 0.08, -0.12, 0.08], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def _initial_omega() -> np.ndarray:
    return np.array([0.015, -0.012, 0.009], dtype=np.float64)


def _initial_joint_positions(njoint: int) -> np.ndarray:
    q = np.zeros(njoint, dtype=np.float64)
    q[: min(4, njoint)] = BIMANUAL_POSE[: min(4, njoint)]
    return q


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


def _mjorbit_body_origin_eci_states(
    data: Any,
    body_ids: Sequence[int],
    *,
    mujoco_module: Any | None = None,
    measurement_model: Any | None = None,
    measurement_data: Any | None = None,
    measurement_body_ids: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    positions = []
    velocities = []
    r0_m = np.asarray(data.orbit.R_eci, dtype=np.float64).reshape(-1, 3)[0] * 1.0e3
    v0_m_s = np.asarray(data.orbit.V_eci, dtype=np.float64).reshape(-1, 3)[0] * 1.0e3
    xpos = np.asarray(data.xpos, dtype=np.float64)
    xipos = np.asarray(data.xipos, dtype=np.float64)
    cvel = np.asarray(data.cvel, dtype=np.float64)
    if xpos.ndim == 3:
        xpos = xpos[0]
        xipos = xipos[0]
        cvel = cvel[0]
    qpos = np.asarray(data.qpos, dtype=np.float64)
    qvel = np.asarray(data.qvel, dtype=np.float64)
    use_body_jacobian = (
        mujoco_module is not None
        and measurement_model is not None
        and measurement_data is not None
        and measurement_body_ids is not None
    )
    if use_body_jacobian:
        qpos_sample = qpos.reshape(-1, int(measurement_model.nq))[0]
        qvel_sample = qvel.reshape(-1, int(measurement_model.nv))[0]
        measurement_data.qpos[:] = qpos_sample
        measurement_data.qvel[:] = qvel_sample
        mujoco_module.mj_forward(measurement_model, measurement_data)
        jacp = np.empty((3, int(measurement_model.nv)), dtype=np.float64)
        jacr = np.empty((3, int(measurement_model.nv)), dtype=np.float64)

    active_body_ids = measurement_body_ids if use_body_jacobian else body_ids
    for body_id in active_body_ids:
        if use_body_jacobian:
            origin_world_m = np.asarray(measurement_data.xpos[body_id], dtype=np.float64)
            jacp.fill(0.0)
            jacr.fill(0.0)
            mujoco_module.mj_jacBody(measurement_model, measurement_data, jacp, jacr, body_id)
            origin_velocity_world_m_s = jacp @ qvel_sample
        else:
            origin_world_m = xpos[body_id]
            com_world_m = xipos[body_id]
            com_velocity_world_m_s = cvel[body_id, 3:6]
            omega_world_rad_s = cvel[body_id, 0:3]
            origin_velocity_world_m_s = com_velocity_world_m_s - np.cross(
                omega_world_rad_s,
                com_world_m - origin_world_m,
            )
        positions.append((r0_m + origin_world_m) * 1.0e-3)
        velocities.append((v0_m_s + origin_velocity_world_m_s) * 1.0e-3)
    return np.vstack(positions), np.vstack(velocities)


def _interp_columns(x: np.ndarray, xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(x, xp, yp[:, idx]) for idx in range(yp.shape[1])])


def _interp_body_vectors(x: np.ndarray, xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
    out = np.empty((x.size, yp.shape[1], yp.shape[2]), dtype=np.float64)
    for body_idx in range(yp.shape[1]):
        out[:, body_idx, :] = _interp_columns(x, xp, yp[:, body_idx, :])
    return out


def _interp_quat(x: np.ndarray, xp: np.ndarray, quat: np.ndarray) -> np.ndarray:
    aligned = np.asarray(quat, dtype=np.float64).copy()
    for idx in range(1, aligned.shape[0]):
        if float(np.dot(aligned[idx - 1], aligned[idx])) < 0.0:
            aligned[idx] *= -1.0
    out = _interp_columns(x, xp, aligned)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norm, 1.0e-15)


def _squeeze_state_column(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    return arr.reshape((arr.shape[0], -1))[:, 0]


def _skipped_trajectory(backend: str, precision: str, message: str) -> Trajectory:
    return Trajectory(
        backend=backend,
        precision=precision,
        times_s=np.zeros(0),
        body_r_eci_km=np.zeros((0, 0, 3)),
        body_v_eci_km_s=np.zeros((0, 0, 3)),
        hub_quat_world_body=np.zeros((0, 4)),
        joint_angles_rad=np.zeros((0, 0)),
        joint_rates_rad_s=np.zeros((0, 0)),
        summary={"available": False, "ran": False, "message": message},
    )


def maybe_write_figure(
    out_dir: Path,
    accuracy: dict[str, Any],
    throughput: dict[str, Any],
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    comparisons = accuracy["comparisons"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)

    ax = axes[0]
    if comparisons:
        for name, comp in comparisons.items():
            label = _pretty_comparison_label(name)
            ax.semilogy(comp["times_s"], comp["body_relative_position_error_m"], label=label)
    else:
        ax.text(0.5, 0.5, "Basilisk unavailable", ha="center", va="center")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("max rel. body pos. error [m]")
    ax.grid(True, which="both", alpha=0.3)
    if comparisons:
        ax.legend(fontsize=8)

    ax = axes[1]
    bars = _throughput_bars(throughput)
    if bars:
        labels = [label for label, _ in bars]
        values = [value for _, value in bars]
        ax.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b"][: len(values)])
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=25)
    else:
        ax.text(0.5, 0.5, "No throughput runs", ha="center", va="center")
    ax.set_ylabel("sim steps / s")
    ax.grid(True, axis="y", which="both", alpha=0.3)

    path = out_dir / "basilisk_multibody_summary.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _pretty_comparison_label(name: str) -> str:
    return {
        "mjorbit_cpu_vs_basilisk": "mjorbit CPU",
        "mjorbit_warp_vs_basilisk": "mjorbit GPU",
        "mjorbit_warp_vs_cpu": "mjorbit GPU vs CPU",
    }.get(name, name)


def _throughput_bars(throughput: dict[str, Any]) -> list[tuple[str, float]]:
    bars = []
    for label, key in (
        ("Basilisk CPU", "basilisk_threads"),
        ("mjorbit CPU", "mjorbit_cpu"),
        ("mjorbit GPU", "mjorbit_warp"),
    ):
        runs = throughput.get(key, {}).get("runs", [])
        values = [run.get("sim_steps_per_s") for run in runs if "sim_steps_per_s" in run]
        if values:
            bars.append((label, float(max(values))))
    return bars


def _print_summary(summary: dict[str, Any]) -> None:
    print("=" * 72)
    print("Basilisk multibody validation")
    print("=" * 72)
    accuracy = summary["accuracy"]
    print("Accuracy:")
    for name, comp in accuracy["comparisons"].items():
        print(f"  {name}:")
        for key, value in comp.items():
            print(f"    {key}: {value:.6e}")
    throughput = summary["throughput"]
    print("\nThroughput best sim-steps/s:")
    for label, value in _throughput_bars(throughput):
        print(f"  {label}: {value:.6e}")
    print(f"\nWrote {OUT_DIR / 'basilisk_multibody_summary.json'}")
    if "figure" in summary:
        print(f"Wrote {summary['figure']}")


__all__ = [
    "ACTUATED_JOINTS",
    "BIMANUAL_POSE",
    "ExperimentAssets",
    "make_random_control_batch",
    "make_random_control_profile",
    "make_smooth_random_controls",
    "write_bimanual_assets",
]


if __name__ == "__main__":
    main()
