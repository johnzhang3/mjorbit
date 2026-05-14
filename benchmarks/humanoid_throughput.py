"""Humanoid throughput sweep — single scenario, vary ``nworld``.

Scenario: the standard MuJoCo humanoid (28 qpos / 27 qvel / 21 actuators /
17 bodies + ground plane, capsule friction, condim=3 floor) with random
control commands held constant across the run. The humanoid falls under
gravity, contacts the ground, and the actuators hold a non-zero pose so
the contact + actuation pipeline is exercised every step.

Backends and ``nworld`` sweeps:

  - CPU: ``mujoco.rollout`` (pure) and ``mujoco_orbit.rollout`` (overlay).
    Sweep ``nworld in [1, 4, 16, 64, 256]``. CPU thread count is set to
    ``min(nworld, 20)`` (i7-12700K logical-core ceiling).
  - GPU: ``mjw.step`` (pure) and ``mjo_step`` (overlay). Each captured
    once into a CUDA Graph and replayed via ``wp.capture_launch`` per
    step. Sweep ``nworld in [64, 128, 512, 1024, 2048, 4096]``.

The orbit overlay's actual contribution to the humanoid scene is small
(it adds tidal/GG forces relative to a 400 km LEO chief), but the
computational cost of the overlay's per-step kernels is the point of the
benchmark.

Output: a JSON file with per-(nworld, backend) wall time + steps/s.

Usage:
    pixi run -e warp python benchmarks/humanoid_throughput.py \\
        --json benchmarks/results/humanoid_throughput.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import mujoco
import mujoco.rollout

import mujoco_orbit as mjo_cpu
from mujoco_orbit import OrbitInit, SurfaceSpec
from mujoco_orbit.constants import GM_EARTH, R_EARTH
from mujoco_orbit.rollout import mjo_get_state, rollout
from mujoco_orbit.spec import MjoSpec

XML_PATH = Path(__file__).parent / "humanoid.xml"


# ----------------------------------------------------------------------
# Orbit physics config: surfaces for drag + SRP per body
# ----------------------------------------------------------------------
# Approximate cross-section areas (m^2) per body so the drag/SRP kernels
# do real work. Normals are body-fixed and roughly perpendicular to the
# limb's long axis. These are coarse but representative of a humanoid-
# sized spacecraft.
_HUMANOID_BODY_SURFACES: list[tuple[str, np.ndarray, float]] = [
    ("torso",           np.array([0.0, 0.0, 1.0]), 0.40),
    ("head",            np.array([0.0, 0.0, 1.0]), 0.04),
    ("waist_lower",     np.array([0.0, 0.0, 1.0]), 0.20),
    ("pelvis",          np.array([0.0, 0.0, 1.0]), 0.20),
    ("thigh_right",     np.array([0.0, 1.0, 0.0]), 0.10),
    ("shin_right",      np.array([0.0, 1.0, 0.0]), 0.07),
    ("foot_right",      np.array([0.0, 0.0, 1.0]), 0.04),
    ("thigh_left",      np.array([0.0, -1.0, 0.0]), 0.10),
    ("shin_left",       np.array([0.0, -1.0, 0.0]), 0.07),
    ("foot_left",       np.array([0.0, 0.0, 1.0]), 0.04),
    ("upper_arm_right", np.array([0.0, -1.0, 0.0]), 0.05),
    ("lower_arm_right", np.array([0.0, -1.0, 0.0]), 0.03),
    ("hand_right",      np.array([0.0, 0.0, 1.0]), 0.01),
    ("upper_arm_left",  np.array([0.0, 1.0, 0.0]), 0.05),
    ("lower_arm_left",  np.array([0.0, 1.0, 0.0]), 0.03),
    ("hand_left",       np.array([0.0, 0.0, 1.0]), 0.01),
]


def _humanoid_surfaces() -> list[SurfaceSpec]:
    return [
        SurfaceSpec(
            body_name=name,
            center_of_pressure_body=np.zeros(3),
            normal_body=normal,
            area=area,
            drag_coeff=2.2,
            srp_coeff=1.5,
            use_drag=True,
            use_srp=True,
            name=f"surf_{name}",
        )
        for name, normal, area in _HUMANOID_BODY_SURFACES
    ]


# ----------------------------------------------------------------------
# Initial state and control
# ----------------------------------------------------------------------


def _set_initial_state(qpos: np.ndarray, qvel: np.ndarray) -> None:
    """Humanoid at the chief origin, identity orientation, at rest.

    With gravity disabled and no floor, the humanoid tumbles under random
    actuator torques alone. Contact still triggers via limb-on-limb self
    collisions (capsule geoms with default contype/conaffinity).
    """
    qpos[:] = 0.0
    qpos[3] = 1.0  # quaternion w
    qvel[:] = 0.0


def _orbit_init() -> OrbitInit:
    a = R_EARTH + 400.0
    R_eci = np.array([a, 0.0, 0.0])
    v = float(np.sqrt(GM_EARTH / a))
    V_eci = np.array([0.0, v, 0.0])
    return OrbitInit(R_eci=R_eci, V_eci=V_eci)


def _build_ctrl_traj(nworld: int, nstep: int, nu: int, seed: int = 0) -> np.ndarray:
    """Per-world per-step random control trajectory, shape (nworld, nstep, nu).

    Each world gets its own independent uniform-random control sequence so
    every rollout evolves independently — no degenerate SIMD coherence
    across worlds on GPU and no identical-rollout artifacts on CPU.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.4, 0.4, size=(nworld, nstep, nu)).astype(np.float64)


# ----------------------------------------------------------------------
# CPU benchmarks
# ----------------------------------------------------------------------


def _benchmark_cpu_pure(
    *, nworlds: list[int], nstep: int, ctrl_seed: int, nthread_cap: int,
    trials: int,
) -> dict[str, Any]:
    """Bare ``mujoco.rollout`` (no orbit overlay), per-world random ctrls."""
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    nu = int(model.nu)
    nstate = int(mujoco.mj_stateSize(model, int(mujoco.mjtState.mjSTATE_FULLPHYSICS)))

    data = mujoco.MjData(model)
    _set_initial_state(data.qpos, data.qvel)
    mujoco.mj_forward(model, data)
    initial_state = np.zeros(nstate)
    mujoco.mj_getState(model, data, initial_state, int(mujoco.mjtState.mjSTATE_FULLPHYSICS))

    runs: list[dict[str, Any]] = []
    for nworld in nworlds:
        initial_batch = np.tile(initial_state, (nworld, 1))
        ctrl_batch = _build_ctrl_traj(nworld, nstep, nu, seed=ctrl_seed)
        state_buf = np.empty((nworld, nstep, nstate), dtype=np.float64)
        sensor_buf = np.empty((nworld, nstep, int(model.nsensordata)), dtype=np.float64)

        nthread = min(nworld, nthread_cap)
        thread_datas = [data] + [mujoco.MjData(model) for _ in range(nthread - 1)]

        # warm up
        mujoco.rollout.rollout(model, thread_datas, initial_batch[:nthread],
                               control=ctrl_batch[:nthread, :8], nstep=8)

        walls: list[float] = []
        for _ in range(trials):
            t0 = time.perf_counter()
            mujoco.rollout.rollout(
                model, thread_datas, initial_batch, control=ctrl_batch,
                nstep=nstep, state=state_buf, sensordata=sensor_buf,
            )
            walls.append(time.perf_counter() - t0)
        wall = float(np.median(walls))
        runs.append({
            "nworld": nworld, "nthread": nthread,
            "wall_s": wall, "wall_min_s": float(min(walls)),
            "trials": trials,
            "sim_steps": nworld * nstep,
            "sim_steps_per_s": (nworld * nstep) / wall,
            "sim_steps_per_s_best": (nworld * nstep) / min(walls),
        })

    return {"backend": "mujoco.rollout (pure CPU)", "nstep": nstep, "runs": runs}


def _build_cpu_orbit_model() -> mjo_cpu.MjoModel:
    """Compile humanoid + orbit overlay (drag + SRP active, GG enabled)."""
    spec = MjoSpec.from_xml_path(str(XML_PATH))
    spec.mjorbit.use_j2 = True
    spec.mjorbit.use_drag = True
    spec.mjorbit.use_srp = True
    spec.mjorbit.use_magnetic = False
    spec.mjorbit.use_gravity_gradient = True
    for surface in _humanoid_surfaces():
        spec.mjorbit.add_surface(surface)
    return spec.compile()


def _benchmark_cpu_orbit(
    *, nworlds: list[int], nstep: int, ctrl_seed: int, nthread_cap: int,
    trials: int,
) -> dict[str, Any]:
    """Orbit-aware ``mujoco_orbit.rollout`` with drag + SRP via per-body surfaces."""
    model = _build_cpu_orbit_model()
    data = model.make_data(orbit=_orbit_init())
    _set_initial_state(data.qpos, data.qvel)
    mjo_cpu.mjo_forward(model, data)
    nu = int(model.nu)

    initial_state = mjo_get_state(model, data)

    runs: list[dict[str, Any]] = []
    for nworld in nworlds:
        initial_batch = np.tile(initial_state, (nworld, 1))
        ctrl_batch = _build_ctrl_traj(nworld, nstep, nu, seed=ctrl_seed)
        state_buf = np.empty((nworld, nstep, initial_state.shape[0]), dtype=np.float64)
        sensor_buf = np.empty((nworld, nstep, int(model.nsensordata)), dtype=np.float64)

        nthread = min(nworld, nthread_cap)
        # warm up
        rollout(model, data, initial_batch[:nthread],
                control=ctrl_batch[:nthread, :8], nstep=8, nthread=nthread)

        walls: list[float] = []
        for _ in range(trials):
            t0 = time.perf_counter()
            rollout(
                model, data, initial_batch, control=ctrl_batch, nstep=nstep,
                state=state_buf, sensordata=sensor_buf, nthread=nthread,
            )
            walls.append(time.perf_counter() - t0)
        wall = float(np.median(walls))
        runs.append({
            "nworld": nworld, "nthread": nthread,
            "wall_s": wall, "wall_min_s": float(min(walls)),
            "trials": trials,
            "sim_steps": nworld * nstep,
            "sim_steps_per_s": (nworld * nstep) / wall,
            "sim_steps_per_s_best": (nworld * nstep) / min(walls),
        })

    return {"backend": "mujoco_orbit.rollout (orbit CPU)", "nstep": nstep, "runs": runs}


# ----------------------------------------------------------------------
# GPU benchmarks (CUDA Graph captured)
# ----------------------------------------------------------------------


def _benchmark_gpu_pure(
    *, nworlds: list[int], nstep: int, ctrl_seed: int, dt: float,
    trials: int,
) -> dict[str, Any]:
    """Bare ``mjw.step`` captured into a CUDA Graph, per-world per-step ctrls."""
    try:
        import mujoco_warp as mjw
        import warp as wp
    except ImportError as exc:
        return {"available": False, "message": str(exc)}

    mj_model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    mj_model.opt.timestep = dt
    nu = int(mj_model.nu)

    runs: list[dict[str, Any]] = []
    for nworld in nworlds:
        host_data = mujoco.MjData(mj_model)
        _set_initial_state(host_data.qpos, host_data.qvel)
        mujoco.mj_forward(mj_model, host_data)

        warp_model = mjw.put_model(mj_model)
        warp_data = mjw.put_data(mj_model, host_data, nworld=nworld)

        # Per-world per-step ctrls, generated as float32 contiguous so each
        # per-step slice can be uploaded without extra copies.
        ctrl_traj_f32 = _build_ctrl_traj(nworld, nstep, nu, seed=ctrl_seed).astype(np.float32)
        ctrl_traj_f32 = np.ascontiguousarray(ctrl_traj_f32.transpose(1, 0, 2))  # (nstep, nworld, nu)
        warp_data.ctrl.assign(ctrl_traj_f32[0])

        # warm up + JIT
        mjw.forward(warp_model, warp_data)
        for _ in range(4):
            mjw.step(warp_model, warp_data)
        wp.synchronize()

        # CUDA Graph capture (reads warp_data.ctrl at launch time, so
        # updating that device buffer between launches is safe).
        with wp.ScopedCapture() as capture:
            mjw.step(warp_model, warp_data)
        graph = capture.graph
        wp.synchronize()

        walls: list[float] = []
        for _ in range(trials):
            t0 = time.perf_counter()
            for k in range(nstep):
                warp_data.ctrl.assign(ctrl_traj_f32[k])
                wp.capture_launch(graph)
            wp.synchronize()
            walls.append(time.perf_counter() - t0)
        wall = float(np.median(walls))
        runs.append({
            "nworld": nworld, "wall_s": wall, "wall_min_s": float(min(walls)),
            "trials": trials,
            "sim_steps": nworld * nstep,
            "sim_steps_per_s": (nworld * nstep) / wall,
            "sim_steps_per_s_best": (nworld * nstep) / min(walls),
        })

    return {"available": True, "backend": "mjw.step (pure GPU)",
            "nstep": nstep, "runs": runs}


def _benchmark_gpu_orbit(
    *, nworlds: list[int], nstep: int, ctrl_seed: int, dt: float,
    trials: int,
) -> dict[str, Any]:
    """Orbit ``mjo_step`` (drag + SRP via surfaces) captured into a CUDA Graph."""
    try:
        import warp as wp

        import mujoco_orbit_warp as mjo_warp
    except ImportError as exc:
        return {"available": False, "message": str(exc)}

    runs: list[dict[str, Any]] = []
    for nworld in nworlds:
        model = mjo_warp.MjoModel.from_xml_path(
            str(XML_PATH),
            mj_timestep=dt,
            surfaces=_humanoid_surfaces(),
            use_j2=True,
            use_drag=True,
            use_srp=True,
            use_magnetic=False,
            use_gravity_gradient=True,
        )
        nu = int(model.nu)
        orbit_inits = [_orbit_init() for _ in range(nworld)]
        warp_orbit_inits = [
            mjo_warp.OrbitInit(R_eci=o.R_eci.copy(), V_eci=o.V_eci.copy(), t=o.t)
            for o in orbit_inits
        ]
        data = model.make_data(orbit=warp_orbit_inits, nworld=nworld)

        qpos0 = np.zeros(model.nq)
        qvel0 = np.zeros(model.nv)
        _set_initial_state(qpos0, qvel0)
        if nworld == 1:
            np.copyto(data.qpos, qpos0)
            np.copyto(data.qvel, qvel0)
        else:
            data.qpos[:] = qpos0
            data.qvel[:] = qvel0
        mjo_warp.mjo_upload(model, data, fields=("state", "inputs", "core"))
        mjo_warp.mjo_forward(model, data)

        # Per-world per-step ctrls.
        ctrl_traj_f32 = _build_ctrl_traj(nworld, nstep, nu, seed=ctrl_seed).astype(np.float32)
        ctrl_traj_f32 = np.ascontiguousarray(ctrl_traj_f32.transpose(1, 0, 2))  # (nstep, nworld, nu)

        # Seed device-side ctrl buffer for the warmup steps.
        if nworld == 1:
            np.copyto(data.ctrl, ctrl_traj_f32[0, 0])
        else:
            data.ctrl[:] = ctrl_traj_f32[0]
        mjo_warp.mjo_upload(model, data, fields=("ctrl",))

        for _ in range(4):
            mjo_warp.mjo_step(model, data)
        wp.synchronize()

        with wp.ScopedCapture() as capture:
            mjo_warp.mjo_step(model, data)
        graph = capture.graph
        wp.synchronize()

        walls: list[float] = []
        for _ in range(trials):
            t0 = time.perf_counter()
            for k in range(nstep):
                if nworld == 1:
                    np.copyto(data.ctrl, ctrl_traj_f32[k, 0])
                else:
                    data.ctrl[:] = ctrl_traj_f32[k]
                mjo_warp.mjo_upload(model, data, fields=("ctrl",))
                wp.capture_launch(graph)
            wp.synchronize()
            walls.append(time.perf_counter() - t0)
        wall = float(np.median(walls))
        runs.append({
            "nworld": nworld, "wall_s": wall, "wall_min_s": float(min(walls)),
            "trials": trials,
            "sim_steps": nworld * nstep,
            "sim_steps_per_s": (nworld * nstep) / wall,
            "sim_steps_per_s_best": (nworld * nstep) / min(walls),
        })

    return {"available": True, "backend": "mjo_step (orbit GPU)",
            "nstep": nstep, "runs": runs}


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def _print_report(out: dict[str, Any]) -> None:
    print("=" * 76)
    print(f"humanoid_throughput — nstep={out['nstep']}, dt={out['dt_s']}")
    print("=" * 76)
    for key, label in (("cpu_pure", "CPU pure"), ("cpu_orbit", "CPU orbit"),
                       ("gpu_pure", "GPU pure"), ("gpu_orbit", "GPU orbit")):
        block = out[key]
        if not block.get("available", True):
            print(f"\n[{label}] skipped: {block.get('message', '?')}")
            continue
        print(f"\n[{label}] {block['backend']}")
        print(f"  {'nworld':>8}  {'wall (s)':>10}  {'steps/s':>15}")
        for r in block["runs"]:
            print(f"  {r['nworld']:>8}  {r['wall_s']:>10.3f}  {r['sim_steps_per_s']:>15,.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nstep", type=int, default=1000)
    parser.add_argument("--cpu-nworlds", type=int, nargs="+",
                        default=[1, 4, 16, 64, 256])
    parser.add_argument("--gpu-nworlds", type=int, nargs="+",
                        default=[64, 128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--nthread-cap", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of timed trials per (backend, nworld); "
                             "we report the median to suppress measurement noise.")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    mj_model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    nu = int(mj_model.nu)
    dt = float(mj_model.opt.timestep)
    ctrl_seed = 0

    out: dict[str, Any] = {
        "config": vars(args).copy(),
        "nstep": args.nstep,
        "dt_s": dt,
        "model": {"nq": int(mj_model.nq), "nv": int(mj_model.nv),
                  "nu": nu, "nbody": int(mj_model.nbody)},
        "cpu_pure": _benchmark_cpu_pure(
            nworlds=args.cpu_nworlds, nstep=args.nstep, ctrl_seed=ctrl_seed,
            nthread_cap=args.nthread_cap, trials=args.trials),
        "cpu_orbit": _benchmark_cpu_orbit(
            nworlds=args.cpu_nworlds, nstep=args.nstep, ctrl_seed=ctrl_seed,
            nthread_cap=args.nthread_cap, trials=args.trials),
        "gpu_pure": _benchmark_gpu_pure(
            nworlds=args.gpu_nworlds, nstep=args.nstep, ctrl_seed=ctrl_seed,
            dt=dt, trials=args.trials),
        "gpu_orbit": _benchmark_gpu_orbit(
            nworlds=args.gpu_nworlds, nstep=args.nstep, ctrl_seed=ctrl_seed,
            dt=dt, trials=args.trials),
    }
    out["config"]["json"] = str(args.json) if args.json else None
    _print_report(out)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
