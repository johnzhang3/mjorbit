"""Capture-arm throughput benchmark.

Scenario: a chaser spacecraft with a 2-link arm + spherical end-effector and
a 50 kg tumbling target cube initially in contact with the EE. The arm has
two MuJoCo position actuators (shoulder + elbow). Gravity is disabled at the
MuJoCo level so the orbit overlay drives ECI gravity / GG torque / etc.

Two control modes:
  - ``zero``:  data.ctrl[:] = 0 throughout. Position actuators with non-zero
               kp will still push the arm toward the zero target, which keeps
               the EE pressed against the target and exercises the contact
               solver continuously.
  - ``sin``:   sinusoidal sweep on shoulder + elbow targets. The EE swings
               into the target periodically, generating intermittent
               high-impact contact events.

Three backends:
  - ``cpu``:    ``mujoco_orbit.rollout(model, data, control=ctrl, nstep=N,
                nthread=T)`` for several thread counts.
  - ``gpu``:    ``mujoco_orbit_warp.MjoModel.make_data(nworld=W)`` followed by
                a tight ``mjo_step`` loop with per-step ``mjo_upload`` of the
                control vector.
  - ``basilisk``: TODO — Basilisk-MuJoCo integration for a multi-body
                articulated model with actuators is non-trivial. The single-
                body Basilisk benchmark already covers their CPU path. We
                leave this as a stub; the GPU side has no Basilisk
                equivalent.

Usage:
    pixi run -e warp python benchmarks/capture_arm.py
    pixi run -e warp python benchmarks/capture_arm.py --nstep 2000 \\
        --threads 1 2 4 8 --nworlds 1 8 64 512 --modes both
    pixi run -e warp python benchmarks/capture_arm.py --json out/capture_arm.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import mujoco_orbit as mjo_cpu
from mujoco_orbit import OrbitInit
from mujoco_orbit.constants import R_EARTH
from mujoco_orbit.rollout import mjo_get_state, rollout

XML_PATH = Path(__file__).parent / "capture_arm.xml"


# ----------------------------------------------------------------------
# Initial state
# ----------------------------------------------------------------------

# Layout for capture_arm.xml (see model for body order):
#   qpos[0:7]   chaser bus free joint  (3 pos + 4 quat)
#   qpos[7]     shoulder hinge
#   qpos[8]     elbow hinge
#   qpos[9:16]  target free joint (3 pos + 4 quat)
# nu = 2  (shoulder, elbow position actuators)


def _set_initial_state(qpos: np.ndarray, qvel: np.ndarray) -> None:
    """Place chaser at chief origin and target ~at the EE, with a small tumble."""
    qpos[:] = 0.0
    qpos[3] = 1.0  # chaser quat w
    qpos[12] = 1.0  # target quat w
    qpos[9:12] = (2.0, 0.0, 0.0)  # target offset along bus +x

    qvel[:] = 0.0
    # Target tumble (rad/s, body frame): 0.1 / 0.05 / 0.2 about x/y/z.
    qvel[11:14] = (0.1, 0.05, 0.2)


def _orbit_init() -> OrbitInit:
    a = R_EARTH + 400.0
    R_eci = np.array([a, 0.0, 0.0])
    # Equatorial circular: v ≈ sqrt(mu/a)
    from mujoco_orbit.constants import GM_EARTH

    v = float(np.sqrt(GM_EARTH / a))
    V_eci = np.array([0.0, v, 0.0])
    return OrbitInit(R_eci=R_eci, V_eci=V_eci)


# ----------------------------------------------------------------------
# Control profiles
# ----------------------------------------------------------------------


def _build_control(nstep: int, nu: int, dt: float, mode: str) -> np.ndarray:
    """Return a (1, nstep, nu) control array. Tiled across batch by callers."""
    if mode == "zero":
        return np.zeros((1, nstep, nu), dtype=np.float64)
    if mode == "sin":
        t = np.arange(nstep) * dt
        ctrl = np.zeros((1, nstep, nu), dtype=np.float64)
        # Shoulder: 0.5 Hz, ±0.5 rad. Elbow: 0.7 Hz, ±1.0 rad.
        if nu >= 1:
            ctrl[0, :, 0] = 0.5 * np.sin(2.0 * np.pi * 0.5 * t)
        if nu >= 2:
            ctrl[0, :, 1] = 1.0 * np.sin(2.0 * np.pi * 0.7 * t)
        return ctrl
    raise ValueError(f"unknown control mode: {mode!r}")


# ----------------------------------------------------------------------
# CPU benchmark (mujoco_orbit.rollout)
# ----------------------------------------------------------------------


@dataclass
class CpuRun:
    threads: int
    wall_s: float
    sim_steps: int

    @property
    def steps_per_s(self) -> float:
        return self.sim_steps / self.wall_s


def _benchmark_cpu(
    *, nbatch: int, nstep: int, ctrl_per_step: np.ndarray, threads: list[int]
) -> dict[str, Any]:
    model = mjo_cpu.MjoModel.from_xml_path(str(XML_PATH))
    data = model.make_data(orbit=_orbit_init())

    # Warm initial qpos/qvel into data
    _set_initial_state(data.qpos, data.qvel)
    mjo_cpu.mjo_forward(model, data)

    initial_state = mjo_get_state(model, data)
    initial_batch = np.tile(initial_state, (nbatch, 1))
    ctrl_batch = np.broadcast_to(ctrl_per_step, (nbatch, nstep, ctrl_per_step.shape[-1])).copy()

    # Pre-allocate output buffers so allocation cost is excluded from the timed loop.
    state_buf = np.empty((nbatch, nstep, initial_state.shape[0]), dtype=np.float64)
    sensor_buf = np.empty((nbatch, nstep, int(model.nsensordata)), dtype=np.float64)

    # Warm up: small rollout to amortize first-touch costs (thread pool, JIT, page faults).
    rollout(model, data, initial_batch[:1], control=ctrl_batch[:1, :8], nstep=8, nthread=1)

    runs: list[CpuRun] = []
    for nthread in threads:
        nthread_eff = min(max(1, int(nthread)), nbatch)
        t0 = time.perf_counter()
        rollout(
            model,
            data,
            initial_batch,
            control=ctrl_batch,
            nstep=nstep,
            state=state_buf,
            sensordata=sensor_buf,
            nthread=nthread_eff,
        )
        wall = time.perf_counter() - t0
        runs.append(CpuRun(threads=nthread_eff, wall_s=wall, sim_steps=nbatch * nstep))

    return {
        "backend": "mujoco_orbit.rollout",
        "nbatch": nbatch,
        "nstep": nstep,
        "runs": [
            {
                "threads": r.threads,
                "wall_s": r.wall_s,
                "sim_steps": r.sim_steps,
                "sim_steps_per_s": r.steps_per_s,
            }
            for r in runs
        ],
    }


# ----------------------------------------------------------------------
# GPU benchmark (mujoco_orbit_warp batched mjo_step)
# ----------------------------------------------------------------------


@dataclass
class GpuRun:
    nworld: int
    wall_s: float
    sim_steps: int

    @property
    def steps_per_s(self) -> float:
        return self.sim_steps / self.wall_s


def _benchmark_gpu(
    *, nworlds: list[int], nstep: int, ctrl_per_step: np.ndarray, dt: float
) -> dict[str, Any]:
    try:
        import warp as wp

        import mujoco_orbit_warp as mjo_warp
    except ImportError as exc:
        return {"available": False, "message": f"mujoco_orbit_warp unavailable: {exc}"}

    runs: list[GpuRun] = []
    for nworld in nworlds:
        model = mjo_warp.MjoModel.from_xml_path(str(XML_PATH), mj_timestep=dt)
        orbit_inits = [_orbit_init() for _ in range(nworld)]
        warp_orbit_inits = [
            mjo_warp.OrbitInit(R_eci=o.R_eci.copy(), V_eci=o.V_eci.copy(), t=o.t)
            for o in orbit_inits
        ]
        data = model.make_data(orbit=warp_orbit_inits, nworld=nworld)

        # Initial qpos/qvel: broadcast the same values to every world.
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

        nu = int(model.nu)
        # Each world gets the same ctrl trajectory, broadcast on demand.
        ctrl_traj = ctrl_per_step[0]  # (nstep, nu)
        ctrl_zero = bool(np.all(ctrl_traj == 0.0))

        # Warm up so kernel compile is excluded from the measured loop.
        if nu > 0:
            if nworld == 1:
                np.copyto(data.ctrl, ctrl_traj[0])
            else:
                data.ctrl[:] = ctrl_traj[0]
            mjo_warp.mjo_upload(model, data, fields=("ctrl",))
        for _ in range(4):
            mjo_warp.mjo_step(model, data)
        wp.synchronize()

        t0 = time.perf_counter()
        for k in range(nstep):
            if not ctrl_zero and nu > 0:
                if nworld == 1:
                    np.copyto(data.ctrl, ctrl_traj[k])
                else:
                    data.ctrl[:] = ctrl_traj[k]
                mjo_warp.mjo_upload(model, data, fields=("ctrl",))
            mjo_warp.mjo_step(model, data)
        wp.synchronize()
        wall = time.perf_counter() - t0
        runs.append(GpuRun(nworld=nworld, wall_s=wall, sim_steps=nworld * nstep))

    return {
        "available": True,
        "backend": "mujoco_orbit_warp.mjo_step (batched)",
        "nstep": nstep,
        "runs": [
            {
                "nworld": r.nworld,
                "wall_s": r.wall_s,
                "sim_steps": r.sim_steps,
                "sim_steps_per_s": r.steps_per_s,
            }
            for r in runs
        ],
    }


# ----------------------------------------------------------------------
# Basilisk-MuJoCo (stub)
# ----------------------------------------------------------------------


def _basilisk_status() -> dict[str, Any]:
    """Capture-arm + actuators + contact through Basilisk-MuJoCo is not yet wired up.

    The Basilisk-MuJoCo single-body harness in ``comparisons/basilisk_mujoco/`` covers
    the orbit-only / single-rigid-body case. Reproducing the multi-body articulated
    actuated capture scenario in Basilisk's MJScene needs explicit joint target
    wiring and is left for a follow-up.
    """
    try:
        from Basilisk.simulation import mujoco as _mujoco  # noqa: F401

        installed = True
        del _mujoco
    except Exception:
        installed = False
    return {
        "available": False,
        "installed": installed,
        "message": (
            "Basilisk-MuJoCo capture-arm benchmark is not yet implemented; "
            "Basilisk has no GPU-batched path so its only point of comparison is "
            "the CPU rollout backend, which the existing comparisons/basilisk_mujoco "
            "harness already covers for single-body scenarios."
        ),
    }


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def _print_report(summary: dict[str, Any]) -> None:
    print("=" * 78)
    print(f"capture_arm benchmark — mode={summary['mode']!r}, nstep={summary['nstep']}")
    print("=" * 78)

    cpu = summary["cpu"]
    print(f"\n[CPU] {cpu['backend']}  (nbatch={cpu['nbatch']})")
    print(f"  {'threads':>8}  {'wall_s':>10}  {'sim_steps/s':>14}")
    for run in cpu["runs"]:
        print(
            f"  {run['threads']:>8}  {run['wall_s']:>10.4f}  {run['sim_steps_per_s']:>14,.0f}"
        )

    gpu = summary["gpu"]
    if not gpu.get("available"):
        print(f"\n[GPU] skipped: {gpu.get('message', '?')}")
    else:
        print(f"\n[GPU] {gpu['backend']}")
        print(f"  {'nworld':>8}  {'wall_s':>10}  {'sim_steps/s':>14}")
        for run in gpu["runs"]:
            print(
                f"  {run['nworld']:>8}  {run['wall_s']:>10.4f}  {run['sim_steps_per_s']:>14,.0f}"
            )

    print(f"\n[Basilisk-MuJoCo] {summary['basilisk']['message']}")


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def run_for_mode(
    *, mode: str, args: argparse.Namespace
) -> dict[str, Any]:
    nstep = args.nstep
    nu = 2  # shoulder, elbow
    dt = 0.01
    ctrl_per_step = _build_control(nstep, nu, dt, mode)

    cpu_summary = _benchmark_cpu(
        nbatch=args.nbatch, nstep=nstep, ctrl_per_step=ctrl_per_step, threads=args.threads
    )
    gpu_summary = _benchmark_gpu(
        nworlds=args.nworlds, nstep=nstep, ctrl_per_step=ctrl_per_step, dt=dt
    )

    return {
        "mode": mode,
        "nstep": nstep,
        "dt_s": dt,
        "cpu": cpu_summary,
        "gpu": gpu_summary,
        "basilisk": _basilisk_status(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nstep", type=int, default=1000)
    parser.add_argument("--nbatch", type=int, default=64,
                        help="CPU rollout batch size (passed to rollout(); "
                             "doubles as upper bound on threads).")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--nworlds", type=int, nargs="+", default=[1, 8, 64, 512])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("zero", "sin", "both"),
        default=["both"],
        help="Control modes to benchmark.",
    )
    parser.add_argument("--json", type=Path, default=None,
                        help="Write the full result dict to this path as JSON.")
    args = parser.parse_args()

    if "both" in args.modes:
        modes = ["zero", "sin"]
    else:
        modes = list(args.modes)

    out: dict[str, Any] = {"config": vars(args).copy()}
    out["config"]["json"] = str(args.json) if args.json else None

    for mode in modes:
        summary = run_for_mode(mode=mode, args=args)
        out[mode] = summary
        _print_report(summary)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
