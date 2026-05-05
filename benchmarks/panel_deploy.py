"""Panel-deploy throughput benchmark.

Scenario: a chaser bus with two symmetric solar wings, six hinge-jointed
panels each (12 hinges total). Every hinge is a passive torsional spring
(``springref=0``) with damping; panels start partially folded (``qpos = +1.5
rad`` on every hinge) and the springs deploy them toward flat with a hard
range-stop latch at ``qpos = 0``. No MuJoCo actuators — the dynamics are
entirely passive multibody plus (when the orbit overlay is enabled) ECI
gravity, gravity-gradient torque, drag, SRP, and origin-acceleration
compensation.

Single control mode (``passive``); ``nu = 0`` so the control trajectory is a
``(1, nstep, 0)`` array. Backends mirror ``capture_arm.py``:

  - ``mujoco.rollout``         — pure MuJoCo CPU baseline (no orbit overlay)
  - ``mujoco_orbit.rollout``   — orbit overlay on CPU
  - ``mujoco_warp.step``       — pure MJWarp GPU baseline (no orbit overlay)
  - ``mujoco_orbit_warp.mjo_step`` — orbit overlay on GPU, batched

The eventual figure plots steps/s for these four paths across three difficulty
tiers (capture-arm easy / panel-deploy moderate / 7-DOF-arm-plus-arrays hard,
TODO), with Basilisk-MuJoCo as the CPU baseline and smallsatsim.github.io as
the GPU baseline. Both reference baselines are stubbed for now and called out
as TODOs further down.

Usage:
    pixi run -e warp python benchmarks/panel_deploy.py
    pixi run -e warp python benchmarks/panel_deploy.py --nstep 2000 \\
        --threads 1 2 4 8 --nworlds 1 8 64 512
    pixi run -e warp python benchmarks/panel_deploy.py --json out/panel_deploy.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import mujoco
import mujoco.rollout

import mujoco_orbit as mjo_cpu
from mujoco_orbit import OrbitInit
from mujoco_orbit.constants import R_EARTH
from mujoco_orbit.rollout import mjo_get_state, rollout

XML_PATH = Path(__file__).parent / "panel_deploy.xml"


# ----------------------------------------------------------------------
# Initial state
# ----------------------------------------------------------------------

# Layout for panel_deploy.xml (see model for body order):
#   qpos[0:7]    bus free joint  (3 pos + 4 quat)
#   qpos[7:13]   +X wing hinges hp_0 .. hp_5
#   qpos[13:19]  -X wing hinges hn_0 .. hn_5
# nu = 0  (no actuators)

INITIAL_FOLD_RAD = 1.5  # Initial hinge angle on every panel.


def _set_initial_state(qpos: np.ndarray, qvel: np.ndarray) -> None:
    """Bus at chief origin, all panels folded to ``INITIAL_FOLD_RAD`` rad."""
    qpos[:] = 0.0
    qpos[3] = 1.0  # bus quat w
    qpos[7:19] = INITIAL_FOLD_RAD
    qvel[:] = 0.0


def _orbit_init() -> OrbitInit:
    a = R_EARTH + 400.0
    R_eci = np.array([a, 0.0, 0.0])
    from mujoco_orbit.constants import GM_EARTH

    v = float(np.sqrt(GM_EARTH / a))
    V_eci = np.array([0.0, v, 0.0])
    return OrbitInit(R_eci=R_eci, V_eci=V_eci)


# ----------------------------------------------------------------------
# Control profile
# ----------------------------------------------------------------------


def _build_control(nstep: int, nu: int, dt: float, mode: str) -> np.ndarray:
    """Return a ``(1, nstep, nu)`` control array.

    Panel-deploy has ``nu == 0`` so the result is shape ``(1, nstep, 0)``;
    rollout still walks ``nstep`` MuJoCo steps but the empty control vector
    is a no-op every step.
    """
    if mode != "passive":
        raise ValueError(f"unknown control mode: {mode!r}")
    return np.zeros((1, nstep, nu), dtype=np.float64)


# ----------------------------------------------------------------------
# CPU reference: pure mujoco.rollout (no orbit overlay)
# ----------------------------------------------------------------------


def _benchmark_pure_mujoco(
    *, nbatch: int, nstep: int, ctrl_per_step: np.ndarray, threads: list[int]
) -> dict[str, Any]:
    """Bare MuJoCo rollout on the same XML — orbit overlay disabled."""
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    nu = int(model.nu)
    nstate = int(mujoco.mj_stateSize(model, int(mujoco.mjtState.mjSTATE_FULLPHYSICS)))

    data = mujoco.MjData(model)
    _set_initial_state(data.qpos, data.qvel)
    mujoco.mj_forward(model, data)
    initial_state = np.zeros(nstate)
    mujoco.mj_getState(model, data, initial_state, int(mujoco.mjtState.mjSTATE_FULLPHYSICS))
    initial_batch = np.tile(initial_state, (nbatch, 1))

    ctrl_batch = np.broadcast_to(ctrl_per_step, (nbatch, nstep, nu)).copy()
    state_buf = np.empty((nbatch, nstep, nstate), dtype=np.float64)
    sensor_buf = np.empty((nbatch, nstep, int(model.nsensordata)), dtype=np.float64)

    mujoco.rollout.rollout(
        model, data, initial_batch[:1], control=ctrl_batch[:1, :8], nstep=8
    )

    runs: list[dict[str, Any]] = []
    for nthread in threads:
        nthread_eff = min(max(1, int(nthread)), nbatch)
        thread_datas = [data] + [mujoco.MjData(model) for _ in range(nthread_eff - 1)]
        t0 = time.perf_counter()
        mujoco.rollout.rollout(
            model,
            thread_datas,
            initial_batch,
            control=ctrl_batch,
            nstep=nstep,
            state=state_buf,
            sensordata=sensor_buf,
        )
        wall = time.perf_counter() - t0
        runs.append({
            "threads": nthread_eff,
            "wall_s": wall,
            "sim_steps": nbatch * nstep,
            "sim_steps_per_s": (nbatch * nstep) / wall,
        })

    return {"backend": "mujoco.rollout (no orbit overlay)", "nbatch": nbatch,
            "nstep": nstep, "runs": runs}


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

    _set_initial_state(data.qpos, data.qvel)
    mjo_cpu.mjo_forward(model, data)

    initial_state = mjo_get_state(model, data)
    initial_batch = np.tile(initial_state, (nbatch, 1))
    ctrl_batch = np.broadcast_to(
        ctrl_per_step, (nbatch, nstep, ctrl_per_step.shape[-1])
    ).copy()

    state_buf = np.empty((nbatch, nstep, initial_state.shape[0]), dtype=np.float64)
    sensor_buf = np.empty((nbatch, nstep, int(model.nsensordata)), dtype=np.float64)

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
# GPU reference: pure MJWarp (no orbit overlay)
# ----------------------------------------------------------------------


def _benchmark_pure_mjwarp(
    *, nworlds: list[int], nstep: int, ctrl_per_step: np.ndarray, dt: float
) -> dict[str, Any]:
    """Bare MJWarp step loop on the same XML — orbit overlay disabled."""
    try:
        import mujoco_warp as mjw
        import warp as wp
    except ImportError as exc:
        return {"available": False, "message": f"mujoco_warp unavailable: {exc}"}

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

        ctrl_traj = ctrl_per_step[0]  # (nstep, nu); nu = 0 for panel_deploy
        ctrl_zero = True  # nu == 0 → nothing to upload per step

        mjw.forward(warp_model, warp_data)
        if nu > 0:
            ctrl0 = np.broadcast_to(ctrl_traj[0], (nworld, nu)).astype(np.float32)
            warp_data.ctrl.assign(ctrl0)
        for _ in range(4):
            mjw.step(warp_model, warp_data)
        wp.synchronize()

        t0 = time.perf_counter()
        for k in range(nstep):
            if not ctrl_zero and nu > 0:
                ctrl_k = np.broadcast_to(ctrl_traj[k], (nworld, nu)).astype(np.float32)
                warp_data.ctrl.assign(ctrl_k)
            mjw.step(warp_model, warp_data)
        wp.synchronize()
        wall = time.perf_counter() - t0
        runs.append({
            "nworld": nworld,
            "wall_s": wall,
            "sim_steps": nworld * nstep,
            "sim_steps_per_s": (nworld * nstep) / wall,
        })

    return {"available": True, "backend": "mujoco_warp.step (no orbit overlay)",
            "nstep": nstep, "runs": runs}


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
        ctrl_zero = True  # nu == 0 for panel_deploy

        for _ in range(4):
            mjo_warp.mjo_step(model, data)
        wp.synchronize()

        t0 = time.perf_counter()
        for _ in range(nstep):
            if not ctrl_zero and nu > 0:
                # Reserved for future actuated variants of this benchmark.
                pass
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
# Reference baselines (TODOs for the eventual figure)
# ----------------------------------------------------------------------


def _basilisk_status() -> dict[str, Any]:
    """TODO: Basilisk-MuJoCo CPU baseline for the panel-deploy scenario.

    Plan for the figure: reproduce the 12-hinge spring-loaded wing in
    Basilisk's MJScene and run their CPU step loop to get a steps/s number
    on the same hardware. Single-body Basilisk-MuJoCo is already wired up
    under ``comparisons/basilisk_mujoco/`` but that harness only covers
    one rigid body; the multi-hinge passive-deploy port still needs to be
    written.
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
            "TODO: Basilisk-MuJoCo panel-deploy baseline not yet wired up. "
            "Needed for the CPU sub-figure once the eventual throughput plot "
            "lands. Reuse the comparisons/basilisk_mujoco harness as a starting "
            "point and add the 12-hinge wing model."
        ),
    }


def _smallsatsim_status() -> dict[str, Any]:
    """TODO: smallsatsim GPU baseline for the panel-deploy scenario.

    Plan for the figure: take smallsatsim's flexible-array deploy benchmark
    (https://smallsatsim.github.io) and report steps/s on the same hardware
    so the GPU sub-figure has a third bar (pure MJWarp / mujoco_orbit_warp /
    smallsatsim). Their solver and frame conventions differ from MJWarp's so
    apples-to-apples requires care — the right comparison is total wall-clock
    per simulated second of spacecraft time, not raw kernel throughput.
    """
    return {
        "available": False,
        "installed": False,
        "message": (
            "TODO: smallsatsim panel-deploy baseline not yet wired up. "
            "Needed for the GPU sub-figure. Use their flexible-array deploy "
            "demo as the reference scenario and report steps/s on identical "
            "hardware."
        ),
    }


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def _overlay_overhead_pct(orbit_steps_per_s: float, pure_steps_per_s: float) -> str:
    if pure_steps_per_s <= 0.0 or orbit_steps_per_s <= 0.0:
        return "n/a"
    pct = 100.0 * (pure_steps_per_s - orbit_steps_per_s) / pure_steps_per_s
    return f"{pct:+5.1f}%"


def _print_report(summary: dict[str, Any]) -> None:
    print("=" * 86)
    print(f"panel_deploy benchmark — mode={summary['mode']!r}, nstep={summary['nstep']}")
    print("=" * 86)

    cpu_pure = summary["cpu_pure"]
    cpu_orbit = summary["cpu"]
    print(f"\n[CPU]  pure: {cpu_pure['backend']}    "
          f"orbit: {cpu_orbit['backend']}    "
          f"(nbatch={cpu_orbit['nbatch']})")
    print(f"  {'threads':>8}  {'pure steps/s':>14}  {'orbit steps/s':>15}  "
          f"{'overlay tax':>12}")
    pure_by_t = {r["threads"]: r["sim_steps_per_s"] for r in cpu_pure["runs"]}
    for run in cpu_orbit["runs"]:
        pure = pure_by_t.get(run["threads"], 0.0)
        print(
            f"  {run['threads']:>8}  {pure:>14,.0f}  {run['sim_steps_per_s']:>15,.0f}  "
            f"{_overlay_overhead_pct(run['sim_steps_per_s'], pure):>12}"
        )

    gpu_pure = summary.get("gpu_pure", {"available": False})
    gpu_orbit = summary["gpu"]
    if not gpu_orbit.get("available"):
        print(f"\n[GPU] skipped: {gpu_orbit.get('message', '?')}")
    else:
        print(f"\n[GPU]  pure: {gpu_pure.get('backend', 'n/a')}    "
              f"orbit: {gpu_orbit['backend']}")
        if gpu_pure.get("available"):
            pure_by_w = {r["nworld"]: r["sim_steps_per_s"] for r in gpu_pure["runs"]}
        else:
            pure_by_w = {}
        print(f"  {'nworld':>8}  {'pure steps/s':>14}  {'orbit steps/s':>15}  "
              f"{'overlay tax':>12}")
        for run in gpu_orbit["runs"]:
            pure = pure_by_w.get(run["nworld"], 0.0)
            print(
                f"  {run['nworld']:>8}  {pure:>14,.0f}  {run['sim_steps_per_s']:>15,.0f}  "
                f"{_overlay_overhead_pct(run['sim_steps_per_s'], pure):>12}"
            )

    print(f"\n[Basilisk-MuJoCo] {summary['basilisk']['message']}")
    print(f"[smallsatsim]    {summary['smallsatsim']['message']}")


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def run_for_mode(*, mode: str, args: argparse.Namespace) -> dict[str, Any]:
    nstep = args.nstep
    nu = 0  # panel_deploy has no actuators
    dt = 0.005
    ctrl_per_step = _build_control(nstep, nu, dt, mode)

    cpu_pure = _benchmark_pure_mujoco(
        nbatch=args.nbatch, nstep=nstep, ctrl_per_step=ctrl_per_step, threads=args.threads
    )
    cpu_summary = _benchmark_cpu(
        nbatch=args.nbatch, nstep=nstep, ctrl_per_step=ctrl_per_step, threads=args.threads
    )
    gpu_pure = _benchmark_pure_mjwarp(
        nworlds=args.nworlds, nstep=nstep, ctrl_per_step=ctrl_per_step, dt=dt
    )
    gpu_summary = _benchmark_gpu(
        nworlds=args.nworlds, nstep=nstep, ctrl_per_step=ctrl_per_step, dt=dt
    )

    return {
        "mode": mode,
        "nstep": nstep,
        "dt_s": dt,
        "cpu_pure": cpu_pure,
        "cpu": cpu_summary,
        "gpu_pure": gpu_pure,
        "gpu": gpu_summary,
        "basilisk": _basilisk_status(),
        "smallsatsim": _smallsatsim_status(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nstep", type=int, default=1000)
    parser.add_argument("--nbatch", type=int, default=64,
                        help="CPU rollout batch size (also caps thread count).")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--nworlds", type=int, nargs="+", default=[1, 8, 64, 512])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("passive",),
        default=["passive"],
        help="Control modes to benchmark (panel_deploy currently has only one).",
    )
    parser.add_argument("--json", type=Path, default=None,
                        help="Write the full result dict to this path as JSON.")
    args = parser.parse_args()

    out: dict[str, Any] = {"config": vars(args).copy()}
    out["config"]["json"] = str(args.json) if args.json else None

    for mode in args.modes:
        summary = run_for_mode(mode=mode, args=args)
        out[mode] = summary
        _print_report(summary)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
