"""Multi-threaded rollout benchmark: humanoid in LEO via mjorbit vs pure MuJoCo.

Both backends run the same MuJoCo humanoid (no floor, no contact). The
mjorbit run additionally activates the full orbital environment via the
mjorbit plugin: J2, drag (with one drag surface on the torso), SRP, the
Earth magnetic field, and gravity gradient torques.

Frames-per-second is reported for several thread counts. The MuJoCo baseline
uses ``mujoco.rollout.Rollout(nthread=N)``; the mjorbit side uses
``mjorbit.rollout(..., nthread=N)``, which spreads the batch across a
``ThreadPoolExecutor`` of independent ``MjoData`` workers.

Usage:
    pixi run python experiments/rollout_benchmark/run.py
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.rollout as mj_rollout
import numpy as np

from mjorbit import (
    MjoData,
    MjoModel,
    OrbitInit,
    mjo_forward,
    mjo_get_state,
    rollout,
)
from mjorbit.constants import GM_EARTH, R_EARTH

EXPERIMENT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = EXPERIMENT_DIR / "figures"
HUMANOID_XML = EXPERIMENT_DIR / "humanoid.xml"

ALT_KM = 400.0
INCLINATION_DEG = 51.6
NSTEP = 1000
NBATCH = 256
THREAD_COUNTS = (1, 2, 4, 8)
WARMUP_STEPS = 200


@dataclass
class Result:
    label: str
    nthread: int
    fps: float
    wall_seconds: float


def _circular_leo_state() -> tuple[np.ndarray, np.ndarray]:
    """Return (R_eci, V_eci) for a circular LEO at ALT_KM, inclination."""
    a = R_EARTH + ALT_KM
    inc = np.deg2rad(INCLINATION_DEG)
    R_eci = np.array([a, 0.0, 0.0])
    speed = np.sqrt(GM_EARTH / a)
    V_eci = np.array([0.0, speed * np.cos(inc), speed * np.sin(inc)])
    return R_eci, V_eci


def _humanoid_orbit_xml(humanoid_xml_text: str) -> str:
    """Insert a <mjorbit> block enabling the full orbital environment."""
    block = (
        "\n  <mjorbit "
        'use_j2="true" use_drag="true" use_srp="true" '
        'use_magnetic="true" use_gravity_gradient="true">\n'
        '    <surface body="torso" cop="0 0 0" normal="1 0 0" area="0.6" '
        'drag_coeff="2.2" srp_coeff="1.5" use_drag="true" use_srp="true"/>\n'
        '    <magnetic_body body="torso" dipole="0 0 0.05"/>\n'
        "  </mjorbit>\n"
    )
    closing = humanoid_xml_text.rfind("</mujoco>")
    if closing < 0:
        raise RuntimeError("could not find </mujoco> in humanoid xml")
    return humanoid_xml_text[:closing] + block + humanoid_xml_text[closing:]


def _make_mujoco_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(HUMANOID_XML))


def _make_orbit_model() -> MjoModel:
    base_xml = HUMANOID_XML.read_text()
    orbit_xml = _humanoid_orbit_xml(base_xml)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "humanoid_orbit.xml"
        path.write_text(orbit_xml)
        return MjoModel.from_xml_path(str(path))


def _benchmark_mujoco(model: mujoco.MjModel, nthread: int) -> Result:
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value)
    data_list = [mujoco.MjData(model) for _ in range(max(nthread, 1))]
    initial_state = np.tile(np.zeros(nstate)[None, :], (NBATCH, 1))
    base = mujoco.MjData(model)
    mujoco.mj_resetData(model, base)
    mujoco.mj_getState(
        model, base, initial_state[0], mujoco.mjtState.mjSTATE_FULLPHYSICS.value
    )
    initial_state[:] = initial_state[0]

    state = np.empty((NBATCH, NSTEP, nstate))
    sensordata = np.empty((NBATCH, NSTEP, model.nsensordata))

    with mj_rollout.Rollout(nthread=nthread) as pool:
        # warmup
        pool.rollout(
            model,
            data_list,
            initial_state[: max(nthread, 1)],
            nstep=WARMUP_STEPS,
        )
        t0 = time.perf_counter()
        pool.rollout(
            model,
            data_list,
            initial_state,
            nstep=NSTEP,
            state=state,
            sensordata=sensordata,
        )
        elapsed = time.perf_counter() - t0

    total_steps = NBATCH * NSTEP
    return Result("MuJoCo", nthread, total_steps / elapsed, elapsed)


def _benchmark_orbit(model: MjoModel, nthread: int) -> Result:
    R_eci, V_eci = _circular_leo_state()
    data = MjoData(model, orbit=OrbitInit(R_eci=R_eci, V_eci=V_eci, t=0.0))
    mjo_forward(model, data)

    nstate = int(np.asarray(mjo_get_state(model, data)).shape[0])
    initial_state = np.tile(mjo_get_state(model, data)[None, :], (NBATCH, 1))

    state = np.empty((NBATCH, NSTEP, nstate))
    sensordata = np.empty((NBATCH, NSTEP, model.nsensordata))

    # warmup
    rollout(
        model,
        data,
        initial_state=initial_state[:nthread],
        nstep=WARMUP_STEPS,
        nthread=nthread,
    )
    t0 = time.perf_counter()
    rollout(
        model,
        data,
        initial_state=initial_state,
        nstep=NSTEP,
        nthread=nthread,
        state=state,
        sensordata=sensordata,
    )
    elapsed = time.perf_counter() - t0

    total_steps = NBATCH * NSTEP
    return Result("mjorbit", nthread, total_steps / elapsed, elapsed)


def _save_figure(results: list[Result]) -> Path:
    nthreads = sorted({r.nthread for r in results})
    by_label = {}
    for r in results:
        by_label.setdefault(r.label, {})[r.nthread] = r.fps

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    width = 0.4
    x = np.arange(len(nthreads))
    for offset, (label, color) in enumerate(
        [("MuJoCo", "#4c78a8"), ("mjorbit", "#f58518")]
    ):
        fps_per_n = [by_label.get(label, {}).get(n, 0.0) for n in nthreads]
        ax.bar(x + (offset - 0.5) * width, fps_per_n, width=width, label=label, color=color)
        for xi, fps in zip(x + (offset - 0.5) * width, fps_per_n):
            ax.text(xi, fps, f"{fps/1e3:.1f}k", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in nthreads])
    ax.set_xlabel("number of threads")
    ax.set_ylabel("frames per second")
    ax.set_title(
        f"Humanoid rollout throughput  (nbatch={NBATCH}, nstep={NSTEP}, dt={3e-3} s)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")

    path = FIGURE_DIR / "rollout_fps.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    mujoco_model = _make_mujoco_model()
    orbit_model = _make_orbit_model()

    print(f"humanoid: nq={mujoco_model.nq} nv={mujoco_model.nv} nu={mujoco_model.nu}")
    print(
        f"benchmark: nbatch={NBATCH} nstep={NSTEP} timestep={mujoco_model.opt.timestep}"
    )

    results: list[Result] = []
    for nthread in THREAD_COUNTS:
        mj_res = _benchmark_mujoco(mujoco_model, nthread)
        print(
            f"  MuJoCo       nthread={nthread}: "
            f"{mj_res.fps:>11,.0f} fps  ({mj_res.wall_seconds:.2f} s)"
        )
        mjo_res = _benchmark_orbit(orbit_model, nthread)
        print(
            f"  mjorbit nthread={nthread}: "
            f"{mjo_res.fps:>11,.0f} fps  ({mjo_res.wall_seconds:.2f} s)"
        )
        results.extend([mj_res, mjo_res])

    path = _save_figure(results)
    print(f"saved {path.relative_to(EXPERIMENT_DIR)}")


if __name__ == "__main__":
    main()
