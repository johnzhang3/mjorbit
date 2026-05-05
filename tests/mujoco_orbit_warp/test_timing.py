"""Opt-in timing tests for MJWarp versus the orbit-aware MJWarp wrapper."""

from __future__ import annotations

import os
import statistics
import time

import mujoco
import numpy as np
import pytest

pytest.importorskip("mujoco_warp")
pytest.importorskip("warp")

import mujoco_warp as mjw
import warp as wp

import mujoco_orbit_warp as mjow
from mujoco_orbit.constants import R_EARTH
from mujoco_orbit.testdata import HUMANOID_XML
from tests.mujoco_orbit.reference.orbit.elements import keplerian_to_cartesian


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _orbit_init() -> mjow.OrbitInit:
    r_eci, v_eci = keplerian_to_cartesian(
        a=R_EARTH + 400.0,
        e=0.0,
        inc=np.deg2rad(51.6),
        raan=0.0,
        argp=0.0,
        nu=0.0,
    )
    return mjow.OrbitInit(R_eci=r_eci, V_eci=v_eci)


def _time_steps(step_fn, steps: int) -> float:
    wp.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        step_fn()
    wp.synchronize()
    return (time.perf_counter() - start) / steps


def test_humanoid_zero_g_orbit_warp_overhead_benchmark():
    """Compare orbit-aware MJWarp stepping to raw MJWarp on MuJoCo's humanoid.

    This is skipped by default because timing assertions are hardware-sensitive.
    Run with:

        MJO_WARP_TIMING=1 pixi run -e warp pytest tests/mujoco_orbit_warp/test_timing.py -q -s

    Optional environment knobs:
      MJO_WARP_TIMING_NWORLD, MJO_WARP_TIMING_STEPS,
      MJO_WARP_TIMING_WARMUP, MJO_WARP_TIMING_REPEATS,
      MJO_WARP_TIMING_MAX_OVERHEAD_PCT.
    """
    if os.environ.get("MJO_WARP_TIMING") != "1":
        pytest.skip("set MJO_WARP_TIMING=1 to run GPU timing benchmark")
    if not wp.get_device().is_cuda:
        pytest.skip("GPU timing benchmark requires a CUDA Warp device")

    nworld = _env_int("MJO_WARP_TIMING_NWORLD", 128)
    steps = _env_int("MJO_WARP_TIMING_STEPS", 200)
    warmup = _env_int("MJO_WARP_TIMING_WARMUP", 25)
    repeats = _env_int("MJO_WARP_TIMING_REPEATS", 5)

    raw_mjm = mujoco.MjModel.from_xml_path(HUMANOID_XML)
    raw_mjm.opt.gravity[:] = 0.0
    raw_mjd = mujoco.MjData(raw_mjm)
    mujoco.mj_forward(raw_mjm, raw_mjd)
    raw_m = mjw.put_model(raw_mjm)
    raw_d = mjw.put_data(raw_mjm, raw_mjd, nworld=nworld)

    orbit_m = mjow.MjoModel.from_xml_path(
        HUMANOID_XML,
        mj_timestep=None,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    orbit_d = orbit_m.make_data(orbit=_orbit_init(), nworld=nworld)

    for _ in range(warmup):
        mjw.step(raw_m, raw_d)
        mjow.step(orbit_m, orbit_d)
    wp.synchronize()

    raw_times = []
    orbit_times = []
    for _ in range(repeats):
        raw_times.append(_time_steps(lambda: mjw.step(raw_m, raw_d), steps))
        orbit_times.append(_time_steps(lambda: mjow.step(orbit_m, orbit_d), steps))

    raw_median = statistics.median(raw_times)
    orbit_median = statistics.median(orbit_times)
    overhead = orbit_median - raw_median
    overhead_pct = (overhead / raw_median) * 100.0

    print(
        "\nMJWarp humanoid zero-g timing "
        f"(nworld={nworld}, steps={steps}, warmup={warmup}, repeats={repeats})"
    )
    print(f"  raw_mjwarp:    {raw_median * 1e3:.4f} ms/step")
    print(f"  mjorbit_warp:  {orbit_median * 1e3:.4f} ms/step")
    print(f"  overhead:      {overhead * 1e3:.4f} ms/step ({overhead_pct:.2f}%)")
    print(f"  raw samples:   {[round(t * 1e3, 4) for t in raw_times]} ms/step")
    print(f"  orbit samples: {[round(t * 1e3, 4) for t in orbit_times]} ms/step")

    assert raw_median > 0.0
    assert orbit_median > 0.0

    max_overhead_pct = os.environ.get("MJO_WARP_TIMING_MAX_OVERHEAD_PCT")
    if max_overhead_pct is not None:
        assert overhead_pct <= float(max_overhead_pct)
