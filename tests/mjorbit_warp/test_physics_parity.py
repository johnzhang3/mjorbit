"""CPU-vs-MJWarp physics parity tests for the orbit API."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")

import warp as wp

import mjorbit as mjo_cpu
import mjorbit_warp as mjo_warp
from mjorbit.constants import R_EARTH
from mjorbit.testdata import (
    FREE_BODY_XML,
    SPACECRAFT_ARM_XML,
    SPACECRAFT_BIMANUAL_PANELS_XML,
)
from tests.mjorbit._helpers import _xml_with_mjorbit as _cpu_xml_with_mjorbit
from tests.mjorbit.reference.orbit.elements import keplerian_to_cartesian

# CUDA-graph capture/replay needs a CUDA device; on a CPU-only Warp install the
# graph tests are skipped rather than failed.
_HAS_CUDA = any(getattr(d, "is_cuda", False) for d in wp.get_devices())

# MJWarp runs in fp32 by default while the CPU mjorbit backend runs in fp64,
# so cross-backend state divergence is dominated by accumulated fp32 round-off
# in MJWarp's solver. Empirically (no-torque free body, 50 mj_dt steps):
#   d(qvel_lin) ≈ 4.5e-7 per step → ~2.3e-5 at 50 steps, ~4.5e-5 at 100
#   d(qpos)     grows quadratically as the integral of velocity round-off
#   d(qvel_ang) stays at fp64 noise (~1e-9) when there is no torque
#   d(R_eci)    stays in fp64 (mjorbit overlay is fp64) — only the chief
#               feedback path picks up fp32 noise from MJWarp wrenches
# Tolerances are chosen to cover ~100 steps of round-off, not to mask physics.
STATE_ATOL = 1e-4
STATE_RTOL = 1e-4
DERIVED_ATOL = 1e-4


def _orbit_init(alt_km: float = 400.0) -> mjo_cpu.OrbitInit:
    a = R_EARTH + alt_km
    r_eci, v_eci = keplerian_to_cartesian(
        a=a,
        e=0.0,
        inc=np.deg2rad(51.6),
        raan=0.0,
        argp=0.0,
        nu=0.0,
    )
    return mjo_cpu.OrbitInit(R_eci=r_eci, V_eci=v_eci)


def _warp_orbit_init(init: mjo_cpu.OrbitInit) -> mjo_warp.OrbitInit:
    return mjo_warp.OrbitInit(
        R_eci=init.R_eci.copy(),
        V_eci=init.V_eci.copy(),
        t=init.t,
    )


def _make_pair(xml_path: str, *, orbit: mjo_cpu.OrbitInit | None = None, **kwargs: Any):
    defaults: dict[str, Any] = dict(
        mj_timestep=0.01,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    defaults.update(kwargs)

    orbit = _orbit_init() if orbit is None else orbit
    cpu_model = _cpu_model_from_xml_path(xml_path, **defaults)
    warp_model = mjo_warp.MjoModel.from_xml_path(xml_path, **defaults)
    cpu_data = cpu_model.make_data(orbit=orbit)
    warp_data = warp_model.make_data(orbit=_warp_orbit_init(orbit))
    return cpu_model, cpu_data, warp_model, warp_data


def _cpu_model_from_xml_path(xml_path: str, **kwargs: Any):
    mj_timestep = kwargs.pop("mj_timestep", 0.01)
    configured_xml = _cpu_xml_with_mjorbit(
        xml_path,
        orbit_dt=kwargs.pop("orbit_dt", None),
        use_j2=kwargs.pop("use_j2", False),
        use_drag=kwargs.pop("use_drag", False),
        use_srp=kwargs.pop("use_srp", False),
        use_magnetic=kwargs.pop("use_magnetic", False),
        use_gravity_gradient=kwargs.pop("use_gravity_gradient", True),
        surfaces=kwargs.pop("surfaces", ()),
        magnetic_bodies=kwargs.pop("magnetic_bodies", ()),
        reaction_wheels=kwargs.pop("reaction_wheels", ()),
        magnetorquers=kwargs.pop("magnetorquers", ()),
        thrusters=kwargs.pop("thrusters", ()),
        cmgs=kwargs.pop("cmgs", ()),
    )
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected CPU model option(s): {names}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / Path(xml_path).name
        path.write_text(configured_xml)
        return mjo_cpu.MjoModel.from_xml_path(str(path), mj_timestep=mj_timestep)


def _upload_warp_inputs(warp_model, warp_data) -> None:
    mjo_warp.mjo_upload(warp_model, warp_data, fields=("state", "inputs", "core"))


def _forward_and_pull_warp(warp_model, warp_data) -> None:
    mjo_warp.mjo_forward(warp_model, warp_data)
    mjo_warp.mjo_pull(warp_model, warp_data)


def _normalize_quat(qpos: Sequence[float] | np.ndarray) -> np.ndarray:
    out = np.asarray(qpos, dtype=float).copy()
    out[3:7] /= np.linalg.norm(out[3:7])
    return out


def _assert_close(actual, expected, *, atol: float = STATE_ATOL, rtol: float = STATE_RTOL) -> None:
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def _assert_single_world_state_matches(cpu_data, warp_data) -> None:
    _assert_close(warp_data.time, cpu_data.time, atol=1e-6)
    _assert_close(warp_data.qpos, cpu_data.qpos)
    _assert_close(warp_data.qvel, cpu_data.qvel)
    _assert_close(warp_data.xpos, cpu_data.xpos, atol=DERIVED_ATOL)
    _assert_close(warp_data.xquat, cpu_data.xquat, atol=DERIVED_ATOL)
    _assert_close(warp_data.xmat, cpu_data.xmat, atol=DERIVED_ATOL)
    _assert_close(warp_data.cvel, cpu_data.cvel, atol=DERIVED_ATOL)
    _assert_close(warp_data.qacc, cpu_data.qacc, atol=DERIVED_ATOL)


def test_free_body_step_matches_cpu_reference():
    cpu_model, cpu_data, warp_model, warp_data = _make_pair(FREE_BODY_XML)
    qpos = _normalize_quat([0.2, -0.1, 0.05, 0.98, 0.1, -0.15, 0.05])
    qvel = np.array([0.01, -0.02, 0.03, 0.04, -0.03, 0.02])
    cpu_data.qpos[:] = qpos
    warp_data.qpos[:] = qpos
    cpu_data.qvel[:] = qvel
    warp_data.qvel[:] = qvel
    _upload_warp_inputs(warp_model, warp_data)

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    _forward_and_pull_warp(warp_model, warp_data)
    _assert_single_world_state_matches(cpu_data, warp_data)

    for _ in range(50):
        mjo_cpu.mjo_step(cpu_model, cpu_data)
        mjo_warp.mjo_step(warp_model, warp_data)

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    _forward_and_pull_warp(warp_model, warp_data)
    _assert_single_world_state_matches(cpu_data, warp_data)
    _assert_close(warp_data.orbit.R_eci, cpu_data.orbit.R_eci, atol=1e-10)
    _assert_close(warp_data.orbit.V_eci, cpu_data.orbit.V_eci, atol=1e-10)


def test_articulated_position_actuator_matches_cpu_reference():
    cpu_model, cpu_data, warp_model, warp_data = _make_pair(SPACECRAFT_ARM_XML)
    qvel = np.array([0.03, -0.01, 0.02, 0.02, -0.015, 0.01, 0.0, 0.0])
    ctrl = np.array([0.4, -0.25])
    cpu_data.qpos[:] = 0.0
    warp_data.qpos[:] = 0.0
    cpu_data.qpos[3] = 1.0
    warp_data.qpos[3] = 1.0
    cpu_data.qvel[:] = qvel
    warp_data.qvel[:] = qvel
    cpu_data.ctrl[:] = ctrl
    warp_data.ctrl[:] = ctrl
    _upload_warp_inputs(warp_model, warp_data)

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    _forward_and_pull_warp(warp_model, warp_data)
    for _ in range(50):
        mjo_cpu.mjo_step(cpu_model, cpu_data)
        mjo_warp.mjo_step(warp_model, warp_data)

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    _forward_and_pull_warp(warp_model, warp_data)
    _assert_single_world_state_matches(cpu_data, warp_data)
    _assert_close(warp_data.ctrl, cpu_data.ctrl)
    _assert_close(warp_data.actuator_force, cpu_data.actuator_force)
    _assert_close(warp_data.qfrc_actuator, cpu_data.qfrc_actuator)


# TODO(sensors): re-add a Warp sensor parity test once GPU-native sensor
# kernels exist. The CPU sensor parity reference still lives at
# tests/mjorbit/test_sensors.py.


def test_orbit_actuator_coupling_matches_cpu_reference():
    reaction_wheels = [
        mjo_cpu.ReactionWheelSpec(
            body_name="spacecraft",
            axis_body=np.array([0.0, 0.0, 1.0]),
            inertia=0.01,
            torque_limit=0.02,
        )
    ]
    thrusters = [
        mjo_cpu.ThrusterSpec(
            body_name="spacecraft",
            position_body=np.array([0.1, 0.0, 0.0]),
            direction_body=np.array([0.0, 1.0, 0.0]),
            force_limit=10.0,
        )
    ]
    cpu_model, cpu_data, warp_model, warp_data = _make_pair(
        FREE_BODY_XML,
        reaction_wheels=reaction_wheels,
        thrusters=thrusters,
    )
    qpos = np.array([0.1, 0.2, -0.1, 1.0, 0.0, 0.0, 0.0])
    qvel = np.array([0.02, 0.01, -0.01, 0.01, 0.02, 0.03])
    cpu_data.qpos[:] = qpos
    warp_data.qpos[:] = qpos
    cpu_data.qvel[:] = qvel
    warp_data.qvel[:] = qvel
    cpu_data.actuators.rw_speed[0] = 1.5
    warp_data.actuators.rw_speed[0] = 1.5
    cpu_data.actuators.rw_torque_cmd[0] = 0.012
    warp_data.actuators.rw_torque_cmd[0] = 0.012
    cpu_data.actuators.thr_force_cmd[0] = 4.0
    warp_data.actuators.thr_force_cmd[0] = 4.0
    _upload_warp_inputs(warp_model, warp_data)

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    _forward_and_pull_warp(warp_model, warp_data)
    for _ in range(40):
        mjo_cpu.mjo_step(cpu_model, cpu_data)
        mjo_warp.mjo_step(warp_model, warp_data)

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    _forward_and_pull_warp(warp_model, warp_data)
    _assert_single_world_state_matches(cpu_data, warp_data)
    _assert_close(warp_data.xfrc_applied, cpu_data.xfrc_applied, atol=DERIVED_ATOL)
    _assert_close(warp_data.wrench_buffer, cpu_data.wrench_buffer, atol=DERIVED_ATOL)
    _assert_close(warp_data.orbit.R_eci, cpu_data.orbit.R_eci, atol=1e-7)
    _assert_close(warp_data.orbit.V_eci, cpu_data.orbit.V_eci, atol=1e-7)
    _assert_close(warp_data.actuators.rw_speed, cpu_data.actuators.rw_speed)
    _assert_close(warp_data.actuators.rw_momentum, cpu_data.actuators.rw_momentum)
    _assert_close(warp_data.actuators.thr_force_cmd, cpu_data.actuators.thr_force_cmd)


def test_device_resident_steps_match_synced_path_after_final_pull():
    reaction_wheels = [
        mjo_cpu.ReactionWheelSpec(
            body_name="spacecraft",
            axis_body=np.array([0.0, 0.0, 1.0]),
            inertia=0.01,
            torque_limit=0.02,
        )
    ]
    thrusters = [
        mjo_cpu.ThrusterSpec(
            body_name="spacecraft",
            position_body=np.array([0.1, 0.0, 0.0]),
            direction_body=np.array([0.0, 1.0, 0.0]),
            force_limit=10.0,
        )
    ]
    common: dict[str, Any] = dict(
        mj_timestep=0.01,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
        reaction_wheels=reaction_wheels,
        thrusters=thrusters,
    )
    orbit = _orbit_init()
    cpu_model = _cpu_model_from_xml_path(FREE_BODY_XML, **common)
    warp_model = mjo_warp.MjoModel.from_xml_path(FREE_BODY_XML, **common)
    cpu_data = cpu_model.make_data(orbit=orbit)
    warp_synced = warp_model.make_data(orbit=_warp_orbit_init(orbit))
    warp_device = warp_model.make_data(orbit=_warp_orbit_init(orbit))

    qpos = np.array([0.1, 0.2, -0.1, 1.0, 0.0, 0.0, 0.0])
    qvel = np.array([0.02, 0.01, -0.01, 0.01, 0.02, 0.03])
    for data in (cpu_data, warp_synced, warp_device):
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.actuators.rw_speed[0] = 1.5
        data.actuators.rw_torque_cmd[0] = 0.012
        data.actuators.thr_force_cmd[0] = 4.0
        if data is not cpu_data:
            _upload_warp_inputs(warp_model, data)

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    _forward_and_pull_warp(warp_model, warp_synced)
    mjo_warp.mjo_forward(warp_model, warp_device)
    stale_public_time = float(warp_device.time)

    for _ in range(60):
        mjo_cpu.mjo_step(cpu_model, cpu_data)
        mjo_warp.mjo_step(warp_model, warp_synced, sync=True)
        mjo_warp.mjo_step(warp_model, warp_device)

    assert float(warp_device.time) == stale_public_time

    mjo_warp.mjo_forward(warp_model, warp_device, sync=False)
    assert float(warp_device.time) == stale_public_time
    mjo_warp.mjo_pull(warp_model, warp_device)
    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    _forward_and_pull_warp(warp_model, warp_synced)

    _assert_single_world_state_matches(cpu_data, warp_device)
    _assert_close(warp_device.qpos, warp_synced.qpos)
    _assert_close(warp_device.qvel, warp_synced.qvel)
    _assert_close(warp_device.orbit.R_eci, warp_synced.orbit.R_eci, atol=1e-10)
    _assert_close(warp_device.orbit.V_eci, warp_synced.orbit.V_eci, atol=1e-10)
    _assert_close(warp_device.actuators.rw_speed, warp_synced.actuators.rw_speed)
    _assert_close(warp_device.actuators.rw_momentum, warp_synced.actuators.rw_momentum)
    _assert_close(warp_device.wrench_buffer, warp_synced.wrench_buffer, atol=DERIVED_ATOL)


def test_surface_magnetic_and_limited_rw_coupling_matches_cpu_reference():
    surfaces = [
        mjo_cpu.SurfaceSpec(
            body_name="spacecraft",
            center_of_pressure_body=np.array([0.0, 0.0, 0.2]),
            normal_body=np.array([0.0, 1.0, 0.0]),
            area=3.0,
            drag_coeff=2.0,
            srp_coeff=1.4,
        )
    ]
    magnetic_bodies = [
        mjo_cpu.MagneticBodySpec(
            body_name="spacecraft",
            dipole_body=np.array([0.02, -0.01, 0.03]),
        )
    ]
    reaction_wheels = [
        mjo_cpu.ReactionWheelSpec(
            body_name="spacecraft",
            axis_body=np.array([0.0, 0.0, 1.0]),
            inertia=0.02,
            speed_limit=1.0,
            torque_limit=0.01,
        )
    ]
    magnetorquers = [
        mjo_cpu.MagnetorquerSpec(
            body_name="spacecraft",
            axis_body=np.array([1.0, 0.0, 0.0]),
            dipole_limit=0.04,
        )
    ]
    cpu_model, cpu_data, warp_model, warp_data = _make_pair(
        FREE_BODY_XML,
        use_j2=True,
        use_drag=True,
        use_srp=True,
        use_magnetic=True,
        surfaces=surfaces,
        magnetic_bodies=magnetic_bodies,
        reaction_wheels=reaction_wheels,
        magnetorquers=magnetorquers,
    )

    qpos = _normalize_quat([0.05, -0.02, 0.1, 0.98, -0.1, 0.08, 0.05])
    qvel = np.array([0.03, 0.02, -0.01, 0.02, -0.03, 0.04])
    cpu_data.qpos[:] = qpos
    warp_data.qpos[:] = qpos
    cpu_data.qvel[:] = qvel
    warp_data.qvel[:] = qvel
    cpu_data.actuators.rw_speed[0] = 0.995
    warp_data.actuators.rw_speed[0] = 0.995
    cpu_data.actuators.rw_torque_cmd[0] = 0.03
    warp_data.actuators.rw_torque_cmd[0] = 0.03
    cpu_data.actuators.mtq_dipole_cmd[0] = 0.08
    warp_data.actuators.mtq_dipole_cmd[0] = 0.08
    _upload_warp_inputs(warp_model, warp_data)

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    _forward_and_pull_warp(warp_model, warp_data)
    _assert_close(warp_data.wrench_buffer, cpu_data.wrench_buffer, atol=1e-6)
    _assert_close(warp_data.xfrc_applied, cpu_data.xfrc_applied, atol=1e-6)

    for _ in range(20):
        mjo_cpu.mjo_step(cpu_model, cpu_data)
        mjo_warp.mjo_step(warp_model, warp_data)

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    _forward_and_pull_warp(warp_model, warp_data)
    _assert_single_world_state_matches(cpu_data, warp_data)
    _assert_close(warp_data.wrench_buffer, cpu_data.wrench_buffer, atol=1e-5)
    _assert_close(warp_data.actuators.rw_speed, cpu_data.actuators.rw_speed, atol=1e-8)
    _assert_close(warp_data.actuators.rw_momentum, cpu_data.actuators.rw_momentum, atol=1e-8)


def test_magnetorquer_command_applies_without_explicit_upload():
    """Documented public pattern (set cmd, then plain mjo_step/mjo_forward with
    NO explicit upload) must produce torque on the warp backend, matching the
    CPU backend. Regression for issue #10: before the command-input auto-sync,
    warp silently dropped the dipole command (device buffer stayed 0) so this
    asserted-nonzero torque was 0 and the test failed."""
    magnetorquers = [
        mjo_cpu.MagnetorquerSpec(
            body_name="spacecraft",
            axis_body=np.array([1.0, 0.0, 0.0]),
            dipole_limit=10.0,
        )
    ]
    cpu_model, cpu_data, warp_model, warp_data = _make_pair(
        FREE_BODY_XML,
        use_magnetic=True,
        use_gravity_gradient=False,  # isolate the magnetorquer torque
        magnetorquers=magnetorquers,
    )
    bid = cpu_model.body_id("spacecraft")

    # Documented pattern: set the command, then forward — no _upload_warp_inputs.
    cpu_data.actuators.mtq_dipole_cmd[0] = 10.0
    warp_data.actuators.mtq_dipole_cmd[0] = 10.0
    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    _forward_and_pull_warp(warp_model, warp_data)

    b_world = np.asarray(cpu_data.env.mag_field_eci)
    assert np.linalg.norm(b_world) > 0.0, "env magnetic field is zero; cannot test MTQ torque"
    assert np.linalg.norm(warp_data.wrench_buffer[bid, 3:]) > 1e-9, (
        "warp magnetorquer produced no torque without an explicit upload"
    )
    _assert_close(warp_data.wrench_buffer, cpu_data.wrench_buffer, atol=1e-6)

    # Clearing the command must also propagate through the auto-sync.
    warp_data.actuators.mtq_dipole_cmd[0] = 0.0
    _forward_and_pull_warp(warp_model, warp_data)
    np.testing.assert_allclose(warp_data.wrench_buffer[bid, 3:], 0.0, atol=1e-9)

    # End-to-end: plain mjo_step (no upload) must move the body rate in parity
    # with the CPU backend.
    cpu_data.actuators.mtq_dipole_cmd[0] = 10.0
    warp_data.actuators.mtq_dipole_cmd[0] = 10.0
    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    _forward_and_pull_warp(warp_model, warp_data)
    w0 = cpu_data.qvel[3:6].copy()
    for _ in range(50):
        mjo_cpu.mjo_step(cpu_model, cpu_data)
        mjo_warp.mjo_step(warp_model, warp_data)
    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    _forward_and_pull_warp(warp_model, warp_data)
    assert np.linalg.norm(cpu_data.qvel[3:6] - w0) > 1e-9, "CPU body rate did not respond to MTQ"
    _assert_close(warp_data.qvel, cpu_data.qvel)


def test_batched_warp_worlds_match_independent_cpu_runs():
    common: dict[str, Any] = dict(
        mj_timestep=0.01,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    cpu_model = _cpu_model_from_xml_path(FREE_BODY_XML, **common)
    warp_model = mjo_warp.MjoModel.from_xml_path(FREE_BODY_XML, **common)
    orbits = [_orbit_init(400.0), _orbit_init(500.0)]
    cpu_runs = [cpu_model.make_data(orbit=orbit) for orbit in orbits]
    warp_data = warp_model.make_data(
        orbit=[_warp_orbit_init(orbit) for orbit in orbits],
        nworld=2,
    )
    qposes = [
        np.array([0.1, 0.2, -0.1, 1.0, 0.0, 0.0, 0.0]),
        _normalize_quat([-0.2, 0.05, 0.3, 0.98, 0.1, 0.05, -0.1]),
    ]
    qvels = [
        np.array([0.02, 0.01, -0.01, 0.01, 0.02, 0.03]),
        np.array([-0.01, 0.04, 0.02, -0.02, 0.01, -0.03]),
    ]
    for world_id, cpu_data in enumerate(cpu_runs):
        cpu_data.qpos[:] = qposes[world_id]
        cpu_data.qvel[:] = qvels[world_id]
    warp_data.qpos[:] = qposes
    warp_data.qvel[:] = qvels
    _upload_warp_inputs(warp_model, warp_data)

    for cpu_data in cpu_runs:
        mjo_cpu.mjo_forward(cpu_model, cpu_data)
    _forward_and_pull_warp(warp_model, warp_data)

    for _ in range(25):
        for cpu_data in cpu_runs:
            mjo_cpu.mjo_step(cpu_model, cpu_data)
        mjo_warp.mjo_step(warp_model, warp_data)

    for cpu_data in cpu_runs:
        mjo_cpu.mjo_forward(cpu_model, cpu_data)
    _forward_and_pull_warp(warp_model, warp_data)

    for world_id, cpu_data in enumerate(cpu_runs):
        _assert_close(np.asarray(warp_data.time)[world_id], cpu_data.time, atol=1e-6)
        _assert_close(warp_data.qpos[world_id], cpu_data.qpos)
        _assert_close(warp_data.qvel[world_id], cpu_data.qvel)
        _assert_close(warp_data.xpos[world_id], cpu_data.xpos, atol=DERIVED_ATOL)
        _assert_close(warp_data.xmat[world_id], cpu_data.xmat, atol=DERIVED_ATOL)
        _assert_close(warp_data.cvel[world_id], cpu_data.cvel, atol=DERIVED_ATOL)
        _assert_close(warp_data.orbit.R_eci[world_id], cpu_data.orbit.R_eci, atol=1e-10)
        _assert_close(warp_data.orbit.V_eci[world_id], cpu_data.orbit.V_eci, atol=1e-10)


def test_batched_device_resident_steps_match_synced_path_after_final_pull():
    common: dict[str, Any] = dict(
        mj_timestep=0.01,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    warp_model = mjo_warp.MjoModel.from_xml_path(FREE_BODY_XML, **common)
    orbits = [_orbit_init(400.0), _orbit_init(500.0)]
    warp_synced = warp_model.make_data(
        orbit=[_warp_orbit_init(orbit) for orbit in orbits],
        nworld=2,
    )
    warp_device = warp_model.make_data(
        orbit=[_warp_orbit_init(orbit) for orbit in orbits],
        nworld=2,
    )
    qposes = np.stack(
        [
            np.array([0.1, 0.2, -0.1, 1.0, 0.0, 0.0, 0.0]),
            _normalize_quat([-0.2, 0.05, 0.3, 0.98, 0.1, 0.05, -0.1]),
        ],
        axis=0,
    )
    qvels = np.stack(
        [
            np.array([0.02, 0.01, -0.01, 0.01, 0.02, 0.03]),
            np.array([-0.01, 0.04, 0.02, -0.02, 0.01, -0.03]),
        ],
        axis=0,
    )
    for data in (warp_synced, warp_device):
        data.qpos[:] = qposes
        data.qvel[:] = qvels
        _upload_warp_inputs(warp_model, data)

    _forward_and_pull_warp(warp_model, warp_synced)
    mjo_warp.mjo_forward(warp_model, warp_device)
    stale_public_time = warp_device.time.copy()

    for _ in range(30):
        mjo_warp.mjo_step(warp_model, warp_synced, sync=True)
        mjo_warp.step(warp_model, warp_device)

    np.testing.assert_array_equal(warp_device.time, stale_public_time)

    mjo_warp.forward(warp_model, warp_device, sync=False)
    np.testing.assert_array_equal(warp_device.time, stale_public_time)
    mjo_warp.mjo_pull(warp_model, warp_device)
    _forward_and_pull_warp(warp_model, warp_synced)

    _assert_close(warp_device.time, warp_synced.time, atol=1e-6)
    _assert_close(warp_device.qpos, warp_synced.qpos)
    _assert_close(warp_device.qvel, warp_synced.qvel)
    _assert_close(warp_device.xpos, warp_synced.xpos, atol=DERIVED_ATOL)
    _assert_close(warp_device.xmat, warp_synced.xmat, atol=DERIVED_ATOL)
    _assert_close(warp_device.cvel, warp_synced.cvel, atol=DERIVED_ATOL)
    _assert_close(warp_device.orbit.R_eci, warp_synced.orbit.R_eci, atol=1e-10)
    _assert_close(warp_device.orbit.V_eci, warp_synced.orbit.V_eci, atol=1e-10)


def test_initial_contact_dynamics_match_cpu_reference(tmp_path):
    xml_path = tmp_path / "contact_parity.xml"
    xml_path.write_text(
        """\
<mujoco model="contact_parity">
  <option timestep="0.001" gravity="0 0 0" iterations="100" tolerance="1e-12">
    <flag contact="enable"/>
  </option>
  <default>
    <geom condim="1" solref="0.02 1" solimp="0.9 0.95 0.001"/>
  </default>
  <worldbody>
    <body name="a" pos="-0.25 0 0">
      <freejoint/>
      <geom type="sphere" size="0.3" mass="50"/>
    </body>
    <body name="b" pos="0.25 0 0">
      <freejoint/>
      <geom type="sphere" size="0.3" mass="50"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    cpu_model, cpu_data, warp_model, warp_data = _make_pair(
        str(xml_path),
        mj_timestep=0.001,
    )

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    _forward_and_pull_warp(warp_model, warp_data)
    assert cpu_data.ncon == 1
    assert warp_data.ncon == 1
    _assert_close(warp_data.qacc, cpu_data.qacc, atol=1e-4)

    for _ in range(20):
        mjo_cpu.mjo_step(cpu_model, cpu_data)
        mjo_warp.mjo_step(warp_model, warp_data)

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    _forward_and_pull_warp(warp_model, warp_data)
    assert cpu_data.ncon == warp_data.ncon == 1
    # MJWarp's fp32 contact solver leaves a ~1e-5 noise floor in qvel where the
    # fp64 CPU MuJoCo resolves to numerical zero — this is fp32 round-off in
    # the constraint solve, not a physics divergence.
    _assert_close(warp_data.qpos, cpu_data.qpos, atol=1e-4)
    _assert_close(warp_data.qvel, cpu_data.qvel, atol=1e-4)
    # The Warp-side contact_force(0) helper is exercised separately; we don't
    # call mj_contactForce on the CPU MjoData here because the public API
    # deliberately hides the raw mj_model/mj_data handles
    # (src/mjorbit/data.py:53).
    assert np.all(np.isfinite(warp_data.contact_force(0)))


def test_multirate_orbit_step_matches_cpu_reference():
    """orbit_dt > mj_dt: chief follows averaged-feedback schedule (CPU parity)."""
    cpu_model, cpu_data, warp_model, warp_data = _make_pair(
        FREE_BODY_XML,
        use_j2=True,
        orbit_dt=0.05,  # 5x mj_timestep=0.01
    )
    qpos = _normalize_quat([0.2, -0.1, 0.05, 0.98, 0.1, -0.15, 0.05])
    qvel = np.array([0.01, -0.02, 0.03, 0.04, -0.03, 0.02])
    cpu_data.qpos[:] = qpos
    warp_data.qpos[:] = qpos
    cpu_data.qvel[:] = qvel
    warp_data.qvel[:] = qvel
    _upload_warp_inputs(warp_model, warp_data)

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    _forward_and_pull_warp(warp_model, warp_data)

    # Run long enough that several orbit_dt segments elapse and the averaged
    # feedback path is exercised (mj_dt=0.01, orbit_dt=0.05 → 5 mj steps per
    # segment, 100 mj steps = 20 segments).
    for _ in range(100):
        mjo_cpu.mjo_step(cpu_model, cpu_data)
        mjo_warp.mjo_step(warp_model, warp_data)

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    _forward_and_pull_warp(warp_model, warp_data)
    _assert_close(warp_data.orbit.R_eci, cpu_data.orbit.R_eci, atol=1e-5, rtol=1e-7)
    _assert_close(warp_data.orbit.V_eci, cpu_data.orbit.V_eci, atol=1e-6, rtol=1e-5)


def test_atmosphere_config_propagates_to_warp_device_core():
    """CentralBodySpec atmosphere fields must reach the GPU density kernel."""
    from mjorbit.spec import CentralBodySpec, MjoSpec
    from mjorbit_warp.core_gpu import make_device_core_model

    # Default model: defaults from mjorbit.spec.CentralBodySpec.
    default_model = mjo_warp.MjoModel.from_xml_path(
        FREE_BODY_XML,
        mj_timestep=0.01,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    default_core = make_device_core_model(default_model.host_model)
    assert default_core.atm_h0_km == pytest.approx(400.0)
    assert default_core.atm_rho0 == pytest.approx(2.62e-13)
    assert default_core.atm_h_scale_km == pytest.approx(58.2)

    # Build a model with a custom CentralBodySpec by mutating the spec before
    # compile. The atmosphere_* fields are baked in at compile time, so
    # mutating a compiled model's central_body has no effect — the spec is
    # the right place to override.
    spec = MjoSpec.from_xml_path(FREE_BODY_XML)
    custom_central = CentralBodySpec()
    custom_central.atmosphere_scale_height = 30.0
    custom_central.atmosphere_h0 = 350.0
    spec.mjorbit.central_body = custom_central
    custom_host = spec.compile(mj_timestep=0.01)
    custom_core = make_device_core_model(custom_host)
    assert custom_core.atm_h_scale_km == pytest.approx(30.0)
    assert custom_core.atm_h0_km == pytest.approx(350.0)


def test_bimanual_panels_step_matches_cpu_reference():
    """Full coupled parity for the bundled dual-arm bimanual+panels model.

    This is the model ``examples/banner_viewer_gpu.py`` simulates on the GPU:
    14 bodies, an articulated dual-arm tree, 4 position actuators, the
    ``implicitfast`` integrator, and gravity-gradient torque ON. None of those
    were exercised together in the existing parity suite (free body / 2-joint
    arm). The XML self-describes its ``<mjorbit>`` block (J2/drag/SRP/magnetic
    off, gravity_gradient on), so both backends load it directly with no
    injected overlay — the same path the example and benchmarks use, and a
    regression guard for the keep-XML ``use_*``/``mj_timestep`` defaults.

    A tilted initial attitude plus chief-offset arms make the gravity-gradient
    torque measurable, so an accidental gravity_gradient-off (or wrong frame)
    on either backend would diverge the attitude and fail the xquat check.
    """
    orbit = _orbit_init(600.0)
    cpu_model = mjo_cpu.MjoModel.from_xml_path(SPACECRAFT_BIMANUAL_PANELS_XML)
    warp_model = mjo_warp.MjoModel.from_xml_path(SPACECRAFT_BIMANUAL_PANELS_XML)
    cpu_data = cpu_model.make_data(orbit=orbit)
    warp_data = warp_model.make_data(orbit=_warp_orbit_init(orbit))

    nq, nv = int(cpu_model.nq), int(cpu_model.nv)
    quat = np.array([0.98, 0.1, -0.15, 0.05])
    quat /= np.linalg.norm(quat)
    qpos = np.zeros(nq)
    qpos[3:7] = quat
    qpos[7:11] = np.array([-1.1, -0.9, 1.1, 0.9])  # bimanual arm stance
    qvel = np.zeros(nv)
    qvel[3:6] = np.array([0.02, -0.015, 0.01])  # body-frame spin
    qvel[6:10] = np.array([0.05, -0.03, 0.04, -0.02])
    ctrl = np.array([-1.0, -0.8, 1.0, 0.8])

    for data in (cpu_data, warp_data):
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.ctrl[:] = ctrl
    _upload_warp_inputs(warp_model, warp_data)

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    _forward_and_pull_warp(warp_model, warp_data)
    _assert_single_world_state_matches(cpu_data, warp_data)

    for _ in range(100):
        mjo_cpu.mjo_step(cpu_model, cpu_data)
        mjo_warp.mjo_step(warp_model, warp_data)

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    _forward_and_pull_warp(warp_model, warp_data)
    _assert_single_world_state_matches(cpu_data, warp_data)
    _assert_close(warp_data.ctrl, cpu_data.ctrl)
    _assert_close(warp_data.actuator_force, cpu_data.actuator_force)


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA-graph capture requires a CUDA device")
def test_cuda_graph_capture_replay_matches_uncaptured_step():
    """Replaying a CUDA-graph-captured ``mjo_step`` must match the plain loop.

    This is the exact hot-loop construct every GPU example/benchmark depends on
    (``wp.ScopedCapture`` around ``mjo_step``, then ``wp.capture_launch`` per
    frame, with ``ctrl`` re-uploaded between launches) and it was previously
    unverified. Two batched datas start from identical state and are warmed up
    identically; one then advances with uncaptured ``mjo_step`` while the other
    replays a captured graph. Both receive the SAME per-step, per-world ``ctrl``
    sequence via ``mjo_upload(fields=("ctrl",))``. If capture were unsafe, or if
    mid-loop ctrl uploads did not reach the captured graph, the replayed worlds
    would diverge well beyond fp32 round-off.
    """
    orbit = _orbit_init(500.0)
    model = mjo_warp.MjoModel.from_xml_path(
        SPACECRAFT_ARM_XML,
        mj_timestep=0.01,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    nworld = 4
    ref = model.make_data(orbit=_warp_orbit_init(orbit), nworld=nworld)
    cap = model.make_data(orbit=_warp_orbit_init(orbit), nworld=nworld)

    base_ctrl = np.tile(np.array([0.3, -0.2]), (nworld, 1))
    for data in (ref, cap):
        data.qpos[:] = 0.0
        data.qpos[:, 3] = 1.0  # identity quaternion (w,x,y,z)
        data.ctrl[:] = base_ctrl
        _upload_warp_inputs(model, data)
        mjo_warp.mjo_forward(model, data)

    # Identical warmup so both datas sit at the same state before divergence.
    for _ in range(4):
        mjo_warp.mjo_step(model, ref)
        mjo_warp.mjo_step(model, cap)
    wp.synchronize()

    # Capture records the kernel sequence without executing it; capture_launch
    # replays it. The graph is bound to ``cap``'s device buffers, so re-uploaded
    # ctrl lands in the same arrays the graph reads.
    with wp.ScopedCapture() as capture:
        mjo_warp.mjo_step(model, cap)
    graph = capture.graph
    wp.synchronize()

    n_run = 20
    for k in range(n_run):
        # Distinct per-step, per-world control so a dropped upload is visible.
        delta = 0.02 * np.sin(0.3 * k + np.arange(nworld))[:, None]
        new_ctrl = base_ctrl + delta * np.array([1.0, -1.0])
        ref.ctrl[:] = new_ctrl
        mjo_warp.mjo_upload(model, ref, fields=("ctrl",))
        mjo_warp.mjo_step(model, ref)

        cap.ctrl[:] = new_ctrl
        mjo_warp.mjo_upload(model, cap, fields=("ctrl",))
        wp.capture_launch(graph)
    wp.synchronize()

    mjo_warp.mjo_pull(model, ref)
    mjo_warp.mjo_pull(model, cap)

    # Same kernels on the same device with the same inputs: agreement is at the
    # fp32 floor, far tighter than the CPU/warp cross-backend tolerance.
    np.testing.assert_allclose(np.asarray(cap.qpos), np.asarray(ref.qpos), atol=1e-5, rtol=0.0)
    np.testing.assert_allclose(np.asarray(cap.qvel), np.asarray(ref.qvel), atol=1e-5, rtol=0.0)
    np.testing.assert_allclose(np.asarray(cap.xpos), np.asarray(ref.xpos), atol=1e-5, rtol=0.0)
    np.testing.assert_allclose(np.asarray(cap.xquat), np.asarray(ref.xquat), atol=1e-5, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray(cap.orbit.R_eci), np.asarray(ref.orbit.R_eci), atol=1e-6, rtol=0.0
    )


def test_central_body_config_propagates_to_warp_device_core():
    """radius / magnetic_b0 / magnetic_axis must reach the device core.

    Before this fix the warp environment kernels hardcoded Earth's R_EARTH,
    B0_EARTH, and dipole axis (0, 0, -1), so a non-default CentralBodySpec
    silently produced Earth physics on the GPU (wrong eclipse radius, drag
    altitude reference, dipole magnitude, and dipole direction). The axis is
    normalized at upload time, mirroring the CPU dipole_field_eci.
    """
    from mjorbit.spec import CentralBodySpec, MjoSpec
    from mjorbit_warp.core_gpu import make_device_core_model

    default_core = make_device_core_model(
        mjo_warp.MjoModel.from_xml_path(FREE_BODY_XML, mj_timestep=0.01).host_model
    )
    assert default_core.radius_km == pytest.approx(R_EARTH)

    spec = MjoSpec.from_xml_path(FREE_BODY_XML)
    custom = CentralBodySpec()
    custom.radius = 6000.0
    custom.magnetic_b0 = 5.0e-5
    custom.magnetic_axis = (0.2, -0.1, -1.0)  # tilted and intentionally non-unit
    spec.mjorbit.central_body = custom
    core = make_device_core_model(spec.compile(mj_timestep=0.01))

    assert core.radius_km == pytest.approx(6000.0)
    assert core.magnetic_b0 == pytest.approx(5.0e-5)
    axis = np.array([core.magnetic_axis[0], core.magnetic_axis[1], core.magnetic_axis[2]])
    expected = np.array([0.2, -0.1, -1.0])
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(axis, expected, atol=1e-6)


def test_custom_central_body_step_matches_cpu_reference():
    """CPU/warp parity with a NON-default central body + drag + magnetic torque.

    A custom radius shifts the drag/eclipse altitude reference and the dipole
    ``radius_ratio``; a custom ``magnetic_b0`` and tilted axis change the
    B-field magnitude and direction. With the Earth constants formerly hardcoded
    in the warp kernels, the GPU wrenches would diverge from the CPU reference
    here, so this is the regression guard for that fix. Both backends are built
    from one shared spec, guaranteeing identical configuration.
    """
    from mjorbit.spec import CentralBodySpec, MjoSpec

    spec = MjoSpec.from_xml_path(FREE_BODY_XML)
    custom = CentralBodySpec()
    custom.radius = 6000.0
    custom.magnetic_b0 = 5.0e-5
    custom.magnetic_axis = (0.2, -0.1, -1.0)
    # Atmosphere referenced near the test altitude so drag is well above the
    # fp32 floor; with the radius bug the warp altitude would be wrong by
    # (R_EARTH - 6000) km and the density off by orders of magnitude.
    custom.atmosphere_h0 = 800.0
    custom.atmosphere_rho0 = 1.0e-8
    custom.atmosphere_scale_height = 58.2
    spec.mjorbit.central_body = custom
    spec.mjorbit.use_drag = True
    spec.mjorbit.use_srp = True
    spec.mjorbit.use_magnetic = True
    spec.mjorbit.use_gravity_gradient = False
    spec.mjorbit.add_surface(
        mjo_cpu.SurfaceSpec(
            body_name="spacecraft",
            center_of_pressure_body=np.array([0.0, 0.0, 0.0]),
            normal_body=np.array([1.0, 0.0, 0.0]),
            area=5.0,
        )
    )
    spec.mjorbit.add_magnetic_body(
        mjo_cpu.MagneticBodySpec(
            body_name="spacecraft",
            dipole_body=np.array([0.6, -0.3, 0.4]),
        )
    )
    cpu_model = spec.compile(mj_timestep=0.01)
    warp_model = mjo_warp.MjoModel.from_host_model(cpu_model)

    orbit = _orbit_init(450.0)
    cpu_data = cpu_model.make_data(orbit=orbit)
    warp_data = warp_model.make_data(orbit=_warp_orbit_init(orbit))

    qpos = _normalize_quat([0.1, 0.2, -0.1, 0.98, 0.1, -0.15, 0.05])
    qvel = np.array([0.02, 0.01, -0.01, 0.01, 0.02, 0.03])
    cpu_data.qpos[:] = qpos
    warp_data.qpos[:] = qpos
    cpu_data.qvel[:] = qvel
    warp_data.qvel[:] = qvel
    _upload_warp_inputs(warp_model, warp_data)

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    _forward_and_pull_warp(warp_model, warp_data)
    # Initial wrench parity isolates the environment kernels (drag + dipole)
    # from any integration drift.
    _assert_close(warp_data.wrench_buffer, cpu_data.wrench_buffer, atol=DERIVED_ATOL)

    for _ in range(50):
        mjo_cpu.mjo_step(cpu_model, cpu_data)
        mjo_warp.mjo_step(warp_model, warp_data)

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    _forward_and_pull_warp(warp_model, warp_data)
    _assert_single_world_state_matches(cpu_data, warp_data)
    _assert_close(warp_data.wrench_buffer, cpu_data.wrench_buffer, atol=DERIVED_ATOL)


def test_warp_rejects_cmg_models_loudly():
    """CMGs are not implemented on the device core, so a CMG-equipped model must
    raise when uploaded to MJWarp rather than silently dropping the CMG
    gyroscopic/command torque and freezing the gimbal angle (which would
    diverge from the CPU backend with no error)."""
    from mjorbit.spec import MjoSpec

    spec = MjoSpec.from_xml_path(FREE_BODY_XML)
    spec.mjorbit.add_cmg(
        mjo_cpu.ControlMomentGyroSpec(
            body_name="spacecraft",
            gimbal_axis_body=np.array([0.0, 0.0, 1.0]),
            spin_axis_body_0=np.array([1.0, 0.0, 0.0]),  # orthogonal to gimbal axis
            rotor_momentum=0.05,
        )
    )
    cpu_model = spec.compile(mj_timestep=0.01)

    with pytest.raises(NotImplementedError, match="control moment gyro"):
        mjo_warp.MjoModel.from_host_model(cpu_model)
