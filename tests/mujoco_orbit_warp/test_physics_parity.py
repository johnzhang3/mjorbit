"""CPU-vs-MJWarp physics parity tests for the orbit API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mujoco
import numpy as np
import pytest

pytest.importorskip("mujoco_warp")

import mujoco_orbit as mjo_cpu
import mujoco_orbit_warp as mjo_warp
from mujoco_orbit.constants import R_EARTH
from mujoco_orbit.orbit.elements import keplerian_to_cartesian
from mujoco_orbit.testdata import FREE_BODY_SENSORS_XML, FREE_BODY_XML, SPACECRAFT_ARM_XML

STATE_ATOL = 2e-5
STATE_RTOL = 2e-5
DERIVED_ATOL = 5e-5


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
    cpu_model = mjo_cpu.MjoModel.from_xml_path(xml_path, **defaults)
    warp_model = mjo_warp.MjoModel.from_xml_path(xml_path, **defaults)
    cpu_data = cpu_model.make_data(orbit=orbit)
    warp_data = warp_model.make_data(orbit=_warp_orbit_init(orbit))
    return cpu_model, cpu_data, warp_model, warp_data


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

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    mjo_warp.mjo_forward(warp_model, warp_data)
    _assert_single_world_state_matches(cpu_data, warp_data)

    for _ in range(50):
        mjo_cpu.mjo_step(cpu_model, cpu_data)
        mjo_warp.mjo_step(warp_model, warp_data)

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    mjo_warp.mjo_forward(warp_model, warp_data)
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

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    mjo_warp.mjo_forward(warp_model, warp_data)
    for _ in range(50):
        mjo_cpu.mjo_step(cpu_model, cpu_data)
        mjo_warp.mjo_step(warp_model, warp_data)

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    mjo_warp.mjo_forward(warp_model, warp_data)
    _assert_single_world_state_matches(cpu_data, warp_data)
    _assert_close(warp_data.ctrl, cpu_data.ctrl)
    _assert_close(warp_data.actuator_force, cpu_data.actuator_force)
    _assert_close(warp_data.qfrc_actuator, cpu_data.qfrc_actuator)


def test_sensor_truth_matches_cpu_reference():
    cpu_model, cpu_data, warp_model, warp_data = _make_pair(
        FREE_BODY_SENSORS_XML,
        use_magnetic=True,
    )
    qpos = _normalize_quat([0.2, -0.1, 0.05, 0.98, 0.1, -0.15, 0.05])
    qvel = np.array([0.01, -0.02, 0.03, 0.04, -0.03, 0.02])
    cpu_data.qpos[:] = qpos
    warp_data.qpos[:] = qpos
    cpu_data.qvel[:] = qvel
    warp_data.qvel[:] = qvel

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    mjo_warp.mjo_forward(warp_model, warp_data)

    _assert_close(warp_data.sensordata, cpu_data.sensordata, atol=1e-6)
    for name in (
        "gyro_body",
        "acc_body",
        "mag_body",
        "mag_rotated",
        "orbit_sun_body",
        "orbit_horizon_body",
        "orbit_star_body",
    ):
        _assert_close(
            warp_data.sensors.measure(name, noisy=False),
            cpu_data.sensors.measure(name, noisy=False),
            atol=1e-6,
        )


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

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    mjo_warp.mjo_forward(warp_model, warp_data)
    for _ in range(40):
        mjo_cpu.mjo_step(cpu_model, cpu_data)
        mjo_warp.mjo_step(warp_model, warp_data)

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    mjo_warp.mjo_forward(warp_model, warp_data)
    _assert_single_world_state_matches(cpu_data, warp_data)
    _assert_close(warp_data.xfrc_applied, cpu_data.xfrc_applied, atol=DERIVED_ATOL)
    _assert_close(warp_data.wrench_buffer, cpu_data.wrench_buffer, atol=DERIVED_ATOL)
    _assert_close(warp_data.orbit.R_eci, cpu_data.orbit.R_eci, atol=1e-7)
    _assert_close(warp_data.orbit.V_eci, cpu_data.orbit.V_eci, atol=1e-7)
    _assert_close(warp_data.actuators.rw_speed, cpu_data.actuators.rw_speed)
    _assert_close(warp_data.actuators.rw_momentum, cpu_data.actuators.rw_momentum)
    _assert_close(warp_data.actuators.thr_force_cmd, cpu_data.actuators.thr_force_cmd)


def test_batched_warp_worlds_match_independent_cpu_runs():
    common: dict[str, Any] = dict(
        mj_timestep=0.01,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    cpu_model = mjo_cpu.MjoModel.from_xml_path(FREE_BODY_XML, **common)
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

    for cpu_data in cpu_runs:
        mjo_cpu.mjo_forward(cpu_model, cpu_data)
    mjo_warp.mjo_forward(warp_model, warp_data)

    for _ in range(25):
        for cpu_data in cpu_runs:
            mjo_cpu.mjo_step(cpu_model, cpu_data)
        mjo_warp.mjo_step(warp_model, warp_data)

    for cpu_data in cpu_runs:
        mjo_cpu.mjo_forward(cpu_model, cpu_data)
    mjo_warp.mjo_forward(warp_model, warp_data)

    for world_id, cpu_data in enumerate(cpu_runs):
        _assert_close(np.asarray(warp_data.time)[world_id], cpu_data.time, atol=1e-6)
        _assert_close(warp_data.qpos[world_id], cpu_data.qpos)
        _assert_close(warp_data.qvel[world_id], cpu_data.qvel)
        _assert_close(warp_data.xpos[world_id], cpu_data.xpos, atol=DERIVED_ATOL)
        _assert_close(warp_data.xmat[world_id], cpu_data.xmat, atol=DERIVED_ATOL)
        _assert_close(warp_data.cvel[world_id], cpu_data.cvel, atol=DERIVED_ATOL)
        _assert_close(warp_data.orbit.R_eci[world_id], cpu_data.orbit.R_eci, atol=1e-10)
        _assert_close(warp_data.orbit.V_eci[world_id], cpu_data.orbit.V_eci, atol=1e-10)


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
    mjo_warp.mjo_forward(warp_model, warp_data)
    assert cpu_data.ncon == 1
    assert warp_data.ncon == 1
    _assert_close(warp_data.qacc, cpu_data.qacc, atol=1e-4)

    for _ in range(20):
        mjo_cpu.mjo_step(cpu_model, cpu_data)
        mjo_warp.mjo_step(warp_model, warp_data)

    mjo_cpu.mjo_forward(cpu_model, cpu_data)
    mjo_warp.mjo_forward(warp_model, warp_data)
    assert cpu_data.ncon == warp_data.ncon == 1
    _assert_close(warp_data.qpos, cpu_data.qpos, atol=1e-5)
    _assert_close(warp_data.qvel, cpu_data.qvel, atol=1e-5)
    cpu_force = np.zeros(6)
    mj_contact_force = getattr(mujoco, "mj_contactForce")
    mj_contact_force(cpu_model.mj_model, cpu_data.mj_data, 0, cpu_force)
    _assert_close(warp_data.contact_force(0), cpu_force, atol=1e-3)
