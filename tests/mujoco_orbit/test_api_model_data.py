"""Direct tests for the MuJoCo-style ``MjoModel`` / ``MjoData`` API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mujoco_orbit import (
    MjoData,
    MjoModel,
    OrbitInit,
    mjo_forward,
    mjo_step,
)
from mujoco_orbit.constants import R_EARTH
from mujoco_orbit.testdata import FREE_BODY_SENSORS_XML, FREE_BODY_XML
from tests.mujoco_orbit.reference.orbit.elements import keplerian_to_cartesian


def _orbit_init(alt_km: float = 400.0) -> OrbitInit:
    a = R_EARTH + alt_km
    R_eci, V_eci = keplerian_to_cartesian(
        a=a, e=0.0, inc=np.deg2rad(51.6), raan=0.0, argp=0.0, nu=0.0
    )
    return OrbitInit(R_eci=R_eci, V_eci=V_eci)


def _xml_with_mjorbit(
    tmp_path: Path,
    source: str,
    *,
    name: str = "model.xml",
    attrs: str = "",
    children: str = "",
) -> str:
    text = Path(source).read_text()
    attr_text = (
        f'plugin_body="spacecraft" use_j2="false" use_drag="false" '
        f'use_srp="false" use_magnetic="false" {attrs}'
    )
    block = f"\n  <mjorbit {attr_text}>{children}</mjorbit>\n"
    path = tmp_path / name
    path.write_text(text.replace("</mujoco>", block + "</mujoco>"))
    return str(path)


def test_model_and_data_expose_mujoco_fields(tmp_path: Path):
    xml_path = _xml_with_mjorbit(tmp_path, FREE_BODY_XML)
    model = MjoModel.from_xml_path(xml_path, mj_timestep=0.01)
    data = MjoData(model, orbit=_orbit_init())

    assert model.nbody >= 2
    assert model.nplugin == 1
    with pytest.raises(AttributeError):
        _ = model.mj_model
    with pytest.raises(AttributeError):
        _ = data.mj_data
    assert model.backend == "cpu"
    np.testing.assert_allclose(model.opt.gravity, [0.0, 0.0, 0.0])
    assert data.backend == "cpu"
    assert data.nworld == 1
    assert data.qpos.shape[0] == model.nq
    assert data.qvel.shape[0] == model.nv
    assert data.wrench_buffer.shape == (model.nbody, 6)
    assert data.orbit.t == 0.0


def test_model_make_data_matches_direct_constructor(tmp_path: Path):
    xml_path = _xml_with_mjorbit(tmp_path, FREE_BODY_XML)
    model = MjoModel.from_xml_path(xml_path, mj_timestep=0.01)
    via_factory = model.make_data(orbit=_orbit_init())
    via_constructor = MjoData(model, orbit=_orbit_init(alt_km=450.0))

    assert via_factory.model is model
    assert via_constructor.model is model
    assert not np.shares_memory(via_factory.qpos, via_constructor.qpos)


def test_mjo_forward_syncs_derived_state(tmp_path: Path):
    xml_path = _xml_with_mjorbit(tmp_path, FREE_BODY_XML)
    model = MjoModel.from_xml_path(xml_path, mj_timestep=0.01)
    data = MjoData(model, orbit=_orbit_init())
    orbit_position_view = data.orbit.R_eci
    frame_view = data.frame.C_LI
    env_view = data.env.mag_field_eci

    data.qpos[:3] = [2.0, -1.0, 0.5]
    data.qvel[:3] = [0.1, 0.0, -0.05]
    data.orbit.t = 12.0
    a = R_EARTH + 400.0
    R_eci, V_eci = keplerian_to_cartesian(
        a=a,
        e=0.0,
        inc=np.deg2rad(51.6),
        raan=0.0,
        argp=0.0,
        nu=np.deg2rad(30.0),
    )
    data.orbit.R_eci[:] = R_eci
    data.orbit.V_eci[:] = V_eci

    mjo_forward(model, data)

    np.testing.assert_allclose(data.xfrc_applied, data.wrench_buffer)
    assert np.all(np.isfinite(data.xipos))
    assert data.orbit.t == 12.0
    assert np.shares_memory(data.orbit.R_eci, orbit_position_view)
    assert np.shares_memory(data.frame.C_LI, frame_view)
    assert np.shares_memory(data.env.mag_field_eci, env_view)
    np.testing.assert_allclose(data.frame.C_LI[0], R_eci / np.linalg.norm(R_eci), atol=1e-14)
    assert abs(np.linalg.norm(data.env.sun_vector_eci) - 1.0) < 1e-12


def test_sensor_lookup_and_measurement_use_canonical_sensordata(tmp_path: Path):
    xml_path = _xml_with_mjorbit(tmp_path, FREE_BODY_SENSORS_XML, attrs='use_magnetic="true"')
    model = MjoModel.from_xml_path(xml_path, mj_timestep=0.01)
    data = MjoData(model, orbit=_orbit_init())

    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[3:6] = np.array([0.03, -0.02, 0.01])
    mjo_forward(model, data)

    descriptor = model.sensor("gyro_body")
    truth = data.sensordata[descriptor.data_slice].copy()
    np.testing.assert_allclose(data.sensors.measure("gyro_body", noisy=False), truth)

    truth_before = data.sensordata.copy()
    noisy = data.sensors.measure("gyro_body", noisy=True, rng=np.random.default_rng(7))
    truth_after = data.sensordata.copy()

    np.testing.assert_allclose(truth_after, truth_before)
    assert not np.allclose(noisy, truth)


def test_multiple_data_instances_share_model_but_not_runtime_state(tmp_path: Path):
    xml_path = _xml_with_mjorbit(tmp_path, FREE_BODY_SENSORS_XML, attrs='use_magnetic="true"')
    model = MjoModel.from_xml_path(xml_path, mj_timestep=0.01)
    data_a = MjoData(model, orbit=_orbit_init())
    data_b = MjoData(model, orbit=_orbit_init(alt_km=500.0))

    angle = np.pi / 3.0
    data_a.qpos[3:7] = np.array([np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0])
    data_b.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    mjo_forward(model, data_a)
    mjo_forward(model, data_b)

    assert not np.shares_memory(data_a.qpos, data_b.qpos)
    assert data_a.orbit.t == 0.0
    assert data_b.orbit.t == 0.0
    assert not np.allclose(data_a.sensordata, data_b.sensordata)


def test_mjo_step_consumes_new_api_command_arrays(tmp_path: Path):
    children = """
    <reaction_wheel name="rw_z" body="spacecraft" axis="0 0 1"
                    inertia="0.01" speed_limit="100" torque_limit="0.1"/>
    <thruster name="thr_y" body="spacecraft" pos="0 0 0" dir="0 1 0"
              force_limit="10"/>
    """
    xml_path = _xml_with_mjorbit(
        tmp_path,
        FREE_BODY_XML,
        name="actuators.xml",
        children=children,
    )
    model = MjoModel.from_xml_path(xml_path, mj_timestep=0.01)
    data = MjoData(model, orbit=_orbit_init())

    v0 = np.linalg.norm(data.orbit.V_eci)
    data.actuators.rw_torque_cmd[0] = 0.01
    data.actuators.thr_force_cmd[0] = 5.0

    for _ in range(100):
        mjo_step(model, data)

    assert data.actuators.rw_speed[0] > 0.0
    assert np.linalg.norm(data.orbit.V_eci) > v0
    assert np.all(np.isfinite(data.qpos))


def test_orbit_dt_larger_than_mujoco_timestep_stays_time_aligned(tmp_path: Path):
    fine_xml = _xml_with_mjorbit(tmp_path, FREE_BODY_XML, name="fine.xml")
    coarse_xml = _xml_with_mjorbit(
        tmp_path,
        FREE_BODY_XML,
        name="coarse.xml",
        attrs='orbit_dt="0.1"',
    )
    fine_model = MjoModel.from_xml_path(fine_xml, mj_timestep=0.01)
    coarse_model = MjoModel.from_xml_path(coarse_xml, mj_timestep=0.01)
    fine_data = MjoData(fine_model, orbit=_orbit_init())
    coarse_data = MjoData(coarse_model, orbit=_orbit_init())

    for _ in range(20):
        mjo_step(fine_model, fine_data)
        mjo_step(coarse_model, coarse_data)

    assert coarse_data.time == pytest.approx(0.2)
    assert coarse_data.orbit.t == pytest.approx(coarse_data.time)
    assert coarse_data.orbit.rk4_count < fine_data.orbit.rk4_count
    np.testing.assert_allclose(coarse_data.orbit.R_eci, fine_data.orbit.R_eci, atol=1e-5)
    np.testing.assert_allclose(coarse_data.orbit.V_eci, fine_data.orbit.V_eci, atol=1e-8)


def test_orbit_dt_smaller_than_mujoco_timestep_substeps_orbit(tmp_path: Path):
    fine_xml = _xml_with_mjorbit(
        tmp_path,
        FREE_BODY_XML,
        name="fine_substep_reference.xml",
        attrs='orbit_dt="0.01"',
    )
    substep_xml = _xml_with_mjorbit(
        tmp_path,
        FREE_BODY_XML,
        name="substep.xml",
        attrs='orbit_dt="0.01"',
    )
    fine_model = MjoModel.from_xml_path(fine_xml, mj_timestep=0.01)
    substep_model = MjoModel.from_xml_path(substep_xml, mj_timestep=0.05)
    fine_data = MjoData(fine_model, orbit=_orbit_init())
    substep_data = MjoData(substep_model, orbit=_orbit_init())

    for _ in range(5):
        mjo_step(fine_model, fine_data)
    mjo_step(substep_model, substep_data)

    assert substep_data.time == pytest.approx(0.05)
    assert substep_data.orbit.t == pytest.approx(substep_data.time)
    np.testing.assert_allclose(substep_data.orbit.R_eci, fine_data.orbit.R_eci, atol=1e-12)
    np.testing.assert_allclose(substep_data.orbit.V_eci, fine_data.orbit.V_eci, atol=1e-12)
