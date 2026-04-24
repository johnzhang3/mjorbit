"""Direct tests for the MuJoCo-style ``MjoModel`` / ``MjoData`` API."""

from __future__ import annotations

import numpy as np

from mujoco_orbit import (
    MjoData,
    MjoModel,
    OrbitInit,
    ReactionWheelSpec,
    ThrusterSpec,
    mjo_forward,
    mjo_step,
)
from mujoco_orbit.constants import R_EARTH
from mujoco_orbit.orbit.elements import keplerian_to_cartesian
from mujoco_orbit.testdata import FREE_BODY_SENSORS_XML, FREE_BODY_XML


def _orbit_init(alt_km: float = 400.0) -> OrbitInit:
    a = R_EARTH + alt_km
    R_eci, V_eci = keplerian_to_cartesian(
        a=a, e=0.0, inc=np.deg2rad(51.6), raan=0.0, argp=0.0, nu=0.0
    )
    return OrbitInit(R_eci=R_eci, V_eci=V_eci)


def test_model_and_data_expose_mujoco_fields():
    model = MjoModel.from_xml_path(
        FREE_BODY_XML,
        mj_timestep=0.01,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    data = MjoData(model, orbit=_orbit_init())

    assert model.nbody >= 2
    assert model.mj_model.nplugin == 1
    np.testing.assert_allclose(model.opt.gravity, [0.0, 0.0, 0.0])
    assert data.qpos.shape[0] == model.nq
    assert data.qvel.shape[0] == model.nv
    assert data.wrench_buffer.shape == (model.nbody, 6)
    assert data.orbit.t == 0.0


def test_mjo_forward_syncs_derived_state():
    model = MjoModel.from_xml_path(
        FREE_BODY_XML,
        mj_timestep=0.01,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    data = MjoData(model, orbit=_orbit_init())
    orbit_position_view = data.orbit.R_eci
    frame_view = data.frame.C_LI
    env_view = data.env.mag_field_eci

    data.qpos[:3] = [2.0, -1.0, 0.5]
    data.qvel[:3] = [0.1, 0.0, -0.05]
    data.orbit.t = 12.0

    mjo_forward(model, data)

    np.testing.assert_allclose(data.xfrc_applied, data.wrench_buffer)
    assert np.all(np.isfinite(data.xipos))
    assert data.orbit.t == 12.0
    assert data.orbit.R_eci is orbit_position_view
    assert data.frame.C_LI is frame_view
    assert data.env.mag_field_eci is env_view


def test_sensor_lookup_and_measurement_use_canonical_sensordata():
    model = MjoModel.from_xml_path(
        FREE_BODY_SENSORS_XML,
        mj_timestep=0.01,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=True,
    )
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


def test_multiple_data_instances_share_model_but_not_runtime_state():
    model = MjoModel.from_xml_path(
        FREE_BODY_SENSORS_XML,
        mj_timestep=0.01,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=True,
    )
    data_a = MjoData(model, orbit=_orbit_init())
    data_b = MjoData(model, orbit=_orbit_init(alt_km=500.0))

    angle = np.pi / 3.0
    data_a.qpos[3:7] = np.array([np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0])
    data_b.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    mjo_forward(model, data_a)
    mjo_forward(model, data_b)

    assert data_a.mj_data is not data_b.mj_data
    assert data_a.orbit.t == 0.0
    assert data_b.orbit.t == 0.0
    assert not np.allclose(data_a.sensordata, data_b.sensordata)


def test_mjo_step_consumes_new_api_command_arrays():
    model = MjoModel.from_xml_path(
        FREE_BODY_XML,
        mj_timestep=0.01,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
        reaction_wheels=[
            ReactionWheelSpec(
                body_name="spacecraft",
                axis_body=np.array([0.0, 0.0, 1.0]),
                inertia=0.01,
            )
        ],
        thrusters=[
            ThrusterSpec(
                body_name="spacecraft",
                position_body=np.zeros(3),
                direction_body=np.array([0.0, 1.0, 0.0]),
                force_limit=10.0,
            )
        ],
    )
    data = MjoData(model, orbit=_orbit_init())

    v0 = np.linalg.norm(data.orbit.V_eci)
    data.actuators.rw_torque_cmd[0] = 0.01
    data.actuators.thr_force_cmd[0] = 5.0

    for _ in range(100):
        mjo_step(model, data)

    assert data.actuators.rw_speed[0] > 0.0
    assert np.linalg.norm(data.orbit.V_eci) > v0
    assert np.all(np.isfinite(data.qpos))
