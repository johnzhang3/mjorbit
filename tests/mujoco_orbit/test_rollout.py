"""Batched rollout tests for the MuJoCo-style orbit API."""

from __future__ import annotations

import numpy as np

from mujoco_orbit import (
    MjoData,
    ReactionWheelSpec,
    ThrusterSpec,
    mjo_control_size,
    mjo_forward,
    mjo_get_state,
    mjo_set_state,
    mjo_state_size,
    mjo_step,
    rollout,
)
from mujoco_orbit.testdata import FREE_BODY_SENSORS_XML, FREE_BODY_XML

from ._helpers import circular_leo_orbit_init, make_model_data


def _full_state_size(model) -> int:
    orbit_tail = 7 + len(model.reaction_wheels) + 2 * len(model.cmgs)
    return mjo_state_size(model) - orbit_tail


def test_mjo_state_pack_round_trips_orbit_and_actuator_state():
    model, data = make_model_data(
        xml_path=FREE_BODY_XML,
        use_gravity_gradient=False,
        reaction_wheels=[
            ReactionWheelSpec(
                body_name="spacecraft",
                axis_body=np.array([0.0, 0.0, 1.0]),
                inertia=0.01,
            )
        ],
    )
    data.qpos[:3] = [1.0, 2.0, 3.0]
    data.qvel[:6] = [0.1, 0.2, 0.3, 0.01, 0.02, 0.03]
    data.orbit.t = 12.0
    data.actuators.rw_speed[0] = 4.0
    data.actuators.update_rw_momentum(model.rw_inertia)
    mjo_forward(model, data)

    packed = mjo_get_state(model, data)
    assert packed.shape == (mjo_state_size(model),)

    clone = MjoData(model, orbit=circular_leo_orbit_init(alt_km=500.0))
    mjo_set_state(model, clone, packed)

    np.testing.assert_allclose(clone.qpos, data.qpos)
    np.testing.assert_allclose(clone.qvel, data.qvel)
    np.testing.assert_allclose(clone.orbit.R_eci, data.orbit.R_eci)
    np.testing.assert_allclose(clone.orbit.V_eci, data.orbit.V_eci)
    np.testing.assert_allclose(clone.orbit.t, data.orbit.t)
    np.testing.assert_allclose(clone.actuators.rw_speed, data.actuators.rw_speed)
    np.testing.assert_allclose(clone.actuators.rw_momentum, data.actuators.rw_momentum)


def test_rollout_matches_sequential_steps_with_sensors():
    model, data = make_model_data(
        xml_path=FREE_BODY_SENSORS_XML,
        use_magnetic=True,
        use_gravity_gradient=False,
    )
    data.qpos[:3] = [2.0, -1.0, 0.5]
    data.qvel[:6] = [0.1, -0.2, 0.3, 0.01, -0.02, 0.03]
    mjo_forward(model, data)
    initial = mjo_get_state(model, data)

    nstep = 6
    state, sensordata = rollout(model, data, initial, nstep=nstep)

    ref = MjoData(model, orbit=circular_leo_orbit_init())
    mjo_set_state(model, ref, initial)
    expected_state = np.empty_like(state[0])
    expected_sensordata = np.empty_like(sensordata[0])
    for step in range(nstep):
        mjo_step(model, ref)
        expected_state[step] = mjo_get_state(model, ref)
        expected_sensordata[step] = ref.sensordata

    np.testing.assert_allclose(state[0], expected_state, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(sensordata[0], expected_sensordata, rtol=0.0, atol=1e-12)


def test_rollout_resets_orbit_state_for_each_batch_member():
    model, data = make_model_data(
        xml_path=FREE_BODY_XML,
        mj_timestep=0.25,
        use_gravity_gradient=False,
    )
    initial = mjo_get_state(model, data)
    batch = np.vstack([initial, initial])
    batch[1, 0] = 5.0

    state, _ = rollout(model, data, batch, nstep=3)

    orbit_time_index = _full_state_size(model) + 6
    np.testing.assert_allclose(state[:, -1, orbit_time_index], [0.75, 0.75])
    assert state[0, -1, 0] != state[1, -1, 0]


def test_rollout_reinitializes_multirate_orbit_schedule_per_batch_member():
    model, data = make_model_data(
        xml_path=FREE_BODY_XML,
        mj_timestep=0.01,
        orbit_dt=0.1,
        use_gravity_gradient=False,
    )
    first = mjo_get_state(model, data)

    second_data = MjoData(model, orbit=circular_leo_orbit_init(alt_km=500.0))
    mjo_forward(model, second_data)
    second = mjo_get_state(model, second_data)

    initial = np.vstack([first, second])
    state, _ = rollout(model, data, initial, nstep=5)

    for batch_id, packed in enumerate(initial):
        ref = MjoData(model, orbit=circular_leo_orbit_init())
        mjo_set_state(model, ref, packed)
        expected = np.empty_like(state[batch_id])
        for step in range(state.shape[1]):
            mjo_step(model, ref)
            expected[step] = mjo_get_state(model, ref)
        np.testing.assert_allclose(state[batch_id], expected, rtol=0.0, atol=1e-12)


def test_rollout_applies_external_actuator_controls():
    model, data = make_model_data(
        xml_path=FREE_BODY_XML,
        use_gravity_gradient=False,
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
    initial = mjo_get_state(model, data)
    nstep = 8
    control = np.zeros((nstep, mjo_control_size(model)))
    control[:, 0] = 0.01  # reaction wheel torque command
    control[:, 1] = 5.0  # thruster force command

    state, _ = rollout(model, data, initial, control)

    ref = MjoData(model, orbit=circular_leo_orbit_init())
    mjo_set_state(model, ref, initial)
    expected_state = np.empty_like(state[0])
    for step in range(nstep):
        ref.actuators.rw_torque_cmd[0] = control[step, 0]
        ref.actuators.thr_force_cmd[0] = control[step, 1]
        mjo_step(model, ref)
        expected_state[step] = mjo_get_state(model, ref)

    np.testing.assert_allclose(state[0], expected_state, rtol=0.0, atol=1e-12)
    assert state[0, -1, _full_state_size(model) + 7] > 0.0
