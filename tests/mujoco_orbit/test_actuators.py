"""Phase 7 validation: reaction wheels, magnetorquers, thrusters."""

import numpy as np

from mujoco_orbit import MagnetorquerSpec, ReactionWheelSpec, ThrusterSpec, mjo_forward
from mujoco_orbit.coupling.actuators import (
    _apply_magnetorquers,
    _apply_thrusters,
    command_rw_torques,
)
from mujoco_orbit.testdata import FREE_BODY_XML, SPACECRAFT_ARM_XML

from ._helpers import make_model_data


def _make_model_data(**overrides):
    defaults = dict(
        xml_path=FREE_BODY_XML,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=True,
    )
    defaults.update(overrides)
    return make_model_data(**defaults)


class TestReactionWheel:
    def test_torque_changes_wheel_speed(self):
        model, data = _make_model_data(
            reaction_wheels=[
                ReactionWheelSpec(
                    body_name="spacecraft",
                    axis_body=np.array([0.0, 0.0, 1.0]),
                    inertia=0.01,
                )
            ]
        )
        assert data.actuators.rw_speed[0] == 0.0

        data.clear_wrench_buffer()
        command_rw_torques(model, data, np.array([0.005]), dt=0.01)

        np.testing.assert_allclose(data.actuators.rw_speed[0], 0.005, rtol=1e-10)

    def test_equal_opposite_body_torque(self):
        model, data = _make_model_data(
            reaction_wheels=[
                ReactionWheelSpec(
                    body_name="spacecraft",
                    axis_body=np.array([0.0, 0.0, 1.0]),
                    inertia=0.01,
                )
            ]
        )
        data.clear_wrench_buffer()
        tau_cmd = 0.005
        command_rw_torques(model, data, np.array([tau_cmd]), dt=0.01)

        tau_body = data.wrench_buffer[1, 3:]
        np.testing.assert_allclose(tau_body, [0.0, 0.0, -tau_cmd], atol=1e-12)

    def test_no_translational_force(self):
        model, data = _make_model_data(
            reaction_wheels=[
                ReactionWheelSpec(
                    body_name="spacecraft",
                    axis_body=np.array([0.0, 0.0, 1.0]),
                    inertia=0.01,
                )
            ]
        )
        data.clear_wrench_buffer()
        command_rw_torques(model, data, np.array([0.01]), dt=0.01)
        np.testing.assert_allclose(data.wrench_buffer[1, :3], 0.0, atol=1e-20)

    def test_torque_limit(self):
        model, data = _make_model_data(
            reaction_wheels=[
                ReactionWheelSpec(
                    body_name="spacecraft",
                    axis_body=np.array([0.0, 0.0, 1.0]),
                    inertia=0.01,
                    torque_limit=0.003,
                )
            ]
        )
        data.clear_wrench_buffer()
        command_rw_torques(model, data, np.array([0.01]), dt=0.01)
        np.testing.assert_allclose(data.wrench_buffer[1, 5], -0.003, atol=1e-12)

    def test_speed_saturation(self):
        model, data = _make_model_data(
            reaction_wheels=[
                ReactionWheelSpec(
                    body_name="spacecraft",
                    axis_body=np.array([0.0, 0.0, 1.0]),
                    inertia=0.01,
                    speed_limit=100.0,
                )
            ]
        )
        data.actuators.rw_speed[0] = 100.0

        data.clear_wrench_buffer()
        command_rw_torques(model, data, np.array([0.01]), dt=0.01)

        assert data.actuators.rw_speed[0] <= 100.0
        np.testing.assert_allclose(data.wrench_buffer[1, 3:], 0.0, atol=1e-14)

    def test_speed_saturation_allows_decel(self):
        model, data = _make_model_data(
            reaction_wheels=[
                ReactionWheelSpec(
                    body_name="spacecraft",
                    axis_body=np.array([0.0, 0.0, 1.0]),
                    inertia=0.01,
                    speed_limit=100.0,
                )
            ]
        )
        data.actuators.rw_speed[0] = 100.0

        data.clear_wrench_buffer()
        command_rw_torques(model, data, np.array([-0.01]), dt=0.01)

        assert data.actuators.rw_speed[0] < 100.0
        assert abs(data.wrench_buffer[1, 5]) > 1e-6

    def test_momentum_tracked(self):
        model, data = _make_model_data(
            reaction_wheels=[
                ReactionWheelSpec(
                    body_name="spacecraft",
                    axis_body=np.array([0.0, 0.0, 1.0]),
                    inertia=0.05,
                )
            ]
        )
        data.clear_wrench_buffer()
        command_rw_torques(model, data, np.array([0.01]), dt=1.0)
        expected_speed = 0.01 / 0.05
        np.testing.assert_allclose(data.actuators.rw_speed[0], expected_speed)
        np.testing.assert_allclose(data.actuators.rw_momentum[0], expected_speed * 0.05)


class TestMagnetorquer:
    def test_mtq_torque_matches_m_cross_b(self):
        model, data = _make_model_data(
            magnetorquers=[
                MagnetorquerSpec(
                    body_name="spacecraft",
                    axis_body=np.array([1.0, 0.0, 0.0]),
                    dipole_limit=10.0,
                )
            ]
        )
        data.actuators.mtq_dipole_cmd[0] = 5.0
        b_test = np.array([0.0, 0.0, 1e-5])
        data.env.mag_field_eci = data.frame.C_IL @ b_test

        data.clear_wrench_buffer()
        _apply_magnetorquers(model, data)

        np.testing.assert_allclose(data.wrench_buffer[1, 3:], [0, -5e-5, 0], atol=1e-14)

    def test_mtq_dipole_clamped(self):
        model, data = _make_model_data(
            magnetorquers=[
                MagnetorquerSpec(
                    body_name="spacecraft",
                    axis_body=np.array([0.0, 0.0, 1.0]),
                    dipole_limit=3.0,
                )
            ]
        )
        data.actuators.mtq_dipole_cmd[0] = 10.0
        b_test = np.array([1e-5, 0.0, 0.0])
        data.env.mag_field_eci = data.frame.C_IL @ b_test

        data.clear_wrench_buffer()
        _apply_magnetorquers(model, data)

        np.testing.assert_allclose(data.wrench_buffer[1, 3:], [0, 3e-5, 0], atol=1e-14)

    def test_mtq_no_translational_force(self):
        model, data = _make_model_data(
            magnetorquers=[
                MagnetorquerSpec(
                    body_name="spacecraft",
                    axis_body=np.array([1.0, 0.0, 0.0]),
                    dipole_limit=10.0,
                )
            ]
        )
        data.actuators.mtq_dipole_cmd[0] = 5.0
        data.clear_wrench_buffer()
        _apply_magnetorquers(model, data)
        np.testing.assert_allclose(data.wrench_buffer[1, :3], 0.0, atol=1e-20)

    def test_mtq_disabled(self):
        model, data = _make_model_data(
            magnetorquers=[
                MagnetorquerSpec(
                    body_name="spacecraft",
                    axis_body=np.array([1.0, 0.0, 0.0]),
                    dipole_limit=10.0,
                )
            ],
            use_magnetic=False,
        )
        data.actuators.mtq_dipole_cmd[0] = 5.0
        data.clear_wrench_buffer()
        _apply_magnetorquers(model, data)
        np.testing.assert_allclose(data.wrench_buffer[1], 0.0)


class TestThruster:
    def test_force_at_com(self):
        model, data = _make_model_data(
            thrusters=[
                ThrusterSpec(
                    body_name="spacecraft",
                    position_body=np.array([0.0, 0.0, 0.0]),
                    direction_body=np.array([1.0, 0.0, 0.0]),
                    force_limit=10.0,
                )
            ]
        )
        data.actuators.thr_force_cmd[0] = 5.0

        data.clear_wrench_buffer()
        _apply_thrusters(model, data)

        np.testing.assert_allclose(data.wrench_buffer[1, :3], [5.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(data.wrench_buffer[1, 3:], 0.0, atol=1e-14)

    def test_force_at_offset_produces_torque(self):
        model, data = _make_model_data(
            thrusters=[
                ThrusterSpec(
                    body_name="spacecraft",
                    position_body=np.array([0.0, 0.0, 1.0]),
                    direction_body=np.array([1.0, 0.0, 0.0]),
                    force_limit=10.0,
                )
            ]
        )
        data.actuators.thr_force_cmd[0] = 5.0

        data.clear_wrench_buffer()
        _apply_thrusters(model, data)

        np.testing.assert_allclose(data.wrench_buffer[1, :3], [5.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(data.wrench_buffer[1, 3:], [0.0, 5.0, 0.0], atol=1e-12)

    def test_force_clamped(self):
        model, data = _make_model_data(
            thrusters=[
                ThrusterSpec(
                    body_name="spacecraft",
                    position_body=np.zeros(3),
                    direction_body=np.array([1.0, 0.0, 0.0]),
                    force_limit=5.0,
                )
            ]
        )
        data.actuators.thr_force_cmd[0] = 100.0

        data.clear_wrench_buffer()
        _apply_thrusters(model, data)

        np.testing.assert_allclose(data.wrench_buffer[1, :3], [5.0, 0.0, 0.0], atol=1e-12)

    def test_negative_thrust_clamped_to_zero(self):
        model, data = _make_model_data(
            thrusters=[
                ThrusterSpec(
                    body_name="spacecraft",
                    position_body=np.zeros(3),
                    direction_body=np.array([1.0, 0.0, 0.0]),
                    force_limit=5.0,
                )
            ]
        )
        data.actuators.thr_force_cmd[0] = -1.0

        data.clear_wrench_buffer()
        _apply_thrusters(model, data)

        np.testing.assert_allclose(data.wrench_buffer[1], 0.0, atol=1e-14)

    def test_rotated_body(self):
        model, data = _make_model_data(
            thrusters=[
                ThrusterSpec(
                    body_name="spacecraft",
                    position_body=np.zeros(3),
                    direction_body=np.array([0.0, 0.0, 1.0]),
                    force_limit=10.0,
                )
            ]
        )
        data.actuators.thr_force_cmd[0] = 5.0

        angle = np.pi / 2
        data.qpos[3] = np.cos(angle / 2)
        data.qpos[4] = 0.0
        data.qpos[5] = np.sin(angle / 2)
        data.qpos[6] = 0.0
        mjo_forward(model, data)

        data.clear_wrench_buffer()
        _apply_thrusters(model, data)

        np.testing.assert_allclose(data.wrench_buffer[1, :3], [5.0, 0.0, 0.0], atol=0.01)

    def test_off_com_body_uses_com_moment_arm(self):
        model, data = _make_model_data(
            xml_path=SPACECRAFT_ARM_XML,
            thrusters=[
                ThrusterSpec(
                    body_name="link1",
                    position_body=np.array([0.4, 0.0, 0.0]),
                    direction_body=np.array([0.0, 0.0, 1.0]),
                    force_limit=10.0,
                )
            ],
        )
        data.actuators.thr_force_cmd[0] = 5.0

        data.clear_wrench_buffer()
        _apply_thrusters(model, data)

        expected_force = data.xmat[2].reshape(3, 3) @ np.array([0.0, 0.0, 5.0])
        np.testing.assert_allclose(data.wrench_buffer[2, :3], expected_force, atol=1e-12)
        np.testing.assert_allclose(data.wrench_buffer[2, 3:], 0.0, atol=1e-12)
