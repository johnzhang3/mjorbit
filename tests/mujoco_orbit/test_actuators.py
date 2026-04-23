"""Phase 7 validation: reaction wheels, magnetorquers, thrusters, CMGs."""

import mujoco
import numpy as np
import pytest

from mujoco_orbit import (
    ControlMomentGyroSpec,
    MagnetorquerSpec,
    ReactionWheelSpec,
    ThrusterSpec,
    mjo_forward,
    mjo_step,
)
from mujoco_orbit.coupling.actuators import (
    _apply_cmgs,
    _apply_magnetorquers,
    _apply_reaction_wheels,
    _apply_thrusters,
    command_cmg_gimbal_rates,
    command_rw_torques,
)
from mujoco_orbit.testdata import FREE_BODY_XML, SPACECRAFT_ARM_XML, TWO_BODIES_XML

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

    def test_gyro_coupling_uses_host_body_angular_velocity(self):
        model, data = _make_model_data(
            xml_path=TWO_BODIES_XML,
            reaction_wheels=[
                ReactionWheelSpec(
                    body_name="body_b",
                    axis_body=np.array([0.0, 0.0, 1.0]),
                    inertia=0.02,
                )
            ],
        )

        joint_a = mujoco.mj_name2id(model.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "jnt_a")
        joint_b = mujoco.mj_name2id(model.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "jnt_b")
        data.qvel[model.jnt_dofadr[joint_a] + 3 : model.jnt_dofadr[joint_a] + 6] = 0.0
        data.qvel[model.jnt_dofadr[joint_b] + 3 : model.jnt_dofadr[joint_b] + 6] = [1.0, 0.0, 0.0]
        mjo_forward(model, data)

        data.actuators.rw_speed[0] = 200.0
        data.clear_wrench_buffer()
        _apply_reaction_wheels(model, data)

        expected_tau = model.rw_inertia[0] * data.actuators.rw_speed[0]
        np.testing.assert_allclose(
            data.wrench_buffer[model.body_id("body_b"), 3:],
            [0.0, expected_tau, 0.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            data.wrench_buffer[model.body_id("body_a"), 3:],
            0.0,
            atol=1e-12,
        )


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
        data.env.mag_field_eci = b_test

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
        data.env.mag_field_eci = b_test

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


class TestControlMomentGyro:
    """Control-moment-gyro physics.

    Conventions for most tests below: a single SGCMG on ``spacecraft`` with
        gimbal axis ĝ  = +z body
        spin axis ŝ₀  = +x body   (gimbal angle θ = 0)
        torque axis t̂₀ = ĝ × ŝ₀ = +y body
    As θ increases, ŝ(θ) rotates from +x toward +y (right-hand rule about +z),
    and the output torque on the body from gimbal rate θ̇ is ``-h·θ̇·t̂(θ)``.
    """

    @staticmethod
    def _make_default_cmg(**kw):
        defaults = dict(
            body_name="spacecraft",
            gimbal_axis_body=np.array([0.0, 0.0, 1.0]),
            spin_axis_body_0=np.array([1.0, 0.0, 0.0]),
            rotor_momentum=4.0,
        )
        defaults.update(kw)
        return ControlMomentGyroSpec(**defaults)

    # ---- static construction / validation --------------------------------

    def test_non_orthogonal_axes_rejected(self):
        with pytest.raises(ValueError, match="orthogonal"):
            _make_model_data(
                cmgs=[
                    ControlMomentGyroSpec(
                        body_name="spacecraft",
                        gimbal_axis_body=np.array([0.0, 0.0, 1.0]),
                        spin_axis_body_0=np.array([1.0, 0.0, 1.0]),
                        rotor_momentum=4.0,
                    )
                ]
            )

    def test_nonpositive_rotor_momentum_rejected(self):
        with pytest.raises(ValueError, match="rotor_momentum"):
            _make_model_data(
                cmgs=[
                    ControlMomentGyroSpec(
                        body_name="spacecraft",
                        gimbal_axis_body=np.array([0.0, 0.0, 1.0]),
                        spin_axis_body_0=np.array([1.0, 0.0, 0.0]),
                        rotor_momentum=0.0,
                    )
                ]
            )

    def test_axes_are_normalized(self):
        model, _ = _make_model_data(
            cmgs=[
                ControlMomentGyroSpec(
                    body_name="spacecraft",
                    gimbal_axis_body=np.array([0.0, 0.0, 2.0]),
                    spin_axis_body_0=np.array([3.0, 0.0, 0.0]),
                    rotor_momentum=4.0,
                )
            ]
        )
        np.testing.assert_allclose(model.cmgs[0].gimbal_axis_body, [0, 0, 1])
        np.testing.assert_allclose(model.cmgs[0].spin_axis_body_0, [1, 0, 0])
        np.testing.assert_allclose(model.cmgs[0].torque_axis_body_0, [0, 1, 0])

    # ---- output torque formula  -----------------------------------------

    def test_output_torque_at_theta_zero(self):
        """At θ=0, positive gimbal rate yields τ_body = -h·θ̇·ŷ."""
        model, data = _make_model_data(cmgs=[self._make_default_cmg()])
        dt = 0.01
        theta_dot = 0.2
        h = 4.0

        data.clear_wrench_buffer()
        command_cmg_gimbal_rates(model, data, np.array([theta_dot]), dt=dt)

        expected = np.array([0.0, -h * theta_dot, 0.0])
        np.testing.assert_allclose(data.wrench_buffer[1, 3:], expected, atol=1e-12)

    def test_output_torque_after_90deg_rotation(self):
        """At θ=π/2, ŝ=+ŷ, t̂=-x̂, so τ_body = +h·θ̇·x̂."""
        model, data = _make_model_data(cmgs=[self._make_default_cmg()])
        data.actuators.cmg_gimbal_angle[0] = np.pi / 2
        dt = 0.01
        theta_dot = 0.2
        h = 4.0

        data.clear_wrench_buffer()
        command_cmg_gimbal_rates(model, data, np.array([theta_dot]), dt=dt)

        expected = np.array([h * theta_dot, 0.0, 0.0])
        np.testing.assert_allclose(data.wrench_buffer[1, 3:], expected, atol=1e-12)

    def test_output_torque_reverses_sign(self):
        model, data = _make_model_data(cmgs=[self._make_default_cmg()])
        dt = 0.01
        h = 4.0

        data.clear_wrench_buffer()
        command_cmg_gimbal_rates(model, data, np.array([-0.5]), dt=dt)

        expected = np.array([0.0, h * 0.5, 0.0])
        np.testing.assert_allclose(data.wrench_buffer[1, 3:], expected, atol=1e-12)

    def test_no_translational_force(self):
        model, data = _make_model_data(cmgs=[self._make_default_cmg()])
        data.clear_wrench_buffer()
        command_cmg_gimbal_rates(model, data, np.array([0.5]), dt=0.01)
        np.testing.assert_allclose(data.wrench_buffer[1, :3], 0.0, atol=1e-20)

    # ---- gimbal-angle integration ---------------------------------------

    def test_gimbal_angle_integrates(self):
        model, data = _make_model_data(cmgs=[self._make_default_cmg()])
        dt = 0.05
        theta_dot = 0.4
        data.clear_wrench_buffer()
        command_cmg_gimbal_rates(model, data, np.array([theta_dot]), dt=dt)
        np.testing.assert_allclose(
            data.actuators.cmg_gimbal_angle[0], theta_dot * dt, rtol=1e-10
        )

        # Second call continues integration
        data.clear_wrench_buffer()
        command_cmg_gimbal_rates(model, data, np.array([theta_dot]), dt=dt)
        np.testing.assert_allclose(
            data.actuators.cmg_gimbal_angle[0], 2 * theta_dot * dt, rtol=1e-10
        )

    # ---- limits ---------------------------------------------------------

    def test_gimbal_rate_limit(self):
        model, data = _make_model_data(
            cmgs=[self._make_default_cmg(gimbal_rate_limit=0.1)]
        )
        dt = 0.01
        h = 4.0
        data.clear_wrench_buffer()
        command_cmg_gimbal_rates(model, data, np.array([1.0]), dt=dt)

        # Rate clipped to 0.1, so τ_y = -h·0.1
        np.testing.assert_allclose(
            data.wrench_buffer[1, 3:], [0.0, -h * 0.1, 0.0], atol=1e-12
        )
        np.testing.assert_allclose(data.actuators.cmg_gimbal_angle[0], 0.1 * dt, rtol=1e-10)

    def test_gimbal_angle_limit_freezes_rate(self):
        model, data = _make_model_data(
            cmgs=[self._make_default_cmg(gimbal_angle_limit=0.5)]
        )
        data.actuators.cmg_gimbal_angle[0] = 0.5  # at positive limit
        data.clear_wrench_buffer()
        command_cmg_gimbal_rates(model, data, np.array([1.0]), dt=0.01)

        # Pushing further into the limit → zero torque, angle stays
        np.testing.assert_allclose(data.wrench_buffer[1, 3:], 0.0, atol=1e-12)
        np.testing.assert_allclose(data.actuators.cmg_gimbal_angle[0], 0.5)

    def test_gimbal_angle_limit_allows_reverse(self):
        model, data = _make_model_data(
            cmgs=[self._make_default_cmg(gimbal_angle_limit=0.5)]
        )
        data.actuators.cmg_gimbal_angle[0] = 0.5
        data.clear_wrench_buffer()
        command_cmg_gimbal_rates(model, data, np.array([-1.0]), dt=0.01)

        # Negative rate is allowed at +limit
        assert data.actuators.cmg_gimbal_angle[0] < 0.5
        assert abs(data.wrench_buffer[1, 4]) > 1e-8  # y-torque nonzero

    def test_gimbal_angle_limit_uses_effective_rate_at_crossing(self):
        model, data = _make_model_data(
            cmgs=[self._make_default_cmg(gimbal_angle_limit=0.5)]
        )
        theta_old = 0.49
        dt = 0.01
        h = 4.0
        data.actuators.cmg_gimbal_angle[0] = theta_old
        data.clear_wrench_buffer()
        command_cmg_gimbal_rates(model, data, np.array([10.0]), dt=dt)

        theta_dot_effective = (0.5 - theta_old) / dt
        t_axis_body = np.array([-np.sin(theta_old), np.cos(theta_old), 0.0])
        expected = -h * theta_dot_effective * t_axis_body

        np.testing.assert_allclose(data.actuators.cmg_gimbal_angle[0], 0.5)
        np.testing.assert_allclose(data.wrench_buffer[1, 3:], expected, atol=1e-12)

    # ---- gyroscopic coupling --------------------------------------------

    def test_gyroscopic_coupling_with_body_rate(self):
        """With body rate ω=ωx·x̂ and rotor momentum h·ŝ(θ=0)=h·x̂, expect -ω×h = 0.
        With ω=ωy·ŷ, expect -ωy·ŷ × h·x̂ = +h·ωy·ẑ.
        """
        model, data = _make_model_data(cmgs=[self._make_default_cmg()])
        h = 4.0
        # Body-frame angular velocity in +y
        data.qvel[3:6] = [0.0, 0.3, 0.0]  # body-frame angular velocity
        mjo_forward(model, data)

        data.clear_wrench_buffer()
        _apply_cmgs(model, data)

        # -ω × h = -0.3·ŷ × h·x̂ = -0.3·h·(ŷ×x̂) = -0.3·h·(-ẑ) = +0.3·h·ẑ
        expected = np.array([0.0, 0.0, 0.3 * h])
        np.testing.assert_allclose(data.wrench_buffer[1, 3:], expected, atol=1e-10)

    def test_gyroscopic_coupling_depends_on_gimbal_angle(self):
        """At θ=π/2, rotor h points +ŷ; with ω=ωx·x̂, -ω×h = -ωx·x̂×h·ŷ = -h·ωx·ẑ."""
        model, data = _make_model_data(cmgs=[self._make_default_cmg()])
        h = 4.0
        data.actuators.cmg_gimbal_angle[0] = np.pi / 2
        data.qvel[3:6] = [0.3, 0.0, 0.0]
        mjo_forward(model, data)

        data.clear_wrench_buffer()
        _apply_cmgs(model, data)

        expected = np.array([0.0, 0.0, -h * 0.3])
        np.testing.assert_allclose(data.wrench_buffer[1, 3:], expected, atol=1e-10)

    # ---- coupled dynamics: angular momentum conservation -----------------

    def test_inertial_angular_momentum_conserved_under_gimbaling(self):
        """Closed-system check: CMG gimbal motion on a torque-free body should
        preserve total inertial angular momentum L = R·(J·ω + h_body) to
        integrator accuracy."""
        model, data = _make_model_data(
            mj_timestep=0.001,
            cmgs=[self._make_default_cmg(rotor_momentum=2.0)],
        )
        body_id = model.body_id("spacecraft")
        # Body inertia J (diagonal) from MuJoCo, in body frame
        J = np.diag(model.body_inertia[body_id])

        # Give the body some initial rate, and command a steady gimbal rate.
        data.qvel[3:6] = [0.02, -0.01, 0.015]
        mjo_forward(model, data)
        data.actuators.cmg_gimbal_rate_cmd[0] = 0.5

        def total_L_inertial():
            R = data.xmat[body_id].reshape(3, 3)
            w_body = R.T @ data.cvel[body_id, :3]
            theta = float(data.actuators.cmg_gimbal_angle[0])
            cmg = model.cmgs[0]
            h_body = data.actuators.cmg_rotor_momentum[0] * (
                np.cos(theta) * cmg.spin_axis_body_0
                + np.sin(theta) * cmg.torque_axis_body_0
            )
            return R @ (J @ w_body + h_body)

        L0 = total_L_inertial()
        for _ in range(100):  # 0.1 s
            mjo_step(model, data)
        L1 = total_L_inertial()

        # First-order accuracy: forward-Euler gimbal integration on top of
        # MuJoCo's semi-implicit step. Drift should scale with dt and sim time.
        err = np.linalg.norm(L1 - L0) / np.linalg.norm(L0)
        assert err < 5e-4, f"relative L drift = {err:.3e} (L0 = {L0})"

    def test_cmg_body_reaction_no_external_force(self):
        """Applied wrench must be a pure torque on the host body: no force,
        and no effect on any other body."""
        model, data = _make_model_data(cmgs=[self._make_default_cmg()])
        data.clear_wrench_buffer()
        command_cmg_gimbal_rates(model, data, np.array([0.5]), dt=0.01)
        _apply_cmgs(model, data)

        # No force on host body, no wrench on worldbody
        np.testing.assert_allclose(data.wrench_buffer[1, :3], 0.0, atol=1e-20)
        np.testing.assert_allclose(data.wrench_buffer[0], 0.0, atol=1e-20)
