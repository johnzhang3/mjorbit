"""Phase 7 validation: reaction wheels, magnetorquers, thrusters."""

import mujoco
import numpy as np

from mujoco_orbit import compile
from mujoco_orbit.constants import R_EARTH
from mujoco_orbit.core.config import (
    MagnetorquerCfg,
    MuJoCoCfg,
    OrbitCfg,
    ReactionWheelCfg,
    ScenarioCfg,
    ThrusterCfg,
)
from mujoco_orbit.coupling.actuators import (
    _apply_magnetorquers,
    _apply_thrusters,
    command_rw_torques,
)
from mujoco_orbit.orbit.elements import keplerian_to_cartesian
from mujoco_orbit.testdata import FREE_BODY_XML, SPACECRAFT_ARM_XML


def _leo_cfg(**overrides) -> ScenarioCfg:
    a = R_EARTH + 400.0
    R, V = keplerian_to_cartesian(a, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0)
    defaults = dict(
        orbit=OrbitCfg(R_eci=R, V_eci=V),
        mujoco=MuJoCoCfg(xml_path=FREE_BODY_XML, dt=0.01),
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=True,
    )
    defaults.update(overrides)
    return ScenarioCfg(**defaults)


# =========================================================================
# Reaction wheels
# =========================================================================

class TestReactionWheel:
    def test_torque_changes_wheel_speed(self):
        """Commanding torque on wheel should change its speed."""
        cfg = _leo_cfg(
            reaction_wheels=[
                ReactionWheelCfg(
                    body_name="spacecraft",
                    axis_body=np.array([0.0, 0.0, 1.0]),
                    inertia=0.01,
                )
            ]
        )
        sc = compile(cfg)
        assert sc.actuator_state.rw_speed[0] == 0.0

        sc.clear_wrench_buffer()
        command_rw_torques(sc, np.array([0.005]), dt=0.01)  # 5 mN·m

        # alpha = 0.005 / 0.01 = 0.5 rad/s^2
        # speed = 0 + 0.5 * 0.01 = 0.005 rad/s
        np.testing.assert_allclose(sc.actuator_state.rw_speed[0], 0.005, rtol=1e-10)

    def test_equal_opposite_body_torque(self):
        """Reaction torque on body should be equal and opposite to wheel torque."""
        cfg = _leo_cfg(
            reaction_wheels=[
                ReactionWheelCfg(
                    body_name="spacecraft",
                    axis_body=np.array([0.0, 0.0, 1.0]),
                    inertia=0.01,
                )
            ]
        )
        sc = compile(cfg)
        sc.clear_wrench_buffer()
        tau_cmd = 0.005  # N·m on wheel
        command_rw_torques(sc, np.array([tau_cmd]), dt=0.01)

        tau_body = sc._wrench_buffer[1, 3:]
        # Reaction: -I*alpha * axis = -(0.01 * 0.5) * [0,0,1] = [0, 0, -0.005]
        np.testing.assert_allclose(tau_body, [0.0, 0.0, -tau_cmd], atol=1e-12)

    def test_no_translational_force(self):
        """Reaction wheels should produce zero translational force."""
        cfg = _leo_cfg(
            reaction_wheels=[
                ReactionWheelCfg(
                    body_name="spacecraft",
                    axis_body=np.array([0.0, 0.0, 1.0]),
                    inertia=0.01,
                )
            ]
        )
        sc = compile(cfg)
        sc.clear_wrench_buffer()
        command_rw_torques(sc, np.array([0.01]), dt=0.01)
        F = sc._wrench_buffer[1, :3]
        np.testing.assert_allclose(F, 0.0, atol=1e-20)

    def test_torque_limit(self):
        """Commanded torque should be clamped to torque_limit."""
        cfg = _leo_cfg(
            reaction_wheels=[
                ReactionWheelCfg(
                    body_name="spacecraft",
                    axis_body=np.array([0.0, 0.0, 1.0]),
                    inertia=0.01,
                    torque_limit=0.003,
                )
            ]
        )
        sc = compile(cfg)
        sc.clear_wrench_buffer()
        command_rw_torques(sc, np.array([0.01]), dt=0.01)  # 10 mN·m > 3 mN·m limit

        # Should be clamped to 3 mN·m
        tau_body = sc._wrench_buffer[1, 3:]
        np.testing.assert_allclose(tau_body[2], -0.003, atol=1e-12)

    def test_speed_saturation(self):
        """When at speed limit, should not accelerate further in that direction."""
        cfg = _leo_cfg(
            reaction_wheels=[
                ReactionWheelCfg(
                    body_name="spacecraft",
                    axis_body=np.array([0.0, 0.0, 1.0]),
                    inertia=0.01,
                    speed_limit=100.0,
                )
            ]
        )
        sc = compile(cfg)
        sc.actuator_state.rw_speed[0] = 100.0  # at limit

        sc.clear_wrench_buffer()
        command_rw_torques(sc, np.array([0.01]), dt=0.01)  # positive (would go beyond)

        # Speed should not increase
        assert sc.actuator_state.rw_speed[0] <= 100.0
        # Body torque should be zero (no acceleration)
        np.testing.assert_allclose(sc._wrench_buffer[1, 3:], 0.0, atol=1e-14)

    def test_speed_saturation_allows_decel(self):
        """At positive speed limit, negative torque should still work."""
        cfg = _leo_cfg(
            reaction_wheels=[
                ReactionWheelCfg(
                    body_name="spacecraft",
                    axis_body=np.array([0.0, 0.0, 1.0]),
                    inertia=0.01,
                    speed_limit=100.0,
                )
            ]
        )
        sc = compile(cfg)
        sc.actuator_state.rw_speed[0] = 100.0

        sc.clear_wrench_buffer()
        command_rw_torques(sc, np.array([-0.01]), dt=0.01)

        # Negative torque should decelerate wheel
        assert sc.actuator_state.rw_speed[0] < 100.0
        # Body torque should be nonzero (reaction from deceleration)
        assert abs(sc._wrench_buffer[1, 5]) > 1e-6

    def test_momentum_tracked(self):
        """After commanding torque, momentum should match speed * inertia."""
        cfg = _leo_cfg(
            reaction_wheels=[
                ReactionWheelCfg(
                    body_name="spacecraft",
                    axis_body=np.array([0.0, 0.0, 1.0]),
                    inertia=0.05,
                )
            ]
        )
        sc = compile(cfg)
        sc.clear_wrench_buffer()
        command_rw_torques(sc, np.array([0.01]), dt=1.0)
        expected_speed = 0.01 / 0.05 * 1.0  # alpha * dt = 0.2 rad/s
        np.testing.assert_allclose(sc.actuator_state.rw_speed[0], expected_speed)
        np.testing.assert_allclose(sc.actuator_state.rw_momentum[0], expected_speed * 0.05)


# =========================================================================
# Magnetorquers
# =========================================================================

class TestMagnetorquer:
    def test_mtq_torque_matches_m_cross_b(self):
        """Magnetorquer torque should match tau = m × B."""
        cfg = _leo_cfg(
            magnetorquers=[
                MagnetorquerCfg(
                    body_name="spacecraft",
                    axis_body=np.array([1.0, 0.0, 0.0]),
                    dipole_limit=10.0,
                )
            ]
        )
        sc = compile(cfg)

        # Command 5 A·m^2 on the x-axis magnetorquer
        sc.actuator_state.mtq_dipole[0] = 5.0

        # Set B to known value
        B_test = np.array([0.0, 0.0, 1e-5])
        sc.env_cache.mag_field_eci = sc.frame_cache.C_IL @ B_test

        sc.clear_wrench_buffer()
        _apply_magnetorquers(sc)

        tau = sc._wrench_buffer[1, 3:]
        # m = [5, 0, 0], B = [0, 0, 1e-5]
        # m × B = [0*1e-5 - 0*0, 0*0 - 5*1e-5, 5*0 - 0*0] = [0, -5e-5, 0]
        np.testing.assert_allclose(tau, [0, -5e-5, 0], atol=1e-14)

    def test_mtq_dipole_clamped(self):
        """Commanded dipole should be clamped to limit."""
        cfg = _leo_cfg(
            magnetorquers=[
                MagnetorquerCfg(
                    body_name="spacecraft",
                    axis_body=np.array([0.0, 0.0, 1.0]),
                    dipole_limit=3.0,
                )
            ]
        )
        sc = compile(cfg)
        sc.actuator_state.mtq_dipole[0] = 10.0  # exceeds limit

        B_test = np.array([1e-5, 0.0, 0.0])
        sc.env_cache.mag_field_eci = sc.frame_cache.C_IL @ B_test

        sc.clear_wrench_buffer()
        _apply_magnetorquers(sc)

        tau = sc._wrench_buffer[1, 3:]
        # m = [0, 0, 3] (clamped), B = [1e-5, 0, 0]
        # m × B = [0*0 - 3*0, 3*1e-5 - 0*0, 0*0 - 0*1e-5] = [0, 3e-5, 0]
        np.testing.assert_allclose(tau, [0, 3e-5, 0], atol=1e-14)

    def test_mtq_no_translational_force(self):
        cfg = _leo_cfg(
            magnetorquers=[
                MagnetorquerCfg(
                    body_name="spacecraft",
                    axis_body=np.array([1.0, 0.0, 0.0]),
                    dipole_limit=10.0,
                )
            ]
        )
        sc = compile(cfg)
        sc.actuator_state.mtq_dipole[0] = 5.0
        sc.clear_wrench_buffer()
        _apply_magnetorquers(sc)
        np.testing.assert_allclose(sc._wrench_buffer[1, :3], 0.0, atol=1e-20)

    def test_mtq_disabled(self):
        """When use_magnetic=False, MTQ should not apply torque."""
        cfg = _leo_cfg(
            magnetorquers=[
                MagnetorquerCfg(
                    body_name="spacecraft",
                    axis_body=np.array([1.0, 0.0, 0.0]),
                    dipole_limit=10.0,
                )
            ],
            use_magnetic=False,
        )
        sc = compile(cfg)
        sc.actuator_state.mtq_dipole[0] = 5.0
        sc.clear_wrench_buffer()
        _apply_magnetorquers(sc)
        np.testing.assert_allclose(sc._wrench_buffer[1], 0.0)


# =========================================================================
# Thrusters
# =========================================================================

class TestThruster:
    def test_force_at_com(self):
        """Thruster at COM should produce pure force, zero torque."""
        cfg = _leo_cfg(
            thrusters=[
                ThrusterCfg(
                    body_name="spacecraft",
                    position_body=np.array([0.0, 0.0, 0.0]),
                    direction_body=np.array([1.0, 0.0, 0.0]),
                    force_limit=10.0,
                )
            ]
        )
        sc = compile(cfg)
        sc.actuator_state.thr_force[0] = 5.0

        sc.clear_wrench_buffer()
        _apply_thrusters(sc)

        F = sc._wrench_buffer[1, :3]
        tau = sc._wrench_buffer[1, 3:]
        # At identity pose: body x = world x
        np.testing.assert_allclose(F, [5.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(tau, 0.0, atol=1e-14)

    def test_force_at_offset_produces_torque(self):
        """Thruster offset from COM should produce force + torque."""
        cfg = _leo_cfg(
            thrusters=[
                ThrusterCfg(
                    body_name="spacecraft",
                    position_body=np.array([0.0, 0.0, 1.0]),  # 1 m along body z
                    direction_body=np.array([1.0, 0.0, 0.0]),  # thrust along body x
                    force_limit=10.0,
                )
            ]
        )
        sc = compile(cfg)
        sc.actuator_state.thr_force[0] = 5.0

        sc.clear_wrench_buffer()
        _apply_thrusters(sc)

        F = sc._wrench_buffer[1, :3]
        tau = sc._wrench_buffer[1, 3:]

        np.testing.assert_allclose(F, [5.0, 0.0, 0.0], atol=1e-12)
        # tau = r × F = [0,0,1] × [5,0,0] = [0*0-1*0, 1*5-0*0, 0*0-0*5] = [0, 5, 0]
        np.testing.assert_allclose(tau, [0.0, 5.0, 0.0], atol=1e-12)

    def test_force_clamped(self):
        """Thrust should be clamped to [0, force_limit]."""
        cfg = _leo_cfg(
            thrusters=[
                ThrusterCfg(
                    body_name="spacecraft",
                    position_body=np.zeros(3),
                    direction_body=np.array([1.0, 0.0, 0.0]),
                    force_limit=5.0,
                )
            ]
        )
        sc = compile(cfg)
        sc.actuator_state.thr_force[0] = 100.0  # exceeds limit

        sc.clear_wrench_buffer()
        _apply_thrusters(sc)

        F = sc._wrench_buffer[1, :3]
        np.testing.assert_allclose(F, [5.0, 0.0, 0.0], atol=1e-12)

    def test_negative_thrust_clamped_to_zero(self):
        """Negative thrust should be clamped to zero."""
        cfg = _leo_cfg(
            thrusters=[
                ThrusterCfg(
                    body_name="spacecraft",
                    position_body=np.zeros(3),
                    direction_body=np.array([1.0, 0.0, 0.0]),
                    force_limit=5.0,
                )
            ]
        )
        sc = compile(cfg)
        sc.actuator_state.thr_force[0] = -1.0

        sc.clear_wrench_buffer()
        _apply_thrusters(sc)

        np.testing.assert_allclose(sc._wrench_buffer[1], 0.0, atol=1e-14)

    def test_rotated_body(self):
        """Thruster on a rotated body should produce force in the correct world direction."""
        cfg = _leo_cfg(
            thrusters=[
                ThrusterCfg(
                    body_name="spacecraft",
                    position_body=np.zeros(3),
                    direction_body=np.array([0.0, 0.0, 1.0]),  # thrust along body z
                    force_limit=10.0,
                )
            ]
        )
        sc = compile(cfg)
        sc.actuator_state.thr_force[0] = 5.0

        # Rotate body 90 deg about y-axis: body z → world x
        angle = np.pi / 2
        sc.mjd.qpos[3] = np.cos(angle / 2)
        sc.mjd.qpos[4] = 0.0
        sc.mjd.qpos[5] = np.sin(angle / 2)
        sc.mjd.qpos[6] = 0.0
        mujoco.mj_forward(sc.mjm, sc.mjd)

        sc.clear_wrench_buffer()
        _apply_thrusters(sc)

        F = sc._wrench_buffer[1, :3]
        # After 90 deg rotation about y: body z → world x
        np.testing.assert_allclose(F, [5.0, 0.0, 0.0], atol=0.01)

    def test_off_com_body_uses_com_moment_arm(self):
        """Thruster torque should be computed about the MuJoCo body COM."""
        cfg = _leo_cfg(
            mujoco=MuJoCoCfg(xml_path=SPACECRAFT_ARM_XML, dt=0.01),
            thrusters=[
                ThrusterCfg(
                    body_name="link1",
                    position_body=np.array([0.4, 0.0, 0.0]),  # link1 COM in body frame
                    direction_body=np.array([0.0, 0.0, 1.0]),
                    force_limit=10.0,
                )
            ],
        )
        sc = compile(cfg)
        sc.actuator_state.thr_force[0] = 5.0

        sc.clear_wrench_buffer()
        _apply_thrusters(sc)

        expected_force = sc.mjd.ximat[2].reshape(3, 3) @ np.array([0.0, 0.0, 5.0])
        np.testing.assert_allclose(sc._wrench_buffer[2, :3], expected_force, atol=1e-12)
        np.testing.assert_allclose(sc._wrench_buffer[2, 3:], 0.0, atol=1e-12)
