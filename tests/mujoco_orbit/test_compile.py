"""Phase 0 + Phase 3 tests: compile, scenario state, body helpers."""

import numpy as np
import pytest

from mujoco_orbit import compile, step
from mujoco_orbit.constants import R_EARTH
from mujoco_orbit.core.config import (
    MagneticBodyCfg,
    MagnetorquerCfg,
    MuJoCoCfg,
    OrbitCfg,
    ReactionWheelCfg,
    ScenarioCfg,
    SurfaceCfg,
    ThrusterCfg,
)
from mujoco_orbit.orbit.elements import keplerian_to_cartesian
from mujoco_orbit.testdata import FREE_BODY_XML


def _circular_leo_cfg(alt_km: float = 400.0, **overrides) -> ScenarioCfg:
    a = R_EARTH + alt_km
    R, V = keplerian_to_cartesian(a=a, e=0.0, inc=np.deg2rad(51.6), raan=0.0, argp=0.0, nu=0.0)
    defaults = dict(
        orbit=OrbitCfg(R_eci=R, V_eci=V),
        mujoco=MuJoCoCfg(xml_path=FREE_BODY_XML, dt=0.01),
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    defaults.update(overrides)
    return ScenarioCfg(**defaults)


# =========================================================================
# Phase 0 exit criteria
# =========================================================================

class TestPhase0:
    def test_compile_returns_scenario(self):
        cfg = _circular_leo_cfg()
        scenario = compile(cfg)
        assert scenario.mjm is not None
        assert scenario.mjd is not None

    def test_gravity_disabled(self):
        cfg = _circular_leo_cfg()
        scenario = compile(cfg)
        np.testing.assert_allclose(scenario.mjm.opt.gravity, [0, 0, 0])

    def test_step_once_no_nan(self):
        cfg = _circular_leo_cfg()
        scenario = compile(cfg)
        step(scenario)
        assert np.all(np.isfinite(scenario.mjd.qpos))
        assert np.all(np.isfinite(scenario.mjd.qvel))

    def test_step_multiple_no_nan(self):
        cfg = _circular_leo_cfg()
        scenario = compile(cfg)
        for _ in range(100):
            step(scenario)
        assert np.all(np.isfinite(scenario.mjd.qpos))


# =========================================================================
# Phase 3: MuJoCo integration
# =========================================================================

class TestPhase3:
    def test_timestep_override(self):
        cfg = _circular_leo_cfg()
        scenario = compile(cfg)
        assert scenario.mjm.opt.timestep == 0.01

    def test_orbit_initialized(self):
        cfg = _circular_leo_cfg()
        scenario = compile(cfg)
        assert scenario.orbit.t == 0.0
        assert np.linalg.norm(scenario.orbit.R_eci) > R_EARTH

    def test_frame_cache_initialized(self):
        cfg = _circular_leo_cfg()
        scenario = compile(cfg)
        fc = scenario.frame_cache
        # Should be a proper rotation
        np.testing.assert_allclose(fc.C_LI @ fc.C_LI.T, np.eye(3), atol=1e-14)

    def test_env_cache_initialized(self):
        cfg = _circular_leo_cfg()
        scenario = compile(cfg)
        ec = scenario.env_cache
        assert abs(np.linalg.norm(ec.sun_vector_eci) - 1.0) < 1e-10

    def test_body_helpers(self):
        cfg = _circular_leo_cfg()
        scenario = compile(cfg)
        bid = scenario.body_id("spacecraft")
        assert bid >= 1

        pos = scenario.body_com_pos(bid)
        assert pos.shape == (3,)

        quat = scenario.body_com_quat(bid)
        assert quat.shape == (4,)
        assert abs(np.linalg.norm(quat) - 1.0) < 1e-10

        rotmat = scenario.body_com_rotmat(bid)
        assert rotmat.shape == (3, 3)
        np.testing.assert_allclose(rotmat @ rotmat.T, np.eye(3), atol=1e-10)

        vel = scenario.body_com_vel(bid)
        assert vel.shape == (3,)

        angvel = scenario.body_com_angvel(bid)
        assert angvel.shape == (3,)

        mass = scenario.body_mass(bid)
        assert mass == 100.0  # from free_body.xml

    def test_body_id_unknown_raises(self):
        cfg = _circular_leo_cfg()
        scenario = compile(cfg)
        with pytest.raises(ValueError):
            scenario.body_id("nonexistent_body")

    def test_wrench_buffer_shape(self):
        cfg = _circular_leo_cfg()
        scenario = compile(cfg)
        assert scenario._wrench_buffer.shape == (scenario.mjm.nbody, 6)

    def test_clear_wrench_buffer(self):
        cfg = _circular_leo_cfg()
        scenario = compile(cfg)
        scenario._wrench_buffer[1, :] = 999.0
        scenario.mjd.xfrc_applied[1, :] = 999.0
        scenario.clear_wrench_buffer()
        np.testing.assert_allclose(scenario._wrench_buffer, 0.0)
        np.testing.assert_allclose(scenario.mjd.xfrc_applied, 0.0)


class TestPhase3SurfaceMetadata:
    def test_surface_resolved(self):
        cfg = _circular_leo_cfg(
            surfaces=[
                SurfaceCfg(
                    body_name="spacecraft",
                    center_of_pressure_body=np.array([0.5, 0.0, 0.0]),
                    normal_body=np.array([1.0, 0.0, 0.0]),
                    area=2.0,
                )
            ]
        )
        scenario = compile(cfg)
        assert len(scenario.surfaces) == 1
        s = scenario.surfaces[0]
        assert s.body_id >= 1
        np.testing.assert_allclose(s.normal_body, [1, 0, 0])
        assert s.area == 2.0

    def test_surface_normal_normalized(self):
        cfg = _circular_leo_cfg(
            surfaces=[
                SurfaceCfg(
                    body_name="spacecraft",
                    center_of_pressure_body=np.zeros(3),
                    normal_body=np.array([3.0, 4.0, 0.0]),
                    area=1.0,
                )
            ]
        )
        scenario = compile(cfg)
        np.testing.assert_allclose(np.linalg.norm(scenario.surfaces[0].normal_body), 1.0)

    def test_surface_bad_body(self):
        cfg = _circular_leo_cfg(
            surfaces=[
                SurfaceCfg(
                    body_name="nonexistent",
                    center_of_pressure_body=np.zeros(3),
                    normal_body=np.array([1, 0, 0]),
                    area=1.0,
                )
            ]
        )
        with pytest.raises(ValueError, match="not found"):
            compile(cfg)


class TestPhase3MagneticMetadata:
    def test_magnetic_resolved(self):
        cfg = _circular_leo_cfg(
            magnetic_bodies=[
                MagneticBodyCfg(body_name="spacecraft", dipole_body=np.array([0, 0, 0.1]))
            ]
        )
        scenario = compile(cfg)
        assert len(scenario.magnetic_bodies) == 1
        assert scenario.magnetic_bodies[0].body_id >= 1

    def test_magnetic_bad_body(self):
        cfg = _circular_leo_cfg(
            magnetic_bodies=[
                MagneticBodyCfg(body_name="nonexistent", dipole_body=np.zeros(3))
            ]
        )
        with pytest.raises(ValueError, match="not found"):
            compile(cfg)


class TestPhase3Actuators:
    def test_no_actuators(self):
        cfg = _circular_leo_cfg()
        scenario = compile(cfg)
        assert scenario.actuator_state.rw_speed.shape == (0,)
        assert scenario.actuator_state.mtq_dipole.shape == (0,)
        assert scenario.actuator_state.thr_force.shape == (0,)

    def test_with_rw(self):
        cfg = _circular_leo_cfg(
            reaction_wheels=[
                ReactionWheelCfg(body_name="spacecraft", axis_body=np.array([0, 0, 1]),
                                 inertia=0.01),
                ReactionWheelCfg(body_name="spacecraft", axis_body=np.array([0, 1, 0]),
                                 inertia=0.02),
            ]
        )
        scenario = compile(cfg)
        assert scenario.actuator_state.rw_speed.shape == (2,)
        np.testing.assert_allclose(scenario.actuator_state.rw_inertia, [0.01, 0.02])
        np.testing.assert_allclose(scenario.actuator_state.rw_speed, [0, 0])

    def test_with_mtq(self):
        cfg = _circular_leo_cfg(
            magnetorquers=[
                MagnetorquerCfg(body_name="spacecraft", axis_body=np.array([1, 0, 0]),
                                dipole_limit=5.0),
            ]
        )
        scenario = compile(cfg)
        assert scenario.actuator_state.mtq_dipole.shape == (1,)

    def test_with_thr(self):
        cfg = _circular_leo_cfg(
            thrusters=[
                ThrusterCfg(body_name="spacecraft", position_body=np.array([0, 0, -0.5]),
                            direction_body=np.array([0, 0, 1]), force_limit=10.0),
            ]
        )
        scenario = compile(cfg)
        assert scenario.actuator_state.thr_force.shape == (1,)

    def test_rw_momentum_update(self):
        cfg = _circular_leo_cfg(
            reaction_wheels=[
                ReactionWheelCfg(body_name="spacecraft", axis_body=np.array([0, 0, 1]),
                                 inertia=0.05),
            ]
        )
        scenario = compile(cfg)
        scenario.actuator_state.rw_speed[0] = 100.0  # rad/s
        scenario.actuator_state.update_rw_momentum()
        np.testing.assert_allclose(scenario.actuator_state.rw_momentum[0], 5.0)
