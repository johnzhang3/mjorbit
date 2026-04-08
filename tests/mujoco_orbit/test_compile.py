"""Model/data construction tests for the MuJoCo-style API."""

import numpy as np
import pytest

from mujoco_orbit import (
    MagneticBodySpec,
    MagnetorquerSpec,
    ReactionWheelSpec,
    SurfaceSpec,
    ThrusterSpec,
    mjo_step,
)
from mujoco_orbit.constants import R_EARTH
from mujoco_orbit.testdata import FREE_BODY_XML

from ._helpers import make_model_data


def _make_model_data(**overrides):
    defaults = dict(
        xml_path=FREE_BODY_XML,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    defaults.update(overrides)
    return make_model_data(**defaults)


class TestPhase0:
    def test_constructs_model_and_data(self):
        model, data = _make_model_data()
        assert model.mj_model is not None
        assert data.mj_data is not None

    def test_gravity_disabled(self):
        model, _ = _make_model_data()
        np.testing.assert_allclose(model.opt.gravity, [0, 0, 0])

    def test_step_once_no_nan(self):
        model, data = _make_model_data()
        mjo_step(model, data)
        assert np.all(np.isfinite(data.qpos))
        assert np.all(np.isfinite(data.qvel))

    def test_step_multiple_no_nan(self):
        model, data = _make_model_data()
        for _ in range(100):
            mjo_step(model, data)
        assert np.all(np.isfinite(data.qpos))


class TestPhase3:
    def test_timestep_override(self):
        model, _ = _make_model_data(mj_timestep=0.01)
        assert model.opt.timestep == 0.01

    def test_orbit_initialized(self):
        _, data = _make_model_data()
        assert data.orbit.t == 0.0
        assert np.linalg.norm(data.orbit.R_eci) > R_EARTH

    def test_frame_cache_initialized(self):
        _, data = _make_model_data()
        np.testing.assert_allclose(data.frame.C_LI @ data.frame.C_LI.T, np.eye(3), atol=1e-14)

    def test_env_cache_initialized(self):
        _, data = _make_model_data()
        assert abs(np.linalg.norm(data.env.sun_vector_eci) - 1.0) < 1e-10

    def test_body_fields(self):
        model, data = _make_model_data()
        bid = model.body_id("spacecraft")
        assert bid >= 1

        pos = data.xipos[bid].copy()
        assert pos.shape == (3,)

        quat = data.xquat[bid].copy()
        assert quat.shape == (4,)
        assert abs(np.linalg.norm(quat) - 1.0) < 1e-10

        rotmat = data.xmat[bid].reshape(3, 3).copy()
        assert rotmat.shape == (3, 3)
        np.testing.assert_allclose(rotmat @ rotmat.T, np.eye(3), atol=1e-10)

        vel = data.cvel[bid, 3:].copy()
        assert vel.shape == (3,)

        angvel = data.cvel[bid, :3].copy()
        assert angvel.shape == (3,)

        assert model.body_mass[bid] == 100.0

    def test_body_id_unknown_raises(self):
        model, _ = _make_model_data()
        with pytest.raises(ValueError):
            model.body_id("nonexistent_body")

    def test_wrench_buffer_shape(self):
        model, data = _make_model_data()
        assert data.wrench_buffer.shape == (model.nbody, 6)

    def test_clear_wrench_buffer(self):
        _, data = _make_model_data()
        data.wrench_buffer[1, :] = 999.0
        data.xfrc_applied[1, :] = 999.0
        data.clear_wrench_buffer()
        np.testing.assert_allclose(data.wrench_buffer, 0.0)
        np.testing.assert_allclose(data.xfrc_applied, 0.0)


class TestPhase3SurfaceMetadata:
    def test_surface_resolved(self):
        model, _ = _make_model_data(
            surfaces=[
                SurfaceSpec(
                    body_name="spacecraft",
                    center_of_pressure_body=np.array([0.5, 0.0, 0.0]),
                    normal_body=np.array([1.0, 0.0, 0.0]),
                    area=2.0,
                )
            ]
        )
        assert len(model.surfaces) == 1
        surface = model.surfaces[0]
        assert surface.body_id >= 1
        np.testing.assert_allclose(surface.normal_body, [1, 0, 0])
        assert surface.area == 2.0

    def test_surface_normal_normalized(self):
        model, _ = _make_model_data(
            surfaces=[
                SurfaceSpec(
                    body_name="spacecraft",
                    center_of_pressure_body=np.zeros(3),
                    normal_body=np.array([3.0, 4.0, 0.0]),
                    area=1.0,
                )
            ]
        )
        np.testing.assert_allclose(np.linalg.norm(model.surfaces[0].normal_body), 1.0)

    def test_surface_bad_body(self):
        with pytest.raises(ValueError, match="not found"):
            _make_model_data(
                surfaces=[
                    SurfaceSpec(
                        body_name="nonexistent",
                        center_of_pressure_body=np.zeros(3),
                        normal_body=np.array([1, 0, 0]),
                        area=1.0,
                    )
                ]
            )


class TestPhase3MagneticMetadata:
    def test_magnetic_resolved(self):
        model, _ = _make_model_data(
            magnetic_bodies=[
                MagneticBodySpec(body_name="spacecraft", dipole_body=np.array([0, 0, 0.1]))
            ]
        )
        assert len(model.magnetic_bodies) == 1
        assert model.magnetic_bodies[0].body_id >= 1

    def test_magnetic_bad_body(self):
        with pytest.raises(ValueError, match="not found"):
            _make_model_data(
                magnetic_bodies=[
                    MagneticBodySpec(body_name="nonexistent", dipole_body=np.zeros(3))
                ]
            )


class TestPhase3Actuators:
    def test_no_actuators(self):
        _, data = _make_model_data()
        assert data.actuators.rw_speed.shape == (0,)
        assert data.actuators.mtq_dipole_cmd.shape == (0,)
        assert data.actuators.thr_force_cmd.shape == (0,)

    def test_with_rw(self):
        _, data = _make_model_data(
            reaction_wheels=[
                ReactionWheelSpec(
                    body_name="spacecraft",
                    axis_body=np.array([0, 0, 1]),
                    inertia=0.01,
                ),
                ReactionWheelSpec(
                    body_name="spacecraft",
                    axis_body=np.array([0, 1, 0]),
                    inertia=0.02,
                ),
            ]
        )
        assert data.actuators.rw_speed.shape == (2,)
        np.testing.assert_allclose(data.actuators.rw_inertia, [0.01, 0.02])
        np.testing.assert_allclose(data.actuators.rw_speed, [0, 0])

    def test_with_mtq(self):
        _, data = _make_model_data(
            magnetorquers=[
                MagnetorquerSpec(
                    body_name="spacecraft",
                    axis_body=np.array([1, 0, 0]),
                    dipole_limit=5.0,
                )
            ]
        )
        assert data.actuators.mtq_dipole_cmd.shape == (1,)

    def test_with_thr(self):
        _, data = _make_model_data(
            thrusters=[
                ThrusterSpec(
                    body_name="spacecraft",
                    position_body=np.array([0, 0, -0.5]),
                    direction_body=np.array([0, 0, 1]),
                    force_limit=10.0,
                )
            ]
        )
        assert data.actuators.thr_force_cmd.shape == (1,)

    def test_rw_momentum_update(self):
        _, data = _make_model_data(
            reaction_wheels=[
                ReactionWheelSpec(
                    body_name="spacecraft",
                    axis_body=np.array([0, 0, 1]),
                    inertia=0.05,
                )
            ]
        )
        data.actuators.rw_speed[0] = 100.0
        data.actuators.update_rw_momentum()
        np.testing.assert_allclose(data.actuators.rw_momentum[0], 5.0)
