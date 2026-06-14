"""Phase 4 validation: chief-inertial gravity coupling and derived LVLH motion."""

import os
import tempfile

import numpy as np

from mjorbit import OrbitInit, mjo_forward, mjo_step
from mjorbit.constants import GM_EARTH, R_EARTH
from mjorbit.testdata import FREE_BODY_XML
from tests.mjorbit.reference.coupling.inertial import apply_inertial_wrenches
from tests.mjorbit.reference.orbit.elements import keplerian_to_cartesian
from tests.mjorbit.reference.orbit.gravity import total_accel
from tests.mjorbit.reference.orbit.lvlh import update_frame_cache

from ._helpers import get_freejoint_lvlh_state, make_model_data, set_freejoint_lvlh_state


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


class TestInertialWrenchBasics:
    def test_wrench_finite(self):
        model, data = _make_model_data()
        mjo_step(model, data)
        assert np.all(np.isfinite(data.wrench_buffer))

    def test_zero_offset_has_zero_differential_gravity(self):
        model, data = _make_model_data()
        data.clear_wrench_buffer()
        apply_inertial_wrenches(model, data)
        np.testing.assert_allclose(data.wrench_buffer[1, :3], 0.0, atol=1e-12)

    def test_radial_offset_positive_radial_gradient(self):
        model, data = _make_model_data()
        set_freejoint_lvlh_state(data, slice(0, 3), slice(0, 3), [1.0, 0.0, 0.0])
        mjo_forward(model, data)

        data.clear_wrench_buffer()
        apply_inertial_wrenches(model, data)
        force_lvlh = data.frame.C_LI @ data.wrench_buffer[1, :3]

        a_km = R_EARTH + 400.0
        n = np.sqrt(GM_EARTH / a_km**3)
        force_cw = 100.0 * 2.0 * n**2 * 1e-3 * 1e3

        assert force_lvlh[0] > 0
        np.testing.assert_allclose(force_lvlh[0], force_cw, rtol=0.01)

    def test_body_force_uses_absolute_eci_position(self):
        model, data = _make_model_data()
        set_freejoint_lvlh_state(data, slice(0, 3), slice(0, 3), [0.0, 1.0, 0.0])
        mjo_forward(model, data)
        data.clear_wrench_buffer()
        apply_inertial_wrenches(model, data)
        r_body_eci = data.orbit.R_eci + data.xipos[1] * 1e-3
        # Direct-subtraction reference loses ~log10(||r_chief||/||rho||) digits
        # to floating-point cancellation, while the Encke form used internally
        # is exact, so the two agree only down to that cancellation floor.
        expected = (
            model.body_mass[1]
            * (total_accel(r_body_eci, use_j2=False) - total_accel(data.orbit.R_eci, use_j2=False))
            * 1e3
        )
        np.testing.assert_allclose(data.wrench_buffer[1, :3], expected, rtol=1e-6, atol=1e-12)


class TestCWLimit:
    def _run_free_drift(
        self,
        x0_m: float,
        y0_m: float,
        z0_m: float,
        vx0_ms: float,
        vy0_ms: float,
        vz0_ms: float,
        t_sec: float,
        dt: float = 0.001,
    ) -> tuple[np.ndarray, np.ndarray]:
        model, data = _make_model_data(mj_timestep=dt)
        set_freejoint_lvlh_state(
            data,
            slice(0, 3),
            slice(0, 3),
            [x0_m, y0_m, z0_m],
            [vx0_ms, vy0_ms, vz0_ms],
        )
        mjo_forward(model, data)

        for _ in range(int(t_sec / dt)):
            mjo_step(model, data)

        return get_freejoint_lvlh_state(data, slice(0, 3), slice(0, 3))

    def _cw_analytical(self, x0, y0, z0, vx0, vy0, vz0, n, t):
        nt = n * t
        cn = np.cos(nt)
        sn = np.sin(nt)

        x = (4.0 - 3.0 * cn) * x0 + sn / n * vx0 + 2.0 / n * (1.0 - cn) * vy0
        y = 6.0 * (sn - nt) * x0 + y0 - 2.0 / n * (1.0 - cn) * vx0 + (4.0 * sn - 3.0 * nt) / n * vy0
        z = z0 * cn + vz0 / n * sn

        vx = 3.0 * n * sn * x0 + cn * vx0 + 2.0 * sn * vy0
        vy = 6.0 * n * (cn - 1.0) * x0 - 2.0 * sn * vx0 + (4.0 * cn - 3.0) * vy0
        vz = -z0 * n * sn + vz0 * cn

        return np.array([x, y, z]), np.array([vx, vy, vz])

    def test_radial_offset_drift(self):
        a_km = R_EARTH + 400.0
        n = np.sqrt(GM_EARTH / a_km**3)
        pos, vel = self._run_free_drift(10.0, 0, 0, 0, 0, 0, 10.0, dt=0.001)
        pos_cw, vel_cw = self._cw_analytical(10.0, 0, 0, 0, 0, 0, n, 10.0)

        np.testing.assert_allclose(pos, pos_cw, rtol=0.01, atol=0.05)
        np.testing.assert_allclose(vel, vel_cw, rtol=0.01, atol=3e-4)

    def test_along_track_velocity(self):
        a_km = R_EARTH + 400.0
        n = np.sqrt(GM_EARTH / a_km**3)
        pos, vel = self._run_free_drift(0, 0, 0, 0, 0.1, 0, 10.0, dt=0.001)
        pos_cw, vel_cw = self._cw_analytical(0, 0, 0, 0, 0.1, 0, n, 10.0)

        np.testing.assert_allclose(pos, pos_cw, rtol=0.01, atol=0.05)
        np.testing.assert_allclose(vel, vel_cw, rtol=0.01, atol=1e-5)

    def test_cross_track_oscillation(self):
        a_km = R_EARTH + 400.0
        n = np.sqrt(GM_EARTH / a_km**3)
        pos, _ = self._run_free_drift(0, 0, 5.0, 0, 0, 0, 30.0, dt=0.001)
        pos_cw, _ = self._cw_analytical(0, 0, 5.0, 0, 0, 0, n, 30.0)
        np.testing.assert_allclose(pos, pos_cw, rtol=0.01, atol=0.15)

    def test_combined_motion(self):
        a_km = R_EARTH + 400.0
        n = np.sqrt(GM_EARTH / a_km**3)
        pos, vel = self._run_free_drift(5.0, -3.0, 2.0, 0.01, -0.02, 0.005, 20.0, dt=0.001)
        pos_cw, vel_cw = self._cw_analytical(5.0, -3.0, 2.0, 0.01, -0.02, 0.005, n, 20.0)

        np.testing.assert_allclose(pos, pos_cw, rtol=0.02, atol=1e-3)
        np.testing.assert_allclose(vel, vel_cw, rtol=0.02, atol=1e-4)


class TestGravityGradient:
    def test_gravity_gradient_torque_direction(self):
        xml = """<mujoco model="dumbbell">
          <option timestep="0.01" gravity="0 0 0"/>
          <worldbody>
            <body name="upper" pos="0.5 0 0">
              <freejoint/>
              <geom type="sphere" size="0.1" mass="50"/>
            </body>
            <body name="lower" pos="-0.5 0 0">
              <freejoint/>
              <geom type="sphere" size="0.1" mass="50"/>
            </body>
          </worldbody>
        </mujoco>"""
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as file:
            file.write(xml)
            xml_path = file.name

        try:
            model, data = _make_model_data(xml_path=xml_path, mj_timestep=0.01)
            data.clear_wrench_buffer()
            apply_inertial_wrenches(model, data)

            force_upper = data.frame.C_LI @ data.wrench_buffer[1, :3]
            force_lower = data.frame.C_LI @ data.wrench_buffer[2, :3]

            a_km = R_EARTH + 400.0
            n = np.sqrt(GM_EARTH / a_km**3)
            force_expected = 50.0 * 2.0 * n**2 * 0.5e-3 * 1e3

            assert force_upper[0] > 0
            assert force_lower[0] < 0
            np.testing.assert_allclose(abs(force_upper[0]), force_expected, rtol=0.01)
            np.testing.assert_allclose(abs(force_lower[0]), force_expected, rtol=0.01)
        finally:
            os.unlink(xml_path)

    def test_j2_bodywise_radial_asymmetry(self):
        model, data = _make_model_data(use_j2=True)

        set_freejoint_lvlh_state(data, slice(0, 3), slice(0, 3), [1.0, 0.0, 0.0])
        mjo_forward(model, data)
        data.clear_wrench_buffer()
        apply_inertial_wrenches(model, data)
        force_plus = data.frame.C_LI @ data.wrench_buffer[1, :3]

        set_freejoint_lvlh_state(data, slice(0, 3), slice(0, 3), [-1.0, 0.0, 0.0])
        mjo_forward(model, data)
        data.clear_wrench_buffer()
        apply_inertial_wrenches(model, data)
        force_minus = data.frame.C_LI @ data.wrench_buffer[1, :3]

        assert abs(force_plus[0]) > 0
        assert abs(force_minus[0]) > 0
        asymmetry = abs(force_plus[0] + force_minus[0]) / (
            abs(force_plus[0]) + abs(force_minus[0])
        )
        assert asymmetry < 0.01

    def test_j2_force_at_reference_is_finite(self):
        model, data = _make_model_data(use_j2=True)
        data.clear_wrench_buffer()
        apply_inertial_wrenches(model, data)
        assert np.all(np.isfinite(data.wrench_buffer[1, :3]))


class TestStepIntegration:
    def test_orbit_propagates(self):
        model, data = _make_model_data()
        t0 = data.orbit.t
        for _ in range(10):
            mjo_step(model, data)
        assert data.orbit.t > t0

    def test_orbit_dt_overrides_mujoco_timestep(self):
        model, data = _make_model_data(orbit_dt=0.25)
        mjo_step(model, data)
        np.testing.assert_allclose(data.orbit.t, model.opt.timestep, atol=1e-12)

    def test_step_recomputes_frame_cache_with_j2(self):
        a = R_EARTH + 700.0
        r_eci, v_eci = keplerian_to_cartesian(
            a=a,
            e=0.1,
            inc=np.deg2rad(63.4),
            raan=np.deg2rad(15.0),
            argp=np.deg2rad(25.0),
            nu=np.deg2rad(70.0),
        )
        model, data = _make_model_data(use_j2=True)
        data.reset(OrbitInit(R_eci=r_eci, V_eci=v_eci))

        mjo_step(model, data)

        fc_j2 = update_frame_cache(data.orbit, use_j2=True)
        fc_two_body = update_frame_cache(data.orbit, use_j2=False)
        np.testing.assert_allclose(data.frame.omega_dot_lvlh, fc_j2.omega_dot_lvlh, atol=1e-16)
        assert not np.allclose(data.frame.omega_dot_lvlh, fc_two_body.omega_dot_lvlh, atol=1e-16)

    def test_stationary_body_stays_near_origin(self):
        model, data = _make_model_data()
        for _ in range(1000):
            mjo_step(model, data)
        assert np.linalg.norm(data.lvlh_position_from_world(data.qpos[:3])) < 10.0

    def test_long_run_finite(self):
        model, data = _make_model_data()
        set_freejoint_lvlh_state(data, slice(0, 3), slice(0, 3), [5.0, 0.0, 0.0])
        mjo_forward(model, data)
        for _ in range(3000):
            mjo_step(model, data)
        assert np.all(np.isfinite(data.qpos))
        assert np.all(np.isfinite(data.qvel))
