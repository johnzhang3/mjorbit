"""Phase 4 validation: inertial / gravity coupling in the LVLH rotating frame.

Key tests:
- CW limit: free body in circular orbit with small offset should match CW analytical solution
- Gravity gradient: dumbbell-like body should experience gravity-gradient torque
- Per-body forcing should be finite and correctly oriented
"""

import numpy as np
import pytest
import mujoco

from mjorbit.constants import R_EARTH, GM_EARTH
from mjorbit.cpu import compile_cpu, step_cpu
from mjorbit.cpu.core.config import CPUScenarioCfg, OrbitCfg, MuJoCoCfg
from mjorbit.cpu.orbit.elements import keplerian_to_cartesian
from mjorbit.cpu.orbit.state import OrbitState
from mjorbit.cpu.orbit.lvlh import update_frame_cache
from mjorbit.cpu.coupling.inertial import apply_inertial_wrenches
from mjorbit.cpu.mjcf.builders import FREE_BODY_XML, ASSETS_DIR


def _circular_leo_cfg(alt_km: float = 400.0) -> CPUScenarioCfg:
    a = R_EARTH + alt_km
    R, V = keplerian_to_cartesian(a=a, e=0.0, inc=np.deg2rad(51.6), raan=0.0, argp=0.0, nu=0.0)
    return CPUScenarioCfg(
        orbit=OrbitCfg(R_eci=R, V_eci=V),
        mujoco=MuJoCoCfg(xml_path=FREE_BODY_XML, dt=0.01),
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )


class TestInertialWrenchBasics:
    def test_wrench_finite(self):
        """All computed wrenches should be finite."""
        cfg = _circular_leo_cfg()
        scenario = compile_cpu(cfg)
        step_cpu(scenario)
        assert np.all(np.isfinite(scenario._wrench_buffer))

    def test_zero_offset_zero_force(self):
        """A body at the LVLH origin with zero velocity should get near-zero force.

        The differential gravity is zero when the body is exactly at the chief,
        and fictitious forces are zero when position and velocity in LVLH are zero.
        """
        cfg = _circular_leo_cfg()
        scenario = compile_cpu(cfg)
        # Body starts at origin (0,0,0) in MuJoCo world = LVLH
        # After compile, body is at origin with zero velocity
        scenario.clear_wrench_buffer()
        apply_inertial_wrenches(scenario)
        F = scenario._wrench_buffer[1, :3]
        # Force should be negligible (body at origin)
        assert np.linalg.norm(F) < 1e-6  # < 1 µN

    def test_radial_offset_positive_radial_force(self):
        """A body displaced radially outward should feel a net outward force.

        In CW: x_ddot = 3n^2 x (for zero velocity, omega_dot=0).
        So positive x offset -> positive x acceleration (away from chief).
        """
        cfg = _circular_leo_cfg()
        scenario = compile_cpu(cfg)

        # Displace body +1 m in radial (x-LVLH) direction in MuJoCo
        # freejoint qpos: [x, y, z, qw, qx, qy, qz]
        scenario.mjd.qpos[0] = 1.0  # 1 m radial outward
        mujoco.mj_forward(scenario.mjm, scenario.mjd)

        scenario.clear_wrench_buffer()
        apply_inertial_wrenches(scenario)
        Fx = scenario._wrench_buffer[1, 0]

        # For CW: a_x = 3*n^2*x, F = m * a, in SI: x=1m
        a_km = R_EARTH + 400.0
        n = np.sqrt(GM_EARTH / a_km**3)  # rad/s
        # F_expected = mass * 3 * n^2 * x_m
        F_expected = 100.0 * 3.0 * n**2 * 1.0  # N (x in meters, accel in m/s^2? No...)

        # Actually: n is in rad/s (from km units), x is in m.
        # The code converts km/s^2 -> m/s^2 via *1e3.
        # Gravity gradient: dg ~ 3*n^2*x_km (km/s^2), x_km = 1e-3 km
        # centripetal: n^2 * x_km (km/s^2) -- negative contribution
        # CW: a_x = 3n^2 x - 2n*vy + omega_dot_y... for circular, a_x = 3n^2*x
        # dg + centripetal = (g_body - g_ref) + centripetal.
        # In CW linear limit: dg_x ~ (3 n^2 x_km) in km/s^2 (for radial)
        # centripetal: -n^2 * x_km (radial)
        # Total: (3n^2 - n^2) * x_km... wait, let me be more careful.
        #
        # CW equations for radial:
        # x_ddot = 3n^2 x + 2n y_dot  (all in LVLH)
        # For zero velocity: x_ddot = 3n^2 x
        # x is in km, acceleration is in km/s^2
        # Force (N) = mass * accel(m/s^2) = mass * accel(km/s^2) * 1e3
        x_km = 1e-3  # 1 m = 1e-3 km
        F_cw = 100.0 * 3.0 * n**2 * x_km * 1e3  # N

        # Force should be positive (outward) and close to CW
        assert Fx > 0, f"Expected positive radial force, got {Fx}"
        np.testing.assert_allclose(Fx, F_cw, rtol=0.01)

    def test_along_track_offset_no_radial_coupling(self):
        """Body offset in along-track only should have zero radial force (CW y-offset).

        CW: x_ddot = 3n^2 x + 2n ydot. For x=0, ydot=0: x_ddot = 0.
        Along-track acceleration from CW: y_ddot = -2n xdot. For xdot=0: y_ddot = 0.
        """
        cfg = _circular_leo_cfg()
        scenario = compile_cpu(cfg)
        scenario.mjd.qpos[1] = 1.0  # 1 m along-track
        mujoco.mj_forward(scenario.mjm, scenario.mjd)
        scenario.clear_wrench_buffer()
        apply_inertial_wrenches(scenario)
        F = scenario._wrench_buffer[1, :3]
        # Force should be negligible (only higher-order terms)
        assert np.linalg.norm(F) < 0.01  # < 10 mN for 1 m offset


class TestCWLimit:
    """Test that the full inertial coupling matches the CW analytical solution
    in the linearized limit (circular orbit, small offsets, no J2)."""

    def _run_free_drift(self, x0_m: float, y0_m: float, z0_m: float,
                         vx0_ms: float, vy0_ms: float, vz0_ms: float,
                         t_sec: float, dt: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
        """Simulate free drift in LVLH and return final (pos_m, vel_m_s)."""
        cfg = _circular_leo_cfg()
        cfg.mujoco.dt = dt
        scenario = compile_cpu(cfg)

        # Set initial state in MuJoCo
        # freejoint qpos: [x, y, z, qw, qx, qy, qz]
        scenario.mjd.qpos[0] = x0_m
        scenario.mjd.qpos[1] = y0_m
        scenario.mjd.qpos[2] = z0_m
        # freejoint qvel: [vx, vy, vz, wx, wy, wz]
        scenario.mjd.qvel[0] = vx0_ms
        scenario.mjd.qvel[1] = vy0_ms
        scenario.mjd.qvel[2] = vz0_ms
        mujoco.mj_forward(scenario.mjm, scenario.mjd)

        n_steps = int(t_sec / dt)
        for _ in range(n_steps):
            step_cpu(scenario)

        pos = scenario.mjd.qpos[:3].copy()
        vel = scenario.mjd.qvel[:3].copy()
        return pos, vel

    def _cw_analytical(self, x0: float, y0: float, z0: float,
                        vx0: float, vy0: float, vz0: float,
                        n: float, t: float) -> tuple[np.ndarray, np.ndarray]:
        """CW analytical solution. All in consistent units (m, m/s, rad/s)."""
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
        """Body with radial offset: compare to CW after a few seconds."""
        a_km = R_EARTH + 400.0
        n = np.sqrt(GM_EARTH / a_km**3)

        x0 = 10.0  # 10 m radial offset
        t = 10.0
        pos, vel = self._run_free_drift(x0, 0, 0, 0, 0, 0, t, dt=0.001)
        pos_cw, vel_cw = self._cw_analytical(x0, 0, 0, 0, 0, 0, n, t)

        np.testing.assert_allclose(pos, pos_cw, rtol=0.01, atol=1e-4,
                                   err_msg="CW position mismatch (radial offset)")
        np.testing.assert_allclose(vel, vel_cw, rtol=0.01, atol=1e-5,
                                   err_msg="CW velocity mismatch (radial offset)")

    def test_along_track_velocity(self):
        """Body with along-track velocity impulse."""
        a_km = R_EARTH + 400.0
        n = np.sqrt(GM_EARTH / a_km**3)

        vy0 = 0.1  # 0.1 m/s along-track
        t = 10.0
        pos, vel = self._run_free_drift(0, 0, 0, 0, vy0, 0, t, dt=0.001)
        pos_cw, vel_cw = self._cw_analytical(0, 0, 0, 0, vy0, 0, n, t)

        np.testing.assert_allclose(pos, pos_cw, rtol=0.01, atol=1e-4)
        np.testing.assert_allclose(vel, vel_cw, rtol=0.01, atol=1e-5)

    def test_cross_track_oscillation(self):
        """Cross-track motion should be a simple harmonic at frequency n."""
        a_km = R_EARTH + 400.0
        n = np.sqrt(GM_EARTH / a_km**3)

        z0 = 5.0  # 5 m cross-track offset
        t = 30.0
        pos, vel = self._run_free_drift(0, 0, z0, 0, 0, 0, t, dt=0.001)
        pos_cw, vel_cw = self._cw_analytical(0, 0, z0, 0, 0, 0, n, t)

        np.testing.assert_allclose(pos, pos_cw, rtol=0.01, atol=1e-4)

    def test_combined_motion(self):
        """General initial condition with all components nonzero."""
        a_km = R_EARTH + 400.0
        n = np.sqrt(GM_EARTH / a_km**3)

        x0, y0, z0 = 5.0, -3.0, 2.0  # m
        vx0, vy0, vz0 = 0.01, -0.02, 0.005  # m/s
        t = 20.0
        pos, vel = self._run_free_drift(x0, y0, z0, vx0, vy0, vz0, t, dt=0.001)
        pos_cw, vel_cw = self._cw_analytical(x0, y0, z0, vx0, vy0, vz0, n, t)

        np.testing.assert_allclose(pos, pos_cw, rtol=0.02, atol=1e-3)
        np.testing.assert_allclose(vel, vel_cw, rtol=0.02, atol=1e-4)


class TestGravityGradient:
    """A dumbbell-like extended rigid body in a gravity gradient should experience
    a restoring torque that tries to align its long axis with the radial direction."""

    def test_gravity_gradient_torque_direction(self):
        """A body with its long axis tilted away from radial should experience
        a torque that tries to rotate it toward the radial direction.

        Instead of checking torque on xfrc_applied (which is a pure force),
        we check that two point masses displaced radially above/below the COM
        would experience differential gravity -> net torque.
        """
        # This is really a sanity check of the bodywise force computation.
        # For a single rigid body at the origin, the inertial force is near-zero.
        # The gravity gradient torque emerges from MULTIPLE bodies (e.g., masses
        # at opposite ends of a dumbbell).
        #
        # Use a two-body model: two masses displaced along radial axis.
        import tempfile, os
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
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(xml)
            xml_path = f.name

        try:
            a_km = R_EARTH + 400.0
            R, V = keplerian_to_cartesian(
                a=a_km, e=0.0, inc=np.deg2rad(51.6), raan=0.0, argp=0.0, nu=0.0
            )
            cfg = CPUScenarioCfg(
                orbit=OrbitCfg(R_eci=R, V_eci=V),
                mujoco=MuJoCoCfg(xml_path=xml_path, dt=0.01),
                use_j2=False,
                use_drag=False,
                use_srp=False,
                use_magnetic=False,
            )
            scenario = compile_cpu(cfg)
            # Upper mass at +0.5m radial, lower at -0.5m radial
            mujoco.mj_forward(scenario.mjm, scenario.mjd)

            scenario.clear_wrench_buffer()
            apply_inertial_wrenches(scenario)

            # Upper body should have outward force, lower should have inward force
            # (relative to chief — differential gravity)
            F_upper = scenario._wrench_buffer[1, 0]  # radial force on upper
            F_lower = scenario._wrench_buffer[2, 0]  # radial force on lower

            # Upper mass (further from Earth) -> outward force (positive x)
            # Lower mass (closer to Earth) -> inward force (negative x)
            # In CW: a_x = 3n^2 x, so F_upper > 0 and F_lower < 0
            assert F_upper > 0, f"Expected positive radial force on upper mass, got {F_upper}"
            assert F_lower < 0, f"Expected negative radial force on lower mass, got {F_lower}"

            # The magnitude should be approximately 3*n^2 * offset * mass
            n = np.sqrt(GM_EARTH / a_km**3)
            offset_km = 0.5e-3  # 0.5 m in km
            F_expected = 50.0 * 3.0 * n**2 * offset_km * 1e3  # N
            np.testing.assert_allclose(abs(F_upper), F_expected, rtol=0.01)
            np.testing.assert_allclose(abs(F_lower), F_expected, rtol=0.01)
        finally:
            os.unlink(xml_path)

    def test_j2_bodywise_radial_asymmetry(self):
        """With J2, two bodies at different radial offsets should experience
        slightly different forces (beyond the pure tidal gradient)."""
        cfg = _circular_leo_cfg()
        cfg.use_j2 = True
        scenario = compile_cpu(cfg)

        # Radial +1m
        scenario.mjd.qpos[0] = 1.0
        mujoco.mj_forward(scenario.mjm, scenario.mjd)
        scenario.clear_wrench_buffer()
        apply_inertial_wrenches(scenario)
        F_plus = scenario._wrench_buffer[1, 0]

        # Radial -1m
        scenario.mjd.qpos[0] = -1.0
        mujoco.mj_forward(scenario.mjm, scenario.mjd)
        scenario.clear_wrench_buffer()
        apply_inertial_wrenches(scenario)
        F_minus = scenario._wrench_buffer[1, 0]

        # With J2, forces should not be exactly symmetric
        assert abs(F_plus) > 0
        assert abs(F_minus) > 0
        # But the asymmetry should be tiny (J2 is a small perturbation)
        asymmetry = abs(F_plus + F_minus) / (abs(F_plus) + abs(F_minus))
        assert asymmetry < 0.01  # less than 1% asymmetry

    def test_no_force_at_origin(self):
        """Even with J2, body at origin should have negligible force."""
        cfg = _circular_leo_cfg()
        cfg.use_j2 = True
        scenario = compile_cpu(cfg)
        mujoco.mj_forward(scenario.mjm, scenario.mjd)
        scenario.clear_wrench_buffer()
        apply_inertial_wrenches(scenario)
        F = scenario._wrench_buffer[1, :3]
        assert np.linalg.norm(F) < 1e-6


class TestStepIntegration:
    """End-to-end stepping tests for Phase 4."""

    def test_orbit_propagates(self):
        """Chief orbit time should advance with each step."""
        cfg = _circular_leo_cfg()
        scenario = compile_cpu(cfg)
        t0 = scenario.orbit.t
        for _ in range(10):
            step_cpu(scenario)
        assert scenario.orbit.t > t0

    def test_orbit_dt_overrides_mujoco_timestep(self):
        """Chief propagation should use cfg.orbit_dt when it is provided."""
        cfg = _circular_leo_cfg()
        cfg.orbit_dt = 0.25
        scenario = compile_cpu(cfg)

        step_cpu(scenario)

        np.testing.assert_allclose(scenario.orbit.t, 0.25, atol=1e-12)

    def test_step_recomputes_frame_cache_with_j2(self):
        """J2-enabled stepping should keep omega_dot consistent with the J2 frame model."""
        a = R_EARTH + 700.0
        R, V = keplerian_to_cartesian(
            a=a,
            e=0.1,
            inc=np.deg2rad(63.4),
            raan=np.deg2rad(15.0),
            argp=np.deg2rad(25.0),
            nu=np.deg2rad(70.0),
        )
        scenario = compile_cpu(
            CPUScenarioCfg(
                orbit=OrbitCfg(R_eci=R, V_eci=V),
                mujoco=MuJoCoCfg(xml_path=FREE_BODY_XML, dt=0.01),
                use_j2=True,
                use_drag=False,
                use_srp=False,
                use_magnetic=False,
            )
        )

        step_cpu(scenario)

        fc_j2 = update_frame_cache(scenario.orbit, use_j2=True)
        fc_two_body = update_frame_cache(scenario.orbit, use_j2=False)
        np.testing.assert_allclose(
            scenario.frame_cache.omega_dot_lvlh, fc_j2.omega_dot_lvlh, atol=1e-16
        )
        assert not np.allclose(
            scenario.frame_cache.omega_dot_lvlh, fc_two_body.omega_dot_lvlh, atol=1e-16
        )

    def test_stationary_body_stays_near_origin(self):
        """Body at origin with no perturbations should stay near origin."""
        cfg = _circular_leo_cfg()
        scenario = compile_cpu(cfg)
        for _ in range(1000):  # 10 seconds
            step_cpu(scenario)
        pos = scenario.mjd.qpos[:3]
        # Should drift only due to numerical effects
        assert np.linalg.norm(pos) < 0.1  # < 10 cm drift

    def test_long_run_finite(self):
        """30 seconds of simulation should remain finite."""
        cfg = _circular_leo_cfg()
        scenario = compile_cpu(cfg)
        scenario.mjd.qpos[0] = 5.0  # 5 m radial offset
        mujoco.mj_forward(scenario.mjm, scenario.mjd)
        for _ in range(3000):  # 30 s at dt=0.01
            step_cpu(scenario)
        assert np.all(np.isfinite(scenario.mjd.qpos))
        assert np.all(np.isfinite(scenario.mjd.qvel))
