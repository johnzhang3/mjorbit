"""Phase 9 end-to-end coupled simulation tests.

Validates:
- Full step remains finite
- External wrench bookkeeping is inspectable per body
- No-contact articulated motions behave sensibly
- Internal manipulator motion alone does not change COM orbit
- Thruster and drag forces do change COM orbit in the expected direction
- Contact scenario remains stable with external loads
- Free bodies drift correctly (CW golden test)
"""

import tempfile
import os

import numpy as np
import pytest
import mujoco

from mjorbit.constants import R_EARTH, GM_EARTH
from mjorbit.cpu import compile_cpu, step_cpu
from mjorbit.cpu.core.config import (
    CPUScenarioCfg, OrbitCfg, MuJoCoCfg,
    SurfaceCfg, ThrusterCfg, ReactionWheelCfg,
)
from mjorbit.cpu.orbit.elements import keplerian_to_cartesian
from mjorbit.cpu.mjcf.builders import FREE_BODY_XML, SPACECRAFT_ARM_XML


def _leo_cfg(xml_path: str = FREE_BODY_XML, alt_km: float = 400.0, **kw) -> CPUScenarioCfg:
    a = R_EARTH + alt_km
    R, V = keplerian_to_cartesian(a=a, e=0.0, inc=np.deg2rad(51.6), raan=0.0, argp=0.0, nu=0.0)
    defaults = dict(use_j2=False, use_drag=False, use_srp=False, use_magnetic=False)
    defaults.update(kw)
    return CPUScenarioCfg(
        orbit=OrbitCfg(R_eci=R, V_eci=V),
        mujoco=MuJoCoCfg(xml_path=xml_path, dt=0.01),
        **defaults,
    )


class TestFullStepFinite:
    """Full coupled step should remain finite under various conditions."""

    def test_free_body_long_run(self):
        """60 s of free-body simulation with radial offset stays finite."""
        cfg = _leo_cfg()
        scenario = compile_cpu(cfg)
        scenario.mjd.qpos[0] = 10.0  # 10 m radial
        mujoco.mj_forward(scenario.mjm, scenario.mjd)
        for _ in range(6000):  # 60 s
            step_cpu(scenario)
        assert np.all(np.isfinite(scenario.mjd.qpos))
        assert np.all(np.isfinite(scenario.mjd.qvel))

    def test_arm_model_long_run(self):
        """30 s of articulated arm model stays finite."""
        cfg = _leo_cfg(xml_path=SPACECRAFT_ARM_XML)
        scenario = compile_cpu(cfg)
        ctrl = np.array([0.5, -0.3])
        for _ in range(15000):  # 30 s
            step_cpu(scenario, ctrl=ctrl)
        assert np.all(np.isfinite(scenario.mjd.qpos))
        assert np.all(np.isfinite(scenario.mjd.qvel))

    def test_with_j2_enabled(self):
        """J2-enabled free body stays finite."""
        cfg = _leo_cfg(use_j2=True)
        scenario = compile_cpu(cfg)
        scenario.mjd.qpos[0] = 5.0
        mujoco.mj_forward(scenario.mjm, scenario.mjd)
        for _ in range(3000):  # 30 s
            step_cpu(scenario)
        assert np.all(np.isfinite(scenario.mjd.qpos))
        assert np.all(np.isfinite(scenario.mjd.qvel))


class TestWrenchBookkeeping:
    """Wrench buffer should be inspectable per body."""

    def test_wrench_shape(self):
        cfg = _leo_cfg()
        scenario = compile_cpu(cfg)
        assert scenario._wrench_buffer.shape == (scenario.mjm.nbody, 6)

    def test_clear_resets(self):
        cfg = _leo_cfg()
        scenario = compile_cpu(cfg)
        step_cpu(scenario)
        scenario.clear_wrench_buffer()
        assert np.all(scenario._wrench_buffer == 0.0)
        assert np.all(scenario.mjd.xfrc_applied == 0.0)


class TestInternalMotionConservation:
    """Internal manipulator motion should not change the COM orbit."""

    def test_arm_self_motion_preserves_orbit(self):
        """Arm moving under its own joints should not perturb the orbit.

        The orbit feedback should be essentially zero because all wrenches
        are internal (gravity/inertial forces cancel at the system level
        for small offsets, and there are no external forces like drag).
        """
        cfg = _leo_cfg(xml_path=SPACECRAFT_ARM_XML)
        cfg.mujoco.dt = 0.002
        scenario = compile_cpu(cfg)
        orbit_0 = scenario.orbit.copy()
        v0 = np.linalg.norm(orbit_0.V_eci)

        ctrl = np.array([1.0, -0.5])
        for _ in range(5000):  # 10 s
            step_cpu(scenario, ctrl=ctrl)

        v1 = np.linalg.norm(scenario.orbit.V_eci)
        # Speed change should be negligible (internal motion only)
        dv = abs(v1 - v0)
        assert dv < 1e-9, f"Speed changed by {dv} km/s — internal motion leaking into orbit"

    def test_reaction_wheel_does_not_change_orbit(self):
        """Spinning up a reaction wheel should change attitude but not orbit speed."""
        cfg = _leo_cfg()
        cfg.reaction_wheels = [
            ReactionWheelCfg(
                body_name="spacecraft",
                axis_body=np.array([0.0, 0.0, 1.0]),
                inertia=0.01,  # kg·m^2
                speed_limit=6000.0,
                torque_limit=0.1,
            ),
        ]
        scenario = compile_cpu(cfg)
        orbit_0 = scenario.orbit.copy()
        v0 = np.linalg.norm(orbit_0.V_eci)

        rw_cmd = np.array([0.05])  # N·m
        for _ in range(1000):  # 10 s
            step_cpu(scenario, rw_torques=rw_cmd)

        v1 = np.linalg.norm(scenario.orbit.V_eci)
        assert abs(v1 - v0) < 1e-10, "RW changed orbit speed"

        # But wheel speed should have changed
        assert scenario.actuator_state.rw_speed[0] != 0.0, "RW didn't spin up"

        # And body should have some angular velocity
        angvel = scenario.mjd.qvel[3:6]
        assert np.linalg.norm(angvel) > 1e-4, "Body should have angular velocity from RW"


class TestThrusterChangesOrbit:
    """Thruster forces should feed back into the orbit."""

    def test_prograde_thrust_increases_speed(self):
        """A prograde thruster should increase orbital speed."""
        cfg = _leo_cfg()
        cfg.thrusters = [
            ThrusterCfg(
                body_name="spacecraft",
                position_body=np.array([0.0, 0.0, 0.0]),  # at COM
                direction_body=np.array([0.0, 1.0, 0.0]),  # along-track
                force_limit=10.0,
            ),
        ]
        scenario = compile_cpu(cfg)
        v0 = np.linalg.norm(scenario.orbit.V_eci)

        # Command 5 N prograde thrust
        scenario.actuator_state.thr_force[0] = 5.0
        for _ in range(1000):  # 10 s
            step_cpu(scenario)

        v1 = np.linalg.norm(scenario.orbit.V_eci)
        dv = v1 - v0
        assert dv > 0, f"Expected prograde thrust to increase speed, got dv={dv:.2e}"

        # Rough check: dv ~ F*t/m in km/s
        # F=5N, t=10s, m=100kg -> dv ~ 0.5 m/s = 5e-4 km/s
        expected_dv = 5.0 * 10.0 / 100.0 * 1e-3  # km/s
        np.testing.assert_allclose(dv, expected_dv, rtol=0.1)

    def test_retrograde_thrust_decreases_speed(self):
        """A retrograde thruster should decrease orbital speed."""
        cfg = _leo_cfg()
        cfg.thrusters = [
            ThrusterCfg(
                body_name="spacecraft",
                position_body=np.array([0.0, 0.0, 0.0]),
                direction_body=np.array([0.0, -1.0, 0.0]),  # anti-along-track
                force_limit=10.0,
            ),
        ]
        scenario = compile_cpu(cfg)
        v0 = np.linalg.norm(scenario.orbit.V_eci)

        scenario.actuator_state.thr_force[0] = 5.0
        for _ in range(1000):
            step_cpu(scenario)

        v1 = np.linalg.norm(scenario.orbit.V_eci)
        assert v1 < v0, "Retrograde thrust should decrease speed"


class TestDragChangesOrbit:
    """Drag should decelerate the orbit."""

    def test_drag_reduces_speed(self):
        """Drag on a forward-facing panel should reduce orbital speed."""
        cfg = _leo_cfg(use_drag=True)
        cfg.surfaces = [
            SurfaceCfg(
                body_name="spacecraft",
                center_of_pressure_body=np.array([0.0, 0.0, 0.0]),
                normal_body=np.array([0.0, 1.0, 0.0]),  # facing along-track
                area=10.0,  # m^2, large for visible effect
                drag_coeff=2.2,
                use_drag=True,
                use_srp=False,
            ),
        ]
        scenario = compile_cpu(cfg)
        v0 = np.linalg.norm(scenario.orbit.V_eci)

        for _ in range(3000):  # 30 s
            step_cpu(scenario)

        v1 = np.linalg.norm(scenario.orbit.V_eci)
        # Drag should slow the spacecraft down
        assert v1 < v0, f"Expected drag to slow orbit, got dv={v1-v0:.2e} km/s"


class TestCWGoldenDrift:
    """Golden test: free body should match CW analytical solution."""

    def _cw(self, x0, y0, z0, vx0, vy0, vz0, n, t):
        nt = n * t
        cn, sn = np.cos(nt), np.sin(nt)
        x = (4 - 3*cn)*x0 + sn/n*vx0 + 2/n*(1-cn)*vy0
        y = 6*(sn-nt)*x0 + y0 - 2/n*(1-cn)*vx0 + (4*sn-3*nt)/n*vy0
        z = z0*cn + vz0/n*sn
        return np.array([x, y, z])

    def test_radial_offset_golden(self):
        """10 m radial offset, 30 s run -> matches CW within 0.5%."""
        a_km = R_EARTH + 400.0
        n = np.sqrt(GM_EARTH / a_km**3)

        cfg = _leo_cfg()
        cfg.mujoco.dt = 0.001
        scenario = compile_cpu(cfg)

        x0 = 10.0
        scenario.mjd.qpos[0] = x0
        mujoco.mj_forward(scenario.mjm, scenario.mjd)

        t_end = 30.0
        for _ in range(int(t_end / 0.001)):
            step_cpu(scenario)

        pos = scenario.mjd.qpos[:3]
        pos_cw = self._cw(x0, 0, 0, 0, 0, 0, n, t_end)
        np.testing.assert_allclose(pos, pos_cw, rtol=0.005, atol=1e-4)

    def test_combined_ic_golden(self):
        """General IC, 20 s run -> matches CW within 2%."""
        a_km = R_EARTH + 400.0
        n = np.sqrt(GM_EARTH / a_km**3)

        cfg = _leo_cfg()
        cfg.mujoco.dt = 0.001
        scenario = compile_cpu(cfg)

        x0, y0, z0 = 5.0, -3.0, 2.0
        vx0, vy0, vz0 = 0.01, -0.02, 0.005
        scenario.mjd.qpos[:3] = [x0, y0, z0]
        scenario.mjd.qvel[:3] = [vx0, vy0, vz0]
        mujoco.mj_forward(scenario.mjm, scenario.mjd)

        t_end = 20.0
        for _ in range(int(t_end / 0.001)):
            step_cpu(scenario)

        pos = scenario.mjd.qpos[:3]
        pos_cw = self._cw(x0, y0, z0, vx0, vy0, vz0, n, t_end)
        np.testing.assert_allclose(pos, pos_cw, rtol=0.02, atol=1e-3)


class TestContactStability:
    """Contact scenarios should remain stable with external loads."""

    def test_two_bodies_contact(self):
        """Two free bodies placed close together should remain stable under gravity coupling."""
        xml = """<mujoco model="contact_test">
          <option timestep="0.002" gravity="0 0 0"/>
          <worldbody>
            <body name="body_a" pos="0 0 0">
              <freejoint/>
              <geom type="sphere" size="0.2" mass="50"/>
            </body>
            <body name="body_b" pos="0.5 0 0">
              <freejoint/>
              <geom type="sphere" size="0.2" mass="50"/>
            </body>
          </worldbody>
        </mujoco>"""
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(xml)
            xml_path = f.name

        try:
            cfg = _leo_cfg(xml_path=xml_path)
            cfg.mujoco.dt = 0.002
            scenario = compile_cpu(cfg)

            for _ in range(5000):  # 10 s
                step_cpu(scenario)

            assert np.all(np.isfinite(scenario.mjd.qpos))
            assert np.all(np.isfinite(scenario.mjd.qvel))
        finally:
            os.unlink(xml_path)

    def test_arm_with_surface_loads(self):
        """Articulated arm with drag surface remains stable."""
        cfg = _leo_cfg(xml_path=SPACECRAFT_ARM_XML, use_drag=True)
        cfg.mujoco.dt = 0.002
        cfg.surfaces = [
            SurfaceCfg(
                body_name="spacecraft",
                center_of_pressure_body=np.array([0.0, 0.0, 0.0]),
                normal_body=np.array([0.0, 1.0, 0.0]),
                area=5.0,
                drag_coeff=2.2,
                use_drag=True,
                use_srp=False,
            ),
        ]
        scenario = compile_cpu(cfg)
        ctrl = np.array([0.3, -0.2])

        for _ in range(5000):  # 10 s
            step_cpu(scenario, ctrl=ctrl)

        assert np.all(np.isfinite(scenario.mjd.qpos))
        assert np.all(np.isfinite(scenario.mjd.qvel))
