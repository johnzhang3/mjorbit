"""Phase 9 end-to-end coupled simulation tests."""

import os
import tempfile

import numpy as np

from mujoco_orbit import ReactionWheelSpec, SurfaceSpec, ThrusterSpec, mjo_forward, mjo_step
from mujoco_orbit.constants import GM_EARTH, R_EARTH
from mujoco_orbit.testdata import FREE_BODY_XML, SPACECRAFT_ARM_XML

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


class TestFullStepFinite:
    def test_free_body_long_run(self):
        model, data = _make_model_data()
        data.qpos[0] = 10.0
        mjo_forward(model, data)
        for _ in range(6000):
            mjo_step(model, data)
        assert np.all(np.isfinite(data.qpos))
        assert np.all(np.isfinite(data.qvel))

    def test_arm_model_long_run(self):
        model, data = _make_model_data(xml_path=SPACECRAFT_ARM_XML)
        data.ctrl[:] = np.array([0.5, -0.3])
        for _ in range(15000):
            mjo_step(model, data)
        assert np.all(np.isfinite(data.qpos))
        assert np.all(np.isfinite(data.qvel))

    def test_with_j2_enabled(self):
        model, data = _make_model_data(use_j2=True)
        data.qpos[0] = 5.0
        mjo_forward(model, data)
        for _ in range(3000):
            mjo_step(model, data)
        assert np.all(np.isfinite(data.qpos))
        assert np.all(np.isfinite(data.qvel))


class TestWrenchBookkeeping:
    def test_wrench_shape(self):
        model, data = _make_model_data()
        assert data.wrench_buffer.shape == (model.nbody, 6)

    def test_clear_resets(self):
        model, data = _make_model_data()
        mjo_step(model, data)
        data.clear_wrench_buffer()
        assert np.all(data.wrench_buffer == 0.0)
        assert np.all(data.xfrc_applied == 0.0)


class TestInternalMotionConservation:
    def test_arm_self_motion_preserves_orbit(self):
        model, data = _make_model_data(xml_path=SPACECRAFT_ARM_XML, mj_timestep=0.002)
        v0 = np.linalg.norm(data.orbit.V_eci)

        data.ctrl[:] = np.array([1.0, -0.5])
        for _ in range(5000):
            mjo_step(model, data)

        dv = abs(np.linalg.norm(data.orbit.V_eci) - v0)
        assert dv < 1e-7, f"Speed changed by {dv} km/s — internal motion leaking into orbit"

    def test_reaction_wheel_does_not_change_orbit(self):
        model, data = _make_model_data(
            reaction_wheels=[
                ReactionWheelSpec(
                    body_name="spacecraft",
                    axis_body=np.array([0.0, 0.0, 1.0]),
                    inertia=0.01,
                    speed_limit=6000.0,
                    torque_limit=0.1,
                )
            ]
        )
        v0 = np.linalg.norm(data.orbit.V_eci)

        data.actuators.rw_torque_cmd[0] = 0.05
        for _ in range(1000):
            mjo_step(model, data)

        assert abs(np.linalg.norm(data.orbit.V_eci) - v0) < 1e-10
        assert data.actuators.rw_speed[0] != 0.0
        assert np.linalg.norm(data.qvel[3:6]) > 1e-4


class TestThrusterChangesOrbit:
    def test_prograde_thrust_increases_speed(self):
        model, data = _make_model_data(
            thrusters=[
                ThrusterSpec(
                    body_name="spacecraft",
                    position_body=np.array([0.0, 0.0, 0.0]),
                    direction_body=np.array([0.0, 1.0, 0.0]),
                    force_limit=10.0,
                )
            ]
        )
        v0 = np.linalg.norm(data.orbit.V_eci)
        data.actuators.thr_force_cmd[0] = 5.0

        for _ in range(1000):
            mjo_step(model, data)

        dv = np.linalg.norm(data.orbit.V_eci) - v0
        assert dv > 0
        expected_dv = 5.0 * 10.0 / 100.0 * 1e-3
        np.testing.assert_allclose(dv, expected_dv, rtol=0.1)

    def test_retrograde_thrust_decreases_speed(self):
        model, data = _make_model_data(
            thrusters=[
                ThrusterSpec(
                    body_name="spacecraft",
                    position_body=np.array([0.0, 0.0, 0.0]),
                    direction_body=np.array([0.0, -1.0, 0.0]),
                    force_limit=10.0,
                )
            ]
        )
        v0 = np.linalg.norm(data.orbit.V_eci)
        data.actuators.thr_force_cmd[0] = 5.0

        for _ in range(1000):
            mjo_step(model, data)

        assert np.linalg.norm(data.orbit.V_eci) < v0


class TestDragChangesOrbit:
    def test_drag_reduces_speed(self):
        model, data = _make_model_data(
            use_drag=True,
            surfaces=[
                SurfaceSpec(
                    body_name="spacecraft",
                    center_of_pressure_body=np.array([0.0, 0.0, 0.0]),
                    normal_body=np.array([0.0, 1.0, 0.0]),
                    area=10.0,
                    drag_coeff=2.2,
                    use_drag=True,
                    use_srp=False,
                )
            ],
        )
        v0 = np.linalg.norm(data.orbit.V_eci)

        for _ in range(3000):
            mjo_step(model, data)

        assert np.linalg.norm(data.orbit.V_eci) < v0


class TestCWGoldenDrift:
    def _cw(self, x0, y0, z0, vx0, vy0, vz0, n, t):
        nt = n * t
        cn, sn = np.cos(nt), np.sin(nt)
        x = (4 - 3 * cn) * x0 + sn / n * vx0 + 2 / n * (1 - cn) * vy0
        y = 6 * (sn - nt) * x0 + y0 - 2 / n * (1 - cn) * vx0 + (4 * sn - 3 * nt) / n * vy0
        z = z0 * cn + vz0 / n * sn
        return np.array([x, y, z])

    def test_radial_offset_golden(self):
        a_km = R_EARTH + 400.0
        n = np.sqrt(GM_EARTH / a_km**3)

        model, data = _make_model_data(mj_timestep=0.001)
        x0 = 10.0
        data.qpos[0] = x0
        mjo_forward(model, data)

        t_end = 30.0
        for _ in range(int(t_end / 0.001)):
            mjo_step(model, data)

        np.testing.assert_allclose(
            data.qpos[:3],
            self._cw(x0, 0, 0, 0, 0, 0, n, t_end),
            rtol=0.005,
            atol=1e-4,
        )

    def test_combined_ic_golden(self):
        a_km = R_EARTH + 400.0
        n = np.sqrt(GM_EARTH / a_km**3)

        model, data = _make_model_data(mj_timestep=0.001)
        x0, y0, z0 = 5.0, -3.0, 2.0
        vx0, vy0, vz0 = 0.01, -0.02, 0.005
        data.qpos[:3] = [x0, y0, z0]
        data.qvel[:3] = [vx0, vy0, vz0]
        mjo_forward(model, data)

        t_end = 20.0
        for _ in range(int(t_end / 0.001)):
            mjo_step(model, data)

        np.testing.assert_allclose(
            data.qpos[:3],
            self._cw(x0, y0, z0, vx0, vy0, vz0, n, t_end),
            rtol=0.02,
            atol=1e-3,
        )


class TestContactStability:
    def test_two_bodies_contact(self):
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
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as file:
            file.write(xml)
            xml_path = file.name

        try:
            model, data = _make_model_data(xml_path=xml_path, mj_timestep=0.002)
            for _ in range(5000):
                mjo_step(model, data)

            assert np.all(np.isfinite(data.qpos))
            assert np.all(np.isfinite(data.qvel))
        finally:
            os.unlink(xml_path)

    def test_arm_with_surface_loads(self):
        model, data = _make_model_data(
            xml_path=SPACECRAFT_ARM_XML,
            mj_timestep=0.002,
            use_drag=True,
            surfaces=[
                SurfaceSpec(
                    body_name="spacecraft",
                    center_of_pressure_body=np.array([0.0, 0.0, 0.0]),
                    normal_body=np.array([0.0, 1.0, 0.0]),
                    area=5.0,
                    drag_coeff=2.2,
                    use_drag=True,
                    use_srp=False,
                )
            ],
        )
        data.ctrl[:] = np.array([0.3, -0.2])

        for _ in range(5000):
            mjo_step(model, data)

        assert np.all(np.isfinite(data.qpos))
        assert np.all(np.isfinite(data.qvel))
