"""Phase 6 validation: magnetic torques (tau = m × B)."""

import numpy as np

from mujoco_orbit import MagneticBodySpec, mjo_forward
from mujoco_orbit.coupling.magnetic import apply_magnetic_wrenches
from mujoco_orbit.testdata import FREE_BODY_XML

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


class TestMagneticTorque:
    def test_cross_product_sign(self):
        model, data = _make_model_data(
            magnetic_bodies=[
                MagneticBodySpec(body_name="spacecraft", dipole_body=np.array([1.0, 0.0, 0.0]))
            ]
        )
        b_test = np.array([0.0, 0.0, 1.0])
        data.env.mag_field_eci = data.frame.C_IL @ b_test

        data.clear_wrench_buffer()
        apply_magnetic_wrenches(model, data)

        tau = data.wrench_buffer[1, 3:]
        np.testing.assert_allclose(tau, [0.0, -1.0, 0.0], atol=1e-12)

    def test_parallel_dipole_zero_torque(self):
        model, data = _make_model_data(
            magnetic_bodies=[
                MagneticBodySpec(body_name="spacecraft", dipole_body=np.array([0.0, 0.0, 1.0]))
            ]
        )
        b_test = np.array([0.0, 0.0, 1.0])
        data.env.mag_field_eci = data.frame.C_IL @ b_test

        data.clear_wrench_buffer()
        apply_magnetic_wrenches(model, data)

        np.testing.assert_allclose(data.wrench_buffer[1, 3:], 0.0, atol=1e-14)

    def test_no_translational_force(self):
        model, data = _make_model_data(
            magnetic_bodies=[
                MagneticBodySpec(body_name="spacecraft", dipole_body=np.array([1.0, 0.0, 0.0]))
            ]
        )
        data.clear_wrench_buffer()
        apply_magnetic_wrenches(model, data)
        np.testing.assert_allclose(data.wrench_buffer[1, :3], 0.0, atol=1e-20)

    def test_torque_magnitude(self):
        dipole_mag = 5.0
        b_mag = 1e-5
        model, data = _make_model_data(
            magnetic_bodies=[
                MagneticBodySpec(
                    body_name="spacecraft",
                    dipole_body=np.array([dipole_mag, 0.0, 0.0]),
                )
            ]
        )
        b_test = np.array([0.0, b_mag, 0.0])
        data.env.mag_field_eci = data.frame.C_IL @ b_test

        data.clear_wrench_buffer()
        apply_magnetic_wrenches(model, data)

        expected_mag = dipole_mag * b_mag
        np.testing.assert_allclose(
            np.linalg.norm(data.wrench_buffer[1, 3:]),
            expected_mag,
            rtol=1e-10,
        )

    def test_rotated_body_frame(self):
        model, data = _make_model_data(
            magnetic_bodies=[
                MagneticBodySpec(body_name="spacecraft", dipole_body=np.array([0.0, 0.0, 1.0]))
            ]
        )

        angle = np.pi / 2
        data.qpos[3] = np.cos(angle / 2)
        data.qpos[4] = 0.0
        data.qpos[5] = np.sin(angle / 2)
        data.qpos[6] = 0.0
        mjo_forward(model, data)

        b_test = np.array([0.0, 0.0, 1.0])
        data.env.mag_field_eci = data.frame.C_IL @ b_test

        data.clear_wrench_buffer()
        apply_magnetic_wrenches(model, data)

        np.testing.assert_allclose(data.wrench_buffer[1, 3:], [0, -1, 0], atol=0.01)

    def test_disabled(self):
        model, data = _make_model_data(
            magnetic_bodies=[
                MagneticBodySpec(body_name="spacecraft", dipole_body=np.array([1.0, 0.0, 0.0]))
            ],
            use_magnetic=False,
        )
        data.clear_wrench_buffer()
        apply_magnetic_wrenches(model, data)
        np.testing.assert_allclose(data.wrench_buffer[1], 0.0)
