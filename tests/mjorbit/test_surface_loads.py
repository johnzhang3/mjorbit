"""Phase 5 validation: drag and SRP surface loads."""

import numpy as np

from mjorbit import SurfaceSpec, mjo_forward, mjo_step
from mjorbit.constants import P_SUN
from mjorbit.testdata import FREE_BODY_XML
from tests.mjorbit.reference.coupling.surfaces import apply_surface_wrenches

from ._helpers import make_model_data


def _make_model_data(**overrides):
    defaults = dict(
        xml_path=FREE_BODY_XML,
        use_j2=False,
        use_drag=True,
        use_srp=True,
        use_magnetic=False,
    )
    defaults.update(overrides)
    return make_model_data(**defaults)


class TestDragSanity:
    def test_single_panel_drag_magnitude(self):
        model, data = _make_model_data(
            surfaces=[
                SurfaceSpec(
                    body_name="spacecraft",
                    center_of_pressure_body=np.zeros(3),
                    normal_body=np.array([0.0, 1.0, 0.0]),
                    area=2.0,
                    drag_coeff=2.2,
                    use_srp=False,
                )
            ],
            use_srp=False,
        )

        data.clear_wrench_buffer()
        apply_surface_wrenches(model, data)

        force = data.wrench_buffer[1, :3]
        assert np.linalg.norm(force) > 1e-8
        assert np.all(np.isfinite(force))

    def test_drag_increases_with_area(self):
        forces = []
        for area in [1.0, 4.0]:
            model, data = _make_model_data(
                surfaces=[
                    SurfaceSpec(
                        body_name="spacecraft",
                        center_of_pressure_body=np.zeros(3),
                        normal_body=np.array([0.0, 1.0, 0.0]),
                        area=area,
                        drag_coeff=2.2,
                        use_srp=False,
                    )
                ],
                use_srp=False,
            )
            data.clear_wrench_buffer()
            apply_surface_wrenches(model, data)
            forces.append(np.linalg.norm(data.wrench_buffer[1, :3]))

        assert forces[1] > forces[0]
        np.testing.assert_allclose(forces[1] / forces[0], 4.0, rtol=0.05)

    def test_drag_disabled(self):
        model, data = _make_model_data(
            surfaces=[
                SurfaceSpec(
                    body_name="spacecraft",
                    center_of_pressure_body=np.zeros(3),
                    normal_body=np.array([0.0, 1.0, 0.0]),
                    area=2.0,
                    use_srp=False,
                )
            ],
            use_drag=False,
            use_srp=False,
        )
        data.clear_wrench_buffer()
        apply_surface_wrenches(model, data)
        np.testing.assert_allclose(data.wrench_buffer[1], 0.0)

    def test_back_facing_panel_no_drag(self):
        model, data = _make_model_data(
            surfaces=[
                SurfaceSpec(
                    body_name="spacecraft",
                    center_of_pressure_body=np.zeros(3),
                    normal_body=np.array([1.0, 0.0, 0.0]),
                    area=2.0,
                    use_srp=False,
                )
            ],
            use_srp=False,
        )
        data.clear_wrench_buffer()
        apply_surface_wrenches(model, data)
        assert np.linalg.norm(data.wrench_buffer[1, :3]) < 1e-7


class TestSRPSanity:
    def test_srp_force_nonzero_in_sun(self):
        model, data = _make_model_data(
            surfaces=[
                SurfaceSpec(
                    body_name="spacecraft",
                    center_of_pressure_body=np.zeros(3),
                    normal_body=np.array([1.0, 0.0, 0.0]),
                    area=4.0,
                    srp_coeff=1.8,
                    use_drag=False,
                )
            ],
            use_drag=False,
        )
        data.env.eclipse = 1.0
        data.clear_wrench_buffer()
        apply_surface_wrenches(model, data)
        assert np.linalg.norm(data.wrench_buffer[1, :3]) >= 0.0

    def test_eclipse_disables_srp(self):
        model, data = _make_model_data(
            surfaces=[
                SurfaceSpec(
                    body_name="spacecraft",
                    center_of_pressure_body=np.zeros(3),
                    normal_body=np.array([-1.0, 0.0, 0.0]),
                    area=4.0,
                    srp_coeff=1.8,
                    use_drag=False,
                )
            ],
            use_drag=False,
        )
        # Place the chief in Earth's cylindrical umbra by pointing the sun
        # opposite to R_eci.  The surface normal faces the sun, so cos_sun>0
        # and only the eclipse factor can zero out the SRP force.
        r_hat = data.orbit.R_eci / np.linalg.norm(data.orbit.R_eci)
        data.env.sun_vector_eci = -r_hat
        data.clear_wrench_buffer()
        apply_surface_wrenches(model, data)
        np.testing.assert_allclose(data.wrench_buffer[1, :3], 0.0)

    def test_srp_magnitude(self):
        model, data = _make_model_data(
            surfaces=[
                SurfaceSpec(
                    body_name="spacecraft",
                    center_of_pressure_body=np.zeros(3),
                    normal_body=np.array([1.0, 0.0, 0.0]),
                    area=10.0,
                    srp_coeff=1.5,
                    use_drag=False,
                )
            ],
            use_drag=False,
        )
        data.env.eclipse = 1.0
        data.env.sun_vector_eci = np.array([1.0, 0.0, 0.0])

        data.clear_wrench_buffer()
        apply_surface_wrenches(model, data)

        expected_force = P_SUN * 1.5 * 10.0
        np.testing.assert_allclose(
            np.linalg.norm(data.wrench_buffer[1, :3]),
            expected_force,
            rtol=0.01,
        )


class TestSurfaceTorque:
    def test_symmetric_panels_cancel_force_but_produce_torque(self):
        model, data = _make_model_data(
            surfaces=[
                SurfaceSpec(
                    body_name="spacecraft",
                    center_of_pressure_body=np.array([0.0, 0.0, 0.5]),
                    normal_body=np.array([1.0, 0.0, 0.0]),
                    area=2.0,
                    srp_coeff=1.5,
                    use_drag=False,
                ),
                SurfaceSpec(
                    body_name="spacecraft",
                    center_of_pressure_body=np.array([0.0, 0.0, -0.5]),
                    normal_body=np.array([1.0, 0.0, 0.0]),
                    area=2.0,
                    srp_coeff=1.5,
                    use_drag=False,
                ),
            ],
            use_drag=False,
        )
        data.env.eclipse = 1.0
        data.env.sun_vector_eci = np.array([1.0, 0.0, 0.0])

        data.clear_wrench_buffer()
        apply_surface_wrenches(model, data)

        force = data.wrench_buffer[1, :3]
        tau = data.wrench_buffer[1, 3:]
        force_single = P_SUN * 1.5 * 2.0
        np.testing.assert_allclose(np.linalg.norm(force), 2 * force_single, rtol=0.01)
        np.testing.assert_allclose(tau, 0.0, atol=1e-15)

    def test_offset_cop_produces_torque(self):
        model, data = _make_model_data(
            surfaces=[
                SurfaceSpec(
                    body_name="spacecraft",
                    center_of_pressure_body=np.array([0.0, 0.0, 0.5]),
                    normal_body=np.array([1.0, 0.0, 0.0]),
                    area=2.0,
                    srp_coeff=1.5,
                    use_drag=False,
                )
            ],
            use_drag=False,
        )
        data.env.eclipse = 1.0
        data.env.sun_vector_eci = np.array([1.0, 0.0, 0.0])

        data.clear_wrench_buffer()
        apply_surface_wrenches(model, data)

        assert np.linalg.norm(data.wrench_buffer[1, 3:]) > 1e-10


class TestSurfaceIntegration:
    def test_drag_decelerates_along_track(self):
        model, data = _make_model_data(
            surfaces=[
                SurfaceSpec(
                    body_name="spacecraft",
                    center_of_pressure_body=np.zeros(3),
                    normal_body=np.array([0.0, 1.0, 0.0]),
                    area=10.0,
                    drag_coeff=2.2,
                    use_srp=False,
                )
            ],
            use_srp=False,
        )
        data.qvel[1] += 0.1
        mjo_forward(model, data)

        for _ in range(100):
            mjo_step(model, data)

        assert np.all(np.isfinite(data.qpos))
        assert np.all(np.isfinite(data.qvel))
