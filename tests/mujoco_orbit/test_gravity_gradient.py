"""Rotational gravity-gradient torque tests."""

from __future__ import annotations

import os
import tempfile

import numpy as np

from mujoco_orbit import MjoData, MjoModel, mjo_forward
from mujoco_orbit.constants import GM_EARTH, R_EARTH
from mujoco_orbit.coupling.gravity_gradient import apply_gravity_gradient_torques

from ._helpers import make_model_data


def _write_xml(xml: str) -> str:
    file = tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False)
    try:
        file.write(xml)
        return file.name
    finally:
        file.close()


def _single_body_xml(inertia: str) -> str:
    return f"""<mujoco model="gravity_gradient_body">
      <compiler angle="radian" balanceinertia="true"/>
      <option timestep="0.01" gravity="0 0 0"/>
      <worldbody>
        <body name="body" pos="0 0 0">
          <freejoint/>
          <inertial pos="0 0 0" mass="100" {inertia}/>
          <geom type="box" size="0.1 0.1 0.1" mass="0"/>
        </body>
      </worldbody>
    </mujoco>"""


def _apply_only_gravity_gradient(model: MjoModel, data: MjoData) -> None:
    data.clear_wrench_buffer()
    apply_gravity_gradient_torques(model, data)


def _expected_torque_from_matrix(
    J_world: np.ndarray,
    r_hat_world: np.ndarray,
    radius_km: float,
) -> np.ndarray:
    coeff = 3.0 * GM_EARTH / radius_km**3
    return coeff * np.cross(r_hat_world, J_world @ r_hat_world)


def test_single_rigid_body_matches_analytic_torque() -> None:
    """A rotated asymmetric single body should get the standard GG torque."""
    xml_path = _write_xml(_single_body_xml('diaginertia="4000000 5000000 6000000"'))
    try:
        model, data = make_model_data(
            xml_path=xml_path,
            alt_km=400.0,
            use_j2=False,
            use_drag=False,
            use_srp=False,
            use_magnetic=False,
        )

        theta = np.deg2rad(30.0)
        data.qpos[3:7] = [np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)]
        mjo_forward(model, data)
        _apply_only_gravity_gradient(model, data)

        body_id = model.body_id("body")
        R_wb = data.xmat[body_id].reshape(3, 3)
        J_body = np.diag([4.0e6, 5.0e6, 6.0e6])
        J_world = R_wb @ J_body @ R_wb.T
        expected = _expected_torque_from_matrix(
            J_world,
            np.array([1.0, 0.0, 0.0]),
            R_EARTH + 400.0,
        )

        np.testing.assert_allclose(data.wrench_buffer[body_id, :3], 0.0, atol=1e-20)
        np.testing.assert_allclose(data.wrench_buffer[body_id, 3:], expected, rtol=1e-12)
        assert data.wrench_buffer[body_id, 5] < 0.0
    finally:
        os.unlink(xml_path)


def test_torque_is_zero_when_radial_direction_is_principal_axis() -> None:
    """Principal-axis nadir alignment is a gravity-gradient equilibrium."""
    xml_path = _write_xml(_single_body_xml('diaginertia="4000000 5000000 6000000"'))
    try:
        model, data = make_model_data(
            xml_path=xml_path,
            alt_km=400.0,
            use_j2=False,
            use_drag=False,
            use_srp=False,
            use_magnetic=False,
        )

        mjo_forward(model, data)
        _apply_only_gravity_gradient(model, data)

        body_id = model.body_id("body")
        np.testing.assert_allclose(data.wrench_buffer[body_id], 0.0, atol=1e-20)
    finally:
        os.unlink(xml_path)


def test_use_gravity_gradient_false_disables_rotational_torque() -> None:
    xml_path = _write_xml(_single_body_xml('diaginertia="4000000 5000000 6000000"'))
    try:
        model, data = make_model_data(
            xml_path=xml_path,
            alt_km=400.0,
            use_j2=False,
            use_drag=False,
            use_srp=False,
            use_magnetic=False,
            use_gravity_gradient=False,
        )

        theta = np.deg2rad(30.0)
        data.qpos[3:7] = [np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)]
        mjo_forward(model, data)
        _apply_only_gravity_gradient(model, data)

        body_id = model.body_id("body")
        np.testing.assert_allclose(data.wrench_buffer[body_id], 0.0, atol=1e-20)
    finally:
        os.unlink(xml_path)


def test_fullinertia_uses_inertia_frame_orientation() -> None:
    """Non-diagonal XML inertia must be reconstructed via ximat, not xmat."""
    J_body = np.array(
        [
            [5.0e6, 1.0e6, 0.0],
            [1.0e6, 4.0e6, 0.0],
            [0.0, 0.0, 3.0e6],
        ]
    )
    xml_path = _write_xml(_single_body_xml('fullinertia="5000000 4000000 3000000 1000000 0 0"'))
    try:
        model, data = make_model_data(
            xml_path=xml_path,
            alt_km=400.0,
            use_j2=False,
            use_drag=False,
            use_srp=False,
            use_magnetic=False,
        )

        mjo_forward(model, data)
        _apply_only_gravity_gradient(model, data)

        body_id = model.body_id("body")
        expected = _expected_torque_from_matrix(
            J_body,
            np.array([1.0, 0.0, 0.0]),
            R_EARTH + 400.0,
        )
        torque_if_xmat_were_used = _expected_torque_from_matrix(
            np.diag(model.body_inertia[body_id]),
            np.array([1.0, 0.0, 0.0]),
            R_EARTH + 400.0,
        )

        np.testing.assert_allclose(data.wrench_buffer[body_id, 3:], expected, rtol=1e-12)
        np.testing.assert_allclose(torque_if_xmat_were_used, 0.0, atol=1e-20)
        assert abs(data.wrench_buffer[body_id, 5]) > 0.0
    finally:
        os.unlink(xml_path)


def test_torque_strength_scales_with_inverse_orbit_radius_cubed() -> None:
    xml_path = _write_xml(_single_body_xml('diaginertia="4000000 5000000 6000000"'))
    try:
        torques = []
        for alt_km in (400.0, 800.0):
            model, data = make_model_data(
                xml_path=xml_path,
                alt_km=alt_km,
                use_j2=False,
                use_drag=False,
                use_srp=False,
                use_magnetic=False,
            )
            theta = np.deg2rad(30.0)
            data.qpos[3:7] = [np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)]
            mjo_forward(model, data)
            _apply_only_gravity_gradient(model, data)
            torques.append(abs(data.wrench_buffer[model.body_id("body"), 5]))

        expected_ratio = ((R_EARTH + 800.0) / (R_EARTH + 400.0)) ** 3
        np.testing.assert_allclose(torques[0] / torques[1], expected_ratio, rtol=1e-12)
    finally:
        os.unlink(xml_path)
