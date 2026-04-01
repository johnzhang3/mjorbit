"""Phase 6 validation: magnetic torques (tau = m × B)."""

import numpy as np
import pytest
import mujoco

from mjorbit.constants import R_EARTH, B0_EARTH
from mjorbit.cpu import compile_cpu, step_cpu
from mjorbit.cpu.core.config import (
    CPUScenarioCfg, OrbitCfg, MuJoCoCfg, MagneticBodyCfg,
)
from mjorbit.cpu.orbit.elements import keplerian_to_cartesian
from mjorbit.cpu.coupling.magnetic import apply_magnetic_wrenches
from mjorbit.cpu.mjcf.builders import FREE_BODY_XML


def _leo_cfg(**overrides) -> CPUScenarioCfg:
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
    return CPUScenarioCfg(**defaults)


class TestMagneticTorque:
    def test_cross_product_sign(self):
        """tau = m × B: simple axis-aligned case."""
        cfg = _leo_cfg(
            magnetic_bodies=[
                MagneticBodyCfg(
                    body_name="spacecraft",
                    dipole_body=np.array([1.0, 0.0, 0.0]),  # dipole along body x
                )
            ]
        )
        sc = compile_cpu(cfg)

        # Set B to be along z in body frame (= world frame at identity pose)
        # Override env cache for controlled test
        B_test = np.array([0.0, 0.0, 1.0])  # T
        sc.env_cache.mag_field_eci = sc.frame_cache.C_IL @ B_test  # store as ECI

        sc.clear_wrench_buffer()
        apply_magnetic_wrenches(sc)

        tau = sc._wrench_buffer[1, 3:]
        # m × B = [1,0,0] × [0,0,1] = [0*1 - 0*0, 0*0 - 1*1, 1*0 - 0*0] = [0, -1, 0]
        np.testing.assert_allclose(tau, [0.0, -1.0, 0.0], atol=1e-12)

    def test_parallel_dipole_zero_torque(self):
        """If m is parallel to B, torque should be zero."""
        cfg = _leo_cfg(
            magnetic_bodies=[
                MagneticBodyCfg(
                    body_name="spacecraft",
                    dipole_body=np.array([0.0, 0.0, 1.0]),
                )
            ]
        )
        sc = compile_cpu(cfg)

        B_test = np.array([0.0, 0.0, 1.0])
        sc.env_cache.mag_field_eci = sc.frame_cache.C_IL @ B_test

        sc.clear_wrench_buffer()
        apply_magnetic_wrenches(sc)

        tau = sc._wrench_buffer[1, 3:]
        np.testing.assert_allclose(tau, 0.0, atol=1e-14)

    def test_no_translational_force(self):
        """Magnetic dipole should produce zero translational force."""
        cfg = _leo_cfg(
            magnetic_bodies=[
                MagneticBodyCfg(
                    body_name="spacecraft",
                    dipole_body=np.array([1.0, 0.0, 0.0]),
                )
            ]
        )
        sc = compile_cpu(cfg)
        sc.clear_wrench_buffer()
        apply_magnetic_wrenches(sc)

        F = sc._wrench_buffer[1, :3]
        np.testing.assert_allclose(F, 0.0, atol=1e-20)

    def test_torque_magnitude(self):
        """Torque magnitude = |m| * |B| * sin(angle)."""
        dipole_mag = 5.0
        B_mag = 1e-5
        cfg = _leo_cfg(
            magnetic_bodies=[
                MagneticBodyCfg(
                    body_name="spacecraft",
                    dipole_body=np.array([dipole_mag, 0.0, 0.0]),
                )
            ]
        )
        sc = compile_cpu(cfg)

        B_test = np.array([0.0, B_mag, 0.0])
        sc.env_cache.mag_field_eci = sc.frame_cache.C_IL @ B_test

        sc.clear_wrench_buffer()
        apply_magnetic_wrenches(sc)

        tau = sc._wrench_buffer[1, 3:]
        # m × B = [d,0,0] × [0,B,0] = [0,0,d*B]
        expected_mag = dipole_mag * B_mag
        np.testing.assert_allclose(np.linalg.norm(tau), expected_mag, rtol=1e-10)

    def test_rotated_body_frame(self):
        """When body is rotated, dipole in body frame maps correctly to world frame."""
        cfg = _leo_cfg(
            magnetic_bodies=[
                MagneticBodyCfg(
                    body_name="spacecraft",
                    dipole_body=np.array([0.0, 0.0, 1.0]),  # dipole along body z
                )
            ]
        )
        sc = compile_cpu(cfg)

        # Rotate body by 90 degrees about y-axis
        # quat for 90 deg about y: [cos(45), 0, sin(45), 0] = [0.707, 0, 0.707, 0]
        angle = np.pi / 2
        sc.mjd.qpos[3] = np.cos(angle / 2)  # w
        sc.mjd.qpos[4] = 0.0  # x
        sc.mjd.qpos[5] = np.sin(angle / 2)  # y
        sc.mjd.qpos[6] = 0.0  # z
        mujoco.mj_forward(sc.mjm, sc.mjd)

        # B in world frame = [0, 0, 1]
        B_test = np.array([0.0, 0.0, 1.0])
        sc.env_cache.mag_field_eci = sc.frame_cache.C_IL @ B_test

        sc.clear_wrench_buffer()
        apply_magnetic_wrenches(sc)

        tau = sc._wrench_buffer[1, 3:]

        # After rotation: body z → world x (approx, 90 deg about y rotates z to x)
        # dipole_world ≈ [1, 0, 0], B_world = [0, 0, 1]
        # tau_world = [1,0,0] × [0,0,1] = [0,-1,0]
        np.testing.assert_allclose(tau, [0, -1, 0], atol=0.01)

    def test_disabled(self):
        """When use_magnetic=False, no magnetic torque applied."""
        cfg = _leo_cfg(
            magnetic_bodies=[
                MagneticBodyCfg(
                    body_name="spacecraft",
                    dipole_body=np.array([1.0, 0.0, 0.0]),
                )
            ],
            use_magnetic=False,
        )
        sc = compile_cpu(cfg)
        sc.clear_wrench_buffer()
        apply_magnetic_wrenches(sc)
        np.testing.assert_allclose(sc._wrench_buffer[1], 0.0)
