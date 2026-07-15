"""Phase 2 validation: environment models (sun, eclipse, magnetic field, atmosphere)."""

import numpy as np
import pytest

from mjorbit.constants import B0_EARTH, OMEGA_EARTH, R_EARTH
from tests.mjorbit.reference.orbit.environment import (
    atm_density,
    atmosphere_relative_velocity_eci,
    dipole_field_eci,
    eclipse_factor,
    sun_vector_eci,
    update_environment_cache,
)
from tests.mjorbit.reference.orbit.state import OrbitState


class TestSunVector:
    def test_unit_vector(self):
        for t in [0.0, 86400.0, 365.25 * 86400.0]:
            s = sun_vector_eci(t)
            np.testing.assert_allclose(np.linalg.norm(s), 1.0, atol=1e-10)

    def test_ecliptic_plane(self):
        """Sun should always be near the ecliptic plane (z small relative to xy)."""
        s = sun_vector_eci(0.0)
        # At J2000, obliquity is ~23.4 deg, so z can be up to sin(23.4) ≈ 0.4
        assert abs(s[2]) < 0.5


class TestEclipse:
    def test_sunward_side(self):
        """Spacecraft on sun side: not eclipsed."""
        sun_hat = np.array([1.0, 0.0, 0.0])
        R = np.array([R_EARTH + 400.0, 0.0, 0.0])  # on sun side
        assert eclipse_factor(R, sun_hat) == 1.0

    def test_behind_earth(self):
        """Spacecraft behind Earth relative to sun: eclipsed."""
        sun_hat = np.array([1.0, 0.0, 0.0])
        R = np.array([-(R_EARTH + 400.0), 0.0, 0.0])  # behind Earth
        assert eclipse_factor(R, sun_hat) == 0.0

    def test_behind_but_outside_shadow(self):
        """Spacecraft behind Earth but off-axis: not eclipsed."""
        sun_hat = np.array([1.0, 0.0, 0.0])
        # Behind Earth but far enough off-axis to miss shadow
        R = np.array([-R_EARTH - 400.0, R_EARTH + 100.0, 0.0])
        assert eclipse_factor(R, sun_hat) == 1.0

    def test_on_shadow_boundary(self):
        """Spacecraft right at shadow cylinder edge."""
        sun_hat = np.array([1.0, 0.0, 0.0])
        # Behind Earth, perpendicular distance = R_EARTH - epsilon
        R = np.array([-R_EARTH * 2.0, R_EARTH - 10.0, 0.0])
        assert eclipse_factor(R, sun_hat) == 0.0

        R = np.array([-R_EARTH * 2.0, R_EARTH + 10.0, 0.0])
        assert eclipse_factor(R, sun_hat) == 1.0


class TestDipoleField:
    def test_north_pole_direction(self):
        """At geographic north pole, B should point radially inward (-z)."""
        R = np.array([0.0, 0.0, R_EARTH + 400.0])
        B = dipole_field_eci(R, 0.0)
        # With m_hat = [0,0,-1], at north pole (r_hat = [0,0,1]):
        # B = factor * (3*dot(-z, +z)*z - (-z)) = factor * (-3z + z) = factor * (-2z)
        assert B[2] < 0  # points inward (south) at north pole
        assert abs(B[0]) < 1e-20
        assert abs(B[1]) < 1e-20

    def test_equatorial_outward(self):
        """At equator, B should point roughly northward (+z) but with outward radial."""
        R = np.array([R_EARTH + 400.0, 0.0, 0.0])
        B = dipole_field_eci(R, 0.0)
        # With m_hat = [0,0,-1], at equator r_hat = [1,0,0]:
        # B = factor * (3*dot(-z, x)*x - (-z)) = factor * (0 + z) = factor * z
        assert B[2] > 0  # points northward at equator
        assert abs(B[0]) < 1e-20

    def test_inverse_cube_scaling(self):
        """B magnitude should scale as r^-3."""
        R1 = np.array([R_EARTH + 400.0, 0.0, 0.0])
        R2 = np.array([R_EARTH + 800.0, 0.0, 0.0])
        B1 = np.linalg.norm(dipole_field_eci(R1, 0.0))
        B2 = np.linalg.norm(dipole_field_eci(R2, 0.0))
        r1 = np.linalg.norm(R1)
        r2 = np.linalg.norm(R2)
        np.testing.assert_allclose(B1 / B2, (r2 / r1) ** 3, rtol=1e-10)

    def test_surface_magnitude(self):
        """At equator surface, |B| should be approximately B0_EARTH."""
        R = np.array([R_EARTH, 0.0, 0.0])
        B = dipole_field_eci(R, 0.0)
        # At equator for aligned dipole: B = factor * m_hat = B0 * [0,0,-1]
        # Wait: factor*(3*dot(m,r_hat)*r_hat - m) = B0*(3*0*x - (-z)) = B0*z
        # |B| = B0
        np.testing.assert_allclose(np.linalg.norm(B), B0_EARTH, rtol=1e-10)


class TestAtmosphere:
    def test_density_at_reference(self):
        """Density at 400 km should be approximately the reference value."""
        R = np.array([R_EARTH + 400.0, 0.0, 0.0])
        rho = atm_density(R)
        np.testing.assert_allclose(rho, 2.62e-13, rtol=0.01)

    def test_density_decreases_with_altitude(self):
        rho_400 = atm_density(np.array([R_EARTH + 400.0, 0.0, 0.0]))
        rho_500 = atm_density(np.array([R_EARTH + 500.0, 0.0, 0.0]))
        assert rho_500 < rho_400

    def test_density_nonnegative(self):
        """Density should never be negative."""
        for alt in [100.0, 400.0, 800.0, 2000.0]:
            rho = atm_density(np.array([R_EARTH + alt, 0.0, 0.0]))
            assert rho >= 0.0


class TestAtmRelVel:
    def test_eci_correction(self):
        """Atmosphere co-rotates with Earth, so v_rel should differ from v_eci."""
        R = np.array([R_EARTH + 400.0, 0.0, 0.0])
        V = np.array([0.0, 7.67, 0.0])
        v_rel = atmosphere_relative_velocity_eci(V, R)
        # v_atm = omega_earth x R = [0,0,omega]*[r,0,0] = [0, omega*r, 0]
        v_atm_y = OMEGA_EARTH * (R_EARTH + 400.0)  # ~0.48 km/s at equator
        np.testing.assert_allclose(v_rel[1], V[1] - v_atm_y, rtol=1e-10)


class TestEnvironmentCache:
    def test_cache_update(self):
        from tests.mjorbit.reference.orbit.elements import keplerian_to_cartesian
        from tests.mjorbit.reference.orbit.lvlh import update_frame_cache

        a = R_EARTH + 400.0
        R, V = keplerian_to_cartesian(a, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0)
        orbit = OrbitState(R, V, 0.0)
        fc = update_frame_cache(orbit)
        ec = update_environment_cache(orbit, fc)

        assert abs(np.linalg.norm(ec.sun_vector_eci) - 1.0) < 1e-10
        assert 0.0 <= ec.eclipse <= 1.0
        assert ec.atm_density > 0
        assert np.all(np.isfinite(ec.mag_field_eci))


class TestMagneticAxisCoRotation:
    """The dipole axis is Earth-fixed and co-rotates (issue #23)."""

    def test_default_aligned_axis_is_time_independent(self):
        from tests.mjorbit.reference.orbit.environment import magnetic_axis_eci

        for t in [0.0, 3600.0, 8.3e8]:
            np.testing.assert_allclose(
                magnetic_axis_eci(t), np.array([0.0, 0.0, -1.0]), atol=1e-15
            )

    def test_tilted_axis_rotates_about_spin_axis(self):
        from mjorbit.constants import ERA_J2000
        from tests.mjorbit.reference.orbit.environment import magnetic_axis_eci

        axis_ecef = np.array([np.sin(0.2), 0.0, -np.cos(0.2)])  # ~11.5 deg tilt

        # At t* with theta(t*) = 2*pi the ECEF axis coincides with its ECI image.
        t_star = (2.0 * np.pi - ERA_J2000) / OMEGA_EARTH
        np.testing.assert_allclose(magnetic_axis_eci(t_star, axis_ecef), axis_ecef, atol=1e-12)

        # A quarter sidereal turn later the equatorial component has moved to +y,
        # the polar component is unchanged, and the axis is still unit length.
        quarter = 0.5 * np.pi / OMEGA_EARTH
        m = magnetic_axis_eci(t_star + quarter, axis_ecef)
        np.testing.assert_allclose(
            m, [0.0, np.sin(0.2), -np.cos(0.2)], atol=1e-9
        )
        assert np.linalg.norm(m) == pytest.approx(1.0)

    def test_dipole_field_uses_rotated_axis(self):
        from tests.mjorbit.reference.orbit.environment import magnetic_axis_eci

        axis_ecef = np.array([1.0, 0.0, 0.0])
        R = np.array([R_EARTH + 500.0, 2000.0, -1500.0])
        t = 1.23e4
        m_hat = magnetic_axis_eci(t, axis_ecef)
        r = np.linalg.norm(R)
        r_hat = R / r
        expected = (
            B0_EARTH * (R_EARTH / r) ** 3 * (3.0 * np.dot(m_hat, r_hat) * r_hat - m_hat)
        )
        np.testing.assert_allclose(dipole_field_eci(R, t, axis_ecef), expected, atol=1e-20)


class TestProductionDipoleCoRotation:
    """The C++ environment cache co-rotates a tilted central-body dipole."""

    @staticmethod
    def _make_data(t: float):
        from pathlib import Path

        from mjorbit import MjoSpec, OrbitInit
        from mjorbit.testdata import FREE_BODY_XML

        text = Path(FREE_BODY_XML).read_text()
        block = """
  <mjorbit plugin_body="spacecraft" use_j2="false" use_drag="false" use_srp="false">
    <central_body magnetic_axis="0.3 0 -1"/>
  </mjorbit>
"""
        spec = MjoSpec.from_xml_string(text.replace("</mujoco>", block + "</mujoco>"))
        model = spec.compile(mj_timestep=0.01)
        R = np.array([R_EARTH + 550.0, 300.0, -800.0])
        V = np.array([0.1, 7.4, 0.4])
        return model.make_data(orbit=OrbitInit(R_eci=R, V_eci=V, t=t)), R

    def test_mag_field_matches_reference_and_rotates(self):
        from tests.mjorbit.reference.orbit.environment import dipole_field_eci as ref_dipole

        axis_ecef = np.array([0.3, 0.0, -1.0])
        fields = {}
        for t in (0.0, 3.0e4):
            data, R = self._make_data(t)
            np.testing.assert_allclose(
                data.env.mag_field_eci, ref_dipole(R, t, axis_ecef), rtol=1e-12
            )
            fields[t] = np.asarray(data.env.mag_field_eci).copy()

        # The tilted dipole actually moved between the two epochs.
        delta = np.linalg.norm(fields[0.0] - fields[3.0e4])
        assert delta > 1e-3 * np.linalg.norm(fields[0.0])
