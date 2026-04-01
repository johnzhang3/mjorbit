"""Phase 2 validation: environment models (sun, eclipse, magnetic field, atmosphere)."""

import numpy as np
import pytest

from mjorbit.constants import R_EARTH, B0_EARTH, OMEGA_EARTH
from mjorbit.cpu.orbit.environment import (
    sun_vector_eci,
    eclipse_factor,
    dipole_field_eci,
    atm_density,
    atmosphere_relative_velocity_eci,
    update_environment_cache,
)
from mjorbit.cpu.orbit.state import OrbitState, FrameCache


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
        from mjorbit.cpu.orbit.lvlh import update_frame_cache
        from mjorbit.cpu.orbit.elements import keplerian_to_cartesian

        a = R_EARTH + 400.0
        R, V = keplerian_to_cartesian(a, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0)
        orbit = OrbitState(R, V, 0.0)
        fc = update_frame_cache(orbit)
        ec = update_environment_cache(orbit, fc)

        assert abs(np.linalg.norm(ec.sun_vector_eci) - 1.0) < 1e-10
        assert 0.0 <= ec.eclipse <= 1.0
        assert ec.atm_density > 0
        assert np.all(np.isfinite(ec.mag_field_eci))
