"""Phase 9: Environment and gravity parity tests.

These tests verify that the default simulator's gravity, J2, magnetic field,
eclipse, and atmosphere models produce values consistent with known
analytical references (OrbitX-equivalent fidelity).

Each test compares a computed quantity against an analytically derived
or well-known reference value.
"""

import numpy as np
import pytest

from mujoco_orbit.constants import B0_EARTH, GM_EARTH, J2_EARTH, OMEGA_EARTH, R_EARTH
from tests.mujoco_orbit.reference.orbit.elements import (
    cartesian_to_keplerian,
    keplerian_to_cartesian,
)
from tests.mujoco_orbit.reference.orbit.environment import (
    atm_density,
    atmosphere_relative_velocity_eci,
    dipole_field_eci,
    eclipse_factor,
    sun_vector_eci,
)
from tests.mujoco_orbit.reference.orbit.gravity import j2_accel, point_mass_accel
from tests.mujoco_orbit.reference.orbit.propagator import propagate_rk4
from tests.mujoco_orbit.reference.orbit.state import OrbitState

# -----------------------------------------------------------------------
# Gravity model parity
# -----------------------------------------------------------------------


class TestGravityParity:
    """Point-mass and J2 gravity against analytical references."""

    def test_point_mass_magnitude_at_surface(self):
        """g at Earth surface ~ 9.80665e-3 km/s^2."""
        R = np.array([R_EARTH, 0.0, 0.0])
        a = point_mass_accel(R)
        g = np.linalg.norm(a)
        # GM/R^2 = 398600.4418 / 6378.137^2 ~ 9.798 e-3 km/s^2
        np.testing.assert_allclose(g, GM_EARTH / R_EARTH**2, rtol=1e-10)

    def test_point_mass_inverse_square(self):
        """g should follow inverse-square law."""
        R1 = np.array([R_EARTH + 200.0, 0.0, 0.0])
        R2 = np.array([R_EARTH + 800.0, 0.0, 0.0])
        g1 = np.linalg.norm(point_mass_accel(R1))
        g2 = np.linalg.norm(point_mass_accel(R2))
        r1 = np.linalg.norm(R1)
        r2 = np.linalg.norm(R2)
        ratio = g1 / g2
        expected = (r2 / r1) ** 2
        np.testing.assert_allclose(ratio, expected, rtol=1e-10)

    def test_j2_zero_at_equatorial_radial(self):
        """J2 radial component: at equator on x-axis, ay and az from J2 should have specific values.

        For R on equator (z=0):
          a_J2_x = (3/2) * J2 * mu * Re^2 / r^5 * x * (0 - 1) = -factor * x
          a_J2_z = (3/2) * J2 * mu * Re^2 / r^5 * z * (0 - 3) = 0 (z=0)
        """
        r = R_EARTH + 400.0
        R = np.array([r, 0.0, 0.0])
        a = j2_accel(R)
        factor = 1.5 * J2_EARTH * GM_EARTH * R_EARTH**2 / r**5
        expected_ax = factor * r * (0.0 - 1.0)
        np.testing.assert_allclose(a[0], expected_ax, rtol=1e-10)
        np.testing.assert_allclose(a[1], 0.0, atol=1e-20)
        np.testing.assert_allclose(a[2], 0.0, atol=1e-20)

    def test_j2_polar_vs_equatorial(self):
        """J2 acceleration magnitude should differ between polar and equatorial positions."""
        r = R_EARTH + 400.0
        R_eq = np.array([r, 0.0, 0.0])
        R_pol = np.array([0.0, 0.0, r])
        a_eq = np.linalg.norm(j2_accel(R_eq))
        a_pol = np.linalg.norm(j2_accel(R_pol))
        # They should differ (J2 is latitude-dependent)
        assert a_pol != pytest.approx(a_eq, rel=0.01)
        # Polar J2 should be larger in magnitude
        assert a_pol > a_eq

    def test_j2_order_of_magnitude(self):
        """J2 should be ~1e-3 of point-mass acceleration at LEO."""
        r = R_EARTH + 400.0
        R = np.array([r, 0.0, 0.0])
        g_pm = np.linalg.norm(point_mass_accel(R))
        g_j2 = np.linalg.norm(j2_accel(R))
        ratio = g_j2 / g_pm
        assert 1e-4 < ratio < 1e-2


class TestOrbitalElementsParity:
    """Keplerian element conversions should roundtrip correctly."""

    def test_circular_leo_elements(self):
        """Circular LEO: e~0, a matches input."""
        a_in = R_EARTH + 400.0
        R, V = keplerian_to_cartesian(a_in, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0)
        elems = cartesian_to_keplerian(R, V)
        np.testing.assert_allclose(elems["a"], a_in, rtol=1e-10)
        assert elems["e"] < 1e-10

    def test_eccentric_orbit_roundtrip(self):
        """Eccentric orbit roundtrips through Cartesian."""
        a = R_EARTH + 500.0
        e = 0.1
        inc = np.deg2rad(45.0)
        raan = np.deg2rad(30.0)
        argp = np.deg2rad(60.0)
        nu = np.deg2rad(120.0)
        R, V = keplerian_to_cartesian(a, e, inc, raan, argp, nu)
        elems = cartesian_to_keplerian(R, V)
        np.testing.assert_allclose(elems["a"], a, rtol=1e-8)
        np.testing.assert_allclose(elems["e"], e, rtol=1e-8)
        np.testing.assert_allclose(elems["inc"], inc, rtol=1e-8)

    def test_circular_velocity(self):
        """Circular velocity = sqrt(mu/a)."""
        a = R_EARTH + 400.0
        R, V = keplerian_to_cartesian(a, 0.0, np.deg2rad(28.5), 0.0, 0.0, 0.0)
        v = np.linalg.norm(V)
        v_expected = np.sqrt(GM_EARTH / a)
        np.testing.assert_allclose(v, v_expected, rtol=1e-10)


class TestPropagatorParity:
    """Orbit propagator should conserve energy and match known rates."""

    def test_two_body_energy_conservation(self):
        """No J2: energy should be conserved over one orbit."""
        a = R_EARTH + 400.0
        R, V = keplerian_to_cartesian(a, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0)
        state = OrbitState(R, V, 0.0)

        E0 = 0.5 * np.dot(V, V) - GM_EARTH / np.linalg.norm(R)
        period = 2 * np.pi * np.sqrt(a**3 / GM_EARTH)
        dt = 1.0
        n_steps = int(period / dt)
        for _ in range(n_steps):
            state = propagate_rk4(state, dt, use_j2=False)

        r1 = np.linalg.norm(state.R_eci)
        v1 = np.linalg.norm(state.V_eci)
        E1 = 0.5 * v1**2 - GM_EARTH / r1
        np.testing.assert_allclose(E1, E0, rtol=1e-8)

    def test_j2_raan_drift_rate(self):
        """J2 secular RAAN drift: Omega_dot ~ -1.5 * n * J2 * (Re/a)^2 * cos(i) / (1-e^2)^2.

        For 400 km circular, 51.6 deg: about -5 deg/day.
        """
        a = R_EARTH + 400.0
        inc = np.deg2rad(51.6)
        R, V = keplerian_to_cartesian(a, 0.0, inc, 0.0, 0.0, 0.0)
        state = OrbitState(R, V, 0.0)

        # Propagate for 1 day
        dt = 10.0
        n_steps = int(86400 / dt)
        for _ in range(n_steps):
            state = propagate_rk4(state, dt, use_j2=True)

        elems = cartesian_to_keplerian(state.R_eci, state.V_eci)
        raan_final = elems["raan"]

        # Expected RAAN drift
        n = np.sqrt(GM_EARTH / a**3)
        raan_dot = -1.5 * n * J2_EARTH * (R_EARTH / a) ** 2 * np.cos(inc)
        expected_raan = raan_dot * 86400.0  # rad after 1 day
        # Should be around -5 deg/day => -0.087 rad/day
        np.testing.assert_allclose(raan_final, expected_raan, rtol=0.05)


# -----------------------------------------------------------------------
# Environment model parity
# -----------------------------------------------------------------------


class TestEclipseParity:
    """Eclipse model should match geometric expectations."""

    def test_sunlit_position(self):
        """Spacecraft on sunward side is in full sun."""
        sun_hat = np.array([1.0, 0.0, 0.0])
        R = np.array([R_EARTH + 400.0, 0.0, 0.0])  # toward sun
        assert eclipse_factor(R, sun_hat) == 1.0

    def test_shadow_position(self):
        """Spacecraft behind Earth on anti-sun axis is in shadow."""
        sun_hat = np.array([1.0, 0.0, 0.0])
        R = np.array([-(R_EARTH + 400.0), 0.0, 0.0])  # behind Earth
        assert eclipse_factor(R, sun_hat) == 0.0

    def test_perpendicular_position_sunlit(self):
        """Spacecraft perpendicular to sun axis is in sunlight."""
        sun_hat = np.array([1.0, 0.0, 0.0])
        R = np.array([0.0, R_EARTH + 400.0, 0.0])
        # Not behind Earth (projection on -sun_hat is 0 -> sunward side)
        assert eclipse_factor(R, sun_hat) == 1.0

    def test_partial_shadow_geometry(self):
        """Spacecraft behind Earth but outside umbra cylinder is sunlit."""
        sun_hat = np.array([1.0, 0.0, 0.0])
        # Behind Earth but offset well beyond R_EARTH in y
        R = np.array([-1000.0, R_EARTH + 1000.0, 0.0])
        assert eclipse_factor(R, sun_hat) == 1.0


class TestMagneticFieldParity:
    """Centered-dipole magnetic field against analytical expressions."""

    def test_north_pole_magnitude(self):
        """At geographic north pole (ECI +z), field should point inward (toward Earth)
        and have magnitude 2*B0*(Re/r)^3."""
        r = R_EARTH + 400.0
        R = np.array([0.0, 0.0, r])
        B = dipole_field_eci(R, 0.0)
        # Dipole axis along -z: at +z pole, field = 2*B0*(Re/r)^3 * (-z_hat)
        expected_mag = 2.0 * B0_EARTH * (R_EARTH / r) ** 3
        np.testing.assert_allclose(np.linalg.norm(B), expected_mag, rtol=1e-6)
        # Should point in -z direction (inward at north pole for -z dipole)
        assert B[2] < 0

    def test_equatorial_magnitude(self):
        """At equator, field magnitude = B0*(Re/r)^3."""
        r = R_EARTH + 400.0
        R = np.array([r, 0.0, 0.0])
        B = dipole_field_eci(R, 0.0)
        expected_mag = B0_EARTH * (R_EARTH / r) ** 3
        np.testing.assert_allclose(np.linalg.norm(B), expected_mag, rtol=1e-6)

    def test_inverse_cube_scaling(self):
        """Field should scale as 1/r^3."""
        R1 = np.array([R_EARTH + 200.0, 0.0, 0.0])
        R2 = np.array([R_EARTH + 800.0, 0.0, 0.0])
        B1 = np.linalg.norm(dipole_field_eci(R1, 0.0))
        B2 = np.linalg.norm(dipole_field_eci(R2, 0.0))
        r1, r2 = np.linalg.norm(R1), np.linalg.norm(R2)
        np.testing.assert_allclose(B1 / B2, (r2 / r1) ** 3, rtol=1e-6)

    def test_surface_field_magnitude(self):
        """At equatorial surface, |B| ~ B0_EARTH = 3.12e-5 T."""
        R = np.array([R_EARTH, 0.0, 0.0])
        B = dipole_field_eci(R, 0.0)
        np.testing.assert_allclose(np.linalg.norm(B), B0_EARTH, rtol=1e-6)


class TestAtmosphereParity:
    """Atmosphere density model against expected values."""

    def test_reference_altitude_density(self):
        """At 400 km altitude, density should match the reference value."""
        R = np.array([R_EARTH + 400.0, 0.0, 0.0])
        rho = atm_density(R)
        # Reference: 2.62e-13 kg/m^3 at 400 km
        np.testing.assert_allclose(rho, 2.62e-13, rtol=0.01)

    def test_density_decreases_with_altitude(self):
        """Density at 500 km should be less than at 400 km."""
        R400 = np.array([R_EARTH + 400.0, 0.0, 0.0])
        R500 = np.array([R_EARTH + 500.0, 0.0, 0.0])
        assert atm_density(R500) < atm_density(R400)

    def test_density_positive_everywhere(self):
        """Density should be positive at all altitudes."""
        for alt in [200, 400, 600, 800, 1000, 2000]:
            R = np.array([R_EARTH + alt, 0.0, 0.0])
            assert atm_density(R) >= 0.0

    def test_exponential_decay_rate(self):
        """Density ratio over one scale height should be ~1/e."""
        H = 58.2  # km, scale height
        R1 = np.array([R_EARTH + 400.0, 0.0, 0.0])
        R2 = np.array([R_EARTH + 400.0 + H, 0.0, 0.0])
        ratio = atm_density(R2) / atm_density(R1)
        np.testing.assert_allclose(ratio, np.exp(-1.0), rtol=1e-6)


class TestAtmRelVelParity:
    """Atmosphere-relative velocity computation."""

    def test_corotation_correction(self):
        """Atmosphere co-rotation should reduce along-track relative velocity."""
        R = np.array([R_EARTH + 400.0, 0.0, 0.0])
        v_circ = np.sqrt(GM_EARTH / np.linalg.norm(R))
        V = np.array([0.0, v_circ, 0.0])

        v_rel = atmosphere_relative_velocity_eci(V, R)
        # Atmosphere velocity here is omega_earth x R = [0, omega * r, 0].
        v_atm = OMEGA_EARTH * np.linalg.norm(R)  # km/s
        expected_rel = v_circ - v_atm
        np.testing.assert_allclose(v_rel[1], expected_rel, rtol=1e-10)


class TestSunVectorParity:
    """Sun vector should satisfy basic geometric constraints."""

    def test_unit_vector(self):
        """Sun direction should be a unit vector."""
        for t in [0.0, 86400.0, 86400.0 * 365.25]:
            s = sun_vector_eci(t)
            np.testing.assert_allclose(np.linalg.norm(s), 1.0, rtol=1e-10)

    def test_ecliptic_plane(self):
        """Sun vector should primarily lie in ecliptic plane (z component bounded by obliquity)."""
        for t in np.linspace(0, 86400 * 365.25, 12):
            s = sun_vector_eci(t)
            # Obliquity ~ 23.4 deg, so |z| <= sin(23.4 deg) ~ 0.4
            assert abs(s[2]) <= 0.42
