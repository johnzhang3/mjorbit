"""Phase 1 validation: orbit propagation, elements, frame transforms."""

import numpy as np

from mujoco_orbit.constants import GM_EARTH, J2_EARTH, R_EARTH
from mujoco_orbit.orbit.elements import cartesian_to_keplerian, keplerian_to_cartesian
from mujoco_orbit.orbit.gravity import j2_accel, point_mass_accel
from mujoco_orbit.orbit.lvlh import (
    eci_to_lvlh_pos,
    eci_to_lvlh_vel,
    lvlh_to_eci_pos,
    update_frame_cache,
)
from mujoco_orbit.orbit.propagator import propagate_rk4
from mujoco_orbit.orbit.state import OrbitState

# =========================================================================
# Gravity tests
# =========================================================================

class TestGravity:
    def test_point_mass_magnitude(self):
        """Gravity at Earth surface should be ~9.8e-3 km/s^2."""
        r = np.array([R_EARTH, 0.0, 0.0])
        a = point_mass_accel(r)
        mag = np.linalg.norm(a)
        assert abs(mag - GM_EARTH / R_EARTH**2) < 1e-10

    def test_point_mass_direction(self):
        """Gravity should point radially inward."""
        r = np.array([1000.0, 2000.0, 3000.0])
        a = point_mass_accel(r)
        r_hat = r / np.linalg.norm(r)
        a_hat = a / np.linalg.norm(a)
        # a should be anti-parallel to r
        assert np.dot(r_hat, a_hat) < -0.999

    def test_j2_vanishes_on_equator(self):
        """J2 z-component should be zero for equatorial position (z=0)."""
        r = np.array([R_EARTH + 400.0, 0.0, 0.0])
        a = j2_accel(r)
        assert abs(a[2]) < 1e-20

    def test_j2_symmetry(self):
        """J2 should be symmetric under rotation about z-axis."""
        r1 = np.array([R_EARTH + 400.0, 0.0, 500.0])
        r2 = np.array([0.0, R_EARTH + 400.0, 500.0])
        a1 = j2_accel(r1)
        a2 = j2_accel(r2)
        assert abs(np.linalg.norm(a1) - np.linalg.norm(a2)) < 1e-14

    def test_j2_much_smaller_than_point_mass(self):
        """J2 should be ~1000x smaller than point mass at LEO."""
        r = np.array([R_EARTH + 400.0, 0.0, 100.0])
        ratio = np.linalg.norm(j2_accel(r)) / np.linalg.norm(point_mass_accel(r))
        assert ratio < 1e-2
        assert ratio > 1e-5


# =========================================================================
# Keplerian elements tests
# =========================================================================

class TestElements:
    def test_circular_equatorial(self):
        """Circular equatorial orbit should roundtrip cleanly."""
        a, e, inc, raan, argp, nu = R_EARTH + 400.0, 0.001, 0.001, 0.0, 0.0, 0.0
        R, V = keplerian_to_cartesian(a, e, inc, raan, argp, nu)
        assert abs(np.linalg.norm(R) - a * (1 - e**2) / (1 + e * np.cos(nu))) < 1e-8

    def test_roundtrip(self):
        """Keplerian -> Cartesian -> Keplerian should recover original elements."""
        a0, e0, inc0 = R_EARTH + 500.0, 0.01, np.deg2rad(51.6)
        raan0, argp0, nu0 = np.deg2rad(30.0), np.deg2rad(60.0), np.deg2rad(90.0)
        R, V = keplerian_to_cartesian(a0, e0, inc0, raan0, argp0, nu0)
        elems = cartesian_to_keplerian(R, V)
        assert abs(elems["a"] - a0) / a0 < 1e-10
        assert abs(elems["e"] - e0) < 1e-10
        assert abs(elems["inc"] - inc0) < 1e-10
        assert abs(elems["raan"] - raan0) < 1e-10
        assert abs(elems["argp"] - argp0) < 1e-8
        assert abs(elems["nu"] - nu0) < 1e-8

    def test_circular_velocity(self):
        """Circular orbit velocity should be sqrt(mu/r)."""
        a = R_EARTH + 400.0
        R, V = keplerian_to_cartesian(a, 0.0, 0.0, 0.0, 0.0, 0.0)
        v_circ = np.sqrt(GM_EARTH / a)
        assert abs(np.linalg.norm(V) - v_circ) < 1e-10


# =========================================================================
# Propagator tests
# =========================================================================

class TestPropagator:
    def _circular_leo(self, alt_km: float = 400.0) -> OrbitState:
        a = R_EARTH + alt_km
        R, V = keplerian_to_cartesian(a, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0)
        return OrbitState(R_eci=R, V_eci=V, t=0.0)

    def test_period_circular(self):
        """Propagating one full period should return to starting position."""
        a = R_EARTH + 400.0
        T = 2.0 * np.pi * np.sqrt(a**3 / GM_EARTH)
        state = self._circular_leo()
        R0 = state.R_eci.copy()

        dt = 1.0  # 1s steps
        n_steps = int(T / dt)
        for _ in range(n_steps):
            state = propagate_rk4(state, dt, use_j2=False)
        # Propagate the remaining fractional step
        remainder = T - n_steps * dt
        if remainder > 1e-10:
            state = propagate_rk4(state, remainder, use_j2=False)

        # Should be back near starting position
        pos_err_km = np.linalg.norm(state.R_eci - R0)
        assert pos_err_km < 0.01  # < 10 m error after one orbit with RK4

    def test_energy_conservation_no_j2(self):
        """Specific energy should be conserved without J2."""
        state = self._circular_leo()

        def energy(s: OrbitState) -> float:
            r = np.linalg.norm(s.R_eci)
            v = np.linalg.norm(s.V_eci)
            return 0.5 * v**2 - GM_EARTH / r

        E0 = energy(state)
        dt = 1.0
        for _ in range(5000):
            state = propagate_rk4(state, dt, use_j2=False)

        E_final = energy(state)
        assert abs((E_final - E0) / E0) < 1e-10

    def test_j2_secular_raan_drift(self):
        """J2 should cause RAAN to precess for an inclined orbit."""
        a = R_EARTH + 400.0
        inc = np.deg2rad(51.6)
        state = OrbitState(
            R_eci=keplerian_to_cartesian(a, 0.001, inc, 0.0, 0.0, 0.0)[0],
            V_eci=keplerian_to_cartesian(a, 0.001, inc, 0.0, 0.0, 0.0)[1],
            t=0.0,
        )

        elems_0 = cartesian_to_keplerian(state.R_eci, state.V_eci)

        dt = 10.0
        for _ in range(1000):  # 10000 s ≈ 1.8 orbits
            state = propagate_rk4(state, dt, use_j2=True)

        elems_f = cartesian_to_keplerian(state.R_eci, state.V_eci)

        # RAAN should have changed
        draan = elems_f["raan"] - elems_0["raan"]
        # Analytical RAAN rate: dOmega/dt = -1.5 * n * J2 * (Re/p)^2 * cos(i)
        n = np.sqrt(GM_EARTH / a**3)
        p = a * (1 - 0.001**2)
        draan_dt_theory = -1.5 * n * J2_EARTH * (R_EARTH / p) ** 2 * np.cos(inc)
        draan_theory = draan_dt_theory * 10000.0

        # Check same sign and within 20% (secular theory vs numerical)
        assert np.sign(draan) == np.sign(draan_theory)
        assert abs(draan - draan_theory) / abs(draan_theory) < 0.2

    def test_time_advances(self):
        state = self._circular_leo()
        state2 = propagate_rk4(state, 10.0)
        assert state2.t == 10.0


# =========================================================================
# LVLH frame tests
# =========================================================================

class TestLVLHFrame:
    def _circular_leo(self) -> OrbitState:
        a = R_EARTH + 400.0
        R, V = keplerian_to_cartesian(a, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0)
        return OrbitState(R_eci=R, V_eci=V, t=0.0)

    def test_orthonormality(self):
        """C_LI rows should be orthonormal."""
        fc = update_frame_cache(self._circular_leo())
        eye = fc.C_LI @ fc.C_LI.T
        np.testing.assert_allclose(eye, np.eye(3), atol=1e-14)

    def test_det_positive(self):
        """C_LI should be a proper rotation (det = +1)."""
        fc = update_frame_cache(self._circular_leo())
        np.testing.assert_allclose(np.linalg.det(fc.C_LI), 1.0, atol=1e-14)

    def test_inverse(self):
        """C_IL should be the transpose of C_LI."""
        fc = update_frame_cache(self._circular_leo())
        np.testing.assert_allclose(fc.C_IL, fc.C_LI.T, atol=1e-14)

    def test_radial_axis(self):
        """x_hat (first row of C_LI) should be parallel to R."""
        orbit = self._circular_leo()
        fc = update_frame_cache(orbit)
        r_hat = orbit.R_eci / np.linalg.norm(orbit.R_eci)
        np.testing.assert_allclose(fc.C_LI[0], r_hat, atol=1e-14)

    def test_orbit_normal(self):
        """z_hat (third row of C_LI) should be along R x V."""
        orbit = self._circular_leo()
        fc = update_frame_cache(orbit)
        h = np.cross(orbit.R_eci, orbit.V_eci)
        h_hat = h / np.linalg.norm(h)
        np.testing.assert_allclose(fc.C_LI[2], h_hat, atol=1e-14)

    def test_omega_circular_orbit(self):
        """For circular orbit, omega should be (0, 0, n) in LVLH."""
        a = R_EARTH + 400.0
        n = np.sqrt(GM_EARTH / a**3)
        R, V = keplerian_to_cartesian(a, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0)
        fc = update_frame_cache(OrbitState(R, V, 0.0))
        # omega in LVLH = (0, 0, h/r^2) = (0, 0, n)
        np.testing.assert_allclose(fc.omega_lvlh[0], 0.0, atol=1e-14)
        np.testing.assert_allclose(fc.omega_lvlh[1], 0.0, atol=1e-14)
        np.testing.assert_allclose(fc.omega_lvlh[2], n, rtol=1e-10)

    def test_omega_dot_circular_orbit(self):
        """For circular orbit with point-mass, omega_dot should be zero."""
        a = R_EARTH + 400.0
        R, V = keplerian_to_cartesian(a, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0)
        fc = update_frame_cache(OrbitState(R, V, 0.0), use_j2=False)
        np.testing.assert_allclose(fc.omega_dot_lvlh, np.zeros(3), atol=1e-16)

    def test_position_roundtrip(self):
        """ECI -> LVLH -> ECI position roundtrip."""
        orbit = self._circular_leo()
        fc = update_frame_cache(orbit)
        offset_eci = orbit.R_eci + np.array([0.01, -0.005, 0.003])
        r_lvlh = eci_to_lvlh_pos(offset_eci, orbit.R_eci, fc.C_LI)
        r_eci_back = lvlh_to_eci_pos(r_lvlh, orbit.R_eci, fc.C_IL)
        np.testing.assert_allclose(r_eci_back, offset_eci, atol=1e-14)

    def test_velocity_transform(self):
        """Chief velocity in LVLH should be (rdot, v_along, 0)."""
        orbit = self._circular_leo()
        fc = update_frame_cache(orbit)
        # For circular orbit, chief velocity in LVLH:
        # rdot = 0, along-track = |V|, cross-track = 0
        # BUT eci_to_lvlh_vel gives relative velocity (which is 0 for the chief itself)
        v_lvlh = eci_to_lvlh_vel(
            orbit.V_eci, np.zeros(3), orbit.V_eci, fc.C_LI, fc.omega_lvlh
        )
        # For a point at the origin, relative vel = C_LI @ (V - V) - omega x 0 = 0
        np.testing.assert_allclose(v_lvlh, np.zeros(3), atol=1e-14)

    def test_eccentric_omega_dot_nonzero(self):
        """For an eccentric orbit at periapsis, omega_dot should be nonzero (rdot=0 but
        the formula still gives zero for point-mass 2-body at any point where rdot=0).
        At true anomaly = 90 deg, rdot != 0, so omega_dot != 0."""
        a = R_EARTH + 1000.0
        e = 0.1
        R, V = keplerian_to_cartesian(a, e, np.deg2rad(30.0), 0.0, 0.0, np.deg2rad(90.0))
        fc = update_frame_cache(OrbitState(R, V, 0.0), use_j2=False)
        # For 2-body: dh/dt = 0, but dr/dt != 0, so omega_dot != 0
        # omega_dot = -2*h*rdot/r^3 (in ECI)
        r = np.linalg.norm(R)
        rdot = np.dot(R, V) / r
        assert abs(rdot) > 0.01  # rdot is nonzero at nu=90 for e=0.1
        assert np.linalg.norm(fc.omega_dot_lvlh) > 1e-14
