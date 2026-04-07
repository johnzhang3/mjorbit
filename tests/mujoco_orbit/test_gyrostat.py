"""Gyrostat and rigid-body attitude dynamics tests.

Validates:
- Torque-free rigid body conserves angular momentum and kinetic energy
- Torque-free gyrostat (body + constant-speed reaction wheels) conserves L and E
- Major-axis spin is stable under perturbation
- Intermediate-axis spin is unstable (tennis racket theorem)
- Inertia perturbation via fullinertia XML preserves dynamics correctness
- Gyrostat dynamic balance produces equilibrium
"""

import tempfile

import mujoco
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from mujoco_orbit import compile, step
from mujoco_orbit.constants import R_EARTH
from mujoco_orbit.core.config import (
    MuJoCoCfg,
    OrbitCfg,
    ReactionWheelCfg,
    ScenarioCfg,
)
from mujoco_orbit.orbit.elements import keplerian_to_cartesian


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _perturb_inertia(
    J: np.ndarray,
    eigenvalue_sigma: float = 0.03,
    axis_sigma_deg: float = 3.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Perturb inertia via eigendecomposition: J = VDV^T -> V~D~V~^T."""
    if rng is None:
        rng = np.random.default_rng(42)
    D_vals, V = np.linalg.eigh(J)
    d = rng.standard_normal(3) * eigenvalue_sigma
    D_tilde = np.diag(D_vals * (1.0 + d))
    v = rng.standard_normal(3) * np.deg2rad(axis_sigma_deg)
    V_tilde = V @ expm(_skew(v))
    return V_tilde @ D_tilde @ V_tilde.T


def _make_xml(mass: float, J: np.ndarray) -> str:
    """Generate MuJoCo XML with full inertia tensor via fullinertia."""
    fi = f"{J[0,0]} {J[1,1]} {J[2,2]} {J[0,1]} {J[0,2]} {J[1,2]}"
    return f"""<mujoco model="gyro_test">
  <compiler angle="radian" balanceinertia="true"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="body" pos="0 0 0">
      <freejoint name="base"/>
      <inertial pos="0 0 0" mass="{mass}" fullinertia="{fi}"/>
      <geom type="box" size="0.5 0.5 0.5" mass="0"/>
    </body>
  </worldbody>
</mujoco>"""


def _leo_orbit():
    a = R_EARTH + 400.0
    return keplerian_to_cartesian(
        a=a, e=0.0, inc=np.deg2rad(51.6), raan=0.0, argp=0.0, nu=0.0,
    )


def _compile_scenario(xml_str: str, rw_cfgs=None, dt=0.002):
    f = tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False)
    f.write(xml_str)
    f.flush()
    R_eci, V_eci = _leo_orbit()
    cfg = ScenarioCfg(
        orbit=OrbitCfg(R_eci=R_eci, V_eci=V_eci),
        mujoco=MuJoCoCfg(xml_path=f.name, dt=dt),
        surfaces=[],
        reaction_wheels=rw_cfgs or [],
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    return compile(cfg), f.name


# ---------------------------------------------------------------------------
# Conservation tests — single rigid body, torque-free
# ---------------------------------------------------------------------------

class TestTorqueFreeConservation:
    """Torque-free rigid body should conserve angular momentum and energy."""

    MASS = 100.0
    J = np.diag([3.0, 5.0, 8.0])  # distinct principal MOI

    def _run(self, omega0, t_total=10.0, dt=0.002):
        xml = _make_xml(self.MASS, self.J)
        scenario, _ = _compile_scenario(xml, dt=dt)
        scenario.mjd.qvel[3:6] = omega0
        mujoco.mj_forward(scenario.mjm, scenario.mjd)

        n_steps = int(t_total / dt)
        for _ in range(n_steps):
            step(scenario)

        return scenario.mjd.qvel[3:6].copy()

    def test_energy_conservation_major_axis(self):
        """Spin about major axis (I=8): energy conserved."""
        omega0 = np.array([0.01, 0.01, 5.0])
        w = self._run(omega0)

        E0 = 0.5 * omega0 @ self.J @ omega0
        Ef = 0.5 * w @ self.J @ w
        assert abs(Ef - E0) / E0 < 1e-4, f"Energy drift {(Ef-E0)/E0:.2e}"

    def test_momentum_conservation_major_axis(self):
        """Spin about major axis: |L| conserved."""
        omega0 = np.array([0.01, 0.01, 5.0])
        w = self._run(omega0)

        L0 = np.linalg.norm(self.J @ omega0)
        Lf = np.linalg.norm(self.J @ w)
        assert abs(Lf - L0) / L0 < 1e-4, f"|L| drift {(Lf-L0)/L0:.2e}"

    def test_energy_conservation_oblique_spin(self):
        """Oblique spin: energy conserved despite nutation.

        Explicit Euler drifts more for strongly nutating cases,
        so use a smaller timestep.
        """
        omega0 = np.array([1.0, 2.0, 3.0])
        w = self._run(omega0, t_total=10.0, dt=0.001)

        E0 = 0.5 * omega0 @ self.J @ omega0
        Ef = 0.5 * w @ self.J @ w
        assert abs(Ef - E0) / E0 < 0.05, f"Energy drift {(Ef-E0)/E0:.2e}"

    def test_momentum_conservation_oblique_spin(self):
        """Oblique spin: |L| conserved."""
        omega0 = np.array([1.0, 2.0, 3.0])
        w = self._run(omega0, t_total=10.0, dt=0.001)

        L0 = np.linalg.norm(self.J @ omega0)
        Lf = np.linalg.norm(self.J @ w)
        assert abs(Lf - L0) / L0 < 0.01, f"|L| drift {(Lf-L0)/L0:.2e}"

    def test_matches_scipy_euler_equations(self):
        """MuJoCo should match scipy integration of Euler equations."""
        J = self.J
        J_inv = np.linalg.inv(J)
        omega0 = np.array([0.5, 1.0, 3.0])

        def eom(t, w):
            return J_inv @ (-np.cross(w, J @ w))

        sol = solve_ivp(eom, [0, 5], omega0, method="DOP853",
                        rtol=1e-12, atol=1e-14)
        w_scipy = sol.y[:, -1]

        w_mj = self._run(omega0, t_total=5.0, dt=0.001)

        np.testing.assert_allclose(w_mj, w_scipy, atol=0.1,
                                   err_msg="MuJoCo diverges from scipy Euler equations")


# ---------------------------------------------------------------------------
# Perturbed inertia tests
# ---------------------------------------------------------------------------

class TestPerturbedInertia:
    """Non-diagonal inertia (via fullinertia) should conserve L and E."""

    MASS = 100.0

    def _make_perturbed_J(self, seed=42):
        # Use inertia values that satisfy triangle inequality after perturbation
        J_nom = np.diag([6.0, 8.0, 10.0])
        return _perturb_inertia(J_nom, eigenvalue_sigma=0.02,
                                axis_sigma_deg=2.0,
                                rng=np.random.default_rng(seed))

    def test_fullinertia_preserves_tensor(self):
        """fullinertia XML should reconstruct the correct J tensor."""
        J = self._make_perturbed_J()
        xml = _make_xml(self.MASS, J)
        scenario, _ = _compile_scenario(xml)
        bid = scenario.body_id("body")

        R_iq = np.zeros(9)
        mujoco.mju_quat2Mat(R_iq, scenario.mjm.body_iquat[bid])
        R_iq = R_iq.reshape(3, 3)
        J_recon = R_iq @ np.diag(scenario.mjm.body_inertia[bid]) @ R_iq.T

        np.testing.assert_allclose(J_recon, J, atol=1e-6,
                                   err_msg="fullinertia J reconstruction error")

    def test_conservation_with_perturbed_inertia(self):
        """Perturbed inertia: L and E conserved for torque-free spin."""
        J = self._make_perturbed_J()
        xml = _make_xml(self.MASS, J)
        scenario, _ = _compile_scenario(xml, dt=0.001)

        omega0 = np.array([0.5, 1.0, 3.0])
        scenario.mjd.qvel[3:6] = omega0
        mujoco.mj_forward(scenario.mjm, scenario.mjd)

        for _ in range(10000):  # 10s at dt=0.001
            step(scenario)

        w = scenario.mjd.qvel[3:6].copy()

        L0 = np.linalg.norm(J @ omega0)
        Lf = np.linalg.norm(J @ w)
        E0 = 0.5 * omega0 @ J @ omega0
        Ef = 0.5 * w @ J @ w

        assert abs(Lf - L0) / L0 < 0.01, f"|L| drift {(Lf-L0)/L0:.2e}"
        assert abs(Ef - E0) / E0 < 0.01, f"E drift {(Ef-E0)/E0:.2e}"

    def test_perturbed_matches_scipy(self):
        """Perturbed inertia dynamics should match scipy."""
        J = self._make_perturbed_J()
        J_inv = np.linalg.inv(J)
        omega0 = np.array([0.5, 1.0, 3.0])

        def eom(t, w):
            return J_inv @ (-np.cross(w, J @ w))

        sol = solve_ivp(eom, [0, 5], omega0, method="DOP853",
                        rtol=1e-12, atol=1e-14)
        w_scipy = sol.y[:, -1]

        xml = _make_xml(100.0, J)
        scenario, _ = _compile_scenario(xml, dt=0.001)
        scenario.mjd.qvel[3:6] = omega0
        mujoco.mj_forward(scenario.mjm, scenario.mjd)

        for _ in range(5000):  # 5s
            step(scenario)

        w_mj = scenario.mjd.qvel[3:6].copy()
        np.testing.assert_allclose(w_mj, w_scipy, atol=0.1,
                                   err_msg="MuJoCo perturbed inertia diverges from scipy")


# ---------------------------------------------------------------------------
# Spin stability tests
# ---------------------------------------------------------------------------

class TestSpinStability:
    """Major-axis spin stable, intermediate-axis spin unstable."""

    MASS = 100.0
    J = np.diag([3.0, 5.0, 8.0])  # Ix < Iy < Iz
    PERTURBATION = 0.01  # 1% transverse kick

    def _run_stability(self, spin_axis: int, t_total=20.0, dt=0.002):
        xml = _make_xml(self.MASS, self.J)
        scenario, _ = _compile_scenario(xml, dt=dt)

        omega0 = np.zeros(3)
        omega0[spin_axis] = 5.0
        transverse = [i for i in range(3) if i != spin_axis]
        for i in transverse:
            omega0[i] = self.PERTURBATION * omega0[spin_axis]

        scenario.mjd.qvel[3:6] = omega0
        mujoco.mj_forward(scenario.mjm, scenario.mjd)

        n_steps = int(t_total / dt)
        for _ in range(n_steps):
            step(scenario)

        w = scenario.mjd.qvel[3:6].copy()
        transverse_rate = np.sqrt(sum(w[i]**2 for i in transverse))
        spin_rate = abs(w[spin_axis])
        return transverse_rate, spin_rate

    def test_major_axis_stable(self):
        """Spin about Z (Iz=8, max): transverse rates stay small."""
        trans, spin = self._run_stability(spin_axis=2)
        ratio = trans / spin
        assert ratio < 0.05, (
            f"Major-axis spin unstable: transverse/spin = {ratio:.3f}"
        )

    def test_minor_axis_stable(self):
        """Spin about X (Ix=3, min): transverse rates stay small."""
        trans, spin = self._run_stability(spin_axis=0)
        ratio = trans / spin
        assert ratio < 0.05, (
            f"Minor-axis spin unstable: transverse/spin = {ratio:.3f}"
        )

    def test_intermediate_axis_unstable(self):
        """Spin about Y (Iy=5, intermediate): transverse rates grow."""
        # Use larger perturbation and longer time for the instability to develop
        xml = _make_xml(self.MASS, self.J)
        scenario, _ = _compile_scenario(xml, dt=0.002)

        omega0 = np.array([0.0, 5.0, 0.0])
        omega0[0] = 0.05 * 5.0  # 5% perturbation
        omega0[2] = 0.05 * 5.0
        scenario.mjd.qvel[3:6] = omega0
        mujoco.mj_forward(scenario.mjm, scenario.mjd)

        for _ in range(25000):  # 50s
            step(scenario)

        w = scenario.mjd.qvel[3:6].copy()
        trans = np.sqrt(w[0]**2 + w[2]**2)
        spin = abs(w[1])
        ratio = trans / (spin + trans)  # fraction of energy in transverse
        assert ratio > 0.1, (
            f"Intermediate-axis spin unexpectedly stable: ratio = {ratio:.3f}"
        )


# ---------------------------------------------------------------------------
# Gyrostat conservation tests
# ---------------------------------------------------------------------------

class TestGyrostatConservation:
    """Gyrostat (body + constant-speed wheels) should conserve L and E."""

    MASS = 100.0
    J = np.diag([3.0, 5.0, 8.0])
    RW_INERTIA = 0.1  # kg·m²

    def _compute_dynamic_balance(self, J, omega, ratio=1.2):
        """Compute h for dynamic balance + superspin."""
        omega_hat = omega / np.linalg.norm(omega)
        J_omega = J @ omega

        if abs(omega_hat[0]) < 0.9:
            e1 = np.cross(omega_hat, [1, 0, 0])
        else:
            e1 = np.cross(omega_hat, [0, 1, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(omega_hat, e1)
        P = np.array([e1, e2])
        J_perp = P @ J @ P.T
        I_trans_max = np.max(np.linalg.eigvalsh(J_perp))

        lam = max(ratio * I_trans_max,
                  np.dot(J_omega, omega_hat) / np.linalg.norm(omega))
        h = lam * omega - J_omega
        return h

    def _setup_gyrostat(self, J, omega0, h):
        xml = _make_xml(self.MASS, J)
        rw_cfgs = [
            ReactionWheelCfg(
                body_name="body",
                axis_body=np.array([float(i == j) for j in range(3)]),
                inertia=self.RW_INERTIA,
            )
            for i in range(3)
        ]
        scenario, path = _compile_scenario(xml, rw_cfgs=rw_cfgs)

        for i in range(3):
            scenario.actuator_state.rw_speed[i] = h[i] / self.RW_INERTIA
        scenario.actuator_state.update_rw_momentum()

        scenario.mjd.qvel[3:6] = omega0
        mujoco.mj_forward(scenario.mjm, scenario.mjd)
        return scenario

    def test_gyrostat_energy_conservation(self):
        """Gyrostat with rotor momentum: energy conserved."""
        J = self.J
        omega0 = np.array([0.1, 0.1, 5.0])
        h = self._compute_dynamic_balance(J, np.array([0, 0, 5.0]))

        scenario = self._setup_gyrostat(J, omega0, h)

        E0 = 0.5 * omega0 @ J @ omega0

        for _ in range(5000):  # 10s
            step(scenario)

        w = scenario.mjd.qvel[3:6].copy()
        Ef = 0.5 * w @ J @ w

        assert abs(Ef - E0) / E0 < 1e-3, f"Energy drift {(Ef-E0)/E0:.2e}"

    def test_gyrostat_momentum_conservation(self):
        """Gyrostat: |L_total| = |Jw + h| conserved."""
        J = self.J
        omega0 = np.array([0.1, 0.1, 5.0])
        h = self._compute_dynamic_balance(J, np.array([0, 0, 5.0]))

        scenario = self._setup_gyrostat(J, omega0, h)

        L0 = np.linalg.norm(J @ omega0 + h)

        for _ in range(5000):
            step(scenario)

        w = scenario.mjd.qvel[3:6].copy()
        Lf = np.linalg.norm(J @ w + h)

        assert abs(Lf - L0) / L0 < 1e-4, f"|L| drift {(Lf-L0)/L0:.2e}"

    def test_dynamic_balance_equilibrium(self):
        """At dynamic balance, exact equilibrium should be maintained."""
        J = self.J
        omega_eq = np.array([0.0, 0.0, 5.0])
        h = self._compute_dynamic_balance(J, omega_eq)

        scenario = self._setup_gyrostat(J, omega_eq, h)

        for _ in range(5000):
            step(scenario)

        w = scenario.mjd.qvel[3:6].copy()
        # Transverse rates should stay near zero
        assert abs(w[0]) < 0.01, f"wx grew to {w[0]:.4f}"
        assert abs(w[1]) < 0.01, f"wy grew to {w[1]:.4f}"
        np.testing.assert_allclose(w[2], omega_eq[2], rtol=1e-3)

    def test_gyrostat_stability_perturbed_ic(self):
        """Perturbed IC with dynamic-balanced gyrostat: transverse rates bounded."""
        J_nom = np.diag([6.0, 8.0, 10.0])
        J = _perturb_inertia(J_nom, eigenvalue_sigma=0.02, axis_sigma_deg=2.0,
                             rng=np.random.default_rng(99))
        omega_spin = np.array([0.0, 0.0, 5.0])
        h = self._compute_dynamic_balance(J, omega_spin)

        omega0 = omega_spin.copy()
        omega0[0] += 0.05  # 1% perturbation
        omega0[1] += 0.05

        scenario = self._setup_gyrostat(J, omega0, h)

        for _ in range(10000):  # 20s
            step(scenario)

        w = scenario.mjd.qvel[3:6].copy()
        transverse = np.sqrt(w[0]**2 + w[1]**2)
        assert transverse < 0.5, (
            f"Gyrostat unstable: transverse rate {transverse:.3f} rad/s"
        )
        # Spin rate should still be close to original
        np.testing.assert_allclose(abs(w[2]), omega_spin[2], rtol=0.05)


# ---------------------------------------------------------------------------
# MuJoCo frame convention tests
# ---------------------------------------------------------------------------

class TestMuJoCoFrameConventions:
    """Verify MuJoCo frame conventions for qvel and rotation matrices."""

    def _make_model(self, fullinertia=False):
        if fullinertia:
            inertia = 'fullinertia="5 4 3 1 0 0"'
        else:
            inertia = 'diaginertia="3 4 5"'
        xml = f"""<mujoco><compiler angle="radian" balanceinertia="true"/>
        <option timestep="0.001" gravity="0 0 0"/>
        <worldbody><body name="b"><freejoint/>
        <inertial pos="0 0 0" mass="1" {inertia}/>
        <geom type="box" size=".1 .1 .1" mass="0"/>
        </body></worldbody></mujoco>"""
        f = tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False)
        f.write(xml)
        f.flush()
        m = mujoco.MjModel.from_xml_path(f.name)
        d = mujoco.MjData(m)
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "b")
        return m, d, bid

    def test_qvel_is_body_frame(self):
        """Free-joint qvel[3:6] should be in the body frame."""
        import math
        m, d, bid = self._make_model()

        # Pre-rotate 90° about Z
        d.qpos[3] = math.cos(math.pi / 4)
        d.qpos[6] = math.sin(math.pi / 4)
        d.qvel[3:6] = [1, 0, 0]
        mujoco.mj_forward(m, d)

        w_world = d.cvel[bid, :3]
        R_body = d.xmat[bid].reshape(3, 3)
        expected_world = R_body @ np.array([1, 0, 0])

        np.testing.assert_allclose(w_world, expected_world, atol=1e-10,
                                   err_msg="qvel is not in body frame")

    def test_xmat_vs_ximat_diagonal(self):
        """For diagonal inertia, xmat and ximat should be identical."""
        m, d, bid = self._make_model(fullinertia=False)
        mujoco.mj_forward(m, d)

        xmat = d.xmat[bid].reshape(3, 3)
        ximat = d.ximat[bid].reshape(3, 3)
        np.testing.assert_allclose(xmat, ximat, atol=1e-10)

    def test_xmat_vs_ximat_fullinertia(self):
        """For fullinertia with off-diagonal terms, xmat != ximat."""
        m, d, bid = self._make_model(fullinertia=True)
        mujoco.mj_forward(m, d)

        xmat = d.xmat[bid].reshape(3, 3)
        ximat = d.ximat[bid].reshape(3, 3)

        # They should differ when body_iquat is non-trivial
        if not np.allclose(m.body_iquat[bid], [1, 0, 0, 0]):
            assert not np.allclose(xmat, ximat), (
                "xmat and ximat should differ for non-diagonal inertia"
            )
