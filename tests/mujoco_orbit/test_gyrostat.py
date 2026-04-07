"""Gyrostat and rigid-body attitude dynamics tests."""

import tempfile

import mujoco
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from mujoco_orbit import MjoData, MjoModel, OrbitInit, ReactionWheelSpec, mjo_forward, mjo_step
from mujoco_orbit.constants import R_EARTH
from mujoco_orbit.orbit.elements import keplerian_to_cartesian


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _perturb_inertia(
    J: np.ndarray,
    eigenvalue_sigma: float = 0.03,
    axis_sigma_deg: float = 3.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)
    d_vals, V = np.linalg.eigh(J)
    d = rng.standard_normal(3) * eigenvalue_sigma
    d_tilde = np.diag(d_vals * (1.0 + d))
    v = rng.standard_normal(3) * np.deg2rad(axis_sigma_deg)
    V_tilde = V @ expm(_skew(v))
    return V_tilde @ d_tilde @ V_tilde.T


def _make_xml(mass: float, J: np.ndarray) -> str:
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


def _compile_model_data(xml_str: str, reaction_wheels=None, dt=0.002):
    file = tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False)
    file.write(xml_str)
    file.flush()
    r_eci, v_eci = _leo_orbit()
    model = MjoModel.from_xml_path(
        file.name,
        reaction_wheels=reaction_wheels or [],
        mj_timestep=dt,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    data = MjoData(model, orbit=OrbitInit(R_eci=r_eci, V_eci=v_eci))
    return model, data, file.name


class TestTorqueFreeConservation:
    MASS = 100.0
    J = np.diag([3.0, 5.0, 8.0])

    def _run(self, omega0, t_total=10.0, dt=0.002):
        model, data, _ = _compile_model_data(_make_xml(self.MASS, self.J), dt=dt)
        data.qvel[3:6] = omega0
        mjo_forward(model, data)

        for _ in range(int(t_total / dt)):
            mjo_step(model, data)

        return data.qvel[3:6].copy()

    def test_energy_conservation_major_axis(self):
        omega0 = np.array([0.01, 0.01, 5.0])
        w = self._run(omega0)
        E0 = 0.5 * omega0 @ self.J @ omega0
        Ef = 0.5 * w @ self.J @ w
        assert abs(Ef - E0) / E0 < 1e-4

    def test_momentum_conservation_major_axis(self):
        omega0 = np.array([0.01, 0.01, 5.0])
        w = self._run(omega0)
        L0 = np.linalg.norm(self.J @ omega0)
        Lf = np.linalg.norm(self.J @ w)
        assert abs(Lf - L0) / L0 < 1e-4

    def test_energy_conservation_oblique_spin(self):
        omega0 = np.array([1.0, 2.0, 3.0])
        w = self._run(omega0, t_total=10.0, dt=0.001)
        E0 = 0.5 * omega0 @ self.J @ omega0
        Ef = 0.5 * w @ self.J @ w
        assert abs(Ef - E0) / E0 < 0.05

    def test_momentum_conservation_oblique_spin(self):
        omega0 = np.array([1.0, 2.0, 3.0])
        w = self._run(omega0, t_total=10.0, dt=0.001)
        L0 = np.linalg.norm(self.J @ omega0)
        Lf = np.linalg.norm(self.J @ w)
        assert abs(Lf - L0) / L0 < 0.01

    def test_matches_scipy_euler_equations(self):
        J_inv = np.linalg.inv(self.J)
        omega0 = np.array([0.5, 1.0, 3.0])

        def eom(_, w):
            return J_inv @ (-np.cross(w, self.J @ w))

        sol = solve_ivp(eom, [0, 5], omega0, method="DOP853", rtol=1e-12, atol=1e-14)
        w_scipy = sol.y[:, -1]
        w_mj = self._run(omega0, t_total=5.0, dt=0.001)
        np.testing.assert_allclose(w_mj, w_scipy, atol=0.1)


class TestPerturbedInertia:
    MASS = 100.0

    def _make_perturbed_J(self, seed=42):
        return _perturb_inertia(
            np.diag([6.0, 8.0, 10.0]),
            eigenvalue_sigma=0.02,
            axis_sigma_deg=2.0,
            rng=np.random.default_rng(seed),
        )

    def test_fullinertia_preserves_tensor(self):
        J = self._make_perturbed_J()
        model, _, _ = _compile_model_data(_make_xml(self.MASS, J))
        body_id = model.body_id("body")

        R_iq = np.zeros(9)
        mujoco.mju_quat2Mat(R_iq, model.body_iquat[body_id])
        R_iq = R_iq.reshape(3, 3)
        J_recon = R_iq @ np.diag(model.body_inertia[body_id]) @ R_iq.T

        np.testing.assert_allclose(J_recon, J, atol=1e-6)

    def test_conservation_with_perturbed_inertia(self):
        J = self._make_perturbed_J()
        model, data, _ = _compile_model_data(_make_xml(self.MASS, J), dt=0.001)

        omega0 = np.array([0.5, 1.0, 3.0])
        data.qvel[3:6] = omega0
        mjo_forward(model, data)

        for _ in range(10000):
            mjo_step(model, data)

        w = data.qvel[3:6].copy()
        L0 = np.linalg.norm(J @ omega0)
        Lf = np.linalg.norm(J @ w)
        E0 = 0.5 * omega0 @ J @ omega0
        Ef = 0.5 * w @ J @ w

        assert abs(Lf - L0) / L0 < 0.01
        assert abs(Ef - E0) / E0 < 0.01

    def test_perturbed_matches_scipy(self):
        J = self._make_perturbed_J()
        J_inv = np.linalg.inv(J)
        omega0 = np.array([0.5, 1.0, 3.0])

        def eom(_, w):
            return J_inv @ (-np.cross(w, J @ w))

        sol = solve_ivp(eom, [0, 5], omega0, method="DOP853", rtol=1e-12, atol=1e-14)
        w_scipy = sol.y[:, -1]

        model, data, _ = _compile_model_data(_make_xml(100.0, J), dt=0.001)
        data.qvel[3:6] = omega0
        mjo_forward(model, data)

        for _ in range(5000):
            mjo_step(model, data)

        np.testing.assert_allclose(data.qvel[3:6], w_scipy, atol=0.1)


class TestSpinStability:
    MASS = 100.0
    J = np.diag([3.0, 5.0, 8.0])
    PERTURBATION = 0.01

    def _run_stability(self, spin_axis: int, t_total=20.0, dt=0.002):
        model, data, _ = _compile_model_data(_make_xml(self.MASS, self.J), dt=dt)

        omega0 = np.zeros(3)
        omega0[spin_axis] = 5.0
        transverse = [i for i in range(3) if i != spin_axis]
        for axis in transverse:
            omega0[axis] = self.PERTURBATION * omega0[spin_axis]

        data.qvel[3:6] = omega0
        mjo_forward(model, data)

        for _ in range(int(t_total / dt)):
            mjo_step(model, data)

        w = data.qvel[3:6].copy()
        return np.sqrt(sum(w[i] ** 2 for i in transverse)), abs(w[spin_axis])

    def test_major_axis_stable(self):
        trans, spin = self._run_stability(spin_axis=2)
        assert trans / spin < 0.05

    def test_minor_axis_stable(self):
        trans, spin = self._run_stability(spin_axis=0)
        assert trans / spin < 0.05

    def test_intermediate_axis_unstable(self):
        model, data, _ = _compile_model_data(_make_xml(self.MASS, self.J), dt=0.002)
        omega0 = np.array([0.25, 5.0, 0.25])
        data.qvel[3:6] = omega0
        mjo_forward(model, data)

        for _ in range(25000):
            mjo_step(model, data)

        w = data.qvel[3:6].copy()
        trans = np.sqrt(w[0] ** 2 + w[2] ** 2)
        spin = abs(w[1])
        assert trans / (spin + trans) > 0.1


class TestGyrostatConservation:
    MASS = 100.0
    J = np.diag([3.0, 5.0, 8.0])
    RW_INERTIA = 0.1

    def _compute_dynamic_balance(self, J, omega, ratio=1.2):
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

        lam = max(ratio * I_trans_max, np.dot(J_omega, omega_hat) / np.linalg.norm(omega))
        return lam * omega - J_omega

    def _setup_gyrostat(self, J, omega0, h):
        rw_specs = [
            ReactionWheelSpec(
                body_name="body",
                axis_body=np.array([float(i == j) for j in range(3)]),
                inertia=self.RW_INERTIA,
            )
            for i in range(3)
        ]
        model, data, _ = _compile_model_data(_make_xml(self.MASS, J), reaction_wheels=rw_specs)

        for i in range(3):
            data.actuators.rw_speed[i] = h[i] / self.RW_INERTIA
        data.actuators.update_rw_momentum()

        data.qvel[3:6] = omega0
        mjo_forward(model, data)
        return model, data

    def test_gyrostat_energy_conservation(self):
        omega0 = np.array([0.1, 0.1, 5.0])
        h = self._compute_dynamic_balance(self.J, np.array([0, 0, 5.0]))

        model, data = self._setup_gyrostat(self.J, omega0, h)
        E0 = 0.5 * omega0 @ self.J @ omega0

        for _ in range(5000):
            mjo_step(model, data)

        Ef = 0.5 * data.qvel[3:6] @ self.J @ data.qvel[3:6]
        assert abs(Ef - E0) / E0 < 1e-3

    def test_gyrostat_momentum_conservation(self):
        omega0 = np.array([0.1, 0.1, 5.0])
        h = self._compute_dynamic_balance(self.J, np.array([0, 0, 5.0]))

        model, data = self._setup_gyrostat(self.J, omega0, h)
        L0 = np.linalg.norm(self.J @ omega0 + h)

        for _ in range(5000):
            mjo_step(model, data)

        Lf = np.linalg.norm(self.J @ data.qvel[3:6] + h)
        assert abs(Lf - L0) / L0 < 1e-4

    def test_dynamic_balance_equilibrium(self):
        omega_eq = np.array([0.0, 0.0, 5.0])
        h = self._compute_dynamic_balance(self.J, omega_eq)

        model, data = self._setup_gyrostat(self.J, omega_eq, h)

        for _ in range(5000):
            mjo_step(model, data)

        w = data.qvel[3:6].copy()
        assert abs(w[0]) < 0.01
        assert abs(w[1]) < 0.01
        np.testing.assert_allclose(w[2], omega_eq[2], rtol=1e-3)

    def test_gyrostat_stability_perturbed_ic(self):
        J = _perturb_inertia(
            np.diag([6.0, 8.0, 10.0]),
            eigenvalue_sigma=0.02,
            axis_sigma_deg=2.0,
            rng=np.random.default_rng(99),
        )
        omega_spin = np.array([0.0, 0.0, 5.0])
        h = self._compute_dynamic_balance(J, omega_spin)

        omega0 = omega_spin.copy()
        omega0[0] += 0.05
        omega0[1] += 0.05

        model, data = self._setup_gyrostat(J, omega0, h)

        for _ in range(10000):
            mjo_step(model, data)

        w = data.qvel[3:6].copy()
        assert np.sqrt(w[0] ** 2 + w[1] ** 2) < 0.5
        np.testing.assert_allclose(abs(w[2]), omega_spin[2], rtol=0.05)


class TestMuJoCoFrameConventions:
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
        file = tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False)
        file.write(xml)
        file.flush()
        model = mujoco.MjModel.from_xml_path(file.name)
        data = mujoco.MjData(model)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b")
        return model, data, body_id

    def test_qvel_is_body_frame(self):
        import math

        model, data, body_id = self._make_model()
        data.qpos[3] = math.cos(math.pi / 4)
        data.qpos[6] = math.sin(math.pi / 4)
        data.qvel[3:6] = [1, 0, 0]
        mujoco.mj_forward(model, data)

        w_world = data.cvel[body_id, :3]
        R_body = data.xmat[body_id].reshape(3, 3)
        expected_world = R_body @ np.array([1, 0, 0])

        np.testing.assert_allclose(w_world, expected_world, atol=1e-10)

    def test_xmat_vs_ximat_diagonal(self):
        model, data, body_id = self._make_model(fullinertia=False)
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(
            data.xmat[body_id].reshape(3, 3),
            data.ximat[body_id].reshape(3, 3),
            atol=1e-10,
        )

    def test_xmat_vs_ximat_fullinertia(self):
        model, data, body_id = self._make_model(fullinertia=True)
        mujoco.mj_forward(model, data)

        xmat = data.xmat[body_id].reshape(3, 3)
        ximat = data.ximat[body_id].reshape(3, 3)
        if not np.allclose(model.body_iquat[body_id], [1, 0, 0, 0]):
            assert not np.allclose(xmat, ximat)
