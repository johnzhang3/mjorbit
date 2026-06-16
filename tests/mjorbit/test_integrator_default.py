"""mjorbit defaults the MuJoCo-side integrator to ``implicitfast``.

MuJoCo's stock default is semi-implicit Euler, which does not conserve angular
momentum for freely tumbling / asymmetric rigid bodies (issue #12). mjorbit
targets coupled orbital + attitude dynamics, so it defaults unspecified models
(and models that ask for Euler) to ``implicitfast`` while preserving explicit
``implicit`` / ``implicitfast`` / ``RK4`` choices.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from mjorbit import MjoModel, OrbitInit, mjo_forward, mjo_step

mjINT = mujoco.mjtIntegrator


def _write_model(tmp_path: Path, *, integrator_attr: str = "") -> str:
    option = f'<option timestep="0.01" gravity="0 0 0" {integrator_attr}/>'
    xml = f"""<mujoco model="asym">
  {option}
  <worldbody>
    <body name="sc">
      <freejoint/>
      <geom type="box" size="0.5 0.3 0.2" mass="100"/>
    </body>
  </worldbody>
</mujoco>"""
    path = tmp_path / "model.xml"
    path.write_text(xml)
    return str(path)


def _angular_momentum_world(model: MjoModel, data, bid: int) -> np.ndarray:
    omega_body = np.asarray(data.qvel[3:6], dtype=float)
    R_wb = np.asarray(data.xmat[bid], dtype=float).reshape(3, 3)
    R_wi = np.asarray(data.ximat[bid], dtype=float).reshape(3, 3)
    inertia = np.asarray(model.body_inertia[bid], dtype=float)
    omega_world = R_wb @ omega_body
    return R_wi @ np.diag(inertia) @ R_wi.T @ omega_world


def test_default_integrator_is_implicitfast(tmp_path: Path):
    model = MjoModel.from_xml_path(_write_model(tmp_path))
    assert model.opt.integrator == int(mjINT.mjINT_IMPLICITFAST)


def test_explicit_euler_is_upgraded_to_implicitfast(tmp_path: Path):
    model = MjoModel.from_xml_path(_write_model(tmp_path, integrator_attr='integrator="Euler"'))
    assert model.opt.integrator == int(mjINT.mjINT_IMPLICITFAST)


def test_explicit_non_euler_integrators_are_preserved(tmp_path: Path):
    for name, expected in (
        ("RK4", mjINT.mjINT_RK4),
        ("implicit", mjINT.mjINT_IMPLICIT),
        ("implicitfast", mjINT.mjINT_IMPLICITFAST),
    ):
        model = MjoModel.from_xml_path(
            _write_model(tmp_path, integrator_attr=f'integrator="{name}"')
        )
        assert model.opt.integrator == int(expected), name


def test_default_integrator_conserves_angular_momentum(tmp_path: Path):
    """A freely tumbling asymmetric body must (very nearly) conserve |H|.

    Under the stock Euler default this drifts by tens of percent (issue #12);
    under the implicitfast default it stays at the gravity-gradient floor.
    """
    model = MjoModel.from_xml_path(_write_model(tmp_path))
    assert model.opt.integrator == int(mjINT.mjINT_IMPLICITFAST)
    data = model.make_data(orbit=OrbitInit(R_eci=[7000.0, 0.0, 0.0], V_eci=[0.0, 7.5, 0.0]))
    bid = model.body_id("sc")

    data.qvel[3:6] = [2.0, 1.3, 0.7]  # generic tumble (all principal axes)
    mjo_forward(model, data)
    h0 = float(np.linalg.norm(_angular_momentum_world(model, data, bid)))

    for _ in range(3000):  # 30 s at dt = 0.01
        mjo_step(model, data)

    h1 = float(np.linalg.norm(_angular_momentum_world(model, data, bid)))
    rel_drift = abs(h1 - h0) / h0
    assert rel_drift < 1.0e-3, f"|H| drift {rel_drift:.3e} too large for implicitfast"
