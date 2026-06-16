"""The implicitfast default must reach the uploaded MJWarp device model.

The warp wrapper compiles the device ``mjModel`` straight from the raw XML
(``_compile_raw_mujoco_model``), bypassing the C++ host post-processing in
``MjoModel::FromSpecXml``. The integrator default therefore has to be re-applied
on the device-compile path, otherwise the device silently runs MuJoCo's stock
Euler while the CPU host reports implicitfast (issue #12 / PR #13 review).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import pytest

pytest.importorskip("mujoco_warp")

from mjorbit_warp import MjoModel
from mjorbit_warp.model import _compile_raw_mujoco_model

mjINT = mujoco.mjtIntegrator

_BARE_XML = (
    '<mujoco><option timestep="0.01" gravity="0 0 0" {opt}/>'
    '<worldbody><body name="sc"><freejoint/>'
    '<geom type="box" size="0.5 0.3 0.2" mass="100"/></body></worldbody></mujoco>'
)


def test_device_compile_defaults_to_implicitfast():
    model = _compile_raw_mujoco_model(_BARE_XML.format(opt=""), mj_timestep=0.01)
    assert model.opt.integrator == int(mjINT.mjINT_IMPLICITFAST)


def test_device_compile_upgrades_euler():
    model = _compile_raw_mujoco_model(
        _BARE_XML.format(opt='integrator="Euler"'), mj_timestep=0.01
    )
    assert model.opt.integrator == int(mjINT.mjINT_IMPLICITFAST)


def test_device_compile_preserves_explicit_choice():
    for name, expected in (("RK4", mjINT.mjINT_RK4), ("implicit", mjINT.mjINT_IMPLICIT)):
        model = _compile_raw_mujoco_model(
            _BARE_XML.format(opt=f'integrator="{name}"'), mj_timestep=0.01
        )
        assert model.opt.integrator == int(expected), name


def test_uploaded_device_model_matches_host(tmp_path: Path):
    path = tmp_path / "model.xml"
    path.write_text(_BARE_XML.format(opt=""))
    model = MjoModel.from_xml_path(str(path))
    # Host (reported) and device (uploaded source) must agree on implicitfast.
    assert model.opt.integrator == int(mjINT.mjINT_IMPLICITFAST)
    assert model.mj_model.opt.integrator == int(mjINT.mjINT_IMPLICITFAST)
