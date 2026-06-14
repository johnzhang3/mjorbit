"""Regression tests for the articulated spacecraft arm MJCF."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import numpy.testing as npt

from mjorbit.testdata import SPACECRAFT_ARM_XML


def _load_spacecraft_arm_model(*, fixed_base: bool) -> mujoco.MjModel:
    xml_text = Path(SPACECRAFT_ARM_XML).read_text()
    if fixed_base:
        xml_text = xml_text.replace('<freejoint name="base"/>\n', "")
    return mujoco.MjModel.from_xml_string(xml_text)


def test_spacecraft_arm_joint_ranges_use_radians() -> None:
    model = _load_spacecraft_arm_model(fixed_base=False)

    npt.assert_allclose(
        model.jnt_range[1:],
        np.array([[-3.14, 3.14], [-3.14, 3.14]]),
        atol=1e-6,
    )
    npt.assert_allclose(
        model.actuator_ctrlrange,
        np.array([[-3.14, 3.14], [-3.14, 3.14]]),
        atol=1e-6,
    )


def test_spacecraft_arm_reaches_target_in_pure_mujoco() -> None:
    model = _load_spacecraft_arm_model(fixed_base=True)
    data = mujoco.MjData(model)
    ctrl = np.array([1.0, -0.8])

    for _ in range(int(10.0 / model.opt.timestep)):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

    npt.assert_allclose(data.qpos, ctrl, atol=5e-3)
