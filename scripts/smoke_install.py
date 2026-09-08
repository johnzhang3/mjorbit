"""Exercise an installed release from outside the source tree (no pytest needed)."""

from __future__ import annotations

import subprocess
import sys
import sysconfig
from pathlib import Path

import numpy as np

import mjorbit
from mjorbit import MjoModel, OrbitInit, mjo_step
from mjorbit._native import find_native_plugin_path
from mjorbit.constants import GM_EARTH, R_EARTH
from mjorbit.testdata import FREE_BODY_XML
from viewer.earth import EARTH_TEXTURE_PATH


def main() -> None:
    prefix = Path(sys.prefix).resolve()
    assert Path(mjorbit.__file__).resolve().is_relative_to(prefix), mjorbit.__file__
    plugin = find_native_plugin_path()
    assert plugin is not None and Path(plugin).resolve().is_relative_to(prefix), plugin
    assert Path(FREE_BODY_XML).is_file()
    assert EARTH_TEXTURE_PATH.is_file()

    radius = R_EARTH + 400.0
    model = MjoModel.from_xml_path(FREE_BODY_XML, mj_timestep=0.01)
    data = model.make_data(
        orbit=OrbitInit(
            R_eci=[radius, 0.0, 0.0],
            V_eci=[0.0, np.sqrt(GM_EARTH / radius), 0.0],
        )
    )
    for _ in range(100):
        mjo_step(model, data)
    np.testing.assert_allclose(data.time, 1.0, atol=1e-12)
    assert np.isfinite(data.qpos).all()
    assert data.orbit.R_eci[1] > 7.0

    viewer_command = Path(sysconfig.get_path("scripts")) / "mjo-viewer"
    tasks = subprocess.check_output([str(viewer_command), "--list-tasks"], text=True)
    assert set(tasks.splitlines()) == {
        "free_drift", "arm_reach_mppi", "capture_stabilize_mppi"
    }, tasks
    print(f"Installed-package smoke check passed: {mjorbit.__file__}")


if __name__ == "__main__":
    main()
