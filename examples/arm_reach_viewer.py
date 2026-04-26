"""Visualise the articulated spacecraft arm-reach scenario.

Same orbit and arm setup as ``arm_reach.py``, but rendered in the
browser via the viser-based MjOrbitViewer.

* 0 - 5 s   : free drift (no control)
* 5 - 120 s : arm slew to shoulder=1.0, elbow=-0.8 rad

The end-effector trajectory is drawn as a yellow trail.

Usage:
    pixi install
    pixi run python examples/arm_reach_viewer.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from _orbit_reference import circular_orbit_eci

from mujoco_orbit import MjoData, MjoModel, OrbitInit
from mujoco_orbit.constants import R_EARTH
from mujoco_orbit.testdata import SPACECRAFT_ARM_XML
from viewer import MjOrbitViewer


def _compile_model(xml_path: str, *, mj_timestep: float) -> MjoModel:
    mjorbit = (
        '<mjorbit use_j2="false" use_drag="false" use_srp="false" '
        'use_magnetic="false">\n  </mjorbit>\n'
    )
    xml = Path(xml_path).read_text().replace("</mujoco>", f"  {mjorbit}</mujoco>")
    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as file:
        file.write(xml)
        configured_path = Path(file.name)
    try:
        return MjoModel.from_xml_path(str(configured_path), mj_timestep=mj_timestep)
    finally:
        configured_path.unlink(missing_ok=True)


def main() -> None:
    # ------------------------------------------------------------------
    # Orbit & scenario — identical to arm_reach.py
    # ------------------------------------------------------------------
    alt_km = 400.0
    a_km = R_EARTH + alt_km

    R_eci, V_eci = circular_orbit_eci(a_km, np.deg2rad(51.6))

    model = _compile_model(SPACECRAFT_ARM_XML, mj_timestep=0.002)
    data = MjoData(model, orbit=OrbitInit(R_eci=R_eci, V_eci=V_eci))

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------
    viewer = MjOrbitViewer(
        model,
        data,
        port=8080,
        show_earth=True,
        show_axes=True,
        track_body="ee",
    )

    # ------------------------------------------------------------------
    # Action callback: free drift then arm slew
    # ------------------------------------------------------------------
    target_ctrl = np.array([1.0, -0.8])

    def action_fn(data, t):
        if t > 5.0:
            return target_ctrl
        return None

    print("=" * 50)
    print("Arm-Reach Viewer")
    print("=" * 50)
    print("  0 -  5 s : free drift (no control)")
    print("  5 - 120 s: arm slew (shoulder=1.0, elbow=-0.8)")
    print("  Tracking : end-effector trail")
    print()

    viewer.run(duration=120.0, action_fn=action_fn)


if __name__ == "__main__":
    main()
