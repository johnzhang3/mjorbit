"""Visualise the articulated spacecraft arm-reach scenario.

Same orbit and arm setup as ``cpu_arm_reach.py``, but rendered in the
browser via the viser-based MjOrbitViewer.

* 0 - 5 s   : free drift (no control)
* 5 - 120 s : arm slew to shoulder=1.0, elbow=-0.8 rad

The end-effector trajectory is drawn as a yellow trail.

Usage:
    uv sync --extra viewer
    uv run python examples/cpu_arm_reach_viewer.py
"""

from __future__ import annotations

import numpy as np

from mjorbit.constants import R_EARTH
from mjorbit.cpu import compile_cpu
from mjorbit.cpu.core.config import CPUScenarioCfg, MuJoCoCfg, OrbitCfg
from mjorbit.cpu.mjcf.builders import SPACECRAFT_ARM_XML
from mjorbit.cpu.orbit.elements import keplerian_to_cartesian
from mjorbit.viewer import MjOrbitViewer


def main() -> None:
    # ------------------------------------------------------------------
    # Orbit & scenario — identical to cpu_arm_reach.py
    # ------------------------------------------------------------------
    alt_km = 400.0
    a_km = R_EARTH + alt_km

    R_eci, V_eci = keplerian_to_cartesian(
        a=a_km, e=0.0, inc=np.deg2rad(51.6), raan=0.0, argp=0.0, nu=0.0,
    )

    cfg = CPUScenarioCfg(
        orbit=OrbitCfg(R_eci=R_eci, V_eci=V_eci),
        mujoco=MuJoCoCfg(xml_path=SPACECRAFT_ARM_XML, dt=0.002),
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    scenario = compile_cpu(cfg)

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------
    viewer = MjOrbitViewer(
        scenario,
        port=8080,
        show_earth=True,
        show_axes=True,
        track_body="ee",
    )

    # ------------------------------------------------------------------
    # Action callback: free drift then arm slew
    # ------------------------------------------------------------------
    target_ctrl = np.array([1.0, -0.8])

    def action_fn(scenario, t):
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
