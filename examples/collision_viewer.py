"""Visualise two rigid bodies colliding in orbit.

Two 100 kg cubes in a 400 km circular LEO.  Body A starts with a radial
velocity toward Body B.  After collision they bounce apart and follow
diverging Clohessy-Wiltshire trajectories — each body ends up on a
different orbit, visible as the trails separate over time.

Gold trail  = Body A (initially moving)
Cyan trail  = Body B (initially at rest)

Usage:
    uv sync --extra viewer
    uv run python examples/collision_viewer.py

Controls (browser):
    - Scroll to zoom, drag to orbit the camera
    - Zoom out along -x (toward red axis) to see Earth below
    - Use the speed slider to fast-forward through CW drift
"""

from __future__ import annotations

import numpy as np

from mujoco_orbit import MjoData, MjoModel, OrbitInit
from mujoco_orbit.constants import R_EARTH
from mujoco_orbit.orbit.elements import keplerian_to_cartesian
from mujoco_orbit.testdata import TWO_BODIES_XML
from viewer import MjOrbitViewer


def main() -> None:
    # ------------------------------------------------------------------
    # Orbit — 400 km circular LEO, 51.6° inclination (ISS-like)
    # ------------------------------------------------------------------
    alt_km = 400.0
    a_km = R_EARTH + alt_km

    R_eci, V_eci = keplerian_to_cartesian(
        a=a_km, e=0.0, inc=np.deg2rad(51.6), raan=0.0, argp=0.0, nu=0.0,
    )

    # ------------------------------------------------------------------
    # Model — two 100 kg cubes, 4 m apart along radial axis
    # ------------------------------------------------------------------
    model = MjoModel.from_xml_path(
        TWO_BODIES_XML,
        mj_timestep=0.002,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    data = MjoData(model, orbit=OrbitInit(R_eci=R_eci, V_eci=V_eci))

    # ------------------------------------------------------------------
    # Initial conditions — Body A approaches Body B
    # Body A: qvel[0:6], Body B: qvel[6:12]
    # Give Body A a +x (radial) velocity of 1 m/s toward Body B
    # ------------------------------------------------------------------
    data.qvel[0] = 1.0  # body_a vx = +1 m/s (radial, toward body_b)

    from mujoco_orbit import mjo_forward
    mjo_forward(model, data)

    # ------------------------------------------------------------------
    # Viewer — track both bodies, camera at 15 m to see collision + drift
    # ------------------------------------------------------------------
    viewer = MjOrbitViewer(
        model,
        data,
        port=8080,
        show_earth=True,
        show_axes=True,
        track_bodies=["body_a", "body_b"],
        camera_distance=15.0,
    )

    print("=" * 55)
    print("  Two-Body Orbital Collision")
    print("=" * 55)
    print("  Bodies : 2 × 100 kg cubes, 4 m apart (radial)")
    print("  Orbit  : 400 km circular LEO")
    print("  IC     : Body A → +1 m/s radial toward Body B")
    print()
    print("  After collision the bodies bounce and follow")
    print("  diverging CW trajectories.  Speed up with the")
    print("  slider to watch the along-track drift grow.")
    print()
    print("  Zoom out along -x to see Earth below.")
    print()

    viewer.run(duration=600.0)


if __name__ == "__main__":
    main()
