"""Visualise two rigid bodies colliding in orbit.

Two 100 kg cubes in a 400 km circular LEO.  Body A starts with a radial
velocity toward Body B.  After collision they bounce apart and follow
diverging Clohessy-Wiltshire trajectories — each body ends up on a
different orbit, visible as the trails separate over time.

Gold trail  = Body A (initially moving)
Cyan trail  = Body B (initially at rest)

Usage:
    pixi install
    pixi run python examples/collision_viewer.py

Controls (browser):
    - Scroll to zoom, drag to orbit the camera
    - This example renders in ECI, so the chief-relative pair visibly orbits Earth
    - Use the local-scale controls to grow/shrink the bodies relative to Earth
    - Use the speed slider to fast-forward through CW drift
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from _orbit_reference import circular_orbit_eci

from mjorbit import MjoData, MjoModel, OrbitInit
from mjorbit.constants import R_EARTH
from mjorbit.testdata import TWO_BODIES_XML
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
    # Orbit — 400 km circular LEO, 51.6° inclination (ISS-like)
    # ------------------------------------------------------------------
    alt_km = 400.0
    a_km = R_EARTH + alt_km

    R_eci, V_eci = circular_orbit_eci(a_km, np.deg2rad(51.6))

    # ------------------------------------------------------------------
    # Model — two 100 kg cubes, 4 m apart along radial axis
    # ------------------------------------------------------------------
    model = _compile_model(TWO_BODIES_XML, mj_timestep=0.01)
    data = MjoData(model, orbit=OrbitInit(R_eci=R_eci, V_eci=V_eci))

    # ------------------------------------------------------------------
    # Initial conditions — Body A approaches Body B
    # Body A: qvel[0:6], Body B: qvel[6:12].
    # Add +x chief-inertial velocity to Body A; initially +x is radial.
    # ------------------------------------------------------------------
    data.qvel[0] += 1.0

    from mjorbit import mjo_forward
    mjo_forward(model, data)

    # ------------------------------------------------------------------
    # Viewer — render in ECI so the pair visibly orbits Earth.
    # Start with a large local scale and a correspondingly larger camera distance.
    # ------------------------------------------------------------------
    viewer = MjOrbitViewer(
        model,
        data,
        port=8080,
        show_earth=True,
        show_axes=True,
        track_bodies=["body_a", "body_b"],
        camera_distance=50000.0,
        render_frame="eci",
    )
    viewer.set_local_scene_scale(10000.0)

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
    print("  Render : ECI (chief-relative state translated along the orbit)")
    print("  Scale  : viewer starts at 10000x local scale")
    print()

    viewer.run(duration=600.0)


if __name__ == "__main__":
    main()
