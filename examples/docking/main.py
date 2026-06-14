"""Forward-simulate the ISS in orbit with the browser viewer.

This is a minimal sanity-check script for loading the ISS model in mjorbit,
placing it on an ISS-like circular orbit, and stepping the simulation in the
ECI viewer.

From blender file:
docking port position on ISS (w.r.t. ISS body frame):
(x, y, z) = (-35.278, 0.0, -4.1379) m

docking port position on Soyuz (w.r.t. Soyuz body frame):
(x, y, z) = (1.7743, 0.033833, 0.23575) m
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from mjorbit import MjoModel, OrbitInit, mjo_forward
from mjorbit.constants import GM_EARTH, R_EARTH
from viewer import MjOrbitViewer


def circular_orbit_eci(radius_km: float, inclination_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ECI position/velocity for a circular orbit at true anomaly zero."""
    speed_km_s = np.sqrt(GM_EARTH / radius_km)
    return (
        np.array([radius_km, 0.0, 0.0]),
        np.array(
            [
                0.0,
                speed_km_s * np.cos(inclination_rad),
                speed_km_s * np.sin(inclination_rad),
            ]
        ),
    )


def _compile_model(xml_path: str, *, mj_timestep: float) -> MjoModel:
    mjorbit = (
        '<mjorbit use_j2="true" use_drag="true" use_srp="true" '
        'use_magnetic="true">\n  </mjorbit>\n'
    )
    xml = Path(xml_path).read_text().replace("</mujoco>", f"  {mjorbit}</mujoco>")
    with tempfile.NamedTemporaryFile(
        suffix=".xml",
        mode="w",
        delete=False,
        dir=Path(xml_path).parent,
    ) as file:
        file.write(xml)
        configured_path = Path(file.name)
    try:
        return MjoModel.from_xml_path(str(configured_path), mj_timestep=mj_timestep)
    finally:
        configured_path.unlink(missing_ok=True)


def main() -> None:
    alt_km = 420.0
    inc_deg = 51.64
    orbit_radius_km = R_EARTH + alt_km
    R_eci, V_eci = circular_orbit_eci(orbit_radius_km, np.deg2rad(inc_deg))

    model = _compile_model(str(Path(__file__).with_name("iss.xml")), mj_timestep=0.01)
    data = model.make_data(orbit=OrbitInit(R_eci=R_eci, V_eci=V_eci))
    mjo_forward(model, data)

    viewer = MjOrbitViewer(
        model,
        data,
        port=8080,
        show_earth=True,
        show_axes=True,
        track_body="iss_main",
        camera_distance=500000.0,
        render_frame="eci",
    )
    viewer.set_local_scene_scale(10000.0)

    print("=" * 60)
    print("ISS forward simulation")
    print("=" * 60)
    print(f"Orbit:  {alt_km:.0f} km circular LEO, inc={inc_deg:.2f} deg")
    print("Model:  examples/docking/iss.xml")
    print("View:   ECI, ISS body tracked")
    print("Scale:  10000x local scene")
    print("Browser: http://localhost:8080  (Ctrl+C to stop)")
    print("=" * 60)

    viewer.run(duration=None)


if __name__ == "__main__":
    main()
