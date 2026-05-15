"""Multi-spacecraft browser viewer for the dual-arm + solar-panels bus.

Spins up N independent simulations — each spacecraft has its own MjoModel
and MjoData (so its own orbit, qpos, qvel, controls, and integrator state) —
and renders them together in absolute ECI so they appear at distinct points
around Earth. Intended for paper figures with several different spacecraft
visible at once.

Each spacecraft drives both arms with a phase-offset sinusoid so the figure
shows a variety of arm poses across the constellation. The viewer runs
indefinitely; stop with Ctrl+C in the terminal.

Usage:
    pixi install
    pixi run python examples/arm_reach_viewer.py
    # or with a custom count:
    pixi run python examples/arm_reach_viewer.py --n 8
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from mujoco_orbit import MjoData, MjoModel, OrbitInit, mjo_forward
from mujoco_orbit.constants import GM_EARTH, R_EARTH
from mujoco_orbit.testdata import SPACECRAFT_DUAL_ARM_PANELS_XML
from viewer import MjOrbitMultiViewer


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


def _orbit_eci(
    radius_km: float,
    inclination_rad: float,
    raan_rad: float,
    true_anomaly_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """ECI position/velocity for a circular orbit at the given orientation.

    Builds the position from (radius, true anomaly), then rotates by
    inclination about X and RAAN about Z. Velocity is tangent to the orbit
    plane with magnitude sqrt(GM / r).
    """
    r = radius_km
    speed = np.sqrt(GM_EARTH / r)
    nu = true_anomaly_rad

    r_peri = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    v_peri = np.array([-speed * np.sin(nu), speed * np.cos(nu), 0.0])

    ci, si = np.cos(inclination_rad), np.sin(inclination_rad)
    Rx = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])
    co, so = np.cos(raan_rad), np.sin(raan_rad)
    Rz = np.array([[co, -so, 0], [so, co, 0], [0, 0, 1]])
    R = Rz @ Rx

    return R @ r_peri, R @ v_peri


def _build_entry(
    model: MjoModel,
    idx: int,
    n_total: int,
    radius_km: float,
) -> tuple[MjoModel, MjoData]:
    """Construct an independent (model, data) entry for spacecraft *idx*."""
    # Spread spacecraft uniformly in true anomaly, with mild variations
    # in inclination and RAAN so they don't all sit on a single ring.
    nu = 2.0 * np.pi * idx / n_total
    inc = np.deg2rad(45.0 + 15.0 * np.sin(2.0 * np.pi * idx / n_total))
    raan = 2.0 * np.pi * (idx / n_total) * 0.5

    R_eci, V_eci = _orbit_eci(radius_km, inc, raan, nu)
    data = model.make_data(orbit=OrbitInit(R_eci=R_eci, V_eci=V_eci))

    # Stagger initial arm targets so the figure shows variety, not a
    # chorus line. Position actuators will pull the arms to these angles
    # within a fraction of a second from the rest qpos.
    phase = 2.0 * np.pi * idx / n_total
    sa = 0.6 * np.sin(phase)
    ea = -0.8 * np.cos(phase)
    sb = 0.6 * np.sin(phase + np.pi / 3.0)
    eb = -0.8 * np.cos(phase + np.pi / 3.0)
    data.ctrl[:] = np.array([sa, ea, sb, eb])
    mjo_forward(model, data)
    return model, data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n", type=int, default=6, help="number of spacecraft to simulate"
    )
    parser.add_argument(
        "--alt-km", type=float, default=600.0, help="circular orbit altitude (km)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="viser server port"
    )
    parser.add_argument(
        "--spacecraft-scale",
        type=float,
        default=50000.0,
        help="visual scale applied to each spacecraft (Earth is huge)",
    )
    args = parser.parse_args()

    radius_km = R_EARTH + args.alt_km

    # One compiled model is enough — every entry instantiates its own MjoData,
    # which carries the per-spacecraft runtime state (orbit, qpos, qvel, ctrl).
    model = _compile_model(SPACECRAFT_DUAL_ARM_PANELS_XML, mj_timestep=0.005)
    entries = [_build_entry(model, i, args.n, radius_km) for i in range(args.n)]

    viewer = MjOrbitMultiViewer(
        entries,
        port=args.port,
        show_earth=True,
        spacecraft_scale=args.spacecraft_scale,
    )

    omega = 2.0 * np.pi / 30.0  # arm slew period: 30 s
    base_ctrls = np.stack([np.copy(d.ctrl) for _, d in entries])

    def action_fn(i, model, data, t):
        del model, data  # ctrl is computed from idx + time alone
        phase = 2.0 * np.pi * i / args.n
        delta = 0.5 * np.sin(omega * t + phase)
        ctrl = base_ctrls[i].copy()
        ctrl[0] = ctrl[0] + delta          # shoulder_a
        ctrl[1] = ctrl[1] - 0.5 * delta    # elbow_a
        ctrl[2] = ctrl[2] - delta          # shoulder_b
        ctrl[3] = ctrl[3] + 0.5 * delta    # elbow_b
        return ctrl

    print("=" * 60)
    print(f"Multi-spacecraft viewer  —  N = {args.n}, alt = {args.alt_km:g} km")
    print("Arms slew with a 30 s sinusoid, panels are deployed.")
    print("Browser: http://localhost:{}  (Ctrl+C to stop)".format(args.port))
    print("=" * 60)

    viewer.run(duration=None, action_fn=action_fn)


if __name__ == "__main__":
    main()
