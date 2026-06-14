"""Phase 9 example: free-body drift in circular LEO.

Chief in a 400 km circular orbit at 51.6 deg inclination.
One free body offset 10 m radially with no drag, SRP, or magnetic effects.
Compares the chief-inertial MuJoCo trajectory to a CW analytical reference.

Usage:
    pixi run python examples/free_drift.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from _orbit_reference import circular_orbit_eci

from mjorbit import MjoData, MjoModel, OrbitInit, mjo_forward, mjo_step
from mjorbit.constants import GM_EARTH, R_EARTH
from mjorbit.testdata import FREE_BODY_XML


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


def cw_analytical(
    x0: float, y0: float, z0: float,
    vx0: float, vy0: float, vz0: float,
    n: float, t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Clohessy-Wiltshire analytical solution (m, m/s, rad/s, s)."""
    nt = n * t
    cn, sn = np.cos(nt), np.sin(nt)

    x = (4.0 - 3.0 * cn) * x0 + sn / n * vx0 + 2.0 / n * (1.0 - cn) * vy0
    y = (6.0 * (sn - nt) * x0 + y0
         - 2.0 / n * (1.0 - cn) * vx0
         + (4.0 * sn - 3.0 * nt) / n * vy0)
    z = z0 * cn + vz0 / n * sn

    vx = 3.0 * n * sn * x0 + cn * vx0 + 2.0 * sn * vy0
    vy = 6.0 * n * (cn - 1.0) * x0 - 2.0 * sn * vx0 + (4.0 * cn - 3.0) * vy0
    vz = -z0 * n * sn + vz0 * cn

    return np.array([x, y, z]), np.array([vx, vy, vz])


def main() -> None:
    alt_km = 400.0
    a_km = R_EARTH + alt_km
    n = np.sqrt(GM_EARTH / a_km**3)  # mean motion, rad/s

    R_eci, V_eci = circular_orbit_eci(a_km, np.deg2rad(51.6))

    model = _compile_model(FREE_BODY_XML, mj_timestep=0.01)
    data = MjoData(model, orbit=OrbitInit(R_eci=R_eci, V_eci=V_eci))

    # Initial offset: 10 m radial, 0.05 m/s along-track velocity
    x0, y0, z0 = 10.0, 0.0, 5.0  # m
    vx0, vy0, vz0 = 0.0, 0.05, 0.0  # m/s

    position_lvlh = np.array([x0, y0, z0])
    velocity_lvlh = np.array([vx0, vy0, vz0])
    data.qpos[:3] = data.world_position_from_lvlh(position_lvlh)
    data.qvel[:3] = data.world_velocity_from_lvlh(position_lvlh, velocity_lvlh)
    mjo_forward(model, data)

    dt = 0.01
    t_total = 60.0  # 1 minute
    n_steps = int(t_total / dt)

    # Record trajectory
    times = np.zeros(n_steps + 1)
    pos_sim = np.zeros((n_steps + 1, 3))
    pos_cw = np.zeros((n_steps + 1, 3))

    pos_sim[0] = data.lvlh_position_from_world(data.qpos[:3])
    pos_cw[0] = [x0, y0, z0]

    for i in range(n_steps):
        mjo_step(model, data)
        t = (i + 1) * dt
        times[i + 1] = t
        pos_sim[i + 1] = data.lvlh_position_from_world(data.qpos[:3])
        p_cw, _ = cw_analytical(x0, y0, z0, vx0, vy0, vz0, n, t)
        pos_cw[i + 1] = p_cw

    # Final comparison
    pos_final = data.lvlh_position_from_world(data.qpos[:3])
    pos_cw_final, vel_cw_final = cw_analytical(x0, y0, z0, vx0, vy0, vz0, n, t_total)

    err = np.linalg.norm(pos_final - pos_cw_final)
    err_rel = err / np.linalg.norm(pos_cw_final)

    print("=" * 60)
    print("Free Drift — CW Reference Check")
    print("=" * 60)
    print(f"Orbit:     {alt_km:.0f} km circular LEO, inc={51.6} deg")
    print(f"Duration:  {t_total:.0f} s ({n_steps} steps at dt={dt})")
    print(f"Mean motion: n = {n:.6e} rad/s")
    print()
    print(f"Initial: x={x0} m, y={y0} m, z={z0} m")
    print(f"         vx={vx0} m/s, vy={vy0} m/s, vz={vz0} m/s")
    print()
    print(
        f"Final (sim):  x={pos_final[0]:+.6f} m, y={pos_final[1]:+.6f} m, z={pos_final[2]:+.6f} m"
    )
    final_cw = (
        f"Final (CW):   x={pos_cw_final[0]:+.6f} m, y={pos_cw_final[1]:+.6f} m, "
        f"z={pos_cw_final[2]:+.6f} m"
    )
    print(final_cw)
    print()
    print(f"Position error: {err:.6e} m  (relative: {err_rel:.2e})")

    # Per-axis max error over trajectory
    max_err = np.max(np.abs(pos_sim - pos_cw), axis=0)
    print(f"Max error over run:  x={max_err[0]:.4e} m, y={max_err[1]:.4e} m, z={max_err[2]:.4e} m")

    if err_rel < 0.01:
        print("\nPASS: Simulation matches CW within 1% relative error.")
    else:
        print(f"\nWARN: Relative error {err_rel:.2e} exceeds 1% threshold.")


if __name__ == "__main__":
    main()
