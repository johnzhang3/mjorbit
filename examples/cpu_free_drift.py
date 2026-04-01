"""Phase 9 example: free-body CW drift in circular LEO.

Chief in a 400 km circular orbit at 51.6 deg inclination.
One free body offset 10 m radially with no drag, SRP, or magnetic effects.
Compares simulated trajectory to the CW analytical solution.

Usage:
    uv run python examples/cpu_free_drift.py
"""

from __future__ import annotations

import numpy as np

from mjorbit.constants import R_EARTH, GM_EARTH
from mjorbit.cpu import compile_cpu, step_cpu
from mjorbit.cpu.core.config import CPUScenarioCfg, OrbitCfg, MuJoCoCfg
from mjorbit.cpu.orbit.elements import keplerian_to_cartesian
from mjorbit.cpu.mjcf.builders import FREE_BODY_XML


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

    R_eci, V_eci = keplerian_to_cartesian(
        a=a_km, e=0.0, inc=np.deg2rad(51.6), raan=0.0, argp=0.0, nu=0.0,
    )

    cfg = CPUScenarioCfg(
        orbit=OrbitCfg(R_eci=R_eci, V_eci=V_eci),
        mujoco=MuJoCoCfg(xml_path=FREE_BODY_XML, dt=0.01),
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    scenario = compile_cpu(cfg)

    # Initial offset: 10 m radial, 0.05 m/s along-track velocity
    x0, y0, z0 = 10.0, 0.0, 5.0  # m
    vx0, vy0, vz0 = 0.0, 0.05, 0.0  # m/s

    scenario.mjd.qpos[0] = x0
    scenario.mjd.qpos[1] = y0
    scenario.mjd.qpos[2] = z0
    scenario.mjd.qvel[0] = vx0
    scenario.mjd.qvel[1] = vy0
    scenario.mjd.qvel[2] = vz0

    import mujoco
    mujoco.mj_forward(scenario.mjm, scenario.mjd)

    dt = 0.01
    t_total = 60.0  # 1 minute
    n_steps = int(t_total / dt)

    # Record trajectory
    times = np.zeros(n_steps + 1)
    pos_sim = np.zeros((n_steps + 1, 3))
    pos_cw = np.zeros((n_steps + 1, 3))

    pos_sim[0] = scenario.mjd.qpos[:3]
    pos_cw[0] = [x0, y0, z0]

    for i in range(n_steps):
        step_cpu(scenario)
        t = (i + 1) * dt
        times[i + 1] = t
        pos_sim[i + 1] = scenario.mjd.qpos[:3]
        p_cw, _ = cw_analytical(x0, y0, z0, vx0, vy0, vz0, n, t)
        pos_cw[i + 1] = p_cw

    # Final comparison
    pos_final = scenario.mjd.qpos[:3]
    pos_cw_final, vel_cw_final = cw_analytical(x0, y0, z0, vx0, vy0, vz0, n, t_total)

    err = np.linalg.norm(pos_final - pos_cw_final)
    err_rel = err / np.linalg.norm(pos_cw_final)

    print("=" * 60)
    print("CPU Free Drift — CW Parity Check")
    print("=" * 60)
    print(f"Orbit:     {alt_km:.0f} km circular LEO, inc={51.6} deg")
    print(f"Duration:  {t_total:.0f} s ({n_steps} steps at dt={dt})")
    print(f"Mean motion: n = {n:.6e} rad/s")
    print()
    print(f"Initial: x={x0} m, y={y0} m, z={z0} m")
    print(f"         vx={vx0} m/s, vy={vy0} m/s, vz={vz0} m/s")
    print()
    print(f"Final (sim):  x={pos_final[0]:+.6f} m, y={pos_final[1]:+.6f} m, z={pos_final[2]:+.6f} m")
    print(f"Final (CW):   x={pos_cw_final[0]:+.6f} m, y={pos_cw_final[1]:+.6f} m, z={pos_cw_final[2]:+.6f} m")
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
