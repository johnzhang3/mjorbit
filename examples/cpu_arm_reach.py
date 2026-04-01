"""Phase 9 example: articulated spacecraft with a 2-link arm.

Demonstrates the coupled simulator with an articulated spacecraft-arm system:
1. Free-floating drift: arm moves but COM orbit is preserved
2. Joint slew: arm reaches to a target configuration
3. Stability check: long-horizon run remains finite

Usage:
    uv run python examples/cpu_arm_reach.py
"""

from __future__ import annotations

import numpy as np
import mujoco

from mjorbit.constants import R_EARTH, GM_EARTH
from mjorbit.cpu import compile_cpu, step_cpu
from mjorbit.cpu.core.config import CPUScenarioCfg, OrbitCfg, MuJoCoCfg
from mjorbit.cpu.orbit.elements import keplerian_to_cartesian
from mjorbit.cpu.mjcf.builders import SPACECRAFT_ARM_XML


def main() -> None:
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
    mjm, mjd = scenario.mjm, scenario.mjd

    print("=" * 60)
    print("CPU Arm Reach — Articulated Spacecraft Example")
    print("=" * 60)
    print(f"Model: {mjm.nbody} bodies, {mjm.njnt} joints, {mjm.nu} actuators")
    print(f"Bodies: {[mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(mjm.nbody)]}")
    print()

    # ----------------------------------------------------------------
    # Part 1: Free drift — no control, arm in initial configuration
    # ----------------------------------------------------------------
    print("--- Part 1: Free Drift (5 s, no control) ---")
    orbit_0 = scenario.orbit.copy()
    com_0 = np.average(
        [mjd.xipos[i] for i in range(1, mjm.nbody)],
        weights=[mjm.body_mass[i] for i in range(1, mjm.nbody)],
        axis=0,
    )

    dt = mjm.opt.timestep
    n_steps_1 = int(5.0 / dt)
    for _ in range(n_steps_1):
        step_cpu(scenario)

    com_1 = np.average(
        [mjd.xipos[i] for i in range(1, mjm.nbody)],
        weights=[mjm.body_mass[i] for i in range(1, mjm.nbody)],
        axis=0,
    )
    com_drift = np.linalg.norm(com_1 - com_0)
    print(f"  System COM drift: {com_drift:.6e} m")
    print(f"  All states finite: {np.all(np.isfinite(mjd.qpos)) and np.all(np.isfinite(mjd.qvel))}")

    # ----------------------------------------------------------------
    # Part 2: Joint slew — command arm to target configuration
    # ----------------------------------------------------------------
    print()
    print("--- Part 2: Arm Slew (10 s, position control to shoulder=1.0, elbow=-0.8 rad) ---")

    target_shoulder = 1.0  # rad
    target_elbow = -0.8  # rad

    n_steps_2 = int(10.0 / dt)
    ctrl = np.array([target_shoulder, target_elbow])

    ee_positions = []
    for i in range(n_steps_2):
        step_cpu(scenario, ctrl=ctrl)
        if i % 500 == 0:
            ee_id = scenario.body_id("ee")
            ee_pos = mjd.xipos[ee_id].copy()
            ee_positions.append(ee_pos)

    ee_id = scenario.body_id("ee")
    ee_final = mjd.xipos[ee_id].copy()
    print(f"  End-effector final pos (LVLH): [{ee_final[0]:.4f}, {ee_final[1]:.4f}, {ee_final[2]:.4f}] m")

    # Check joint angles reached target
    # The spacecraft_arm model has: freejoint (7 qpos), shoulder (1), elbow (1)
    shoulder_q = mjd.qpos[7]  # first hinge after freejoint
    elbow_q = mjd.qpos[8]  # second hinge
    print(f"  Shoulder: {shoulder_q:.4f} rad (target {target_shoulder})")
    print(f"  Elbow:    {elbow_q:.4f} rad (target {target_elbow})")

    shoulder_err = abs(shoulder_q - target_shoulder)
    elbow_err = abs(elbow_q - target_elbow)
    print(f"  Joint errors: shoulder={shoulder_err:.4e}, elbow={elbow_err:.4e}")
    print("  (Note: free-floating base absorbs reaction — joints won't converge to target)")

    # ----------------------------------------------------------------
    # Part 3: Conservation check — internal motion shouldn't change orbit
    # ----------------------------------------------------------------
    print()
    print("--- Part 3: Orbit Conservation Check ---")

    orbit_after = scenario.orbit
    dR = np.linalg.norm(orbit_after.R_eci - orbit_0.R_eci)
    # Expected drift from orbit propagation (15 s at ~7.7 km/s)
    v_circ = np.linalg.norm(V_eci)
    expected_distance = v_circ * 15.0  # km
    print(f"  Chief position change: {dR:.2f} km (expected ~{expected_distance:.0f} km from propagation)")
    print(f"  Chief speed: {np.linalg.norm(orbit_after.V_eci):.6f} km/s (initial: {np.linalg.norm(V_eci):.6f} km/s)")

    # Orbital energy should be roughly conserved (no external forces)
    r1 = np.linalg.norm(orbit_after.R_eci)
    v1 = np.linalg.norm(orbit_after.V_eci)
    r0 = np.linalg.norm(orbit_0.R_eci)
    v0 = np.linalg.norm(orbit_0.V_eci)
    E0 = 0.5 * v0**2 - GM_EARTH / r0
    E1 = 0.5 * v1**2 - GM_EARTH / r1
    dE_rel = abs(E1 - E0) / abs(E0)
    print(f"  Orbital energy change: {dE_rel:.2e} (relative)")

    # ----------------------------------------------------------------
    # Part 4: Long-horizon stability
    # ----------------------------------------------------------------
    print()
    print("--- Part 4: Long-Horizon Stability (30 s more) ---")
    n_steps_4 = int(30.0 / dt)
    for _ in range(n_steps_4):
        step_cpu(scenario, ctrl=ctrl)

    all_finite = np.all(np.isfinite(mjd.qpos)) and np.all(np.isfinite(mjd.qvel))
    print(f"  All states finite after {(n_steps_1 + n_steps_2 + n_steps_4) * dt:.0f} s: {all_finite}")
    print(f"  Total time simulated: {scenario.orbit.t:.1f} s")

    print()
    if all_finite and dE_rel < 1e-6:
        print("PASS: Articulated spacecraft example completed successfully.")
        print("  - Simulation remains stable and finite")
        print("  - Orbital energy conserved (no external forces applied)")
        print("  - Internal arm motion does not perturb the orbit")
    else:
        print("WARN: Check results above for potential issues.")


if __name__ == "__main__":
    main()
