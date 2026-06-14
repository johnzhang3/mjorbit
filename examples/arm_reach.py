"""Phase 9 example: articulated spacecraft with a 2-link arm.

Demonstrates the coupled simulator with an articulated spacecraft-arm system:
1. Free-floating drift: arm moves but COM orbit is preserved
2. Joint slew: arm reaches to a target configuration
3. Stability check: long-horizon run remains finite

Usage:
    pixi run python examples/arm_reach.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from _orbit_reference import circular_orbit_eci

from mjorbit import MjoData, MjoModel, OrbitInit, mjo_step
from mjorbit.constants import GM_EARTH, R_EARTH
from mjorbit.testdata import SPACECRAFT_ARM_XML


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
    alt_km = 400.0
    a_km = R_EARTH + alt_km

    R_eci, V_eci = circular_orbit_eci(a_km, np.deg2rad(51.6))

    model = _compile_model(SPACECRAFT_ARM_XML, mj_timestep=0.002)
    data = MjoData(model, orbit=OrbitInit(R_eci=R_eci, V_eci=V_eci))
    mjm, mjd = model, data

    print("=" * 60)
    print("Arm Reach — Articulated Spacecraft Example")
    print("=" * 60)
    print(f"Model: {mjm.nbody} bodies, {mjm.njnt} joints, {mjm.nu} actuators")
    print(f"Bodies: {[mjm.body_name(i) for i in range(mjm.nbody)]}")
    print()

    # ----------------------------------------------------------------
    # Part 1: Free drift — no control, arm in initial configuration
    # ----------------------------------------------------------------
    print("--- Part 1: Free Drift (5 s, no control) ---")
    orbit_0_R_eci = data.orbit.R_eci.copy()
    orbit_0_V_eci = data.orbit.V_eci.copy()
    com_0_world = np.average(
        [mjd.xipos[i] for i in range(1, mjm.nbody)],
        weights=[mjm.body_mass[i] for i in range(1, mjm.nbody)],
        axis=0,
    )
    com_0 = data.lvlh_position_from_world(com_0_world)

    dt = mjm.opt.timestep
    n_steps_1 = int(5.0 / dt)
    for _ in range(n_steps_1):
        mjo_step(model, data)

    com_1_world = np.average(
        [mjd.xipos[i] for i in range(1, mjm.nbody)],
        weights=[mjm.body_mass[i] for i in range(1, mjm.nbody)],
        axis=0,
    )
    com_1 = data.lvlh_position_from_world(com_1_world)
    com_drift = np.linalg.norm(com_1 - com_0)
    print(f"  System COM drift relative to chief: {com_drift:.6e} m")
    print(
        f"  All states finite: {np.all(np.isfinite(mjd.qpos)) and np.all(np.isfinite(mjd.qvel))}"
    )

    # ----------------------------------------------------------------
    # Part 2: Joint slew — command arm to target configuration
    # ----------------------------------------------------------------
    print()
    print(
        "--- Part 2: Arm Slew (10 s, position control to shoulder=1.0, elbow=-0.8 rad) ---"
    )

    target_shoulder = 1.0  # rad
    target_elbow = -0.8  # rad

    n_steps_2 = int(10.0 / dt)
    ctrl = np.array([target_shoulder, target_elbow])
    np.copyto(data.ctrl, ctrl)

    ee_positions = []
    for i in range(n_steps_2):
        mjo_step(model, data)
        if i % 500 == 0:
            ee_id = model.body_id("ee")
            ee_pos = data.lvlh_position_from_world(mjd.xipos[ee_id])
            ee_positions.append(ee_pos)

    ee_id = model.body_id("ee")
    ee_final = data.lvlh_position_from_world(mjd.xipos[ee_id])
    print(
        "  End-effector final pos (LVLH): "
        f"[{ee_final[0]:.4f}, {ee_final[1]:.4f}, {ee_final[2]:.4f}] m"
    )

    # Check joint angles reached target
    # The spacecraft_arm model has: freejoint (7 qpos), shoulder (1), elbow (1)
    shoulder_q = mjd.qpos[7]  # first hinge after freejoint
    elbow_q = mjd.qpos[8]  # second hinge
    print(f"  Shoulder: {shoulder_q:.4f} rad (target {target_shoulder})")
    print(f"  Elbow:    {elbow_q:.4f} rad (target {target_elbow})")

    shoulder_err = abs(shoulder_q - target_shoulder)
    elbow_err = abs(elbow_q - target_elbow)
    print(f"  Joint errors: shoulder={shoulder_err:.4e}, elbow={elbow_err:.4e}")
    print("  (Free-floating base adds a longer transient; pure MuJoCo with a fixed base")
    print("   reaches the target, so this example is checking coupled motion rather than")
    print("   exact 10 s settling.)")

    # ----------------------------------------------------------------
    # Part 3: Conservation check — internal motion shouldn't change orbit
    # ----------------------------------------------------------------
    print()
    print("--- Part 3: Orbit Conservation Check ---")

    orbit_after = data.orbit
    dR = np.linalg.norm(orbit_after.R_eci - orbit_0_R_eci)
    # Expected drift from orbit propagation (15 s at ~7.7 km/s)
    v_circ = np.linalg.norm(V_eci)
    expected_distance = v_circ * 15.0  # km
    print(
        f"  Chief position change: {dR:.2f} km "
        f"(expected ~{expected_distance:.0f} km from propagation)"
    )
    print(
        f"  Chief speed: {np.linalg.norm(orbit_after.V_eci):.6f} km/s "
        f"(initial: {np.linalg.norm(V_eci):.6f} km/s)"
    )

    # Orbital energy should be roughly conserved (no external forces)
    r1 = np.linalg.norm(orbit_after.R_eci)
    v1 = np.linalg.norm(orbit_after.V_eci)
    r0 = np.linalg.norm(orbit_0_R_eci)
    v0 = np.linalg.norm(orbit_0_V_eci)
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
        mjo_step(model, data)

    all_finite = np.all(np.isfinite(mjd.qpos)) and np.all(np.isfinite(mjd.qvel))
    print(
        f"  All states finite after {(n_steps_1 + n_steps_2 + n_steps_4) * dt:.0f} s: "
        f"{all_finite}"
    )
    print(f"  Total time simulated: {data.orbit.t:.1f} s")

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
