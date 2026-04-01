"""Assemble per-body wrenches and write to MjData.xfrc_applied (Phase 8).

Pipeline:
1. Inertial/gravity forcing for each body
2. Surface drag/SRP loads
3. Magnetic torques (residual dipoles)
4. External actuator wrenches (RW, MTQ, thrusters — called separately)
5. Write results into xfrc_applied
"""

from __future__ import annotations

import numpy as np

from mjorbit.cpu.core.scenario import CPUScenario
from mjorbit.cpu.coupling.inertial import apply_inertial_wrenches
from mjorbit.cpu.coupling.surfaces import apply_surface_wrenches
from mjorbit.cpu.coupling.magnetic import apply_magnetic_wrenches


def assemble_and_apply_wrenches(scenario: CPUScenario) -> None:
    """Compute and write all external body wrenches into xfrc_applied.

    Note: Actuator wrenches (RW, MTQ, thrusters) are applied by
    coupling.actuators and called separately from step_cpu, because
    reaction wheel integration needs dt and happens in a specific order.
    """
    # 1. Inertial / gravity forcing
    apply_inertial_wrenches(scenario)

    # 2. Surface loads (drag + SRP)
    apply_surface_wrenches(scenario)

    # 3. Magnetic residual dipole torques
    apply_magnetic_wrenches(scenario)

    # Copy assembled buffer to MuJoCo
    # xfrc_applied shape is (nbody, 6): [fx, fy, fz, tx, ty, tz] in world frame, SI
    np.copyto(scenario.mjd.xfrc_applied, scenario._wrench_buffer)
