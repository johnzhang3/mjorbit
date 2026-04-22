"""Per-body gravity forcing in the ECI MuJoCo world frame.

The MuJoCo model lives directly in an Earth-centered inertial frame with SI
coordinates. Each body's world position is therefore its ECI position in
meters, and the orbit layer's km-based gravity model is converted to SI before
being written to ``xfrc_applied``.

Units convention:
  MuJoCo uses SI (m, s, kg, N).
  Orbit layer uses km, km/s, km/s^2.
  Forces written to xfrc_applied must be in N.

  Conversion: 1 km/s^2 = 1e3 m/s^2
  Force (N) = mass (kg) * accel (m/s^2)
  Orbit accel (km/s^2) * 1e3 -> m/s^2
"""

from __future__ import annotations

from mujoco_orbit.core.runtime import MjoData, MjoModel
from mujoco_orbit.orbit.gravity import total_accel

# MuJoCo world frame = ECI frame. Body COM positions in MuJoCo are meters (SI);
# the orbit layer uses km, so positions are converted before evaluating gravity.

_M_TO_KM = 1e-3
_KM_S2_TO_M_S2 = 1e3  # km/s^2 -> m/s^2


def apply_inertial_wrenches(model: MjoModel, data: MjoData) -> None:
    """Compute per-body ECI gravity accelerations and accumulate forces.

    Writes force contributions (N) into ``data.wrench_buffer[:, :3]``.
    Torque contributions are zero for translational forcing.
    """
    mjm = model.mj_model
    mjd = data.mj_data

    for body_id in range(1, mjm.nbody):  # skip world body (id=0)
        mass = mjm.body_mass[body_id]  # kg
        if mass <= 0.0:
            continue

        # Body COM position in MuJoCo world (= ECI) frame, meters.
        r_body_eci = mjd.xipos[body_id] * _M_TO_KM

        # Gravity at body position, converted from km/s^2 to m/s^2.
        g_body = total_accel(r_body_eci, use_j2=model.use_j2)  # km/s^2
        F_N = mass * g_body * _KM_S2_TO_M_S2

        data.wrench_buffer[body_id, :3] += F_N
