"""Rotational gravity-gradient torque on rigid bodies.

For a rigid body with mass-center position ``r`` (from Earth center) and
inertia tensor ``J`` about the COM, the torque arising from the differential
gravity across the body — integrated analytically in the limit where the
body's extent is small compared to ``|r|`` — is

    τ_gg = (3 μ / |r|^3)  r̂ × (J · r̂)

where r̂ is a unit vector along ``r`` (sign-invariant: (-r̂) × J(-r̂)
= r̂ × J r̂).  This captures the *rotational* gradient effect on each rigid
body about its own COM, independent of the translational gradient that
``coupling.inertial`` applies as differential body-COM forces.

For a single rigid body, the inertial force-differential approach in
``coupling.inertial`` evaluates gravity only at the body COM and therefore
produces no net torque.  The rotational torque below is the missing piece.
For articulated models, each body contributes its own τ_gg about its own
COM — the two effects are complementary and do not double-count.

Units: GM in km³/s², body positions in km, inertias in kg·m², so
coeff = 3μ/|r|³ has units 1/s² and τ_gg has units kg·m²/s² = N·m.
"""

from __future__ import annotations

import numpy as np

from mujoco_orbit.constants import GM_EARTH
from mujoco_orbit.core.runtime import MjoData, MjoModel


def apply_gravity_gradient_torques(model: MjoModel, data: MjoData) -> None:
    """Accumulate per-body rotational gravity-gradient torques into the
    shared wrench buffer (world-frame torques).

    The torque on each body is evaluated using that body's inertia tensor
    (rotated to the world frame via ``ximat``) and the body-COM radial
    direction in the world (LVLH) frame.
    """
    if not model.use_gravity_gradient:
        return

    orbit = data.orbit
    fc = data.frame
    mjm = model.mj_model
    mjd = data.mj_data

    R_ref_eci = orbit.R_eci  # km

    for body_id in range(1, mjm.nbody):
        if mjm.body_mass[body_id] <= 0.0:
            continue

        # Body COM position in ECI (km)
        r_lvlh_km = mjd.xipos[body_id] * 1e-3
        r_body_eci = R_ref_eci + fc.C_IL @ r_lvlh_km
        R_mag = float(np.linalg.norm(r_body_eci))
        if R_mag < 1e-9:
            continue

        # Radial direction in the world (LVLH) frame. The formula is sign invariant.
        r_hat_lvlh = fc.C_LI @ (r_body_eci / R_mag)

        # Inertia tensor in the world frame via the inertia-frame rotation.
        # ximat is world-from-principal-axes; equals xmat only when body_iquat
        # is identity.
        R_wi = mjd.ximat[body_id].reshape(3, 3)
        J_world = R_wi @ np.diag(mjm.body_inertia[body_id]) @ R_wi.T

        coeff = 3.0 * GM_EARTH / R_mag**3  # 1/s²
        tau_world = coeff * np.cross(r_hat_lvlh, J_world @ r_hat_lvlh)

        data.wrench_buffer[body_id, 3:] += tau_world
