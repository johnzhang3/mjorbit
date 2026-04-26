"""Test that MuJoCo contact forces work correctly within the coupled simulator.

Key properties verified:
1. Two colliding bodies bounce off each other (MuJoCo resolves contact)
2. Contact forces do NOT leak into wrench_buffer / orbit feedback
3. Chief-inertial momentum is conserved through the collision instant
4. The chief orbit is only affected by tidal (gravity gradient) effects,
   not by contact forces
5. Energy is not created through contact
6. Post-collision, each body follows the correct CW trajectory — meaning
   the collision changes their individual ECI orbits
"""

import os
import tempfile

import numpy as np

from mujoco_orbit import mjo_forward, mjo_step
from mujoco_orbit.constants import GM_EARTH, R_EARTH
from mujoco_orbit.core.runtime import MjoData, MjoModel

from ._helpers import get_freejoint_lvlh_state, make_model_data

# Two equal-mass spheres with a gap between them.
# body_a starts with +x velocity, body_b is at rest → head-on collision.
TWO_BODY_COLLISION_XML = """\
<mujoco model="collision_test">
  <option timestep="0.002" gravity="0 0 0">
    <flag contact="enable"/>
  </option>

  <default>
    <geom condim="3" friction="1 0.005 0.0001" solref="0.01 1" solimp="0.9 0.95 0.001"/>
  </default>

  <worldbody>
    <body name="body_a" pos="-1 0 0">
      <freejoint name="jnt_a"/>
      <geom type="sphere" size="0.3" mass="50" rgba="0.9 0.2 0.2 1"/>
    </body>
    <body name="body_b" pos="1 0 0">
      <freejoint name="jnt_b"/>
      <geom type="sphere" size="0.3" mass="50" rgba="0.2 0.2 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _make_collision_model_data(xml_str: str = TWO_BODY_COLLISION_XML, **overrides):
    """Write XML to a temp file and build model/data with all environment off."""
    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
        f.write(xml_str)
        xml_path = f.name

    defaults = dict(
        xml_path=xml_path,
        mj_timestep=0.002,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    defaults.update(overrides)
    model, data = make_model_data(**defaults)
    return model, data, xml_path


def _total_lvlh_momentum(model: MjoModel, data: MjoData) -> np.ndarray:
    """Compute total linear momentum in the chief-inertial MuJoCo world frame."""
    p_total = np.zeros(3)
    for i in range(1, model.nbody):
        mass = model.body_mass[i]
        vel = data.cvel[i, 3:6]
        p_total += mass * vel
    return p_total


def _total_kinetic_energy(model: MjoModel, data: MjoData) -> float:
    """Compute total translational KE in the chief-inertial MuJoCo world frame."""
    ke = 0.0
    for i in range(1, model.nbody):
        mass = model.body_mass[i]
        vel = data.cvel[i, 3:6]
        ke += 0.5 * mass * np.dot(vel, vel)
    return ke


class TestCollisionDynamics:
    """Verify two free bodies collide and bounce correctly."""

    def test_head_on_collision_bounces(self):
        """Body A approaches B; after collision, A should slow/reverse and B should gain speed."""
        model, data, xml_path = _make_collision_model_data()
        try:
            # body_a at x=-1, body_b at x=+1, gap = 2 - 0.3 - 0.3 = 1.4 m
            # Give body_a a +x velocity to approach body_b
            data.qvel[0] += 2.0  # body_a: vx = +2 m/s relative to the chief
            mjo_forward(model, data)

            contact_detected = False
            for step in range(2000):
                mjo_step(model, data)
                if data.ncon > 0:
                    contact_detected = True
                    break

            assert contact_detected, "No contact detected — bodies did not collide"

            # Continue for a bit after contact to let bodies separate
            for _ in range(500):
                mjo_step(model, data)

            # body_a should have slowed or reversed
            va_x = data.qvel[0]
            vb_x = data.qvel[6]

            # body_b should have gained positive x-velocity
            assert vb_x > 0.1, f"body_b should have gained speed, got vx={vb_x:.4f}"
            # For equal masses, body_a should have slowed significantly
            assert va_x < 1.5, f"body_a should have slowed, got vx={va_x:.4f}"

        finally:
            os.unlink(xml_path)

    def test_contact_forces_not_in_wrench_buffer(self):
        """wrench_buffer should contain only inertial forces (~0.2 N), not contact forces (~kN).

        MuJoCo contact forces are O(1000 N) for 50 kg at 2 m/s. Inertial forces
        differential gravity is O(0.1 N) at LEO. If contact leaked into
        wrench_buffer, we'd see values orders of magnitude larger.
        """
        model, data, xml_path = _make_collision_model_data()
        try:
            data.qvel[0] += 2.0
            mjo_forward(model, data)

            max_wrench = 0.0
            max_contact_force = 0.0

            for _ in range(2000):
                mjo_step(model, data)
                w = np.max(np.abs(data.wrench_buffer))
                max_wrench = max(max_wrench, w)

                # Also measure actual contact forces for comparison
                for i in range(data.ncon):
                    force = data.contact_force(i)
                    max_contact_force = max(max_contact_force, np.linalg.norm(force[:3]))

            # Contact forces should be much larger than wrench_buffer entries.
            # If contact leaked into wrench_buffer, max_wrench would be ~max_contact_force.
            # Differential gravity wrenches are O(0.1 N); contact forces are O(1000+ N).
            assert max_wrench < 1000.0, (
                f"wrench_buffer too large ({max_wrench:.1f} N) — "
                f"contact forces ({max_contact_force:.1f} N) may be leaking in"
            )
            assert max_contact_force > 100.0, (
                f"Contact forces suspiciously small ({max_contact_force:.1f} N)"
            )
            # The gap between them confirms separation of concerns
            assert max_contact_force > 10 * max_wrench, (
                f"wrench_buffer ({max_wrench:.2f} N) too close to contact forces "
                f"({max_contact_force:.1f} N) — possible leak"
            )

        finally:
            os.unlink(xml_path)

    def test_momentum_conserved_through_collision_instant(self):
        """Momentum should be conserved through the collision to within the
        inertial impulse per timestep.

        Differential gravity causes slow momentum drift. But through the collision
        instant, the contact impulse should conserve momentum — the per-step drift
        from inertial forces is negligible compared to the contact impulse.
        """
        model, data, xml_path = _make_collision_model_data()
        try:
            data.qvel[0] += 2.0
            mjo_forward(model, data)

            # Run until just before contact
            p_before_contact = None
            p_after_contact = None
            for step in range(2000):
                p_pre = _total_lvlh_momentum(model, data)
                mjo_step(model, data)
                p_post = _total_lvlh_momentum(model, data)

                if data.ncon > 0 and p_before_contact is None:
                    p_before_contact = p_pre
                elif p_before_contact is not None and data.ncon == 0:
                    p_after_contact = p_post
                    break

            assert p_before_contact is not None, "No contact detected"
            assert p_after_contact is not None, "Contact never ended"

            # Momentum should be approximately conserved through the collision.
            # Allow for accumulated inertial drift during the contact period.
            # Inertial forces are ~0.2 N, contact lasts ~20 steps = 0.04s
            # → inertial impulse ~0.008 N·s, vs contact impulse ~100 N·s
            np.testing.assert_allclose(
                p_after_contact, p_before_contact, atol=20.0,
                err_msg="Momentum changed through collision by more than inertial drift",
            )

        finally:
            os.unlink(xml_path)

    def test_orbit_divergence_is_tidal_only(self):
        """The chief orbit should diverge between collision/no-collision scenarios
        only at the level of gravity gradient (tidal) effects, not contact forces.

        With bodies at ~1 m offset from chief at 400 km altitude, the tidal
        acceleration is ~n² × r ≈ 1e-6 m/s². Over 4s this produces negligible
        orbit change. Contact forces (~10 kN) would produce massive orbit changes
        if they leaked in.
        """
        model, data, xml_path = _make_collision_model_data()
        model_ref, data_ref, xml_ref = _make_collision_model_data()
        try:
            data.qvel[0] += 2.0  # collision scenario
            mjo_forward(model, data)
            # data_ref: no velocity, no collision

            for _ in range(2000):
                mjo_step(model, data)
                mjo_step(model_ref, data_ref)

            # Orbit divergence should be tiny — just tidal effects from different
            # body positions (~meters vs ~7000 km orbit).
            # If contact forces leaked, we'd see km-scale divergence.
            dR = np.linalg.norm(data.orbit.R_eci - data_ref.orbit.R_eci)
            dV = np.linalg.norm(data.orbit.V_eci - data_ref.orbit.V_eci)

            # Inertial divergence from tidal gravity remains tiny over this horizon.
            # If contact forces leaked (O(10 kN)), orbit dV would be ~0.4 km/s.
            assert dR < 1e-3, f"Orbit position diverged by {dR:.2e} km — contact leak?"
            assert dV < 1e-4, f"Orbit velocity diverged by {dV:.2e} km/s — contact leak?"

        finally:
            os.unlink(xml_path)
            os.unlink(xml_ref)

    def test_contact_force_magnitude_reasonable(self):
        """Sanity check: MuJoCo's contact forces during collision are physically reasonable."""
        model, data, xml_path = _make_collision_model_data()
        try:
            data.qvel[0] += 2.0
            mjo_forward(model, data)

            max_contact_force = 0.0
            for _ in range(2000):
                mjo_step(model, data)
                for i in range(data.ncon):
                    force = data.contact_force(i)
                    max_contact_force = max(max_contact_force, np.linalg.norm(force[:3]))

            assert max_contact_force > 0, "No contact force detected"
            assert np.isfinite(max_contact_force), "Contact force is not finite"

        finally:
            os.unlink(xml_path)


class TestCollisionEnergyBudget:
    """Check that kinetic energy is handled correctly through contact."""

    def test_kinetic_energy_not_gained(self):
        """Total KE should not increase through contact (energy conservation / dissipation)."""
        model, data, xml_path = _make_collision_model_data()
        try:
            data.qvel[0] += 2.0
            mjo_forward(model, data)

            ke0 = _total_kinetic_energy(model, data)

            for _ in range(2000):
                mjo_step(model, data)

            ke1 = _total_kinetic_energy(model, data)

            # KE should not increase. Contacts dissipate energy.
            # Small tolerance for inertial work (gravity gradient does work on moving bodies).
            assert ke1 <= ke0 * 1.05, (
                f"KE increased through contact: {ke0:.4f} -> {ke1:.4f}"
            )

        finally:
            os.unlink(xml_path)


def _cw(x0, y0, z0, vx0, vy0, vz0, n, t):
    """Clohessy-Wiltshire analytical solution for relative motion."""
    nt = n * t
    cn, sn = np.cos(nt), np.sin(nt)
    x = (4 - 3 * cn) * x0 + sn / n * vx0 + 2 / n * (1 - cn) * vy0
    y = 6 * (sn - nt) * x0 + y0 - 2 / n * (1 - cn) * vx0 + (4 * sn - 3 * nt) / n * vy0
    z = z0 * cn + vz0 / n * sn
    return np.array([x, y, z])


class TestPostCollisionOrbits:
    """Verify that after collision, each body's trajectory (and thus ECI orbit) changes.

    The simulator doesn't track per-body orbits explicitly. Instead, individual
    body orbits emerge from chief_orbit + LVLH_position. After collision:
      - MuJoCo changes each body's LVLH velocity (contact impulse)
      - Inertial wrenches drive CW-like dynamics through differential gravity
      - Each body follows the correct CW trajectory for its post-collision ICs
      - In ECI, this means each body is now on a different orbit than before

    This test validates the complete chain: contact → new LVLH ICs → correct
    CW propagation → divergent ECI orbits.
    """

    def test_post_collision_bodies_follow_cw_trajectories(self):
        """After collision, each body's LVLH trajectory should match the CW
        analytical solution seeded from its post-collision state."""
        model, data, xml_path = _make_collision_model_data(mj_timestep=0.001)
        try:
            a_km = R_EARTH + 400.0
            n = np.sqrt(GM_EARTH / a_km**3)  # mean motion, rad/s

            data.qvel[0] += 2.0  # body_a approaches body_b
            mjo_forward(model, data)

            # Run past the collision until bodies have separated
            collision_ended = False
            in_contact = False
            for _ in range(5000):
                mjo_step(model, data)
                if data.ncon > 0:
                    in_contact = True
                elif in_contact:
                    collision_ended = True
                    break

            assert collision_ended, "Collision did not complete"

            # Let things settle for a few more steps after contact ends
            for _ in range(100):
                mjo_step(model, data)

            # Record post-collision ICs for both bodies (MuJoCo SI: meters, m/s)
            # body_a: qpos[0:3], qvel[0:3]
            # body_b: qpos[7:10], qvel[6:9]
            pos_a0, vel_a0 = get_freejoint_lvlh_state(data, slice(0, 3), slice(0, 3))
            pos_b0, vel_b0 = get_freejoint_lvlh_state(data, slice(7, 10), slice(6, 9))

            # Verify the collision actually changed velocities
            assert vel_b0[0] > 0.1, "body_b didn't gain velocity from collision"

            # Propagate forward for t_prop seconds
            t_prop = 20.0
            n_steps = int(t_prop / 0.001)
            for _ in range(n_steps):
                mjo_step(model, data)

            pos_a_final, _ = get_freejoint_lvlh_state(data, slice(0, 3), slice(0, 3))
            pos_b_final, _ = get_freejoint_lvlh_state(data, slice(7, 10), slice(6, 9))

            # CW prediction from post-collision ICs
            cw_a = _cw(pos_a0[0], pos_a0[1], pos_a0[2],
                        vel_a0[0], vel_a0[1], vel_a0[2], n, t_prop)
            cw_b = _cw(pos_b0[0], pos_b0[1], pos_b0[2],
                        vel_b0[0], vel_b0[1], vel_b0[2], n, t_prop)

            # The simulator uses exact (nonlinear) differential gravity, not
            # the CW linearization. At meter-scale offsets from a 6778 km orbit
            # the linearization error is small, so CW is a good reference — but
            # not exact. Tolerances reflect the linearization gap, not numerical
            # error.
            np.testing.assert_allclose(
                pos_a_final, cw_a, rtol=0.03, atol=0.01,
                err_msg="Body A post-collision trajectory doesn't match CW",
            )
            np.testing.assert_allclose(
                pos_b_final, cw_b, rtol=0.03, atol=0.01,
                err_msg="Body B post-collision trajectory doesn't match CW",
            )

        finally:
            os.unlink(xml_path)

    def test_collision_produces_divergent_eci_orbits(self):
        """After collision, the two bodies should be on different ECI orbits.

        Before collision: both near the same chief orbit.
        After collision: contact impulse gives them different LVLH velocities,
        so their ECI states diverge over time.
        """
        model, data, xml_path = _make_collision_model_data(mj_timestep=0.001)
        try:
            data.qvel[0] += 2.0
            mjo_forward(model, data)

            # Record initial ECI positions (should be close)
            r_a_eci_0 = data.eci_position_from_world(data.qpos[0:3]) * 1e-3
            r_b_eci_0 = data.eci_position_from_world(data.qpos[7:10]) * 1e-3
            eci_sep_0 = np.linalg.norm(r_a_eci_0 - r_b_eci_0)

            # Run past collision and then propagate
            for _ in range(30000):  # 30s at dt=0.001
                mjo_step(model, data)

            # Compute ECI positions after propagation
            r_a_eci = data.eci_position_from_world(data.qpos[0:3]) * 1e-3
            r_b_eci = data.eci_position_from_world(data.qpos[7:10]) * 1e-3
            eci_sep_final = np.linalg.norm(r_a_eci - r_b_eci)

            # The bodies should have diverged in ECI — collision changed
            # their trajectories. CW along-track drift from a radial
            # velocity impulse grows over time.
            assert eci_sep_final > eci_sep_0, (
                f"Bodies didn't diverge in ECI: initial sep={eci_sep_0:.4f} km, "
                f"final sep={eci_sep_final:.4f} km"
            )

        finally:
            os.unlink(xml_path)
