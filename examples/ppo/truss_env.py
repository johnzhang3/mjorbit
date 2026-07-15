"""Vectorized free-flyer truss-servicing environment (mjorbit_warp).

A bi-manual free-flyer spacecraft (the banner-figure bus) holds a large,
free-floating truss in a two-arm **hug** -- each hand closes its capsule jaws on
the spar and holds it by contact + friction (NO weld, NO equality), with a wrist
hinge leveling the hand so the jaws straddle the spar with shallow surface
contact -- and slews it into its gravity-gradient-stable attitude (long axis
along the local vertical, i.e. pointing at Earth). The truss is a fully
independent free body; the only coupling is contact, so the policy must keep the
jaws on it. The wrist and grip joints are auto-driven (wrist leveled, grip held
closed), so the policy commands only the 4 arm joints + 6 bus DOFs.

The policy controls 10 actuators -- 4 arm joints and the spacecraft's 6-DOF
control (3 reaction-wheel torques + 3 thruster forces, ideal `motor` actuators
on the bus free joint, with light velocity damping). It keeps the truss hugged,
slews it with the wheels to point at nadir, and station-keeps with the
thrusters. The wheel/thruster authority is capped so the slew is gentle enough
that the truss is not flung out of the hug. Fly-in / fly-away bookends are
scripted at record time.

The big truss is light but strongly elongated (inertia ~18/18/0.1 kg*m^2), so
gravity gradient genuinely shapes the dynamics: its stable equilibrium is the
long axis along the local vertical. The orbit is in the X-Z plane and nadir
sweeps with it. Gravity gradient, drag, and J2 are enabled.

Implements the ``rsl_rl.env.VecEnv`` interface (rsl-rl-lib >= 5.x, TensorDict
observations).
"""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path

import mujoco as mj

# The native bindings must load before torch so they resolve the pixi env's
# libstdc++ rather than the older system library the torch wheel pulls in.
import mjorbit  # noqa: F401

# isort: split

import numpy as np
import torch
from rsl_rl.env import VecEnv
from sim_backend import MJWARP, SimBackend
from tensordict import TensorDict

from mjorbit.constants import GM_EARTH, R_EARTH
from mjorbit_warp import MjoModel, OrbitInit, SurfaceSpec

_XML_PATH = Path(__file__).with_name("spacecraft_truss.xml")

_ARM_ACT = ("shoulder_a_pos", "elbow_a_pos", "shoulder_b_pos", "elbow_b_pos")
# Wrists auto-level the hands and grips auto-close during manipulation -- they are
# not policy actions (the wrist keeps the jaw clamp axis vertical so the jaws
# straddle the spar with shallow surface contact; the closed grip holds it).
_WRIST_ACT = ("wrist_a_pos", "wrist_b_pos")
_GRIP_ACT = ("grip_a_pos", "grip_b_pos")
_GRIP_CLOSED = -0.03
_THRUST_ACT = ("thrust_x", "thrust_y", "thrust_z")
_RW_ACT = ("rw_x", "rw_y", "rw_z")
# Authority kept modest so the slew is gentle -- full wheel torque would fling
# the truss out of the two-arm hug.
_RW_LIMIT = 8.0
_THRUST_LIMIT = 35.0
_CLAW_REACH = 0.12  # jaw midpoint ~ ee body + ee_x * this
_TRUSS_NOMINAL = np.array([0.0, 0.0, 1.0])


_SURFACES = (
    SurfaceSpec("bus", [0.0, 1.1, 0.16], [0.0, 0.0, 1.0], 1.3, name="wing_p_f"),
    SurfaceSpec("bus", [0.0, 1.1, 0.16], [0.0, 0.0, -1.0], 1.3, name="wing_p_b"),
    SurfaceSpec("bus", [0.0, -1.1, 0.16], [0.0, 0.0, 1.0], 1.3, name="wing_n_f"),
    SurfaceSpec("bus", [0.0, -1.1, 0.16], [0.0, 0.0, -1.0], 1.3, name="wing_n_b"),
    SurfaceSpec("truss", [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.9, name="spar_f"),
    SurfaceSpec("truss", [0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 0.9, name="spar_b"),
)


@dataclasses.dataclass
class TrussEnvCfg:
    """Configuration for :class:`TrussReorientEnv`."""

    num_envs: int = 1024
    episode_length_s: float = 120.0
    control_dt: float = 0.1
    physics_dt: float = 0.01

    altitude_km: float = 400.0

    arm_scale: float = 0.5  # small arm adjustments around the hug pose
    bus_vel_damp: float = 40.0

    init_arm_noise: float = 0.02
    init_truss_angvel: float = 0.0

    hug_lost_m: float = 0.6  # jaw-to-spar-axis distance beyond which the hug failed

    # Reward weights.
    w_hug: float = 1.5  # keep the jaws on the spar
    hug_width_m: float = 0.25
    w_align: float = 2.0  # (gated on hug) point the truss at nadir
    w_fine: float = 2.0
    fine_width_rad: float = 0.25
    w_fine_tight: float = 1.5
    fine_tight_width_rad: float = 0.08
    w_progress: float = 14.0
    w_truss_rate: float = 0.04
    w_bus_pos: float = 0.3
    w_bus_rate: float = 0.02
    w_arm_rate: float = 0.05
    w_effort: float = 0.004
    lost_penalty: float = 20.0
    crash_penalty: float = 20.0

    use_gravity_gradient: bool = True
    use_drag: bool = True
    use_j2: bool = True
    use_srp: bool = False
    use_magnetic: bool = False

    # Physics backend: "mjorbit" (full orbital coupling -- also the eval env) or
    # "mjwarp" (bare mujoco_warp rigid-body dynamics: no gravity gradient / drag /
    # J2 / non-inertial frame / orbit propagation). See experiments/sim_fidelity.
    backend: str = "mjorbit"
    # On the bare backend, advance the nadir reference kinematically (a circular
    # Keplerian orbit) so the policy still sees a MOVING target -- isolating the
    # comparison to the orbital DYNAMICS, not the reference signal (the "fair"
    # baseline). Set False for the "naive" baseline (frozen nadir). Ignored on the
    # mjorbit backend, which propagates the true orbit.
    mjwarp_moving_target: bool = True
    # Disable per-world auto-reset on done (the evaluator runs one fixed-horizon
    # episode per world).
    auto_reset: bool = True
    # Re-randomize the orbit phase (hence the initial nadir-pointing error) on
    # every episode reset. The base example keeps a world's orbit across resets;
    # the fidelity study turns this ON for ALL truss conditions so every backend
    # sees an IDENTICAL initial-condition distribution. Without it, the frozen-
    # target naive baseline would re-init each world to the same misalignment
    # every episode and train on a narrower distribution than the others.
    resample_orbit_on_reset: bool = False

    device: str = "cuda"
    seed: int = 0


class TrussReorientEnv(VecEnv):
    """Hug and gravity-gradient-stabilize a large free truss."""

    def __init__(self, cfg: TrussEnvCfg) -> None:
        if cfg.num_envs < 2:
            raise ValueError("num_envs must be >= 2 (nworld=1 uses unbatched buffers)")
        decimation = cfg.control_dt / cfg.physics_dt
        if abs(decimation - round(decimation)) > 1e-9:
            raise ValueError("control_dt must be an integer multiple of physics_dt")

        self.cfg = cfg
        self.device = cfg.device
        self.num_envs = cfg.num_envs
        self.num_actions = 10  # 4 arm + 3 rw + 3 thrust
        self.max_episode_length = int(round(cfg.episode_length_s / cfg.control_dt))
        self.episode_length_buf = torch.zeros(cfg.num_envs, dtype=torch.long, device=cfg.device)
        self._decimation = int(round(decimation))
        self._rng = np.random.default_rng(cfg.seed)
        self._backend = SimBackend(cfg.backend)
        radius = R_EARTH + cfg.altitude_km
        self._orbit_radius = float(radius)
        self._orbit_rate = float(np.sqrt(GM_EARTH / radius**3))
        self._orbit_speed = float(np.sqrt(GM_EARTH / radius))

        self.model = MjoModel.from_xml_path(
            str(_XML_PATH),
            surfaces=_SURFACES,
            mj_timestep=cfg.physics_dt,
            use_j2=cfg.use_j2,
            use_drag=cfg.use_drag,
            use_srp=cfg.use_srp,
            use_magnetic=cfg.use_magnetic,
            use_gravity_gradient=cfg.use_gravity_gradient,
        )
        self.data = self.model.make_data(
            orbit=self._sample_orbits(cfg.num_envs),
            nworld=cfg.num_envs,
            nconmax=48,
            njmax=128,
        )

        m = self.model.mj_model
        self._arm_ctrl = np.array([m.actuator(n).id for n in _ARM_ACT])
        self._wrist_ctrl = np.array([m.actuator(n).id for n in _WRIST_ACT])
        self._grip_ctrl = np.array([m.actuator(n).id for n in _GRIP_ACT])
        self._wrist_jq = np.array([m.jnt_qposadr[m.actuator(n).trnid[0]] for n in _WRIST_ACT])
        self._grip_jq = np.array([m.jnt_qposadr[m.actuator(n).trnid[0]] for n in _GRIP_ACT])
        self._thrust_ctrl = np.array([m.actuator(n).id for n in _THRUST_ACT])
        self._rw_ctrl = np.array([m.actuator(n).id for n in _RW_ACT])
        self._arm_jq = np.array([m.jnt_qposadr[m.actuator(n).trnid[0]] for n in _ARM_ACT])
        self._arm_jv = np.array([m.jnt_dofadr[m.actuator(n).trnid[0]] for n in _ARM_ACT])
        self._bus_q = int(m.jnt_qposadr[m.joint("base").id])
        self._bus_v = int(m.jnt_dofadr[m.joint("base").id])
        self._tr_q = int(m.jnt_qposadr[m.joint("truss_free").id])
        self._tr_v = int(m.jnt_dofadr[m.joint("truss_free").id])
        self._bus_bid = int(m.body("bus").id)
        self._truss_bid = int(m.body("truss").id)
        self._ee_a_bid = int(m.body("ee_a").id)
        self._ee_b_bid = int(m.body("ee_b").id)
        self._qpos0 = m.qpos0.copy()
        self._arm_range = m.jnt_range[[m.actuator(n).trnid[0] for n in _ARM_ACT]].copy()

        self._arm_ready = self._solve_hug_pose(m)

        self._last_actions = np.zeros((cfg.num_envs, self.num_actions))
        self._obs = TensorDict({}, batch_size=[cfg.num_envs])

        self._reset_idx(np.arange(cfg.num_envs))
        self._pull()
        self._prev_align_err = self.truss_align_error()
        self._obs = self._compute_obs()

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    def get_observations(self) -> TensorDict:
        return self._obs

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        acts = actions.detach().to("cpu", torch.float64).clamp(-1.0, 1.0).numpy()
        self._apply_ctrl(acts)
        self._backend.advance(self.model, self.data, self._decimation)
        self._pull()
        if self._backend.kind == MJWARP and self.cfg.mjwarp_moving_target:
            self._propagate_target(self.cfg.control_dt)

        self.episode_length_buf += 1
        rewards, metrics, lost = self._compute_reward(acts)
        crashed = ~np.isfinite(self.data.qpos).all(axis=1)
        crashed |= ~np.isfinite(self.data.qvel).all(axis=1)
        rewards = np.where(crashed, -self.cfg.crash_penalty, rewards)

        time_out = (self.episode_length_buf >= self.max_episode_length).cpu().numpy()
        dones = time_out | crashed | lost
        self._last_actions = acts

        done_idx = np.flatnonzero(dones)
        if done_idx.size and self.cfg.auto_reset:
            self._reset_idx(done_idx)
            self._pull()
            self._prev_align_err[done_idx] = self.truss_align_error()[done_idx]

        self._obs = self._compute_obs()
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        extras = {
            "time_outs": torch.as_tensor(time_out & ~(crashed | lost), dtype=torch.bool,
                                         device=self.device),
            "log": {f"/metrics/{k}": float(v) for k, v in metrics.items()},
        }
        return self._obs, rewards_t, dones_t, extras

    def _apply_ctrl(self, acts: np.ndarray) -> None:
        cfg = self.cfg
        arm = np.clip(
            self._arm_ready[None, :] + cfg.arm_scale * acts[:, :4],
            self._arm_range[:, 0],
            self._arm_range[:, 1],
        )
        rw = acts[:, 4:7] * _RW_LIMIT
        bus_vel = self.data.qvel[:, self._bus_v : self._bus_v + 3]
        thr = np.clip(
            -cfg.bus_vel_damp * bus_vel + acts[:, 7:10] * _THRUST_LIMIT,
            -_THRUST_LIMIT,
            _THRUST_LIMIT,
        )
        # Auto-level the wrists (clamp axis vertical) and keep the grips closed.
        wrist = -(arm[:, 0:1] + arm[:, 1:2])  # arm A: wrist = -(shoulder+elbow)
        wrist = np.concatenate([wrist, -(arm[:, 2:3] + arm[:, 3:4])], axis=1)
        self.data.ctrl[:, self._arm_ctrl] = arm
        self.data.ctrl[:, self._wrist_ctrl] = wrist
        self.data.ctrl[:, self._grip_ctrl] = _GRIP_CLOSED
        self.data.ctrl[:, self._rw_ctrl] = rw
        self.data.ctrl[:, self._thrust_ctrl] = thr

    # ------------------------------------------------------------------
    # Simulation state helpers
    # ------------------------------------------------------------------

    def _pull(self) -> None:
        self._backend.pull(
            self.model,
            self.data,
            fields=("qpos", "qvel", "xpos", "xmat", "orbit", "qacc_warmstart"),
        )

    def _solve_hug_pose(self, m) -> np.ndarray:
        """Coarse IK placing each wrist (ee body) on the spar at +-0.3. With the
        wrist auto-leveled, the jaws then straddle the spar's surface."""
        d = mj.MjData(m)
        grid = np.linspace(-3.0, 3.0, 121)
        targets = {"ee_a": np.array([0.3, 0.0, 1.0]), "ee_b": np.array([-0.3, 0.0, 1.0])}
        names = {"ee_a": ("shoulder_a", "elbow_a"), "ee_b": ("shoulder_b", "elbow_b")}
        pose = {}
        for ee, (sh, el) in names.items():
            sh_q = int(m.jnt_qposadr[m.joint(sh).id])
            el_q = int(m.jnt_qposadr[m.joint(el).id])
            bid = int(m.body(ee).id)
            best = (1e9, 0.0, 0.0)
            for s, e in itertools.product(grid, grid):
                d.qpos[:] = m.qpos0
                d.qpos[sh_q] = s
                d.qpos[el_q] = e
                mj.mj_kinematics(m, d)
                err = np.linalg.norm(d.xpos[bid] - targets[ee])
                if err < best[0]:
                    best = (err, s, e)
            pose[sh], pose[el] = best[1], best[2]
        return np.array([pose["shoulder_a"], pose["elbow_a"], pose["shoulder_b"], pose["elbow_b"]])

    def _sample_orbits(self, n: int) -> list[OrbitInit]:
        radius = R_EARTH + self.cfg.altitude_km
        speed = float(np.sqrt(GM_EARTH / radius))
        phi = self._rng.uniform(0.0, 2.0 * np.pi, n)
        return [
            OrbitInit(
                R_eci=radius * np.array([np.cos(p), 0.0, np.sin(p)]),
                V_eci=speed * np.array([-np.sin(p), 0.0, np.cos(p)]),
            )
            for p in phi
        ]

    def _lvlh_world(self, idx: np.ndarray | slice = slice(None)) -> np.ndarray:
        r_eci = np.asarray(self.data.orbit.R_eci)[idx]
        return -r_eci / np.linalg.norm(r_eci, axis=1, keepdims=True)

    def _propagate_target(self, dt: float) -> None:
        """Advance the nadir reference kinematically on the bare-mjwarp backend.

        ``mujoco_warp`` does not propagate the chief orbit, so without this the
        nadir target (read from ``data.orbit.R_eci``) would be frozen. A vanilla
        MuJoCo-Warp user would still feed a time-varying nadir computed from a
        Keplerian orbit; this rotates the circular orbit in the X-Z plane at the
        orbit rate so the policy sees that MOVING reference -- WITHOUT any of the
        orbital dynamics (gravity gradient, drag, J2, non-inertial frame) that
        only mjorbit_warp provides. The comparison thus isolates the physics, not
        the reference signal. (On the mjorbit backend the real orbit is propagated
        inside ``mjo_step`` and this is never called.)
        """
        theta = self._orbit_rate * dt
        c, s = float(np.cos(theta)), float(np.sin(theta))
        R = np.asarray(self.data.orbit.R_eci, dtype=float)
        V = np.asarray(self.data.orbit.V_eci, dtype=float)
        rx, rz = R[:, 0].copy(), R[:, 2].copy()
        R[:, 0], R[:, 2] = c * rx - s * rz, s * rx + c * rz
        vx, vz = V[:, 0].copy(), V[:, 2].copy()
        V[:, 0], V[:, 2] = c * vx - s * vz, s * vx + c * vz
        R *= self._orbit_radius / np.linalg.norm(R, axis=1, keepdims=True)
        self.data.orbit.R_eci[:] = R
        self.data.orbit.V_eci[:] = V

    def _reset_idx(self, idx: np.ndarray) -> None:
        n = idx.size
        qpos = np.tile(self._qpos0, (n, 1))
        qvel = np.zeros((n, self.model.mj_model.nv))

        qpos[:, self._bus_q + 3 : self._bus_q + 7] = [1.0, 0.0, 0.0, 0.0]
        arm = self._arm_ready[None, :] + self._rng.uniform(
            -self.cfg.init_arm_noise, self.cfg.init_arm_noise, (n, 4)
        )
        arm = np.clip(arm, self._arm_range[:, 0], self._arm_range[:, 1])
        qpos[:, self._arm_jq] = arm
        # Wrists leveled (jaws straddle the spar), grips closed (already hugging).
        qpos[:, self._wrist_jq] = np.stack(
            [-(arm[:, 0] + arm[:, 1]), -(arm[:, 2] + arm[:, 3])], axis=1
        )
        qpos[:, self._grip_jq] = _GRIP_CLOSED
        # Free truss at the spawn point, identity attitude (orbit phase sets the
        # pointing error), already in the arms' hug.
        qpos[:, self._tr_q : self._tr_q + 3] = _TRUSS_NOMINAL
        qpos[:, self._tr_q + 3 : self._tr_q + 7] = [1.0, 0.0, 0.0, 0.0]
        qvel[:, self._tr_v + 3 : self._tr_v + 6] = self._rng.uniform(
            -self.cfg.init_truss_angvel, self.cfg.init_truss_angvel, (n, 3)
        )

        self.data.qpos[idx] = qpos
        self.data.qvel[idx] = qvel
        self.data.ctrl[idx] = 0.0
        self.data.ctrl[np.ix_(idx, self._arm_ctrl)] = arm
        self.data.ctrl[np.ix_(idx, self._wrist_ctrl)] = qpos[:, self._wrist_jq]
        self.data.ctrl[np.ix_(idx, self._grip_ctrl)] = _GRIP_CLOSED
        self.data.qacc_warmstart[idx] = 0.0
        self._last_actions[idx] = 0.0
        self.episode_length_buf[torch.as_tensor(idx, device=self.device)] = 0

        upload_fields = ["qpos", "qvel", "ctrl", "qacc_warmstart"]
        if self.cfg.resample_orbit_on_reset:
            # Fresh circular-orbit phase per reset world -> fresh initial nadir
            # error, identical distribution across backends. mjorbit advances the
            # orbit on-device, so it must be re-uploaded; mjwarp reads host orbit.
            phi = self._rng.uniform(0.0, 2.0 * np.pi, n)
            zeros = np.zeros(n)
            self.data.orbit.R_eci[idx] = self._orbit_radius * np.stack(
                [np.cos(phi), zeros, np.sin(phi)], axis=1
            )
            self.data.orbit.V_eci[idx] = self._orbit_speed * np.stack(
                [-np.sin(phi), zeros, np.cos(phi)], axis=1
            )
            if self._backend.kind != MJWARP:
                upload_fields.append("orbit")
        self._backend.reset_forward(
            self.model,
            self.data,
            upload_fields=tuple(upload_fields),
        )

    # ------------------------------------------------------------------
    # Observations and reward
    # ------------------------------------------------------------------

    def _body_rot(self, bid: int) -> np.ndarray:
        return np.asarray(self.data.xmat)[:, bid].reshape(self.num_envs, 3, 3)

    def _claw_centers(self) -> tuple[np.ndarray, np.ndarray]:
        xpos = np.asarray(self.data.xpos)
        ca = xpos[:, self._ee_a_bid] + self._body_rot(self._ee_a_bid)[:, :, 0] * _CLAW_REACH
        cb = xpos[:, self._ee_b_bid] + self._body_rot(self._ee_b_bid)[:, :, 0] * _CLAW_REACH
        return ca, cb

    def _hug_dist(self) -> np.ndarray:
        """Sum of perpendicular distances from each hand to the truss long axis.

        Small while the jaws are on the spar; grows if the truss slips
        out of the hug.
        """
        ca, cb = self._claw_centers()
        hub = np.asarray(self.data.xpos)[:, self._truss_bid]
        axis = self._body_rot(self._truss_bid)[:, :, 0]
        d = 0.0
        for c in (ca, cb):
            rel = c - hub
            along = np.sum(rel * axis, axis=1, keepdims=True) * axis
            d = d + np.linalg.norm(rel - along, axis=1)
        return d

    def _align(self) -> tuple[np.ndarray, np.ndarray]:
        nadir = self._lvlh_world()
        axis = self._body_rot(self._truss_bid)[:, :, 0]
        cos_align = np.abs(np.sum(axis * nadir, axis=1)).clip(0.0, 1.0)
        return cos_align, np.arccos(cos_align)

    def truss_align_error(self) -> np.ndarray:
        return self._align()[1]

    def _compute_obs(self) -> TensorDict:
        nadir = self._lvlh_world()
        r_bus = self._body_rot(self._bus_bid)
        r_truss = self._body_rot(self._truss_bid)
        bus_pos = self.data.qpos[:, self._bus_q : self._bus_q + 3]
        qvel = self.data.qvel
        xpos = np.asarray(self.data.xpos)
        truss_w_world = _mat_vec(r_truss, qvel[:, self._tr_v + 3 : self._tr_v + 6])

        obs = np.concatenate(
            [
                self.data.qpos[:, self._arm_jq] / 3.14,
                qvel[:, self._arm_jv] / 3.0,
                _mat_t_vec(r_bus, nadir),
                _mat_t_vec(r_bus, r_truss[:, :, 0]),
                _mat_t_vec(r_bus, xpos[:, self._truss_bid] - bus_pos) - [0.0, 0.0, 1.0],
                _mat_t_vec(r_bus, qvel[:, self._tr_v : self._tr_v + 3]
                           - qvel[:, self._bus_v : self._bus_v + 3]) / 0.3,
                _mat_t_vec(r_bus, truss_w_world) / 0.5,
                bus_pos / 1.0,
                qvel[:, self._bus_v : self._bus_v + 3] / 0.3,
                qvel[:, self._bus_v + 3 : self._bus_v + 6] / 0.2,
                self._last_actions,
            ],
            axis=1,
        )
        obs = np.clip(np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0)
        return TensorDict(
            {"policy": torch.as_tensor(obs, dtype=torch.float32, device=self.device)},
            batch_size=[self.num_envs],
        )

    def _compute_reward(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
        cfg = self.cfg
        qvel = self.data.qvel
        cos_align, angle = self._align()

        progress = np.clip(self._prev_align_err - angle, -0.1, 0.1)
        self._prev_align_err = angle.copy()

        hug_dist = self._hug_dist()
        hugged = np.exp(-((hug_dist / cfg.hug_width_m) ** 2))

        truss_rate = np.linalg.norm(qvel[:, self._tr_v + 3 : self._tr_v + 6], axis=1)
        bus_pos_err = np.linalg.norm(self.data.qpos[:, self._bus_q : self._bus_q + 3], axis=1)
        bus_rate = np.linalg.norm(qvel[:, self._bus_v : self._bus_v + 6], axis=1)
        arm_rate = np.mean((actions[:, :4] - self._last_actions[:, :4]) ** 2, axis=1)
        effort = np.mean(actions[:, 4:] ** 2, axis=1)

        lost = hug_dist > cfg.hug_lost_m

        point = (
            cfg.w_align * cos_align
            + cfg.w_fine * np.exp(-((angle / cfg.fine_width_rad) ** 2))
            + cfg.w_fine_tight * np.exp(-((angle / cfg.fine_tight_width_rad) ** 2))
            + cfg.w_progress * progress
            - cfg.w_truss_rate * np.minimum(truss_rate**2 / 0.04, 25.0)
        )
        reward = (
            cfg.w_hug * hugged
            + hugged * point
            - cfg.w_bus_pos * np.minimum(bus_pos_err**2, 9.0)
            - cfg.w_bus_rate * np.minimum(bus_rate**2, 25.0)
            - cfg.w_arm_rate * arm_rate
            - cfg.w_effort * effort
        )
        reward = np.where(lost, reward - cfg.lost_penalty, reward)

        metrics = {
            "align_err_deg": float(np.degrees(np.nanmean(angle))),
            "hugged_frac": float(np.mean(hug_dist < cfg.hug_width_m)),
            "hug_dist": float(np.nanmean(hug_dist)),
            "lost_frac": float(np.mean(lost)),
            "truss_rate": float(np.nanmean(truss_rate)),
        }
        return reward, metrics, lost


# ----------------------------------------------------------------------
# Batched rotation helpers
# ----------------------------------------------------------------------


def _mat_vec(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nj->ni", r, v)


def _mat_t_vec(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum("nji,nj->ni", r, v)
