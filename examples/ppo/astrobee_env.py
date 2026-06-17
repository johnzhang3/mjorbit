"""Vectorized free-flyer grasping environment (mjorbit_warp).

An Astrobee-style free-flyer (NASA's ~32 cm cube ISS robot) starts tumbling and
drifting from an initial disturbance, **detumbles**, flies to a free-floating
cargo module, and **grasps** the cargo's grapple bar with its perching-arm
gripper. The cargo is a fully independent free body; the only coupling is contact
(no weld, no equality), so the policy must fly the open jaws onto the bar and
the grip auto-closes to pinch it.

The task is sequential by construction: the approach / grasp reward is gated on
the bus being detumbled (``exp(-(spin/width)^2)``), so the policy learns to kill
its angular rate before it can collect approach reward -- it stabilizes first,
then flies in and grabs.

The policy controls 8 actuators -- 2 perching-arm joints (shoulder + wrist) and
the spacecraft's 6-DOF control (3 reaction-wheel torques + 3 thruster forces,
ideal ``motor`` actuators on the bus free joint). The two-jaw grip is auto-closed
on capture (when the gripper is on the bar and aligned), mirroring an
underactuated enveloping gripper.

Gravity gradient, drag, and J2 are enabled for orbital context, though at this
body scale / horizon the environmental wrenches are minor compared with the
control authority -- the challenge is the coupled detumble + precision approach.

Implements the ``rsl_rl.env.VecEnv`` interface (rsl-rl-lib >= 5.x, TensorDict
observations).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

# The native bindings must load before torch so they resolve the pixi env's
# libstdc++ rather than the older system library the torch wheel pulls in.
import mjorbit  # noqa: F401

# isort: split

import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from mjorbit.constants import GM_EARTH, R_EARTH
from mjorbit_warp import (
    MjoModel,
    OrbitInit,
    SurfaceSpec,
    mjo_forward,
    mjo_pull,
    mjo_step,
    mjo_upload,
)

_XML_PATH = Path(__file__).with_name("astrobee_grasp.xml")

_ARM_ACT = ("shoulder_pos", "wrist_pos")
_THRUST_ACT = ("thrust_x", "thrust_y", "thrust_z")
_RW_ACT = ("rw_x", "rw_y", "rw_z")

# Parallel-jaw command: the lower pad mirrors the upper (grip_l = -grip_u), so a
# single binary command opens/closes both. Open retracts both pads clear of the
# bar; closed pinches it.
_GRIP_OPEN = 0.06
_GRIP_CLOSED = -0.02
_JAW_REACH = 0.07  # pad-region midpoint along the ee +x axis
_BAR_OFFSET = np.array([-0.205, 0.0, 0.0])  # handle bar center in the cargo body frame

# Authority sized for the ~9 kg cube (matches the XML ctrlranges).
_RW_LIMIT = 0.15
_THRUST_LIMIT = 1.5

# A cube face (0.32 m side) for the aerodynamic surface model.
_FACE_AREA = 0.32 * 0.32
_SURFACES = (
    SurfaceSpec("bus", [0.16, 0.0, 0.0], [1.0, 0.0, 0.0], _FACE_AREA, name="bus_xp"),
    SurfaceSpec("bus", [-0.16, 0.0, 0.0], [-1.0, 0.0, 0.0], _FACE_AREA, name="bus_xn"),
    SurfaceSpec("bus", [0.0, 0.16, 0.0], [0.0, 1.0, 0.0], _FACE_AREA, name="bus_yp"),
    SurfaceSpec("bus", [0.0, -0.16, 0.0], [0.0, -1.0, 0.0], _FACE_AREA, name="bus_yn"),
    SurfaceSpec("bus", [0.0, 0.0, 0.16], [0.0, 0.0, 1.0], _FACE_AREA, name="bus_zp"),
    SurfaceSpec("bus", [0.0, 0.0, -0.16], [0.0, 0.0, -1.0], _FACE_AREA, name="bus_zn"),
)


@dataclasses.dataclass
class AstrobeeEnvCfg:
    """Configuration for :class:`AstrobeeGraspEnv`."""

    num_envs: int = 1024
    episode_length_s: float = 60.0
    control_dt: float = 0.1
    physics_dt: float = 0.01

    altitude_km: float = 400.0

    arm_scale: float = 0.6  # arm joint range around the straight ready pose (rad)

    # Initial disturbance on the robot (the cargo floats calmly).
    init_spin_max: float = 0.4  # body-frame angular rate per axis (rad/s)
    init_drift_max: float = 0.03  # linear drift per axis (m/s)
    init_arm_noise: float = 0.05

    # Cargo spawn (offset the robot must fly to).
    cargo_dist: tuple[float, float] = (1.2, 2.0)
    cargo_lateral: float = 0.15  # +/- Y,Z spread of the cargo position (m)
    cargo_att_noise: float = 0.1  # cargo attitude noise (rad)

    # A grasp counts when the gripper is on the bar (small distance), aligned
    # (jaw axis parallel to the bar), and the policy has commanded the grip
    # closed. The pads are forgiving, so the distance need not be tiny.
    capture_radius: float = 0.07  # gripper-center to bar-center distance for a grasp
    capture_align: float = 0.7  # min |cos(jaw axis, bar axis)| for a grasp
    fail_radius: float = 3.5  # gripper too far from the bar -> failed episode

    # Reward weights.
    settle_width: float = 0.25  # detumble reward width (rad/s)
    gate_width: float = 0.45  # approach reward is gated by exp(-(spin/gate_width)^2)
    w_settle: float = 0.6
    w_progress: float = 12.0  # dense approach progress (reduce gripper->bar distance)
    w_reach: float = 3.0  # (CHANGE 3) reward the gripper being close to the bar
    reach_width: float = 0.35
    w_reach_tight: float = 4.0  # steep pull into the last ~10 cm so it reaches the bar
    reach_tight_width: float = 0.10
    w_align: float = 1.0  # reward the jaw axis aligned with the bar axis (when near)
    w_grasp: float = 6.0  # bonus while grasped (gripper closed on the bar)
    w_hold: float = 2.0  # (while grasped) reward holding STILL -- low absolute speed
    hold_width: float = 0.15  # m/s
    w_brake: float = 1.0  # (while grasped) penalize absolute drift speed -> station-keep
    w_grip_pen: float = 0.3  # discourage closing the grip away from the bar
    w_spin: float = 0.15  # detumble penalty (spin^2)
    w_dock_vel: float = 0.1  # soft-capture: penalize closing speed only near the bar
    dock_width: float = 0.12
    w_effort: float = 0.003
    w_arm_rate: float = 0.02
    fail_penalty: float = 15.0
    crash_penalty: float = 20.0

    use_gravity_gradient: bool = True
    use_drag: bool = True
    use_j2: bool = True
    use_srp: bool = False
    use_magnetic: bool = False

    device: str = "cuda"
    seed: int = 0


class AstrobeeGraspEnv(VecEnv):
    """Detumble, fly to a free-floating cargo module, and grasp its grapple bar."""

    def __init__(self, cfg: AstrobeeEnvCfg) -> None:
        if cfg.num_envs < 2:
            raise ValueError("num_envs must be >= 2 (nworld=1 uses unbatched buffers)")
        decimation = cfg.control_dt / cfg.physics_dt
        if abs(decimation - round(decimation)) > 1e-9:
            raise ValueError("control_dt must be an integer multiple of physics_dt")

        self.cfg = cfg
        self.device = cfg.device
        self.num_envs = cfg.num_envs
        self.num_actions = 9  # 2 arm + 3 rw + 3 thrust + 1 binary grip
        self.max_episode_length = int(round(cfg.episode_length_s / cfg.control_dt))
        self.episode_length_buf = torch.zeros(cfg.num_envs, dtype=torch.long, device=cfg.device)
        self._decimation = int(round(decimation))
        self._rng = np.random.default_rng(cfg.seed)

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
            nconmax=32,
            njmax=96,
        )

        m = self.model.mj_model
        self._arm_ctrl = np.array([m.actuator(n).id for n in _ARM_ACT])
        self._grip_u_ctrl = int(m.actuator("grip_u_pos").id)
        self._grip_l_ctrl = int(m.actuator("grip_l_pos").id)
        self._thrust_ctrl = np.array([m.actuator(n).id for n in _THRUST_ACT])
        self._rw_ctrl = np.array([m.actuator(n).id for n in _RW_ACT])
        self._arm_jq = np.array([m.jnt_qposadr[m.actuator(n).trnid[0]] for n in _ARM_ACT])
        self._arm_jv = np.array([m.jnt_dofadr[m.actuator(n).trnid[0]] for n in _ARM_ACT])
        self._grip_u_jq = int(m.jnt_qposadr[m.joint("grip_u").id])
        self._grip_l_jq = int(m.jnt_qposadr[m.joint("grip_l").id])
        self._bus_q = int(m.jnt_qposadr[m.joint("base").id])
        self._bus_v = int(m.jnt_dofadr[m.joint("base").id])
        self._cargo_q = int(m.jnt_qposadr[m.joint("cargo_free").id])
        self._cargo_v = int(m.jnt_dofadr[m.joint("cargo_free").id])
        self._bus_bid = int(m.body("bus").id)
        self._ee_bid = int(m.body("ee").id)
        self._cargo_bid = int(m.body("cargo").id)
        self._qpos0 = m.qpos0.copy()
        self._arm_range = m.jnt_range[[m.actuator(n).trnid[0] for n in _ARM_ACT]].copy()
        self._arm_ready = np.zeros(2)  # straight: gripper points +x, level

        self._grasped = np.zeros(cfg.num_envs, dtype=bool)
        self._grip_closed = np.zeros(cfg.num_envs, dtype=bool)
        self._last_actions = np.zeros((cfg.num_envs, self.num_actions))
        self._obs = TensorDict({}, batch_size=[cfg.num_envs])

        self._reset_idx(np.arange(cfg.num_envs))
        self._pull()
        self._prev_dist = self._grip_bar_dist()
        self._obs = self._compute_obs()

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    def get_observations(self) -> TensorDict:
        return self._obs

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        acts = actions.detach().to("cpu", torch.float64).clamp(-1.0, 1.0).numpy()
        self._apply_ctrl(acts)
        mjo_upload(self.model, self.data, fields=("ctrl",))
        for _ in range(self._decimation):
            mjo_step(self.model, self.data)
        self._pull()

        self.episode_length_buf += 1
        rewards, metrics, failed = self._compute_reward(acts)
        crashed = ~np.isfinite(self.data.qpos).all(axis=1)
        crashed |= ~np.isfinite(self.data.qvel).all(axis=1)
        rewards = np.where(crashed, -self.cfg.crash_penalty, rewards)

        time_out = (self.episode_length_buf >= self.max_episode_length).cpu().numpy()
        dones = time_out | crashed | failed
        self._last_actions = acts

        done_idx = np.flatnonzero(dones)
        if done_idx.size:
            self._reset_idx(done_idx)
            self._pull()
            self._prev_dist[done_idx] = self._grip_bar_dist()[done_idx]

        self._obs = self._compute_obs()
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        extras = {
            "time_outs": torch.as_tensor(time_out & ~(crashed | failed), dtype=torch.bool,
                                         device=self.device),
            "log": {f"/metrics/{k}": float(v) for k, v in metrics.items()},
        }
        return self._obs, rewards_t, dones_t, extras

    def _apply_ctrl(self, acts: np.ndarray) -> None:
        cfg = self.cfg
        arm = np.clip(
            self._arm_ready[None, :] + cfg.arm_scale * acts[:, :2],
            self._arm_range[:, 0],
            self._arm_range[:, 1],
        )
        rw = acts[:, 2:5] * _RW_LIMIT
        thr = acts[:, 5:8] * _THRUST_LIMIT
        # Binary grip: action[8] > 0 closes the pads, else opens them (the lower
        # pad mirrors the upper, grip_l = -grip_u). The policy decides when to
        # close on the bar.
        self._grip_closed = acts[:, 8] > 0.0
        grip = np.where(self._grip_closed, _GRIP_CLOSED, _GRIP_OPEN)
        self.data.ctrl[:, self._arm_ctrl] = arm
        self.data.ctrl[:, self._grip_u_ctrl] = grip
        self.data.ctrl[:, self._grip_l_ctrl] = -grip
        self.data.ctrl[:, self._rw_ctrl] = rw
        self.data.ctrl[:, self._thrust_ctrl] = thr

    # ------------------------------------------------------------------
    # Simulation state helpers
    # ------------------------------------------------------------------

    def _pull(self) -> None:
        mjo_pull(
            self.model,
            self.data,
            fields=("qpos", "qvel", "xpos", "xmat", "orbit", "qacc_warmstart"),
        )

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

    def _reset_idx(self, idx: np.ndarray) -> None:
        cfg = self.cfg
        n = idx.size
        qpos = np.tile(self._qpos0, (n, 1))
        qvel = np.zeros((n, self.model.mj_model.nv))

        # Robot at the origin, identity attitude, tumbling + drifting.
        qpos[:, self._bus_q + 3 : self._bus_q + 7] = [1.0, 0.0, 0.0, 0.0]
        qvel[:, self._bus_v : self._bus_v + 3] = self._rng.uniform(
            -cfg.init_drift_max, cfg.init_drift_max, (n, 3)
        )
        qvel[:, self._bus_v + 3 : self._bus_v + 6] = self._rng.uniform(
            -cfg.init_spin_max, cfg.init_spin_max, (n, 3)
        )
        arm = self._arm_ready[None, :] + self._rng.uniform(
            -cfg.init_arm_noise, cfg.init_arm_noise, (n, 2)
        )
        arm = np.clip(arm, self._arm_range[:, 0], self._arm_range[:, 1])
        qpos[:, self._arm_jq] = arm
        qpos[:, self._grip_u_jq] = _GRIP_OPEN
        qpos[:, self._grip_l_jq] = -_GRIP_OPEN

        # Cargo floating ahead (within a forward cone), oriented so its grapple
        # bar (cargo -x) faces the robot. Calm: near-zero velocity.
        dist = self._rng.uniform(cfg.cargo_dist[0], cfg.cargo_dist[1], n)
        lat = self._rng.uniform(-cfg.cargo_lateral, cfg.cargo_lateral, (n, 2))
        pos = np.stack([dist, lat[:, 0], lat[:, 1]], axis=1)
        qpos[:, self._cargo_q : self._cargo_q + 3] = pos
        qpos[:, self._cargo_q + 3 : self._cargo_q + 7] = self._aim_quat(pos, cfg.cargo_att_noise, n)

        self.data.qpos[idx] = qpos
        self.data.qvel[idx] = qvel
        self.data.ctrl[idx] = 0.0
        self.data.ctrl[np.ix_(idx, self._arm_ctrl)] = arm
        self.data.ctrl[idx, self._grip_u_ctrl] = _GRIP_OPEN
        self.data.ctrl[idx, self._grip_l_ctrl] = -_GRIP_OPEN
        self.data.qacc_warmstart[idx] = 0.0
        self._grasped[idx] = False
        self._grip_closed[idx] = False
        self._last_actions[idx] = 0.0
        self.episode_length_buf[torch.as_tensor(idx, device=self.device)] = 0

        mjo_upload(self.model, self.data, fields=("qpos", "qvel", "ctrl", "qacc_warmstart"))
        mjo_forward(self.model, self.data)

    def _aim_quat(self, pos: np.ndarray, att_noise: float, n: int) -> np.ndarray:
        """Quaternion orienting the cargo +x along `pos` (so its -x grapple bar
        points back at the robot at the origin), plus small attitude noise."""
        x = pos / np.linalg.norm(pos, axis=1, keepdims=True)
        ref = np.tile([0.0, 0.0, 1.0], (n, 1))
        # Guard against x ~ +/-z.
        bad = np.abs(x[:, 2]) > 0.95
        ref[bad] = [0.0, 1.0, 0.0]
        y = np.cross(ref, x)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        z = np.cross(x, y)
        rot = np.stack([x, y, z], axis=2)  # columns are the body axes in world
        quat = _mat_to_quat(rot)
        if att_noise > 0.0:
            dq = self._rng.normal(0.0, att_noise, (n, 3))
            quat = _quat_mul(quat, _axis_to_quat(dq))
        return quat / np.linalg.norm(quat, axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Geometry: gripper and grapple bar
    # ------------------------------------------------------------------

    def _body_rot(self, bid: int) -> np.ndarray:
        return np.asarray(self.data.xmat)[:, bid].reshape(self.num_envs, 3, 3)

    def _gripper_center(self) -> tuple[np.ndarray, np.ndarray]:
        xpos = np.asarray(self.data.xpos)
        r_ee = self._body_rot(self._ee_bid)
        center = xpos[:, self._ee_bid] + r_ee[:, :, 0] * _JAW_REACH
        return center, r_ee[:, :, 1]  # gripper center, jaw axis (ee +y, world)

    def _bar_state(self) -> tuple[np.ndarray, np.ndarray]:
        xpos = np.asarray(self.data.xpos)
        r_cargo = self._body_rot(self._cargo_bid)
        bar_off = np.tile(_BAR_OFFSET, (self.num_envs, 1))
        center = xpos[:, self._cargo_bid] + _mat_vec(r_cargo, bar_off)
        return center, r_cargo[:, :, 1]  # bar center, bar axis (cargo +y, world)

    def _grip_bar_dist(self) -> np.ndarray:
        gc, _ = self._gripper_center()
        bc, _ = self._bar_state()
        return np.linalg.norm(gc - bc, axis=1)

    # ------------------------------------------------------------------
    # Observations and reward
    # ------------------------------------------------------------------

    def _compute_obs(self) -> TensorDict:
        r_bus = self._body_rot(self._bus_bid)
        r_ee = self._body_rot(self._ee_bid)
        qvel = self.data.qvel
        gc, _ = self._gripper_center()
        bc, bx = self._bar_state()

        bus_w = qvel[:, self._bus_v + 3 : self._bus_v + 6]  # body-frame angular rate
        bus_v = qvel[:, self._bus_v : self._bus_v + 3]
        cargo_v = qvel[:, self._cargo_v : self._cargo_v + 3]
        cargo_w = qvel[:, self._cargo_v + 3 : self._cargo_v + 6]

        obs = np.concatenate(
            [
                self.data.qpos[:, self._arm_jq] / 2.4,
                qvel[:, self._arm_jv] / 3.0,
                bus_w / 0.5,
                _mat_t_vec(r_bus, bus_v) / 0.2,
                _mat_t_vec(r_bus, bc - gc) / 0.5,  # gripper->bar vector (bus frame)
                _mat_t_vec(r_bus, bx),  # bar axis (bus frame)
                _mat_t_vec(r_bus, r_ee[:, :, 1]),  # jaw axis (ee +y, bus frame)
                _mat_t_vec(r_bus, r_ee[:, :, 0]),  # approach axis (ee +x, bus frame)
                _mat_t_vec(r_bus, cargo_v) / 0.2,
                _mat_t_vec(r_bus, cargo_w) / 0.5,
                self._grasped[:, None].astype(np.float64),
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
        gc, jaw_ax = self._gripper_center()  # jaw axis = ee +y
        bc, bar_ax = self._bar_state()  # bar axis = cargo +y

        dist = np.linalg.norm(gc - bc, axis=1)
        align = np.abs(np.sum(jaw_ax * bar_ax, axis=1)).clip(0.0, 1.0)
        spin = np.linalg.norm(qvel[:, self._bus_v + 3 : self._bus_v + 6], axis=1)

        # A grasp = the policy commanded the grip closed, with the gripper on the
        # bar and the jaws aligned to it (binary grip action -> no auto-latch).
        self._grasped = (
            self._grip_closed & (dist < cfg.capture_radius) & (align > cfg.capture_align)
        )

        progress = np.clip(self._prev_dist - dist, -0.05, 0.05)
        self._prev_dist = dist.copy()

        gate = np.exp(-((spin / cfg.gate_width) ** 2))
        rel_vel = np.linalg.norm(
            qvel[:, self._cargo_v : self._cargo_v + 3] - qvel[:, self._bus_v : self._bus_v + 3],
            axis=1,
        )
        bus_speed = np.linalg.norm(qvel[:, self._bus_v : self._bus_v + 3], axis=1)

        settle = cfg.w_settle * np.exp(-((spin / cfg.settle_width) ** 2))
        reach = cfg.w_reach * np.exp(-((dist / cfg.reach_width) ** 2)) + cfg.w_reach_tight * np.exp(
            -((dist / cfg.reach_tight_width) ** 2)
        )  # (CHANGE 3) wide pull + steep pull into the last few cm
        approach = gate * (
            cfg.w_progress * progress
            + reach
            + cfg.w_align * align * np.exp(-((dist / cfg.reach_width) ** 2))
        )
        dock_vel_pen = cfg.w_dock_vel * np.exp(-((dist / cfg.dock_width) ** 2)) * np.minimum(
            rel_vel**2 / 0.04, 25.0
        )
        # Once grasped, reward holding STILL (low absolute speed) and penalize
        # drift, so the robot brakes after the catch and station-keeps instead of
        # coasting off (otherwise the joined assembly drifts away / toward Earth).
        grasp = self._grasped * (
            cfg.w_grasp + cfg.w_hold * np.exp(-((bus_speed / cfg.hold_width) ** 2))
        )
        brake_pen = cfg.w_brake * self._grasped * np.minimum(bus_speed**2, 4.0)
        # Discourage closing the grip away from the bar (flailing the gripper).
        grip_pen = cfg.w_grip_pen * (self._grip_closed & (dist > cfg.capture_radius))

        effort = np.mean(actions[:, 2:8] ** 2, axis=1)
        arm_rate = np.mean((actions[:, :2] - self._last_actions[:, :2]) ** 2, axis=1)

        reward = (
            settle
            + approach
            + grasp
            - cfg.w_spin * np.minimum(spin**2, 4.0)
            - dock_vel_pen
            - brake_pen
            - grip_pen
            - cfg.w_effort * effort
            - cfg.w_arm_rate * arm_rate
        )

        failed = dist > cfg.fail_radius
        reward = np.where(failed, reward - cfg.fail_penalty, reward)

        metrics = {
            "grip_bar_dist": float(np.nanmean(dist)),
            "grasped_frac": float(np.mean(self._grasped)),
            "grip_closed_frac": float(np.mean(self._grip_closed)),
            "spin": float(np.nanmean(spin)),
            "align": float(np.nanmean(align)),
            "rel_vel": float(np.nanmean(rel_vel)),
            "grasped_speed": float(
                np.nansum(bus_speed * self._grasped) / max(self._grasped.sum(), 1)
            ),
            "failed_frac": float(np.mean(failed)),
        }
        return reward, metrics, failed

    # Convenience for play.py / recording.
    def grip_bar_dist(self) -> np.ndarray:
        return self._grip_bar_dist()


# ----------------------------------------------------------------------
# Batched rotation helpers
# ----------------------------------------------------------------------


def _mat_vec(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nj->ni", r, v)


def _mat_t_vec(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum("nji,nj->ni", r, v)


def _mat_to_quat(rot: np.ndarray) -> np.ndarray:
    """Batched rotation-matrix -> wxyz quaternion."""
    n = rot.shape[0]
    q = np.zeros((n, 4))
    t = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]
    pos = t > 0.0
    s = np.where(pos, np.sqrt(np.maximum(t + 1.0, 1e-12)) * 2.0, 1.0)
    q[pos, 0] = 0.25 * s[pos]
    q[pos, 1] = (rot[pos, 2, 1] - rot[pos, 1, 2]) / s[pos]
    q[pos, 2] = (rot[pos, 0, 2] - rot[pos, 2, 0]) / s[pos]
    q[pos, 3] = (rot[pos, 1, 0] - rot[pos, 0, 1]) / s[pos]
    for i in np.flatnonzero(~pos):
        m = rot[i]
        d = np.array([m[0, 0], m[1, 1], m[2, 2]])
        k = int(np.argmax(d))
        ii, jj, kk = (k + 1) % 3, (k + 2) % 3, k
        sk = np.sqrt(max(m[kk, kk] - m[ii, ii] - m[jj, jj] + 1.0, 1e-12)) * 2.0
        qv = np.zeros(3)
        qv[kk] = 0.25 * sk
        qv[ii] = (m[ii, kk] + m[kk, ii]) / sk
        qv[jj] = (m[jj, kk] + m[kk, jj]) / sk
        q[i, 0] = (m[jj, ii] - m[ii, jj]) / sk
        q[i, 1:] = qv
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _axis_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """Batched rotation-vector (axis*angle) -> wxyz quaternion."""
    ang = np.linalg.norm(rotvec, axis=1, keepdims=True)
    small = ang < 1e-9
    axis = np.where(small, np.array([1.0, 0.0, 0.0]), rotvec / np.where(small, 1.0, ang))
    half = 0.5 * ang
    return np.concatenate([np.cos(half), axis * np.sin(half)], axis=1)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=1,
    )
