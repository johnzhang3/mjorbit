"""Vectorized capture-and-reorient environment (mjorbit_warp) with SGS resets.

The teaser/banner bi-manual free-flyer (``capture_reorient.xml``) must

1. **capture** a free-floating client satellite by its grapple bar -- both
   two-jaw hands closed on the bar, held by contact + friction only (no weld,
   no equality) -- and
2. **reorient** the captured stack with its reaction wheels until the client's
   antenna boresight (client +Z) points at nadir, holding it there.

Reward follows Zhang et al., arXiv 2610.12465, Eq. 5 / App. A.3.2 (the shared
manipulation reward): a binary task-success term, a terminal failure penalty,
and three small clamped per-step regularizers (action magnitude, action rate,
arm-joint velocity), with the paper's weights. There is no shaping: no
distance, alignment, or progress terms.

Resets come from a fixed set of task configurations (App. A.2 / Sec. 3.2)
balanced across three reset families, in the spirit of the paper's OmniReset
strategies:

* ``held``      -- client already clamped in both hands (Stable-Grasp / Near-Goal);
* ``pregrasp``  -- client floating with its bar inside the open jaws (Reaching);
* ``approach``  -- client drifting and tumbling 0.05-1.0 m away (graded: a
  per-configuration difficulty scales distance, misalignment, drift and tumble
  together); the servicer must fly to it with its RCS thrusters, align, and
  close the jaws.

Every family also draws the initial nadir-pointing error uniformly in
[0, 180] deg, so configurations span near-goal to full-slew difficulty.
:class:`sgs.SuccessGuidedSampler` picks the configuration for every reset
(``sampler="sgs"``) or draws uniformly from the same set (``"uniform"``).

The spacecraft is actuated through mjorbit's native actuator models: 3 body-axis
reaction wheels (``rw_torque_cmd``, torque/speed limited, momentum stored in the
wheels) and 6 unidirectional RCS thrusters (``thr_force_cmd``). Gravity
gradient, differential gravity, J2, and drag are on; the chief orbit keeps
flying across resets and each configuration's attitude is defined relative to
the LVLH frame at reset time.

Implements the ``rsl_rl.env.VecEnv`` interface (rsl-rl-lib >= 5.x, TensorDict
observations). The 10 physics substeps per control step run as one captured
CUDA graph.
"""

from __future__ import annotations

import csv
import dataclasses
import itertools
from pathlib import Path

import mujoco as mj
import mujoco_warp as mjw

# The native bindings must load before torch so they resolve the pixi env's
# libstdc++ rather than the older system library the torch wheel pulls in.
import mjorbit  # noqa: F401

# isort: split

import numpy as np
import torch
import warp as wp
from rsl_rl.env import VecEnv
from sgs import SuccessGuidedSampler
from tensordict import TensorDict

from mjorbit.constants import GM_EARTH, R_EARTH
from mjorbit_warp import MjoModel, OrbitInit, SurfaceSpec, mjo_pull, mjo_step, mjo_upload

XML_PATH = Path(__file__).with_name("capture_reorient.xml")

_ARM_ACT = ("shoulder_a_pos", "elbow_a_pos", "shoulder_b_pos", "elbow_b_pos")
_WRIST_ACT = ("wrist_a_pos", "wrist_b_pos")
_GRIP_ACT = ("grip_a_pos", "grip_b_pos")
_HANDS = ("ee_a", "ee_b")
_ELBOWS = ("link2_a", "link2_b")

# Upper-jaw slide positions: open, commanded closed (past contact, so the servo
# squeezes), and where it rests on a clamped bar.
_GRIP_OPEN = 0.12
_GRIP_CLOSED = -0.03
_GRIP_TOUCH = 0.01
# Hold pose: wrists at (+-0.5, 0, 1.0) in the bus frame (elbows outboard), jaws
# level. A clamped bar rests on the lower jaws with its center 1 cm above the
# wrists.
_EE_HOLD = {"ee_a": np.array([0.5, 0.0, 1.0]), "ee_b": np.array([-0.5, 0.0, 1.0])}
_BAR_HOLD = np.array([0.0, 0.0, 1.01])
# Jaw-pair midpoint in each (leveled) hand frame; the jaws point inboard (arm A
# along -x, arm B along +x).
_JAW_CENTER = {"ee_a": np.array([-0.12, 0.0, 0.01]), "ee_b": np.array([0.12, 0.0, 0.01])}
_BAR_HALF_LEN = 0.42
_CLIENT_BODY_OFFSET = np.array([0.0, 0.0, 0.62])  # client box center, client frame
_BUS_HALF = np.array([0.5, 0.3, 0.3])

# Match the XML <reaction_wheel torque_limit> / <thruster force_limit>.
_RW_LIMIT = 5.0
_THR_LIMIT = 25.0

HELD, PREGRASP, APPROACH = 0, 1, 2
FAMILIES = ("held", "pregrasp", "approach")

# Flat faces for the drag model (bus and client boxes).
_SURFACES = tuple(
    SurfaceSpec(body, list(c + s * h * e), list(s * e), area, name=f"{body}_{'xyz'[k]}{'pn'[j]}")
    for body, c, half in (
        ("bus", np.zeros(3), _BUS_HALF),
        ("client", _CLIENT_BODY_OFFSET, np.array([0.4, 0.4, 0.3])),
    )
    for k, e in enumerate(np.eye(3))
    for j, s in enumerate((1.0, -1.0))
    for h, area in [(half[k], 4.0 * np.prod(np.delete(half, k)))]
)


@dataclasses.dataclass
class CaptureEnvCfg:
    """Configuration for :class:`CaptureReorientEnv`."""

    num_envs: int = 4096
    episode_length_s: float = 60.0
    control_dt: float = 0.1
    physics_dt: float = 0.01
    altitude_km: float = 500.0

    # ---- Task configurations + sampler (paper Sec. 3.2, manipulation values).
    num_configs: int = 8192
    config_seed: int = 0
    sampler: str = "sgs"  # "sgs" | "uniform"
    sgs_history: int = 100
    sgs_target: float = 0.5
    sgs_kappa: float = 1.0
    sgs_temperature: float = 2.0
    sgs_eps: float = 1e-4

    # ---- Reset distribution.
    max_point_err_deg: float = 180.0  # initial nadir-pointing error ~ U[0, max]
    bus_rate_max: float = 0.005  # servicer residual body rate per axis (rad/s)
    bus_vel_max: float = 0.005  # servicer residual drift per axis (m/s)
    held_arm_noise: float = 0.0
    pre_offset: tuple[float, float, float] = (0.03, 0.03, 0.08)  # bar offset +-x, +-y, z in [0, z]
    pre_att_deg: float = 4.0
    pre_vel: float = 0.01
    pre_rate_deg: float = 0.5
    pre_arm_noise: float = 0.03
    # Graded approach resets: each approach configuration draws a difficulty
    # u ~ U[0, 1] that scales its start distance across app_dist and its
    # misalignment / drift / tumble / arm offsets from the pregrasp levels up to
    # the app_* maxima, so the family spans pregrasp-like to far-and-tumbling.
    # False: every approach parameter is drawn independently over its full range
    # (the first, gap-limited version of this task).
    app_graded: bool = True
    app_dist: tuple[float, float] = (0.05, 1.0)
    app_yaw_deg: float = 30.0
    app_tilt_deg: float = 15.0
    app_vel: float = 0.02
    app_rate_deg: float = 1.5
    app_arm_noise: float = 0.3

    # ---- Binary success: both hands clamped on the bar (grip closed, jaws on
    # the bar, upper jaw squeezed down) and the client antenna within point_tol
    # of nadir, continuously for hold_s.
    point_tol_deg: float = 10.0
    hold_s: float = 3.0
    hold_dist: float = 0.045  # jaw-center to bar-axis distance (m)
    hold_grip_max: float = 0.04  # upper jaw must be down on the bar

    # ---- Failure (terminal penalty): client lost, client box through the bus,
    # abnormal velocities, or a non-finite state.
    lost_dist: float = 4.0
    collide_dist: float = 0.3
    max_joint_vel: float = 6.0  # policy-driven arm joints
    max_wrist_vel: float = 20.0  # auto-leveled wrists follow arm targets quickly
    max_bus_rate: float = 0.5
    max_client_rate: float = 1.0

    # ---- Reward (paper Eq. 5, App. A.3.2 configured weights).
    w_succ: float = 100.0
    w_fail: float = 1.0
    w_act: float = 1e-5
    w_act_rate: float = 1e-4
    w_joint_vel: float = 1e-4
    reg_clamp: float = 1e4

    # ---- Action interface. Arm actions are joint-target increments (the paper's
    # manipulation policies also act through increments), so exploration noise
    # random-walks the targets instead of jerking the arms.
    arm_rate: float = 0.5  # rad/s of target motion at |a| = 1
    arm_span: float = 1.0  # targets stay within +-arm_span of the hold pose
    # Latched gripper: close as soon as a > +h; release only after
    # grip_release_steps consecutive a < -h commands (a release interlock, so
    # exploration noise does not drop a captured client); otherwise keep.
    grip_threshold: float = 0.8
    grip_release_steps: int = 3

    use_gravity_gradient: bool = True
    use_drag: bool = True
    use_j2: bool = True
    use_cuda_graph: bool = True
    auto_reset: bool = True
    nconmax: int = 64
    njmax: int = 192

    # Logging: one CSV row every steps_per_log control steps (= one PPO iteration).
    steps_per_log: int = 32
    log_dir: str | None = None
    sampler_snapshot_every: int = 100  # iterations between sampler-state dumps

    device: str = "cuda"
    seed: int = 0


class CaptureReorientEnv(VecEnv):
    """Capture a free-floating client with two hands and point it at nadir."""

    def __init__(self, cfg: CaptureEnvCfg) -> None:
        if cfg.num_envs < 2:
            raise ValueError("num_envs must be >= 2 (nworld=1 uses unbatched buffers)")
        decimation = cfg.control_dt / cfg.physics_dt
        if abs(decimation - round(decimation)) > 1e-9:
            raise ValueError("control_dt must be an integer multiple of physics_dt")

        self.cfg = cfg
        self.device = cfg.device
        self.num_envs = cfg.num_envs
        self.num_actions = 11  # 4 arm-target increments + 1 grip + 3 rw + 3 thrust
        self.max_episode_length = int(round(cfg.episode_length_s / cfg.control_dt))
        self.episode_length_buf = torch.zeros(cfg.num_envs, dtype=torch.long, device=cfg.device)
        self._decimation = int(round(decimation))
        self._hold_steps = int(round(cfg.hold_s / cfg.control_dt))
        self._rng = np.random.default_rng(cfg.seed)

        self.model = MjoModel.from_xml_path(
            str(XML_PATH),
            surfaces=_SURFACES,
            mj_timestep=cfg.physics_dt,
            use_j2=cfg.use_j2,
            use_drag=cfg.use_drag,
            use_gravity_gradient=cfg.use_gravity_gradient,
        )
        self.data = self.model.make_data(
            orbit=self._sample_orbits(cfg.num_envs),
            nworld=cfg.num_envs,
            nconmax=cfg.nconmax,
            njmax=cfg.njmax,
        )

        m = self.model.mj_model
        jq = lambda name: int(m.jnt_qposadr[m.joint(name).id])  # noqa: E731
        jv = lambda name: int(m.jnt_dofadr[m.joint(name).id])  # noqa: E731
        self._arm_ctrl = np.array([m.actuator(n).id for n in _ARM_ACT])
        self._wrist_ctrl = np.array([m.actuator(n).id for n in _WRIST_ACT])
        self._grip_ctrl = np.array([m.actuator(n).id for n in _GRIP_ACT])
        arm_j = [m.actuator(n).trnid[0] for n in _ARM_ACT]
        self._arm_jq = np.array([m.jnt_qposadr[j] for j in arm_j])
        self._arm_jv = np.array([m.jnt_dofadr[j] for j in arm_j])
        self._arm_range = m.jnt_range[arm_j].copy()
        self._wrist_jq = np.array([jq("wrist_a"), jq("wrist_b")])
        self._wrist_jv = np.array([jv("wrist_a"), jv("wrist_b")])
        self._grip_jq = np.array([jq("grip_a"), jq("grip_b")])
        self._joint_jv = np.concatenate([self._arm_jv, self._wrist_jv])
        self._bus_q, self._bus_v = jq("base"), jv("base")
        self._cl_q, self._cl_v = jq("client_free"), jv("client_free")
        self._bus_bid = int(m.body("bus").id)
        self._cl_bid = int(m.body("client").id)
        self._ee_bid = np.array([m.body(n).id for n in _HANDS])
        self._jaw_center = np.stack([_JAW_CENTER[n] for n in _HANDS])
        self._qpos0 = m.qpos0.copy()
        self._arm_hold = self._solve_hold_pose(m)

        self.configs = self._collision_checked_configs(cfg.num_configs, cfg.config_seed)
        self.sampler = SuccessGuidedSampler(
            len(self.configs["family"]),
            mode=cfg.sampler,
            history=cfg.sgs_history,
            target=cfg.sgs_target,
            kappa=cfg.sgs_kappa,
            temperature=cfg.sgs_temperature,
            eps=cfg.sgs_eps,
            seed=cfg.seed + 1,
        )

        n = cfg.num_envs
        self.env_config = np.zeros(n, dtype=np.int64)
        self._grip_closed = np.zeros(n, dtype=bool)
        self._release_count = np.zeros(n, dtype=np.int64)
        self._arm_target = np.tile(self._arm_hold, (n, 1))
        self._hold_count = np.zeros(n, dtype=np.int64)
        self._last_actions = np.zeros((n, self.num_actions))
        self.last_success = np.zeros(n, dtype=bool)
        self.last_fail = np.zeros(n, dtype=bool)
        self.last_timeout = np.zeros(n, dtype=bool)
        self.last_held = np.zeros(n, dtype=bool)
        self.last_point_err = np.zeros(n)
        self._global_step = 0
        self._win = _Window()
        self._csv_path = None
        if cfg.log_dir is not None:
            Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
            self._csv_path = Path(cfg.log_dir) / "train_metrics.csv"

        self._graph = None
        self._reset_idx(np.arange(n), self.sampler.sample(n))
        if cfg.use_cuda_graph:
            # Warm up (first launches compile/allocate), capture the substep
            # loop once, then start from fresh resets.
            self._advance()
            with wp.ScopedCapture() as capture:
                for _ in range(self._decimation):
                    mjo_step(self.model, self.data)
            self._graph = capture.graph
            self._pull()
            self._reset_idx(np.arange(n), self.sampler.sample(n))
        self._pull()
        self._obs = self._compute_obs(self._geometry())

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    def get_observations(self) -> TensorDict:
        return self._obs

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        cfg = self.cfg
        acts = actions.detach().to("cpu", torch.float64).clamp(-1.0, 1.0).numpy()
        self._apply_ctrl(acts)
        self._advance()
        self._pull()
        self.episode_length_buf += 1

        geo = self._geometry()
        success, fail = self._success_fail(geo)
        joint_vel = np.nan_to_num(
            self.data.qvel[:, self._joint_jv], nan=1e3, posinf=1e3, neginf=1e3
        )
        c = cfg.reg_clamp
        reward = (
            cfg.w_succ * success
            - cfg.w_fail * fail
            - cfg.w_act * np.minimum(np.sum(acts**2, axis=1), c)
            - cfg.w_act_rate * np.minimum(np.sum((acts - self._last_actions) ** 2, axis=1), c)
            - cfg.w_joint_vel * np.minimum(np.sum(joint_vel**2, axis=1), c)
        )
        time_out = (self.episode_length_buf >= self.max_episode_length).cpu().numpy()
        time_out &= ~(success | fail)
        dones = success | fail | time_out
        self._last_actions = acts
        self.last_success, self.last_fail, self.last_timeout = success, fail, time_out

        done_idx = np.flatnonzero(dones)
        if done_idx.size:
            self._win.record(
                self.configs["family"][self.env_config[done_idx]],
                success[done_idx],
                fail[done_idx],
                self.episode_length_buf[torch.as_tensor(done_idx, device=self.device)]
                .cpu()
                .numpy(),
                {k: v[done_idx] for k, v in self._fail_reasons.items()},
            )
            if cfg.auto_reset:
                self.sampler.update(self.env_config[done_idx], success[done_idx])
                self._reset_idx(done_idx, self.sampler.sample(done_idx.size))
                self._pull(post_reset=True)
                geo = self._geometry()

        self._obs = self._compute_obs(geo)
        extras = {
            "time_outs": torch.as_tensor(time_out, dtype=torch.bool, device=self.device),
            "log": self._maybe_flush_log(),
        }
        rewards_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        return self._obs, rewards_t, dones_t, extras

    def reset_to_configs(self, config_idx: np.ndarray) -> TensorDict:
        """Reset every world to the given configuration indices (evaluation)."""
        self._reset_idx(np.arange(self.num_envs), np.asarray(config_idx, dtype=np.int64))
        self._pull(post_reset=True)
        self._obs = self._compute_obs(self._geometry())
        return self._obs

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def _apply_ctrl(self, acts: np.ndarray) -> None:
        cfg = self.cfg
        lo = np.maximum(self._arm_hold - cfg.arm_span, self._arm_range[:, 0])
        hi = np.minimum(self._arm_hold + cfg.arm_span, self._arm_range[:, 1])
        self._arm_target = np.clip(
            self._arm_target + cfg.arm_rate * cfg.control_dt * acts[:, :4], lo, hi
        )
        arm = self._arm_target
        wrist = np.clip(
            -np.stack([arm[:, 0] + arm[:, 1], arm[:, 2] + arm[:, 3]], axis=1), -3.14, 3.14
        )
        g = acts[:, 4]
        self._release_count = np.where(g < -cfg.grip_threshold, self._release_count + 1, 0)
        self._grip_closed = np.where(
            g > cfg.grip_threshold,
            True,
            np.where(self._release_count >= cfg.grip_release_steps, False, self._grip_closed),
        )
        self.data.ctrl[:, self._arm_ctrl] = arm
        self.data.ctrl[:, self._wrist_ctrl] = wrist
        self.data.ctrl[:, self._grip_ctrl] = np.where(self._grip_closed, _GRIP_CLOSED, _GRIP_OPEN)[
            :, None
        ]
        # Wheel command torque spins the rotor; the bus feels -cmd. Map a > 0 to a
        # positive body torque about that axis.
        self.data.actuators.rw_torque_cmd[:] = -acts[:, 5:8] * _RW_LIMIT
        f = acts[:, 8:11] * _THR_LIMIT  # body-frame force request
        self.data.actuators.thr_force_cmd[:] = np.concatenate(
            [np.maximum(f, 0.0), np.maximum(-f, 0.0)], axis=1
        )

    def _advance(self) -> None:
        mjo_upload(self.model, self.data, fields=("ctrl", "rw_torque_cmd", "thr_force_cmd"))
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            for _ in range(self._decimation):
                mjo_step(self.model, self.data)

    def _pull(self, post_reset: bool = False) -> None:
        # After a reset only the kinematics changed on device; the host already
        # holds the qpos/qvel/rw_speed it uploaded.
        fields = (
            ("xpos", "xmat")
            if post_reset
            else ("qpos", "qvel", "xpos", "xmat", "actuators", "orbit", "qacc_warmstart")
        )
        mjo_pull(self.model, self.data, fields=fields)

    # ------------------------------------------------------------------
    # Resets
    # ------------------------------------------------------------------

    def _solve_hold_pose(self, m) -> np.ndarray:
        """Planar IK for the hold pose: each wrist at _EE_HOLD with the elbow
        bent outboard (clear of the bus and the client), coarse grid then a
        fine local grid."""
        d = mj.MjData(m)
        names = {
            "ee_a": ("shoulder_a", "elbow_a", "link2_a", 1.0),
            "ee_b": ("shoulder_b", "elbow_b", "link2_b", -1.0),
        }
        pose = {}
        for ee, (sh, el, elbow_body, side) in names.items():
            sh_q = int(m.jnt_qposadr[m.joint(sh).id])
            el_q = int(m.jnt_qposadr[m.joint(el).id])
            bid, ebid = int(m.body(ee).id), int(m.body(elbow_body).id)

            def fk(s: float, e: float) -> tuple[float, float]:
                d.qpos[:] = m.qpos0
                d.qpos[sh_q], d.qpos[el_q] = s, e
                mj.mj_kinematics(m, d)
                return float(np.linalg.norm(d.xpos[bid] - _EE_HOLD[ee])), side * float(
                    d.xpos[ebid][0]
                )

            coarse = [
                (s, e, *fk(s, e))
                for s, e in itertools.product(np.linspace(-3.1, 3.1, 125), repeat=2)
            ]
            ok = [c for c in coarse if c[2] < 0.06]
            branch = np.sign(max(ok, key=lambda c: c[3])[1])  # elbow sign of the outboard branch
            s0, e0 = min((c for c in ok if np.sign(c[1]) == branch), key=lambda c: c[2])[:2]
            fine = np.linspace(-0.06, 0.06, 121)
            s1, e1 = min(
                ((s0 + a, e0 + b) for a, b in itertools.product(fine, fine)),
                key=lambda se: fk(*se)[0],
            )
            pose[sh], pose[el] = s1, e1
        return np.array([pose["shoulder_a"], pose["elbow_a"], pose["shoulder_b"], pose["elbow_b"]])

    def _collision_checked_configs(self, n: int, seed: int) -> dict[str, np.ndarray]:
        """Draw ``n`` task configurations and redraw (within their family) any whose
        initial servicer/client geometry interpenetrates, like the paper's
        collision-checked reset buffer (App. A.2). Collision is relative, so it
        is checked with the bus at the origin and identity attitude."""
        rng = np.random.default_rng(seed)
        configs = sample_task_configs(self.cfg, rng.permutation(np.arange(n) % 3), rng)
        for _ in range(100):
            bad = self._penetrating(configs)
            if not bad.any():
                return configs
            fresh = sample_task_configs(self.cfg, configs["family"][bad], rng)
            for key, value in fresh.items():
                configs[key][bad] = value
        raise RuntimeError("could not draw collision-free task configurations")

    def _penetrating(self, configs: dict[str, np.ndarray], tol: float = 0.003) -> np.ndarray:
        m = self.model.mj_model
        d = mj.MjData(m)
        bad = np.zeros(len(configs["family"]), dtype=bool)
        for i in range(bad.size):
            held = configs["family"][i] == HELD
            d.qpos[:] = self._qpos0
            d.qpos[self._bus_q : self._bus_q + 7] = [0, 0, 0, 1, 0, 0, 0]
            d.qpos[self._cl_q : self._cl_q + 3] = _BAR_HOLD + configs["client_dp"][i]
            d.qpos[self._cl_q + 3 : self._cl_q + 7] = _mat_to_quat(configs["client_rot"][i][None])[
                0
            ]
            arm = np.clip(
                self._arm_hold + configs["arm_dq"][i], self._arm_range[:, 0], self._arm_range[:, 1]
            )
            d.qpos[self._arm_jq] = arm
            d.qpos[self._wrist_jq] = [-(arm[0] + arm[1]), -(arm[2] + arm[3])]
            d.qpos[self._grip_jq] = _GRIP_TOUCH if held else _GRIP_OPEN
            mj.mj_fwdPosition(m, d)
            bad[i] = any(d.contact[j].dist < -tol for j in range(d.ncon))
        return bad

    def _sample_orbits(self, n: int) -> list[OrbitInit]:
        """Circular orbits with random planes and phases."""
        radius = R_EARTH + self.cfg.altitude_km
        speed = float(np.sqrt(GM_EARTH / radius))
        u = _unit(self._rng.normal(size=(n, 3)))
        w = _unit(np.cross(u, _unit(self._rng.normal(size=(n, 3)))))
        return [OrbitInit(R_eci=radius * u[i], V_eci=speed * w[i]) for i in range(n)]

    def _lvlh(self, idx: np.ndarray) -> np.ndarray:
        """World-from-LVLH rotation (columns: along-track-ish x, -orbit normal y,
        nadir z) for the given worlds, from the current chief orbit."""
        r = np.asarray(self.data.orbit.R_eci, dtype=float)[idx]
        v = np.asarray(self.data.orbit.V_eci, dtype=float)[idx]
        z = -_unit(r)
        y = -_unit(np.cross(r, v))
        x = np.cross(y, z)
        return np.stack([x, y, z], axis=2)

    def _reset_idx(self, idx: np.ndarray, cfg_idx: np.ndarray) -> None:
        c = self.configs
        k = np.asarray(cfg_idx, dtype=np.int64)
        n = idx.size
        fam = c["family"][k]
        held = fam == HELD

        lvlh = self._lvlh(idx)
        r_bus = lvlh @ c["bus_rot"][k]
        w_bus = _mat_vec(r_bus, c["bus_w"][k])  # world-frame rate
        v_bus = _mat_vec(lvlh, c["bus_v"][k])
        p_rel = _BAR_HOLD[None, :] + c["client_dp"][k]  # bus frame
        r_cl = r_bus @ c["client_rot"][k]
        p_cl = _mat_vec(r_bus, p_rel)
        w_cl = w_bus + _mat_vec(r_bus, c["client_w"][k])
        v_cl = v_bus + np.cross(w_bus, p_cl) + _mat_vec(r_bus, c["client_v"][k])

        qpos = np.tile(self._qpos0, (n, 1))
        qvel = np.zeros((n, self.model.mj_model.nv))
        qpos[:, self._bus_q : self._bus_q + 3] = 0.0
        qpos[:, self._bus_q + 3 : self._bus_q + 7] = _mat_to_quat(r_bus)
        qvel[:, self._bus_v : self._bus_v + 3] = v_bus
        qvel[:, self._bus_v + 3 : self._bus_v + 6] = c["bus_w"][k]
        qpos[:, self._cl_q : self._cl_q + 3] = p_cl
        qpos[:, self._cl_q + 3 : self._cl_q + 7] = _mat_to_quat(r_cl)
        qvel[:, self._cl_v : self._cl_v + 3] = v_cl
        qvel[:, self._cl_v + 3 : self._cl_v + 6] = _mat_t_vec(r_cl, w_cl)
        arm = np.clip(
            self._arm_hold[None, :] + c["arm_dq"][k], self._arm_range[:, 0], self._arm_range[:, 1]
        )
        wrist = -np.stack([arm[:, 0] + arm[:, 1], arm[:, 2] + arm[:, 3]], axis=1)
        qpos[:, self._arm_jq] = arm
        qpos[:, self._wrist_jq] = wrist
        qpos[:, self._grip_jq] = np.where(held, _GRIP_TOUCH, _GRIP_OPEN)[:, None]

        self.data.qpos[idx] = qpos
        self.data.qvel[idx] = qvel
        self.data.ctrl[idx] = 0.0
        self.data.ctrl[np.ix_(idx, self._arm_ctrl)] = arm
        self.data.ctrl[np.ix_(idx, self._wrist_ctrl)] = wrist
        self.data.ctrl[np.ix_(idx, self._grip_ctrl)] = np.where(held, _GRIP_CLOSED, _GRIP_OPEN)[
            :, None
        ]
        self.data.qacc_warmstart[idx] = 0.0
        self.data.actuators.rw_speed[idx] = 0.0
        self.data.actuators.rw_torque_cmd[idx] = 0.0
        self.data.actuators.thr_force_cmd[idx] = 0.0

        self.env_config[idx] = k
        self._arm_target[idx] = arm
        self._grip_closed[idx] = held
        self._release_count[idx] = 0
        self._hold_count[idx] = 0
        self._last_actions[idx] = 0.0
        self._last_actions[idx, 4] = np.where(held, 1.0, -1.0)
        self.episode_length_buf[torch.as_tensor(idx, device=self.device)] = 0

        mjo_upload(
            self.model,
            self.data,
            fields=(
                "qpos",
                "qvel",
                "ctrl",
                "qacc_warmstart",
                "rw_speed",
                "rw_torque_cmd",
                "thr_force_cmd",
            ),
        )
        # Kinematics only, for the post-reset observation: every mjo_step
        # recomputes the full forward pass and all orbital wrenches itself, and
        # the chief orbit is untouched by a reset, so a full mjo_forward (~300
        # eager kernel launches) is not needed here.
        mjw.kinematics(self.model.warp_model, self.data.warp_data)

    # ------------------------------------------------------------------
    # Geometry, success, observations
    # ------------------------------------------------------------------

    def _geometry(self) -> dict[str, np.ndarray]:
        n = self.num_envs
        xpos = np.asarray(self.data.xpos)
        xmat = np.asarray(self.data.xmat).reshape(n, -1, 3, 3)
        qvel = self.data.qvel
        r_bus, p_bus = xmat[:, self._bus_bid], xpos[:, self._bus_bid]
        r_cl, p_cl = xmat[:, self._cl_bid], xpos[:, self._cl_bid]
        axis = r_cl[:, :, 0]

        jaw_vec, inside = [], []
        for h, bid in enumerate(self._ee_bid):
            center = xpos[:, bid] + _mat_vec(
                xmat[:, bid], np.broadcast_to(self._jaw_center[h], (n, 3))
            )
            along = np.sum((center - p_cl) * axis, axis=1)
            closest = p_cl + np.clip(along, -_BAR_HALF_LEN, _BAR_HALF_LEN)[:, None] * axis
            jaw_vec.append(closest - center)
            inside.append(np.abs(along) <= _BAR_HALF_LEN)

        w_bus = _mat_vec(r_bus, qvel[:, self._bus_v + 3 : self._bus_v + 6])
        v_bus = qvel[:, self._bus_v : self._bus_v + 3]
        w_cl = _mat_vec(r_cl, qvel[:, self._cl_v + 3 : self._cl_v + 6])
        v_cl = qvel[:, self._cl_v : self._cl_v + 3]
        r_eci = np.asarray(self.data.orbit.R_eci, dtype=float)
        nadir = -_unit(r_eci)
        cos_pt = np.clip(np.sum(r_cl[:, :, 2] * nadir, axis=1), -1.0, 1.0)
        return {
            "r_bus": r_bus,
            "p_bus": p_bus,
            "r_cl": r_cl,
            "p_cl": p_cl,
            "jaw_vec": np.stack(jaw_vec, axis=1),  # (n, 2, 3) jaw center -> bar axis
            "inside": np.stack(inside, axis=1),
            "w_bus": w_bus,
            "v_bus": v_bus,
            "rel_w": w_cl - w_bus,
            "rel_v": v_cl - (v_bus + np.cross(w_bus, p_cl - p_bus)),
            "w_cl": w_cl,
            "nadir": nadir,
            "point_err": np.arccos(cos_pt),
        }

    def _held(self, geo: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        grip_q = self.data.qpos[:, self._grip_jq]
        on_bar = (
            self._grip_closed[:, None]
            & geo["inside"]
            & (np.linalg.norm(geo["jaw_vec"], axis=2) < cfg.hold_dist)
            & (grip_q < cfg.hold_grip_max)
        )
        return on_bar, on_bar.all(axis=1)

    def _success_fail(self, geo: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        _, held = self._held(geo)
        good = held & (geo["point_err"] < np.radians(cfg.point_tol_deg))
        self._hold_count = np.where(good, self._hold_count + 1, 0)
        success = self._hold_count >= self._hold_steps
        self.last_held, self.last_point_err = held, geo["point_err"]

        qpos, qvel = self.data.qpos, self.data.qvel
        nonfinite = ~(np.isfinite(qpos).all(axis=1) & np.isfinite(qvel).all(axis=1))
        lost = np.linalg.norm(geo["p_cl"] - geo["p_bus"], axis=1) > cfg.lost_dist
        body = geo["p_cl"] + _mat_vec(
            geo["r_cl"], np.broadcast_to(_CLIENT_BODY_OFFSET, geo["p_cl"].shape)
        )
        q = _mat_t_vec(geo["r_bus"], body - geo["p_bus"])
        collide = np.linalg.norm(q - np.clip(q, -_BUS_HALF, _BUS_HALF), axis=1) < cfg.collide_dist
        with np.errstate(invalid="ignore"):
            abnormal = (
                (np.abs(qvel[:, self._arm_jv]) > cfg.max_joint_vel).any(axis=1)
                | (np.abs(qvel[:, self._wrist_jv]) > cfg.max_wrist_vel).any(axis=1)
                | (np.linalg.norm(geo["w_bus"], axis=1) > cfg.max_bus_rate)
                | (np.linalg.norm(geo["w_cl"], axis=1) > cfg.max_client_rate)
            )
        fail = (nonfinite | lost | collide | abnormal) & ~success
        self._fail_reasons = {
            "lost": lost & fail,
            "collide": collide & fail,
            "abnormal": abnormal & fail,
            "nonfinite": nonfinite & fail,
        }
        return success, fail

    def _compute_obs(self, geo: dict[str, np.ndarray]) -> TensorDict:
        r_bus = geo["r_bus"]
        qpos, qvel = self.data.qpos, self.data.qvel
        on_bar, _ = self._held(geo)
        jaw = np.concatenate([_mat_t_vec(r_bus, geo["jaw_vec"][:, h]) for h in range(2)], axis=1)
        obs = np.concatenate(
            [
                qpos[:, self._arm_jq] / np.pi,
                qvel[:, self._arm_jv] / 2.0,
                (self._arm_target - self._arm_hold) / self.cfg.arm_span,
                (qpos[:, self._grip_jq] - 0.045) / 0.075,
                qvel[:, self._bus_v + 3 : self._bus_v + 6] / 0.1,  # body-frame rate
                _mat_t_vec(r_bus, geo["v_bus"]) / 0.1,
                np.asarray(self.data.actuators.rw_speed) / 150.0,
                jaw / 0.3,
                (_mat_t_vec(r_bus, geo["p_cl"] - geo["p_bus"]) - _BAR_HOLD) / 0.5,
                _mat_t_vec(r_bus, geo["r_cl"][:, :, 0]),  # bar axis
                _mat_t_vec(r_bus, geo["r_cl"][:, :, 2]),  # antenna boresight
                _mat_t_vec(r_bus, geo["rel_v"]) / 0.05,
                _mat_t_vec(r_bus, geo["rel_w"]) / 0.05,
                _mat_t_vec(r_bus, geo["nadir"]),
                np.where(self._grip_closed, 1.0, -1.0)[:, None],
                on_bar.astype(np.float64),
                (self._hold_count / self._hold_steps)[:, None],
                self._last_actions,
            ],
            axis=1,
        )
        obs = np.clip(np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0)
        return TensorDict(
            {"policy": torch.as_tensor(obs, dtype=torch.float32, device=self.device)},
            batch_size=[self.num_envs],
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _maybe_flush_log(self) -> dict[str, float]:
        self._global_step += 1
        if self._global_step % self.cfg.steps_per_log:
            return {}
        it = self._global_step // self.cfg.steps_per_log - 1
        row = {"iteration": it, "env_steps": self._global_step * self.num_envs}
        row.update(self._win.summary())
        row.update({f"sgs_{k}": v for k, v in self.sampler.stats().items()})
        self._win = _Window()
        if self._csv_path is not None:
            new = not self._csv_path.exists()
            with self._csv_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    writer.writeheader()
                writer.writerow(row)
            if it % self.cfg.sampler_snapshot_every == 0:
                np.savez_compressed(
                    Path(self.cfg.log_dir) / f"sampler_{it:05d}.npz",
                    **self.sampler.state_dict(),
                    **{f"cfg_{k}": v for k, v in self.configs.items()},
                )
        return {
            f"/capture/{k}": float(v) for k, v in row.items() if k not in ("iteration", "env_steps")
        }


class _Window:
    """Episode-outcome counters over one logging window."""

    def __init__(self) -> None:
        self.n = np.zeros(3)
        self.succ = np.zeros(3)
        self.fail = np.zeros(3)
        self.length = 0.0
        self.reasons: dict[str, float] = {}

    def record(
        self,
        fam: np.ndarray,
        success: np.ndarray,
        fail: np.ndarray,
        length: np.ndarray,
        reasons: dict[str, np.ndarray],
    ) -> None:
        np.add.at(self.n, fam, 1)
        np.add.at(self.succ, fam, success)
        np.add.at(self.fail, fam, fail)
        self.length += float(np.sum(length))
        for k, v in reasons.items():
            self.reasons[k] = self.reasons.get(k, 0.0) + float(np.sum(v))

    def summary(self) -> dict[str, float]:
        tot = max(self.n.sum(), 1.0)
        out = {
            "episodes": float(self.n.sum()),
            "success_rate": float(self.succ.sum() / tot),
            "fail_rate": float(self.fail.sum() / tot),
            "ep_len_s": float(self.length / tot * 0.1),
        }
        for k in ("lost", "collide", "abnormal", "nonfinite"):
            out[f"fail_{k}"] = self.reasons.get(k, 0.0) / tot
        for f, name in enumerate(FAMILIES):
            out[f"success_{name}"] = float(self.succ[f] / self.n[f]) if self.n[f] else float("nan")
            out[f"frac_{name}"] = float(self.n[f] / tot)
        return out


# ----------------------------------------------------------------------
# Task configurations
# ----------------------------------------------------------------------


def sample_task_configs(
    cfg: CaptureEnvCfg, fam: np.ndarray, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Draw one task configuration per entry of ``fam`` (reset-family ids).

    Each configuration fully specifies a reset relative to the servicer's hold
    pose and the LVLH frame at reset time:

    * ``bus_rot`` (LVLH-from-bus): bus +Z tilted ``point_err0 ~ U[0, max]`` from
      nadir about a random horizontal axis, with a random yaw about nadir. In the
      hold pose the client antenna is parallel to bus +Z, so this is the initial
      pointing error of a captured client.
    * servicer residual rate / drift, arm-joint offsets from the hold pose;
    * client bar pose offset from the clamped hold position (bus frame), and the
      client's relative velocity and tumble rate.
    """
    fam = np.asarray(fam, dtype=np.int64)
    n = fam.size
    err0 = np.radians(rng.uniform(0.0, cfg.max_point_err_deg, n))
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    tilt_axis = np.stack([np.cos(phi), np.sin(phi), np.zeros(n)], axis=1)
    yaw = rng.uniform(0.0, 2.0 * np.pi, n)
    bus_rot = _rotvec_to_mat(tilt_axis * err0[:, None]) @ _rotvec_to_mat(
        np.stack([np.zeros(n), np.zeros(n), yaw], axis=1)
    )

    def box(lim, shape):
        return rng.uniform(-1.0, 1.0, shape) * np.asarray(lim)

    dp = np.zeros((n, 3))
    rot = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    rate = np.zeros((n, 3))
    arm = np.zeros((n, 4))

    held = fam == HELD
    arm[held] = box(cfg.held_arm_noise, (held.sum(), 4))

    pre = fam == PREGRASP
    k = pre.sum()
    ox, oy, oz = cfg.pre_offset
    dp[pre] = np.stack([box(ox, k), box(oy, k), rng.uniform(0.0, oz, k)], axis=1)
    rot[pre] = box(np.radians(cfg.pre_att_deg), (k, 3))
    vel[pre] = box(cfg.pre_vel, (k, 3))
    rate[pre] = box(np.radians(cfg.pre_rate_deg), (k, 3))
    arm[pre] = box(cfg.pre_arm_noise, (k, 4))

    app = fam == APPROACH
    k = app.sum()
    u = rng.uniform(0.0, 1.0, k)
    difficulty = np.zeros(n)
    difficulty[app] = u if cfg.app_graded else 1.0

    def span(lo: float, hi: float) -> np.ndarray:
        return lo + (hi - lo) * u if cfg.app_graded else np.full(k, hi)

    az = rng.uniform(0.0, 2.0 * np.pi, k)
    el = np.arcsin(rng.uniform(0.0, 1.0, k))  # uniform on the upper hemisphere
    direction = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
    d_lo, d_hi = cfg.app_dist
    dist = d_lo + (d_hi - d_lo) * (u if cfg.app_graded else rng.uniform(0.0, 1.0, k))
    dp[app] = direction * dist[:, None]
    tilt = span(np.radians(cfg.pre_att_deg), np.radians(cfg.app_tilt_deg))[:, None]
    yaw_lim = span(np.radians(cfg.pre_att_deg), np.radians(cfg.app_yaw_deg))[:, None]
    rot[app] = box(1.0, (k, 3)) * np.concatenate([tilt, tilt, yaw_lim], axis=1)
    vel[app] = box(1.0, (k, 3)) * span(cfg.pre_vel, cfg.app_vel)[:, None]
    rate[app] = (
        box(1.0, (k, 3)) * span(np.radians(cfg.pre_rate_deg), np.radians(cfg.app_rate_deg))[:, None]
    )
    arm[app] = box(1.0, (k, 4)) * span(cfg.pre_arm_noise, cfg.app_arm_noise)[:, None]

    return {
        "family": fam.astype(np.int64),
        "difficulty": difficulty,
        "point_err0": err0,
        "bus_rot": bus_rot,
        "bus_w": box(cfg.bus_rate_max, (n, 3)),
        "bus_v": box(cfg.bus_vel_max, (n, 3)),
        "arm_dq": arm,
        "client_dp": dp,
        "client_rot": _rotvec_to_mat(rot),
        "client_v": vel,
        "client_w": rate,
    }


# ----------------------------------------------------------------------
# Batched rotation helpers
# ----------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _mat_vec(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nj->ni", r, v)


def _mat_t_vec(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum("nji,nj->ni", r, v)


def _rotvec_to_mat(rv: np.ndarray) -> np.ndarray:
    """Rodrigues: (n, 3) rotation vectors -> (n, 3, 3) matrices."""
    theta = np.linalg.norm(rv, axis=1)
    k = rv / np.maximum(theta, 1e-12)[:, None]
    kx = np.zeros((len(rv), 3, 3))
    kx[:, 0, 1], kx[:, 0, 2] = -k[:, 2], k[:, 1]
    kx[:, 1, 0], kx[:, 1, 2] = k[:, 2], -k[:, 0]
    kx[:, 2, 0], kx[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = np.sin(theta)[:, None, None], np.cos(theta)[:, None, None]
    return np.eye(3)[None] + s * kx + (1.0 - c) * (kx @ kx)


def _mat_to_quat(r: np.ndarray) -> np.ndarray:
    """(n, 3, 3) world-from-body matrices -> (n, 4) MuJoCo (w, x, y, z) quaternions."""
    out = np.empty((len(r), 4))
    for i, m in enumerate(r):
        mj.mju_mat2Quat(out[i], np.ascontiguousarray(m).reshape(9))
    return out
