# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""MPPI capture + gravity-gradient stabilize, interactive port.

Phase A reaches and gently grasps a co-orbiting payload; at latch the weld
equality is activated by swapping to a recompiled model (the app rebuilds
the rendered scene automatically). Phase B damps libration with slow arm
motion over a long horizon — phase-B replans roll out tens of minutes of
coupled dynamics, so expect a noticeable pause at each replan.
"""

from __future__ import annotations

import os
import tempfile
import time as wall_time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mujoco_orbit import MjoData, MjoModel, OrbitInit, mjo_forward
from mujoco_orbit.constants import GM_EARTH, R_EARTH
from mujoco_orbit.planning import MppiConfig, MppiPlanner
from mujoco_orbit.rollout import mjo_get_state, mjo_set_state
from mujoco_orbit.testdata import SPACECRAFT_CAPTURE_XML

from .base import ViewerTask, build_rollout_traces, ui_field
from .registry import register_task

# Packed rollout state layout for this model (see examples/mppi/capture_stabilize.py)
_BUS_POS = slice(1, 4)
_PAYLOAD_POS = slice(10, 13)
_BUS_WZ = 22
_ARM_QVEL = slice(23, 25)
_R_ECI = slice(31, 34)

# Sensor layout: ee_pos 0:3, ee_quat 3:7, ee_linvel 7:10, payload_pos 10:13,
# payload_linvel 13:16
_EE_POS = slice(0, 3)
_EE_QUAT = slice(3, 7)
_EE_LINVEL = slice(7, 10)
_SENS_PAYLOAD_POS = slice(10, 13)
_PAYLOAD_LINVEL = slice(13, 16)

_GRASP_OFFSET = 0.6  # m, weld relpose offset along the end-effector x-axis


def _quat_to_xaxis(q: np.ndarray) -> np.ndarray:
    """Body x-axis in world frame for quaternions (..., 4) in (w,x,y,z) order."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)],
        axis=-1,
    )


def _grasp_error(sensors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grasp_pt = sensors[..., _EE_POS] + _GRASP_OFFSET * _quat_to_xaxis(sensors[..., _EE_QUAT])
    dist = np.linalg.norm(sensors[..., _SENS_PAYLOAD_POS] - grasp_pt, axis=-1)
    relspeed = np.linalg.norm(
        sensors[..., _PAYLOAD_LINVEL] - sensors[..., _EE_LINVEL], axis=-1
    )
    return dist, relspeed


def _stack_pitch(states: np.ndarray) -> np.ndarray:
    """Signed in-plane angle between the bus->payload axis and local vertical."""
    axis = states[..., _PAYLOAD_POS] - states[..., _BUS_POS]
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    rhat = states[..., _R_ECI]
    rhat = rhat / np.linalg.norm(rhat, axis=-1, keepdims=True)
    sin = rhat[..., 0] * axis[..., 1] - rhat[..., 1] * axis[..., 0]
    cos = np.sum(rhat * axis, axis=-1)
    return np.arctan2(sin, cos)


def _compile_models(xml_path: str) -> tuple[MjoModel, MjoModel]:
    """Compile the capture (weld off) and stack (weld on) variants of one XML."""
    xml = Path(xml_path).read_text()
    models = []
    for active in ("false", "true"):
        text = xml.replace('active="false"', f'active="{active}"')
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(text)
            path = Path(f.name)
        try:
            models.append(MjoModel.from_xml_path(str(path)))
        finally:
            path.unlink(missing_ok=True)
    return models[0], models[1]


@dataclass
class CaptureParams:
    num_rollouts: float = ui_field(48, 16, 128, step=8)
    horizon_b_s: float = ui_field(1500.0, 300.0, 2400.0, step=100.0)
    replan_b_s: float = ui_field(60.0, 20.0, 120.0, step=5.0)


@register_task
class CaptureStabilizeTask(ViewerTask):
    """Capture a free-flyer, then stabilize the stack about local vertical."""

    name = "capture_stabilize_mppi"
    description = (
        "A bus with a slow 6.5 m arm captures a 400 kg payload drifting "
        "~6 m away, then damps libration with gravity-gradient torques and "
        "slow arm motion only. The weld activates at latch by swapping to a "
        "recompiled model. Phase-B replans are heavy — expect pauses."
    )
    track_body = "bus"

    params: CaptureParams

    def make_params(self) -> CaptureParams:
        return CaptureParams()

    def build(self) -> tuple[MjoModel, MjoData]:
        alt_km = 400.0
        r_orbit = R_EARTH + alt_km
        self._omega = float(np.sqrt(GM_EARTH / r_orbit**3))
        v_orbit = float(np.sqrt(GM_EARTH / r_orbit))
        self._R0 = np.array([r_orbit, 0.0, 0.0])
        self._V0 = np.array([0.0, v_orbit, 0.0])  # equatorial: orbit plane = world XY

        self._model_cap, self._model_stk = _compile_models(SPACECRAFT_CAPTURE_XML)

        data = self._model_cap.make_data(orbit=OrbitInit(R_eci=self._R0, V_eci=self._V0))
        # Bus arm points along-track (+Y); attitude tracks the rotating LVLH frame.
        data.qpos[3:7] = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
        data.qpos[7:9] = [1.0, -2.0]  # folded arm: a real reach is needed to capture
        data.qvel[3:6] = [0.0, 0.0, self._omega]
        payload_pos = np.array([0.5, 6.3, 0.0])
        data.qpos[9:12] = payload_pos
        data.qvel[8:11] = np.cross([0.0, 0.0, self._omega], payload_pos)
        data.qvel[11:14] = [0.0, 0.0, 0.002]
        np.copyto(data.ctrl, data.qpos[7:9])
        mjo_forward(self._model_cap, data)

        self.phase = "capture"
        self._planner: MppiPlanner | None = None
        self._planner_stale = True
        self._next_replan_t = -np.inf
        self._last_plan_s = float("nan")
        return self._model_cap, data

    # -- planners ---------------------------------------------------------

    def _nthread(self) -> int:
        return max(1, os.cpu_count() or 1)

    def _make_capture_cost(self, num_timesteps: int):
        term = slice(int(0.8 * num_timesteps), None)

        def cost_fn(states: np.ndarray, sensors: np.ndarray, controls: np.ndarray) -> np.ndarray:
            del controls
            dist, relspeed = _grasp_error(sensors)
            arm_rate_sq = np.sum(states[:, :, _ARM_QVEL] ** 2, axis=-1)
            costs = (
                0.3 * np.mean(dist**2, axis=1)
                + 3.0 * np.mean(dist[:, term] ** 2, axis=1)
                + 100.0 * np.mean(relspeed[:, term] ** 2, axis=1)
                + 20.0 * np.mean(arm_rate_sq, axis=1)
            )
            self._publish_traces(sensors, costs)
            return costs

        return cost_fn

    def _make_stabilize_cost(self, num_timesteps: int):
        term = slice(int(0.6 * num_timesteps), None)
        omega_orbit = self._omega

        def cost_fn(states: np.ndarray, sensors: np.ndarray, controls: np.ndarray) -> np.ndarray:
            del controls
            pitch = _stack_pitch(states)
            align = np.sin(pitch) ** 2  # 0 at either vertical, 1 at horizontal
            libr = ((states[:, :, _BUS_WZ] - omega_orbit) / omega_orbit) ** 2
            arm_rate_sq = np.sum((states[:, :, _ARM_QVEL] / 0.01) ** 2, axis=-1)
            costs = (
                2.0 * np.mean(align[:, term], axis=1)
                + 0.5 * np.mean(libr[:, term], axis=1)
                + 0.3 * np.mean(align, axis=1)
                + 0.005 * np.mean(arm_rate_sq, axis=1)
            )
            self._publish_traces(sensors, costs)
            return costs

        return cost_fn

    def _publish_traces(self, sensors: np.ndarray, costs: np.ndarray) -> None:
        """Predicted end-effector and payload trajectories for the viewer."""
        self.set_traces(
            build_rollout_traces(sensors[:, :, _EE_POS], costs, max_others=8)
            + build_rollout_traces(sensors[:, :, _SENS_PAYLOAD_POS], costs, max_others=8)
        )

    def _rebuild_planner(self) -> None:
        dt = float(self.model.opt.timestep)
        rollouts = int(self.params.num_rollouts)
        if self.phase == "capture":
            horizon, replan, nodes = 300.0, 10.0, 5
            config = MppiConfig(
                horizon=horizon, num_rollouts=rollouts, num_nodes=nodes,
                spline_order="linear", sigma=0.3, temperature=0.05,
                use_noise_ramp=True, noise_ramp=2.5, nthread=self._nthread(), seed=0,
            )
            cost_fn = self._make_capture_cost(int(np.ceil(horizon / dt)))
        else:
            horizon = float(self.params.horizon_b_s)
            replan, nodes = float(self.params.replan_b_s), 6
            config = MppiConfig(
                horizon=horizon, num_rollouts=rollouts, num_nodes=nodes,
                spline_order="linear", sigma=0.25, temperature=0.02,
                use_noise_ramp=True, noise_ramp=2.5, nthread=self._nthread(), seed=1,
            )
            cost_fn = self._make_stabilize_cost(int(np.ceil(horizon / dt)))
        self._replan_period = replan
        self._planner = MppiPlanner(
            self.model, config, cost_fn,
            ctrl_low=np.full(2, -2.8), ctrl_high=np.full(2, 2.8),
        )
        self._planner.reset(
            self.data, nominal_knots=np.tile(np.asarray(self.data.ctrl), (nodes, 1))
        )
        self._planner_stale = False

    # -- step hooks ---------------------------------------------------------

    def pre_step(self) -> None:
        t = float(self.data.time)
        if self._planner_stale or self._planner is None:
            self._rebuild_planner()
            self._next_replan_t = t
        assert self._planner is not None
        if t >= self._next_replan_t:
            start = wall_time.perf_counter()
            self._planner.update_action(self.data)
            self._last_plan_s = wall_time.perf_counter() - start
            self._next_replan_t = t + self._replan_period
        np.copyto(self.data.ctrl, self._planner.action(t))

    def post_step(self) -> None:
        if self.phase != "capture":
            return
        dist, relspeed = _grasp_error(np.asarray(self.data.sensordata))
        if float(dist) < 0.30 and float(relspeed) < 0.04:
            self._latch()

    def _latch(self) -> None:
        """Activate the weld: transfer packed state into the stack model."""
        state = mjo_get_state(self.model, self.data)
        data = self._model_stk.make_data(orbit=OrbitInit(R_eci=self._R0, V_eci=self._V0))
        mjo_set_state(self._model_stk, data, state)
        np.copyto(data.ctrl, np.asarray(data.qpos[7:9]))  # hold joints
        mjo_forward(self._model_stk, data)
        self.model, self.data = self._model_stk, data
        self.phase = "stabilize"
        self._planner_stale = True

    # -- GUI ------------------------------------------------------------------

    def on_param_changed(self, field_name: str) -> None:
        del field_name
        self._planner_stale = True

    def _plant_pitch_deg(self) -> float:
        bus = np.asarray(self.data.qpos[0:3])
        payload = np.asarray(self.data.qpos[9:12])
        axis = payload - bus
        axis = axis / np.linalg.norm(axis)
        rhat = self.data.orbit.R_eci / np.linalg.norm(self.data.orbit.R_eci)
        pitch = np.arctan2(rhat[0] * axis[1] - rhat[1] * axis[0], rhat @ axis)
        return float(np.rad2deg(pitch))

    def status(self) -> str:
        plan = "—" if np.isnan(self._last_plan_s) else f"{self._last_plan_s:.1f} s"
        if self.phase == "capture":
            dist, relspeed = _grasp_error(np.asarray(self.data.sensordata))
            return (
                f"**phase** = capture &nbsp; **grasp dist** = {float(dist):.2f} m &nbsp; "
                f"**rel speed** = {float(relspeed) * 100:.1f} cm/s &nbsp; "
                f"**plan time** = {plan}"
            )
        return (
            f"**phase** = stabilize &nbsp; **pitch from vertical** = "
            f"{self._plant_pitch_deg():+.1f} deg &nbsp; **plan time** = {plan}"
        )
