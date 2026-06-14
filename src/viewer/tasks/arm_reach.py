# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""MPPI arm reach: a free-floating spacecraft arm reaches for a target.

Interactive port of ``examples/mppi/arm_reach.py``. The MPPI replans run
inline in the viewer loop, so large rollout counts can cause brief frame
hitches at each replan.
"""

from __future__ import annotations

import os
import time as wall_time
from dataclasses import dataclass

import numpy as np

from mjorbit import MjoData, MjoModel, OrbitInit, mjo_forward
from mjorbit.planning import MppiConfig, MppiPlanner
from mjorbit.rollout import mjo_control_size
from mjorbit.testdata import SPACECRAFT_ARM_REACH_XML

from .base import ViewerTask, build_rollout_traces, circular_orbit_eci, ui_field
from .registry import register_task


@dataclass
class ArmReachParams:
    num_rollouts: float = ui_field(32, 8, 256, step=8)
    horizon_s: float = ui_field(1.5, 0.5, 3.0, step=0.25)
    replan_period_s: float = ui_field(0.1, 0.05, 0.5, step=0.05)
    sigma: float = ui_field(0.2, 0.05, 0.5, step=0.05)


@register_task
class ArmReachTask(ViewerTask):
    """Spline-knot MPPI drives a 2-link arm on an unactuated base."""

    name = "arm_reach_mppi"
    description = (
        "A 2-link arm on a free-floating bus reaches for a target co-moving "
        "on the chief orbit. Spline-knot MPPI replans inline; planner "
        "changes apply at the next replan."
    )
    track_body = "spacecraft"

    params: ArmReachParams

    def make_params(self) -> ArmReachParams:
        return ArmReachParams()

    def build(self) -> tuple[MjoModel, MjoData]:
        R_eci, V_eci = circular_orbit_eci(400.0, 51.6)
        model = MjoModel.from_xml_path(SPACECRAFT_ARM_REACH_XML)
        data = model.make_data(orbit=OrbitInit(R_eci=R_eci, V_eci=V_eci))
        mjo_forward(model, data)

        self._ee_slice = self._sensor_slice(model, "ee_pos")
        self._target_slice = self._sensor_slice(model, "target_pos")
        self._planner: MppiPlanner | None = None
        self._planner_stale = True
        self._next_replan_t = -np.inf
        self._last_plan_ms = float("nan")
        return model, data

    @staticmethod
    def _sensor_slice(model: MjoModel, name: str) -> slice:
        descriptor = model.sensor(name)
        return slice(int(descriptor.adr), int(descriptor.adr) + int(descriptor.dim))

    def _make_cost_fn(self):
        nq, nv = int(self.model.nq), int(self.model.nv)
        ee_sl, target_sl = self._ee_slice, self._target_slice
        base_rate_sl = slice(1 + nq + 3, 1 + nq + 6)
        arm_qvel_sl = slice(1 + nq + nv - 2, 1 + nq + nv)
        w_running, w_terminal, w_qvel, w_base = 1.0, 10.0, 1.0e-2, 0.5

        def cost_fn(states: np.ndarray, sensors: np.ndarray, controls: np.ndarray) -> np.ndarray:
            del controls
            dist_sq = np.sum((sensors[:, :, ee_sl] - sensors[:, :, target_sl]) ** 2, axis=-1)
            arm_rate_sq = np.sum(states[:, :, arm_qvel_sl] ** 2, axis=-1)
            base_rate_sq = np.sum(states[:, :, base_rate_sl] ** 2, axis=-1)
            costs = (
                w_running * np.mean(dist_sq, axis=1)
                + w_terminal * dist_sq[:, -1]
                + w_qvel * np.mean(arm_rate_sq, axis=1)
                + w_base * np.mean(base_rate_sq, axis=1)
            )
            # Publish predicted end-effector trajectories (judo-style traces).
            self.set_traces(build_rollout_traces(sensors[:, :, ee_sl], costs))
            return costs

        return cost_fn

    def _rebuild_planner(self) -> None:
        config = MppiConfig(
            horizon=float(self.params.horizon_s),
            num_rollouts=int(self.params.num_rollouts),
            num_nodes=4,
            spline_order="linear",
            sigma=float(self.params.sigma),
            temperature=0.05,
            use_noise_ramp=True,
            noise_ramp=2.5,
            nthread=max(1, os.cpu_count() or 1),
            seed=0,
        )
        ncontrol = mjo_control_size(self.model)
        self._planner = MppiPlanner(
            self.model,
            config,
            self._make_cost_fn(),
            ctrl_low=np.full(ncontrol, -3.14),
            ctrl_high=np.full(ncontrol, 3.14),
        )
        self._planner.reset(
            self.data, nominal_knots=np.tile(np.asarray(self.data.ctrl), (4, 1))
        )
        self._planner_stale = False

    def pre_step(self) -> None:
        t = float(self.data.time)
        if self._planner_stale or self._planner is None:
            self._rebuild_planner()
            self._next_replan_t = t
        assert self._planner is not None
        if t >= self._next_replan_t:
            start = wall_time.perf_counter()
            self._planner.update_action(self.data)
            self._last_plan_ms = 1000.0 * (wall_time.perf_counter() - start)
            self._next_replan_t = t + float(self.params.replan_period_s)
        np.copyto(self.data.ctrl, self._planner.action(t))

    def on_param_changed(self, field_name: str) -> None:
        del field_name
        self._planner_stale = True

    def status(self) -> str:
        sensordata = np.asarray(self.data.sensordata)
        distance = float(
            np.linalg.norm(sensordata[self._ee_slice] - sensordata[self._target_slice])
        )
        plan = "—" if np.isnan(self._last_plan_ms) else f"{self._last_plan_ms:.0f} ms"
        return f"**EE-target distance** = {distance:.3f} m &nbsp; **plan time** = {plan}"
