"""Spline-knot MPPI planner on top of mujoco_orbit's batched rollout.

The planner follows judo's sampling-based MPC structure
(https://github.com/rai-opensource/judo): the decision variables are control
knots at a small number of spline nodes spread over the horizon, rollouts
evaluate the interpolated per-step controls, and the nominal knots are updated
with the standard MPPI exponentially-weighted average (Williams et al.,
information-theoretic MPC). Variance scheduling matches judo's noise ramp:
sigma grows linearly across the horizon so late knots explore more than the
knot about to be executed.

The control vector follows ``mujoco_orbit.rollout``'s layout: ``model.nu``
MuJoCo actuator controls followed by the orbital actuator tail
``[rw_torque, mtq_dipole, thr_force, cmg_rate]``. For models without orbital
actuators it is exactly ``data.ctrl``.

The cost callback receives the raw rollout outputs:

    cost_fn(states, sensors, controls) -> costs, shape (num_rollouts,)

where ``states`` is ``(num_rollouts, num_timesteps, nstate)`` with the packed
mjo state layout ``[time, qpos, qvel, act, R_eci, V_eci, t, ...]`` and
``sensors`` is ``(num_rollouts, num_timesteps, nsensordata)``. Lower cost is
better. Note that MuJoCo evaluates sensors before integrating, so ``sensors``
row ``k`` corresponds to the state at the *start* of step ``k`` (one step
behind ``states`` row ``k``); at typical timesteps this is negligible, but
costs mixing the two arrays should be aware of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d

from mujoco_orbit import MjoData, MjoModel, mjo_forward
from mujoco_orbit.rollout import mjo_control_size, mjo_get_state, mjo_set_state, rollout

CostFn = Callable[[np.ndarray, np.ndarray, np.ndarray], ArrayLike]


@dataclass
class MppiConfig:
    """MPPI hyperparameters (judo's MPPIConfig defaults, except a larger num_rollouts)."""

    horizon: float = 2.0
    num_rollouts: int = 32
    num_nodes: int = 4
    spline_order: str = "linear"  # "zero" | "linear" | "cubic"
    sigma: float | ArrayLike = 0.1
    temperature: float = 0.05
    use_noise_ramp: bool = False
    noise_ramp: float = 2.5
    nthread: int = 1
    seed: int | None = None


def make_spline(times: np.ndarray, knots: np.ndarray, spline_order: str) -> interp1d:
    """Interpolate (possibly batched) ``(..., num_nodes, ncontrol)`` knots over time.

    Queries outside ``times`` clamp to the edge knots, so a stale spline can be
    re-queried at shifted times when the plan is re-anchored each replan.
    """
    fill_value = (knots[..., 0, :], knots[..., -1, :])
    return interp1d(
        times,
        knots,
        kind=spline_order,
        axis=-2,
        copy=False,
        bounds_error=False,
        fill_value=fill_value,  # type: ignore[arg-type]
    )


class MppiPlanner:
    """Receding-horizon MPPI over spline control knots.

    Usage::

        planner = MppiPlanner(model, MppiConfig(...), cost_fn)
        planner.reset(data)
        ...
        planner.update_action(data)        # replan from data's current state
        u = planner.action(data.time)      # query the plan as time advances
    """

    def __init__(
        self,
        model: MjoModel,
        config: MppiConfig,
        cost_fn: CostFn,
        *,
        ctrl_low: ArrayLike | None = None,
        ctrl_high: ArrayLike | None = None,
    ) -> None:
        if config.num_rollouts < 2:
            raise ValueError("num_rollouts must be at least 2 (nominal + samples)")
        if config.num_nodes < 2:
            raise ValueError("num_nodes must be at least 2")
        if config.spline_order == "cubic" and config.num_nodes < 4:
            raise ValueError("cubic splines require num_nodes >= 4")
        if config.horizon <= 0.0:
            raise ValueError("horizon must be positive")
        if config.temperature <= 0.0:
            raise ValueError("temperature must be positive")

        self.model = model
        self.config = config
        self.cost_fn = cost_fn
        self.ncontrol = mjo_control_size(model)
        self.dt = float(model.opt.timestep)
        self.num_timesteps = int(np.ceil(config.horizon / self.dt))

        sigma = np.asarray(config.sigma, dtype=np.float64)
        if sigma.ndim == 0:
            sigma = np.full(self.ncontrol, float(sigma))
        if sigma.shape != (self.ncontrol,):
            raise ValueError(f"sigma must be a scalar or shape ({self.ncontrol},)")
        if np.any(sigma < 0.0):
            raise ValueError("sigma must be non-negative")
        self._sigma = sigma

        self._ctrl_low = None if ctrl_low is None else np.asarray(ctrl_low, dtype=np.float64)
        self._ctrl_high = None if ctrl_high is None else np.asarray(ctrl_high, dtype=np.float64)
        for name, bound in (("ctrl_low", self._ctrl_low), ("ctrl_high", self._ctrl_high)):
            if bound is not None and bound.shape != (self.ncontrol,):
                raise ValueError(f"{name} must have shape ({self.ncontrol},)")

        self._rng = np.random.default_rng(config.seed)
        # Knot times relative to the replan instant, judo-style: first knot is
        # executed immediately, last knot sits at the end of the horizon.
        self._knot_offsets = np.linspace(0.0, config.horizon, config.num_nodes, endpoint=True)
        self._step_offsets = self.dt * np.arange(self.num_timesteps)
        self.times = self._knot_offsets.copy()
        self.nominal_knots = np.zeros((config.num_nodes, self.ncontrol))
        self.spline = make_spline(self.times, self.nominal_knots, config.spline_order)

        # Diagnostics from the most recent update_action call.
        self.last_costs: np.ndarray | None = None
        self.last_weights: np.ndarray | None = None

    def reset(self, data: MjoData, nominal_knots: ArrayLike | None = None) -> None:
        """Anchor the plan at ``data``'s current time, optionally seeding the knots."""
        if nominal_knots is None:
            knots = np.zeros((self.config.num_nodes, self.ncontrol))
        else:
            knots = np.array(nominal_knots, dtype=np.float64)
            if knots.shape != (self.config.num_nodes, self.ncontrol):
                raise ValueError(
                    f"nominal_knots must have shape "
                    f"({self.config.num_nodes}, {self.ncontrol}), got {knots.shape}"
                )
        self.times = float(data.time) + self._knot_offsets
        self.nominal_knots = knots
        self.spline = make_spline(self.times, self.nominal_knots, self.config.spline_order)
        self.last_costs = None
        self.last_weights = None

    def action(self, time: float) -> np.ndarray:
        """Evaluate the current nominal plan at ``time`` (clamped to the horizon)."""
        return np.asarray(self.spline(time), dtype=np.float64)

    def update_action(self, data: MjoData) -> np.ndarray:
        """Run one MPPI update from ``data``'s current state.

        Samples knot perturbations around the time-shifted nominal plan, rolls
        them out with ``mujoco_orbit.rollout``, reweights, and rebuilds the
        nominal spline. ``data`` is used as the rollout workspace; its physics
        and orbit state (qpos/qvel/act/time/orbit) and ``data.ctrl`` are
        restored (including a forward pass) before returning. Orbital actuator
        commands and applied-force buffers are reset by the rollout, so
        re-apply inputs (e.g. from :meth:`action`) before stepping the plant.
        Returns the per-rollout costs.
        """
        cfg = self.config
        initial_state = mjo_get_state(self.model, data)
        initial_ctrl = np.array(data.ctrl, dtype=np.float64)
        now = float(initial_state[0])

        # Receding horizon: re-anchor the previous plan at the new knot times.
        new_times = now + self._knot_offsets
        nominal = np.asarray(self.spline(new_times), dtype=np.float64)

        samples = self._sample_knots(nominal)
        if self._ctrl_low is not None or self._ctrl_high is not None:
            samples = np.clip(samples, self._ctrl_low, self._ctrl_high)

        # Knots -> per-step open-loop controls, one row per physics step.
        controls = np.asarray(
            make_spline(new_times, samples, cfg.spline_order)(now + self._step_offsets),
            dtype=np.float64,
        )

        try:
            states, sensors = rollout(
                self.model,
                data,
                initial_state,
                controls,
                nthread=cfg.nthread,
            )
        finally:
            # rollout reuses `data` as a workspace; put the plant back.
            mjo_set_state(self.model, data, initial_state)
            np.copyto(data.ctrl, initial_ctrl)
            mjo_forward(self.model, data)

        costs = np.asarray(self.cost_fn(states, sensors, controls), dtype=np.float64)
        if costs.shape != (cfg.num_rollouts,):
            raise ValueError(
                f"cost_fn must return shape ({cfg.num_rollouts},), got {costs.shape}"
            )

        # MPPI exponentially-weighted average over knot samples.
        beta = float(np.min(costs))
        weights = np.exp(-(costs - beta) / cfg.temperature)
        weights /= np.sum(weights)
        self.nominal_knots = np.einsum("r,rnu->nu", weights, samples)
        self.times = new_times
        self.spline = make_spline(self.times, self.nominal_knots, cfg.spline_order)

        self.last_costs = costs
        self.last_weights = weights
        return costs

    def _sample_knots(self, nominal: np.ndarray) -> np.ndarray:
        """Gaussian knot perturbations; the unperturbed nominal rides as sample 0."""
        cfg = self.config
        sigma = self._sigma
        if cfg.use_noise_ramp:
            # Variance scheduling: scale sigma linearly across the horizon so
            # the imminent knot stays close to the nominal while distant knots
            # explore more (judo's noise ramp).
            ramp = cfg.noise_ramp * np.linspace(
                1.0 / cfg.num_nodes, 1.0, cfg.num_nodes, endpoint=True
            )
            sigma = ramp[:, None] * sigma
        noise = self._rng.standard_normal((cfg.num_rollouts - 1, cfg.num_nodes, self.ncontrol))
        return np.concatenate([nominal[None], nominal[None] + sigma * noise], axis=0)


__all__ = ["CostFn", "MppiConfig", "MppiPlanner", "make_spline"]
