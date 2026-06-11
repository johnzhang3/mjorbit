# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Task abstraction for the interactive viewer app.

A :class:`ViewerTask` owns its simulation: it builds the ``(MjoModel,
MjoData)`` pair, supplies controls before every physics step, and may swap
its own model/data mid-run (e.g. activating a weld at capture). The app
notices model identity changes and rebuilds the rendered scene.

Numeric fields of the task's params dataclass become GUI sliders. Use
:func:`ui_field` to set explicit bounds; bool fields become checkboxes.

Planner-driven tasks publish *predicted* trajectories (judo-style rollout
traces) by filling ``self.traces`` with :class:`PredictedTrace` objects and
bumping ``self.traces_version``; the app renders them as line traces in the
scene (best rollout orange, alternatives purple).
"""

from __future__ import annotations

import abc
import dataclasses
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from mujoco_orbit import MjoData, MjoModel
from mujoco_orbit.constants import GM_EARTH, R_EARTH

#: Trace colors, judo-style: best rollout orange, the rest purple.
TRACE_BEST_COLOR: tuple[int, int, int] = (255, 165, 60)
TRACE_OTHER_COLOR: tuple[int, int, int] = (150, 110, 210)


def circular_orbit_eci(
    altitude_km: float, inclination_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Circular orbit at ascending node: R along +X, V in the Y-Z plane."""
    radius_km = R_EARTH + float(altitude_km)
    speed = np.sqrt(GM_EARTH / radius_km)
    inc = np.deg2rad(float(inclination_deg))
    R_eci = np.array([radius_km, 0.0, 0.0])
    V_eci = speed * np.array([0.0, np.cos(inc), np.sin(inc)])
    return R_eci, V_eci


@dataclass
class PredictedTrace:
    """One predicted trajectory in chief-centered world coordinates (m)."""

    points: np.ndarray  # (T, 3)
    color: tuple[int, int, int] = TRACE_OTHER_COLOR
    line_width: float = 2.0


def build_rollout_traces(
    positions: np.ndarray,
    costs: np.ndarray,
    *,
    max_others: int = 15,
    max_points: int = 80,
) -> list[PredictedTrace]:
    """Judo-style traces from rollout position histories.

    *positions* is ``(num_rollouts, num_timesteps, 3)`` (e.g. a framepos
    sensor channel from the rollout sensordata). The lowest-cost rollout is
    drawn in orange on top of up to *max_others* of the next-best rollouts
    in purple; each trace is subsampled to at most *max_points* points.
    """
    positions = np.asarray(positions, dtype=float)
    costs = np.asarray(costs, dtype=float)
    num_rollouts, num_timesteps = positions.shape[0], positions.shape[1]
    keep = np.unique(np.linspace(0, num_timesteps - 1, max_points).astype(int))

    order = np.argsort(costs)
    traces = [
        PredictedTrace(positions[r, keep], color=TRACE_OTHER_COLOR, line_width=2.0)
        for r in order[1 : 1 + max(0, min(max_others, num_rollouts - 1))]
    ]
    # Best last so it draws on top.
    traces.append(
        PredictedTrace(positions[order[0], keep], color=TRACE_BEST_COLOR, line_width=4.0)
    )
    return traces


@dataclass(frozen=True)
class UiSlider:
    """Slider bounds attached to a params dataclass field."""

    low: float
    high: float
    step: float | None = None
    label: str | None = None


def ui_field(default: Any, low: float, high: float, *, step: float | None = None,
             label: str | None = None) -> Any:
    """Dataclass field with slider metadata for the auto-generated GUI."""
    return dataclasses.field(
        default=default, metadata={"ui": UiSlider(low, high, step=step, label=label)}
    )


class ViewerTask(abc.ABC):
    """One self-contained simulation scenario shown by the viewer app."""

    #: Registry key and dropdown entry.
    name: ClassVar[str] = ""
    #: One-line summary shown in the task panel.
    description: ClassVar[str] = ""
    #: Body framed by the camera and used for auto-scaling; None = body 1.
    track_body: ClassVar[str | None] = None

    model: MjoModel
    data: MjoData

    #: Predicted trajectories (chief-centered world metres) published by the
    #: task, e.g. planner rollouts. Bump ``traces_version`` when replaced.
    traces: list[PredictedTrace]
    traces_version: int

    def __init__(self) -> None:
        self.traces = []
        self.traces_version = 0
        self.params = self.make_params()
        self.initialize()

    # -- construction ---------------------------------------------------

    def make_params(self) -> Any | None:
        """Return a params dataclass instance, or None for no parameters."""
        return None

    @abc.abstractmethod
    def build(self) -> tuple[MjoModel, MjoData]:
        """Compile the model and construct forwarded runtime state."""

    def initialize(self) -> None:
        self.model, self.data = self.build()

    def reset(self) -> None:
        """Restore the initial state; params take effect here by default."""
        self.initialize()
        self.set_traces([])

    # -- per-step hooks ---------------------------------------------------

    def pre_step(self) -> None:
        """Write controls into ``self.data`` before the physics step."""

    def post_step(self) -> None:
        """Inspect state after the step; may swap ``self.model``/``self.data``."""

    # -- GUI ---------------------------------------------------------------

    def set_traces(self, traces: list[PredictedTrace]) -> None:
        """Publish new predicted trajectories for the app to render."""
        self.traces = traces
        self.traces_version += 1

    def status(self) -> str:
        """Markdown status line(s) shown in the task panel."""
        return ""

    def on_param_changed(self, field_name: str) -> None:
        """Called after the GUI writes a new value into ``self.params``."""

    # -- helpers -----------------------------------------------------------

    @property
    def track_body_id(self) -> int:
        if self.track_body is not None:
            return self.model.body_id(self.track_body)
        return 1
