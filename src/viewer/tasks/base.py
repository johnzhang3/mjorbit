# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Task abstraction for the interactive viewer app.

A :class:`ViewerTask` owns its simulation: it builds the ``(MjoModel,
MjoData)`` pair, supplies controls before every physics step, and may swap
its own model/data mid-run (e.g. activating a weld at capture). The app
notices model identity changes and rebuilds the rendered scene.

Numeric fields of the task's params dataclass become GUI sliders. Use
:func:`ui_field` to set explicit bounds; bool fields become checkboxes.
"""

from __future__ import annotations

import abc
import dataclasses
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from mujoco_orbit import MjoData, MjoModel
from mujoco_orbit.constants import GM_EARTH, R_EARTH


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
    #: Bodies whose trajectories are drawn as trails.
    trail_bodies: ClassVar[tuple[str, ...]] = ()

    model: MjoModel
    data: MjoData

    def __init__(self) -> None:
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

    # -- per-step hooks ---------------------------------------------------

    def pre_step(self) -> None:
        """Write controls into ``self.data`` before the physics step."""

    def post_step(self) -> None:
        """Inspect state after the step; may swap ``self.model``/``self.data``."""

    # -- GUI ---------------------------------------------------------------

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
