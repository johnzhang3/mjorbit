# pyright: reportAttributeAccessIssue=false

"""``MjoModel`` — orbit-aware MJWarp model wrapper.

Holds three views of the same physics:
  * ``host_model``: the CPU ``mjorbit.MjoModel`` (with native plugin)
  * ``mj_model``: the raw ``mujoco.MjModel`` (consumed by MJWarp)
  * ``warp_model``: the MJWarp device-side model
plus ``core_model`` (orbit overlay metadata uploaded to the GPU).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mujoco

if TYPE_CHECKING:
    from .data import MjoData

from mjorbit.config import (
    MagneticBodySpec,
    MagnetorquerSpec,
    OrbitInit,
    ReactionWheelSpec,
    SurfaceSpec,
    ThrusterSpec,
)
from mjorbit.model import MjoModel as CpuMjoModel
from mjorbit.spec import MjoSpec, _raw_mujoco_xml

from ._deps import require_mjwarp


def _compile_raw_mujoco_model(raw_xml: str, *, mj_timestep: float | None) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_string(raw_xml)
    if mj_timestep is not None:
        model.opt.timestep = float(mj_timestep)
    model.opt.gravity[:] = 0.0
    return model

@dataclass
class MjoModel:
    """MJWarp-backed model wrapper with host metadata retained."""

    host_model: CpuMjoModel
    mj_model: mujoco.MjModel
    warp_model: Any
    core_model: Any
    backend: str = "warp"

    @classmethod
    def from_host_model(cls, host_model: CpuMjoModel) -> "MjoModel":
        """Upload a CPU mjorbit model into the orbit-aware MJWarp wrapper."""
        mjw, _ = require_mjwarp()
        from .core_gpu import make_device_core_model

        raw_xml = getattr(host_model, "_raw_xml", None)
        if raw_xml is None:
            raise TypeError(
                "Cannot upload this CPU MjoModel to MJWarp because it does not retain raw XML"
            )
        mj_model = _compile_raw_mujoco_model(raw_xml, mj_timestep=float(host_model.opt.timestep))
        return cls(
            host_model=host_model,
            mj_model=mj_model,
            warp_model=mjw.put_model(mj_model),
            core_model=make_device_core_model(host_model),
        )

    @classmethod
    def from_xml_path(
        cls,
        xml_path: str,
        *,
        surfaces: Iterable[SurfaceSpec] = (),
        magnetic_bodies: Iterable[MagneticBodySpec] = (),
        reaction_wheels: Iterable[ReactionWheelSpec] = (),
        magnetorquers: Iterable[MagnetorquerSpec] = (),
        thrusters: Iterable[ThrusterSpec] = (),
        mj_timestep: float | None = 0.01,
        orbit_dt: float | None = None,
        use_j2: bool | None = None,
        use_drag: bool | None = None,
        use_srp: bool | None = None,
        use_magnetic: bool | None = None,
        use_gravity_gradient: bool | None = None,
    ) -> "MjoModel":
        """Compile the host model and upload an MJWarp device model.

        The ``use_*`` flags and ``orbit_dt`` default to ``None``, meaning the
        values parsed from the XML ``<mjorbit>`` element are kept (matching
        the CPU backend); passing an explicit value overrides the XML.
        """
        spec = MjoSpec.from_xml_path(xml_path)
        if use_j2 is not None:
            spec.mjorbit.use_j2 = use_j2
        if use_drag is not None:
            spec.mjorbit.use_drag = use_drag
        if use_srp is not None:
            spec.mjorbit.use_srp = use_srp
        if use_magnetic is not None:
            spec.mjorbit.use_magnetic = use_magnetic
        if use_gravity_gradient is not None:
            spec.mjorbit.use_gravity_gradient = use_gravity_gradient
        if orbit_dt is not None:
            spec.mjorbit.orbit_dt = orbit_dt
        for surface in surfaces:
            spec.mjorbit.add_surface(surface)
        for magnetic_body in magnetic_bodies:
            spec.mjorbit.add_magnetic_body(magnetic_body)
        for wheel in reaction_wheels:
            spec.mjorbit.add_reaction_wheel(wheel)
        for magnetorquer in magnetorquers:
            spec.mjorbit.add_magnetorquer(magnetorquer)
        for thruster in thrusters:
            spec.mjorbit.add_thruster(thruster)

        host_model = spec.compile(mj_timestep=mj_timestep)
        raw_xml = _raw_mujoco_xml(spec.to_xml())
        mj_model = _compile_raw_mujoco_model(raw_xml, mj_timestep=mj_timestep)

        mjw, _ = require_mjwarp()
        from .core_gpu import make_device_core_model

        return cls(
            host_model=host_model,
            mj_model=mj_model,
            warp_model=mjw.put_model(mj_model),
            core_model=make_device_core_model(host_model),
        )

    def __getattr__(self, name: str):  # pragma: no cover - trivial delegation
        return getattr(self.host_model, name)

    @property
    def device_model(self) -> Any:
        """Return the underlying ``mujoco_warp.Model``."""
        return self.warp_model

    def body_id(self, name: str) -> int:
        return self.host_model.body_id(name)

    def sensor(self, name: str):
        return self.host_model.sensor(name)

    def make_data(
        self,
        *,
        orbit: OrbitInit | Sequence[OrbitInit],
        rng_seed: int | Sequence[int | None] | None = None,
        nworld: int = 1,
        nconmax: int | None = None,
        nccdmax: int | None = None,
        njmax: int | None = None,
        naconmax: int | None = None,
        naccdmax: int | None = None,
    ) -> "MjoData":
        """Construct one or more runtime state objects for this compiled model."""
        # Late import: data.py imports MjoModel from this module, so we cannot
        # bring MjoData into the module namespace at top level.
        from .data import MjoData

        return MjoData(
            self,
            orbit=orbit,
            rng_seed=rng_seed,
            nworld=nworld,
            nconmax=nconmax,
            nccdmax=nccdmax,
            njmax=njmax,
            naconmax=naconmax,
            naccdmax=naccdmax,
        )


__all__ = [
    'MjoModel',
]
