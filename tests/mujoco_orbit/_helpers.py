"""Shared helpers for tests using the MuJoCo-style API."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from mujoco_orbit import (
    MagneticBodySpec,
    MagnetorquerSpec,
    MjoData,
    MjoModel,
    OrbitInit,
    ReactionWheelSpec,
    SurfaceSpec,
    ThrusterSpec,
)
from mujoco_orbit.constants import R_EARTH
from mujoco_orbit.orbit.elements import keplerian_to_cartesian


def circular_leo_orbit_init(alt_km: float = 400.0) -> OrbitInit:
    """Return a default circular LEO initialization used across tests."""
    a = R_EARTH + alt_km
    r_eci, v_eci = keplerian_to_cartesian(
        a=a, e=0.0, inc=np.deg2rad(51.6), raan=0.0, argp=0.0, nu=0.0
    )
    return OrbitInit(R_eci=r_eci, V_eci=v_eci)


def make_model_data(
    *,
    xml_path: str,
    alt_km: float = 400.0,
    mj_timestep: float | None = 0.01,
    orbit_dt: float | None = None,
    use_j2: bool = False,
    use_drag: bool = False,
    use_srp: bool = False,
    use_magnetic: bool = False,
    surfaces: Iterable[SurfaceSpec] = (),
    magnetic_bodies: Iterable[MagneticBodySpec] = (),
    reaction_wheels: Iterable[ReactionWheelSpec] = (),
    magnetorquers: Iterable[MagnetorquerSpec] = (),
    thrusters: Iterable[ThrusterSpec] = (),
    rng_seed: int | None = None,
) -> tuple[MjoModel, MjoData]:
    """Build one model/data pair for a standard circular LEO test orbit."""
    model = MjoModel.from_xml_path(
        xml_path,
        surfaces=surfaces,
        magnetic_bodies=magnetic_bodies,
        reaction_wheels=reaction_wheels,
        magnetorquers=magnetorquers,
        thrusters=thrusters,
        mj_timestep=mj_timestep,
        orbit_dt=orbit_dt,
        use_j2=use_j2,
        use_drag=use_drag,
        use_srp=use_srp,
        use_magnetic=use_magnetic,
    )
    data = model.make_data(orbit=circular_leo_orbit_init(alt_km), rng_seed=rng_seed)
    return model, data
