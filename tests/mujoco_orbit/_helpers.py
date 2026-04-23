"""Shared helpers for tests using the MuJoCo-style API."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from mujoco_orbit import (
    ControlMomentGyroSpec,
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
    use_gravity_gradient: bool = True,
    surfaces: Iterable[SurfaceSpec] = (),
    magnetic_bodies: Iterable[MagneticBodySpec] = (),
    reaction_wheels: Iterable[ReactionWheelSpec] = (),
    magnetorquers: Iterable[MagnetorquerSpec] = (),
    thrusters: Iterable[ThrusterSpec] = (),
    cmgs: Iterable[ControlMomentGyroSpec] = (),
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
        cmgs=cmgs,
        mj_timestep=mj_timestep,
        orbit_dt=orbit_dt,
        use_j2=use_j2,
        use_drag=use_drag,
        use_srp=use_srp,
        use_magnetic=use_magnetic,
        use_gravity_gradient=use_gravity_gradient,
    )
    data = MjoData(model, orbit=circular_leo_orbit_init(alt_km), rng_seed=rng_seed)
    return model, data


def set_freejoint_lvlh_state(
    data: MjoData,
    qpos_slice: slice,
    qvel_slice: slice,
    position_lvlh_m: Iterable[float],
    velocity_lvlh_m_s: Iterable[float] = (0.0, 0.0, 0.0),
) -> None:
    """Set a free joint from chief-relative LVLH position/velocity."""
    position = np.asarray(position_lvlh_m, dtype=float)
    velocity = np.asarray(velocity_lvlh_m_s, dtype=float)
    data.qpos[qpos_slice] = data.world_position_from_lvlh(position)
    data.qvel[qvel_slice] = data.world_velocity_from_lvlh(position, velocity)


def get_freejoint_lvlh_state(
    data: MjoData,
    qpos_slice: slice,
    qvel_slice: slice,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a free joint's chief-relative LVLH position/velocity."""
    position = data.lvlh_position_from_world(data.qpos[qpos_slice])
    velocity = data.lvlh_velocity_from_world(data.qpos[qpos_slice], data.qvel[qvel_slice])
    return position, velocity
