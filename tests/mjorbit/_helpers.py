"""Shared helpers for tests using the MuJoCo-style API."""

from __future__ import annotations

import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from mjorbit import (
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
from mjorbit.constants import R_EARTH
from tests.mjorbit.reference.orbit.elements import keplerian_to_cartesian


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
    configured_xml = _xml_with_mjorbit(
        xml_path,
        orbit_dt=orbit_dt,
        use_j2=use_j2,
        use_drag=use_drag,
        use_srp=use_srp,
        use_magnetic=use_magnetic,
        use_gravity_gradient=use_gravity_gradient,
        surfaces=surfaces,
        magnetic_bodies=magnetic_bodies,
        reaction_wheels=reaction_wheels,
        magnetorquers=magnetorquers,
        thrusters=thrusters,
        cmgs=cmgs,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / Path(xml_path).name
        path.write_text(configured_xml)
        model = MjoModel.from_xml_path(str(path), mj_timestep=mj_timestep)
        data = model.make_data(orbit=circular_leo_orbit_init(alt_km), rng_seed=rng_seed)
    return model, data


def _xml_with_mjorbit(
    xml_path: str,
    *,
    orbit_dt: float | None,
    use_j2: bool,
    use_drag: bool,
    use_srp: bool,
    use_magnetic: bool,
    use_gravity_gradient: bool,
    surfaces: Iterable[SurfaceSpec],
    magnetic_bodies: Iterable[MagneticBodySpec],
    reaction_wheels: Iterable[ReactionWheelSpec],
    magnetorquers: Iterable[MagnetorquerSpec],
    thrusters: Iterable[ThrusterSpec],
    cmgs: Iterable[ControlMomentGyroSpec],
) -> str:
    attrs = [
        f'use_j2="{_xml_bool(use_j2)}"',
        f'use_drag="{_xml_bool(use_drag)}"',
        f'use_srp="{_xml_bool(use_srp)}"',
        f'use_magnetic="{_xml_bool(use_magnetic)}"',
        f'use_gravity_gradient="{_xml_bool(use_gravity_gradient)}"',
    ]
    if orbit_dt is not None:
        attrs.append(f'orbit_dt="{float(orbit_dt):.17g}"')

    children: list[str] = []
    for i, surface in enumerate(surfaces):
        children.append(
            f'<surface name="{surface.name or f"surface_{i}"}" body="{surface.body_name}" '
            f'cop="{_vec(surface.center_of_pressure_body)}" '
            f'normal="{_vec(surface.normal_body)}" area="{surface.area:.17g}" '
            f'drag_coeff="{surface.drag_coeff:.17g}" srp_coeff="{surface.srp_coeff:.17g}" '
            f'use_drag="{_xml_bool(surface.use_drag)}" use_srp="{_xml_bool(surface.use_srp)}"/>'
        )
    for i, magnetic in enumerate(magnetic_bodies):
        children.append(
            f'<magnetic_body name="{magnetic.name or f"magnetic_{i}"}" body="{magnetic.body_name}" '
            f'dipole="{_vec(magnetic.dipole_body)}"/>'
        )
    for i, wheel in enumerate(reaction_wheels):
        extra = ""
        if wheel.speed_limit is not None:
            extra += f' speed_limit="{wheel.speed_limit:.17g}"'
        if wheel.torque_limit is not None:
            extra += f' torque_limit="{wheel.torque_limit:.17g}"'
        children.append(
            f'<reaction_wheel name="{wheel.name or f"rw_{i}"}" body="{wheel.body_name}" '
            f'axis="{_vec(wheel.axis_body)}" inertia="{wheel.inertia:.17g}"{extra}/>'
        )
    for i, mtq in enumerate(magnetorquers):
        children.append(
            f'<magnetorquer name="{mtq.name or f"mtq_{i}"}" body="{mtq.body_name}" '
            f'axis="{_vec(mtq.axis_body)}" dipole_limit="{mtq.dipole_limit:.17g}"/>'
        )
    for i, thruster in enumerate(thrusters):
        children.append(
            f'<thruster name="{thruster.name or f"thr_{i}"}" body="{thruster.body_name}" '
            f'pos="{_vec(thruster.position_body)}" dir="{_vec(thruster.direction_body)}" '
            f'force_limit="{thruster.force_limit:.17g}"/>'
        )
    for i, cmg in enumerate(cmgs):
        extra = ""
        if cmg.gimbal_rate_limit is not None:
            extra += f' gimbal_rate_limit="{cmg.gimbal_rate_limit:.17g}"'
        if cmg.gimbal_angle_limit is not None:
            extra += f' gimbal_angle_limit="{cmg.gimbal_angle_limit:.17g}"'
        children.append(
            f'<cmg name="{cmg.name or f"cmg_{i}"}" body="{cmg.body_name}" '
            f'gimbal_axis="{_vec(cmg.gimbal_axis_body)}" '
            f'spin_axis0="{_vec(cmg.spin_axis_body_0)}" '
            f'rotor_momentum="{cmg.rotor_momentum:.17g}"{extra}/>'
        )

    block = "\n  <mjorbit " + " ".join(attrs) + ">\n"
    block += "".join(f"    {child}\n" for child in children)
    block += "  </mjorbit>\n"
    return Path(xml_path).read_text().replace("</mujoco>", block + "</mujoco>")


def _vec(values: Iterable[float]) -> str:
    return " ".join(f"{float(x):.17g}" for x in values)


def _xml_bool(value: bool) -> str:
    return "true" if value else "false"


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
