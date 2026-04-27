"""Programmatic ``MjoSpec`` construction tests."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_orbit import (
    CentralBodySpec,
    ControlMomentGyroSpec,
    MagneticBodySpec,
    MagnetorquerSpec,
    MjoData,
    MjoModel,
    MjoSpec,
    ReactionWheelSpec,
    SurfaceSpec,
    ThrusterSpec,
)
from mujoco_orbit.testdata import FREE_BODY_XML

from ._helpers import circular_leo_orbit_init


def _xml_with_mjorbit(children: str = "", attrs: str = "") -> str:
    text = Path(FREE_BODY_XML).read_text()
    block = f"""
  <mjorbit plugin_body="spacecraft" {attrs}>
    <central_body name="moon" gm="4902.800066" radius="1737.4" j2="0"
                  omega="0 0 2.6617e-6" magnetic_b0="0"
                  magnetic_axis="0 0 1" atmosphere_rho0="0"/>
    {children}
  </mjorbit>
"""
    return text.replace("</mujoco>", block + "</mujoco>")


def test_from_xml_path_compile_keeps_model_wrapper_compatibility() -> None:
    spec = MjoSpec.from_xml_path(FREE_BODY_XML)
    model = spec.compile(mj_timestep=0.02)

    assert model.nbody == MjoModel.from_xml_path(FREE_BODY_XML, mj_timestep=0.02).nbody
    assert model.opt.timestep == 0.02
    assert model.use_j2
    assert model.orbit_plugin_instance >= 0


def test_from_xml_string_round_trips_named_mjorbit_block() -> None:
    xml = _xml_with_mjorbit(
        """
        <surface name="panel_x" body="spacecraft" cop="0.5 0 0"
                 normal="1 0 0" area="2.0"/>
        """,
        attrs='orbit_dt="0.25" use_drag="true"',
    )
    spec = MjoSpec.from_xml_string(xml)

    assert spec.mjorbit.orbit_dt == 0.25
    assert spec.mjorbit.central_body.name == "moon"
    assert spec.mjorbit.surfaces[0].name == "panel_x"

    round_trip = MjoSpec.from_xml_string(spec.to_xml())
    assert round_trip.mjorbit.central_body.gm == pytest.approx(4902.800066)
    assert round_trip.mjorbit.surfaces[0].name == "panel_x"
    assert '<mjorbit plugin_body="spacecraft"' in round_trip.to_xml()


def test_from_mj_spec_accepts_mujoco_programmatic_spec() -> None:
    mj_spec = mujoco.MjSpec.from_string(Path(FREE_BODY_XML).read_text())

    spec = MjoSpec.from_mj_spec(mj_spec)
    spec.mjorbit.use_drag = False
    model = spec.compile()

    assert model.nbody == 2
    assert not model.use_drag


def test_programmatic_surface_mutators_compile() -> None:
    spec = MjoSpec.from_xml_path(FREE_BODY_XML)
    name = spec.mjorbit.add_surface(
        body_name="spacecraft",
        center_of_pressure_body=[0.5, 0.0, 0.0],
        normal_body=[1.0, 0.0, 0.0],
        area=2.0,
    )

    assert name == "surface_0"
    assert spec.mjorbit.surfaces[0].name == "surface_0"

    spec.mjorbit.update_surface(name, area=3.0, name="panel_x")
    model = spec.compile()
    assert len(model.surfaces) == 1
    assert model.surfaces[0].area == 3.0
    assert spec.mjorbit.surfaces[0].name == "panel_x"

    spec.mjorbit.remove_surface("panel_x")
    assert spec.compile().surfaces == []


def test_programmatic_actuator_and_environment_mutators_compile() -> None:
    spec = MjoSpec.from_xml_path(FREE_BODY_XML)
    spec.mjorbit.add_magnetic_body(
        MagneticBodySpec("spacecraft", np.array([0.0, 0.0, 0.1]), name="mag")
    )
    spec.mjorbit.add_reaction_wheel(
        ReactionWheelSpec("spacecraft", np.array([0.0, 0.0, 1.0]), 0.01, name="rw")
    )
    spec.mjorbit.add_magnetorquer(
        MagnetorquerSpec("spacecraft", np.array([1.0, 0.0, 0.0]), 5.0, name="mtq")
    )
    spec.mjorbit.add_thruster(
        ThrusterSpec(
            "spacecraft",
            np.zeros(3),
            np.array([0.0, 1.0, 0.0]),
            10.0,
            name="thr",
        )
    )
    spec.mjorbit.add_cmg(
        ControlMomentGyroSpec(
            "spacecraft",
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            0.2,
            name="cmg",
        )
    )

    spec.mjorbit.update_thruster("thr", force_limit=12.0)
    spec.mjorbit.update_cmg("cmg", gimbal_rate_limit=0.1)
    model = spec.compile()

    assert len(model.magnetic_bodies) == 1
    assert len(model.reaction_wheels) == 1
    assert len(model.magnetorquers) == 1
    assert len(model.thrusters) == 1
    assert len(model.cmgs) == 1
    assert model.thrusters[0].force_limit == 12.0
    assert model.cmgs[0].gimbal_rate_limit == 0.1

    spec.mjorbit.remove_thruster("thr")
    assert spec.compile().thrusters == []


def test_duplicate_names_raise_clear_error() -> None:
    spec = MjoSpec.from_xml_path(FREE_BODY_XML)
    spec.mjorbit.add_surface(
        SurfaceSpec("spacecraft", np.zeros(3), np.array([1.0, 0.0, 0.0]), 1.0, name="panel")
    )

    with pytest.raises(ValueError, match="Duplicate"):
        spec.mjorbit.add_surface(
            SurfaceSpec(
                "spacecraft",
                np.zeros(3),
                np.array([0.0, 1.0, 0.0]),
                1.0,
                name="panel",
            )
        )


def test_compile_validates_body_references_vectors_and_limits() -> None:
    bad_body = MjoSpec.from_xml_path(FREE_BODY_XML)
    bad_body.mjorbit.add_surface(
        SurfaceSpec("missing", np.zeros(3), np.array([1.0, 0.0, 0.0]), 1.0, name="panel")
    )
    with pytest.raises(ValueError, match="not found"):
        bad_body.compile()

    bad_axis = MjoSpec.from_xml_path(FREE_BODY_XML)
    bad_axis.mjorbit.add_reaction_wheel(
        ReactionWheelSpec("spacecraft", np.zeros(3), 0.01, name="rw")
    )
    with pytest.raises(ValueError, match="axis must be non-zero"):
        bad_axis.compile()

    bad_limit = MjoSpec.from_xml_path(FREE_BODY_XML)
    bad_limit.mjorbit.add_thruster(
        ThrusterSpec("spacecraft", np.zeros(3), np.array([1.0, 0.0, 0.0]), -1.0, name="thr")
    )
    with pytest.raises(ValueError, match="force_limit must be positive"):
        bad_limit.compile()


def test_central_body_constants_are_compiled_into_data_runtime() -> None:
    spec = MjoSpec.from_xml_path(FREE_BODY_XML)
    spec.mjorbit.central_body = CentralBodySpec(omega=[0.0, 0.0, 1.0e-4])
    spec.mjorbit.central_body.gm = 12345.0

    model = spec.compile()
    data = MjoData(model, orbit=circular_leo_orbit_init())

    assert model.central_body.gm == 12345.0
    np.testing.assert_allclose(data.env.atmosphere_omega_eci, [0.0, 0.0, 1.0e-4])
