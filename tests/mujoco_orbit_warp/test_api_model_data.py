"""Direct tests for the MJWarp-backed ``MjoModel`` / ``MjoData`` API."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")

import mujoco_orbit_warp as mjow
from mujoco_orbit.constants import R_EARTH
from mujoco_orbit.orbit.elements import keplerian_to_cartesian
from mujoco_orbit.testdata import FREE_BODY_SENSORS_XML, FREE_BODY_XML
from mujoco_orbit_warp import MjoModel, OrbitInit, mjo_forward, mjo_step


def _orbit_init(alt_km: float = 400.0) -> OrbitInit:
    a = R_EARTH + alt_km
    r_eci, v_eci = keplerian_to_cartesian(
        a=a,
        e=0.0,
        inc=np.deg2rad(51.6),
        raan=0.0,
        argp=0.0,
        nu=0.0,
    )
    return OrbitInit(R_eci=r_eci, V_eci=v_eci)


def test_model_make_data_exposes_backend_and_single_world():
    model = MjoModel.from_xml_path(
        FREE_BODY_XML,
        mj_timestep=0.01,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    data = model.make_data(orbit=_orbit_init())

    assert model.backend == "warp"
    assert data.backend == "warp"
    assert data.nworld == 1
    assert data.qpos.shape == (model.nq,)
    assert data.xfrc_applied.shape == (model.nbody, 6)
    assert data.host_data(0) is data.mj_data


def test_model_make_data_batches_worlds():
    model = MjoModel.from_xml_path(
        FREE_BODY_XML,
        mj_timestep=0.01,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    orbit_a = _orbit_init(400.0)
    orbit_b = _orbit_init(500.0)
    data = model.make_data(orbit=[orbit_a, orbit_b], nworld=2, nconmax=16, njmax=32)

    assert data.nworld == 2
    assert data.warp_data.naconmax == 32
    assert data.warp_data.njmax == 32
    assert data.qpos.shape == (2, model.nq)
    assert data.qvel.shape == (2, model.nv)
    assert data.xfrc_applied.shape == (2, model.nbody, 6)
    assert data.xpos.shape == (2, model.nbody, 3)
    assert data.geom_xpos.shape == (2, model.ngeom, 3)
    assert data.wrench_buffer.shape == (2, model.nbody, 6)
    assert data.ncon.shape == (2,)
    assert data.orbit.R_eci.shape == (2, 3)
    np.testing.assert_allclose(data.orbit.R_eci[0], orbit_a.R_eci)
    np.testing.assert_allclose(data.orbit.R_eci[1], orbit_b.R_eci)


def test_forward_and_step_smoke_for_single_world():
    model = MjoModel.from_xml_path(
        FREE_BODY_SENSORS_XML,
        mj_timestep=0.01,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=True,
    )
    data = model.make_data(orbit=_orbit_init())

    data.qpos[:3] = [0.2, -0.1, 0.05]
    data.qvel[:3] = [0.01, 0.0, -0.02]
    mjo_forward(model, data)
    mjo_step(model, data)

    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))
    assert np.all(np.isfinite(data.xmat))
    truth = data.sensors.measure("gyro_body", noisy=False)
    assert truth.shape == (3,)


def test_batched_step_smoke():
    model = MjoModel.from_xml_path(
        FREE_BODY_XML,
        mj_timestep=0.01,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    data = model.make_data(orbit=[_orbit_init(400.0), _orbit_init(500.0)], nworld=2)

    time0 = data.time.copy()
    data.qvel[:, 0] = [0.1, -0.2]
    mjo_step(model, data)

    np.testing.assert_allclose(data.time, time0 + model.opt.timestep)
    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.xpos))
    assert data.nefc.shape == (2,)


def test_mujoco_warp_style_forward_and_step_aliases_use_orbit_wrapper():
    model = MjoModel.from_xml_path(
        FREE_BODY_XML,
        mj_timestep=0.01,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    data = model.make_data(orbit=_orbit_init())

    data.qpos[:3] = [0.1, -0.2, 0.3]
    mjow.forward(model, data)
    np.testing.assert_allclose(data.xpos[1], data.qpos[:3], atol=1e-6)

    time0 = data.time
    mjow.step(model, data)
    assert data.time == pytest.approx(time0 + model.opt.timestep)
    assert np.all(np.isfinite(data.qpos))


def test_standard_mujoco_warp_api_is_available_from_orbit_package():
    model = MjoModel.from_xml_path(
        FREE_BODY_XML,
        mj_timestep=0.01,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    data = model.make_data(orbit=_orbit_init())

    assert mjow.put_model(model) is model.warp_model
    assert mjow.make_data(model.mj_model, nworld=1).nworld == 1
    assert mjow.Model is not None
    assert mjow.Data is not None
    assert mjow.GeomType is not None

    data.qpos[:3] = [0.4, 0.2, -0.1]
    mjow.kinematics(model, data)
    np.testing.assert_allclose(data.xpos[1], data.qpos[:3], atol=1e-6)
    assert data.device_data is data.warp_data
    assert model.device_model is model.warp_model
