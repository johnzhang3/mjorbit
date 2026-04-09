"""Direct tests for the MJWarp-backed ``MjoModel`` / ``MjoData`` API."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")

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
    data = model.make_data(orbit=[orbit_a, orbit_b], nworld=2)

    assert data.nworld == 2
    assert data.qpos.shape == (2, model.nq)
    assert data.qvel.shape == (2, model.nv)
    assert data.xfrc_applied.shape == (2, model.nbody, 6)
    assert data.wrench_buffer.shape == (2, model.nbody, 6)
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
