"""Direct tests for the MJWarp-backed ``MjoModel`` / ``MjoData`` API."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

pytest.importorskip("mujoco_warp")

import mjorbit as mjo_cpu
import mjorbit_warp as mjow
from mjorbit.constants import R_EARTH
from mjorbit.testdata import FREE_BODY_SENSORS_XML, FREE_BODY_XML
from mjorbit_warp import (
    MagneticBodySpec,
    MagnetorquerSpec,
    MjoModel,
    OrbitInit,
    ReactionWheelSpec,
    SurfaceSpec,
    ThrusterSpec,
    mjo_forward,
    mjo_pull,
    mjo_step,
    mjo_upload,
)
from tests.mjorbit._helpers import _xml_with_mjorbit as _cpu_xml_with_mjorbit
from tests.mjorbit.reference.orbit.elements import keplerian_to_cartesian


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
    data.upload(fields="state")
    mjo_forward(model, data)
    mjo_step(model, data)
    mjo_forward(model, data)
    mjo_pull(model, data)

    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))
    assert np.all(np.isfinite(data.xmat))
    # TODO(sensors): re-add data.sensors.measure(...) check once Warp-side
    # sensors are reimplemented. CPU sensor stack is the current reference.


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
    data.upload(fields="state")
    mjo_step(model, data)
    mjo_forward(model, data)
    data.pull(fields=("state", "xpos", "nefc"))

    np.testing.assert_allclose(data.time, time0 + model.opt.timestep)
    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.xpos))
    assert data.nefc.shape == (2,)


def test_selective_pull_refreshes_state_without_host_shadow(monkeypatch):
    import mujoco_warp as mjw

    model = MjoModel.from_xml_path(
        FREE_BODY_XML,
        mj_timestep=0.01,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
        reaction_wheels=[
            ReactionWheelSpec(
                body_name="spacecraft",
                axis_body=np.array([0.0, 0.0, 1.0]),
                inertia=0.01,
                torque_limit=0.02,
            )
        ],
    )
    orbit = _orbit_init()
    synced = model.make_data(orbit=orbit)
    device = model.make_data(orbit=OrbitInit(R_eci=orbit.R_eci.copy(), V_eci=orbit.V_eci.copy()))

    qvel = np.array([0.02, -0.01, 0.03, 0.01, 0.02, -0.01])
    for data in (synced, device):
        data.qvel[:] = qvel
        data.actuators.rw_speed[0] = 1.0
        data.actuators.rw_torque_cmd[0] = 0.01
        data.upload(fields=("state", "actuators"))

    mjo_forward(model, synced)
    mjo_forward(model, device)
    stale_public_time = float(device.time)
    stale_host_time = device.host_data().time

    for _ in range(12):
        mjo_step(model, synced)
        mjo_step(model, device)

    assert float(device.time) == stale_public_time
    synced.pull(fields=("state", "orbit", "actuators"))

    def fail_get_data_into(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("selective pull should not download full host MjData")

    monkeypatch.setattr(mjw, "get_data_into", fail_get_data_into)
    device.pull(fields=("state", "orbit", "actuators"))

    assert device.time == pytest.approx(synced.time)
    assert device.host_data().time == stale_host_time
    np.testing.assert_allclose(device.qpos, synced.qpos)
    np.testing.assert_allclose(device.qvel, synced.qvel)
    np.testing.assert_allclose(device.orbit.R_eci, synced.orbit.R_eci)
    np.testing.assert_allclose(device.orbit.V_eci, synced.orbit.V_eci)
    np.testing.assert_allclose(device.actuators.rw_speed, synced.actuators.rw_speed)
    np.testing.assert_allclose(device.actuators.rw_momentum, synced.actuators.rw_momentum)


def test_batched_selective_pull_refreshes_public_state_only(monkeypatch):
    import mujoco_warp as mjw

    model = MjoModel.from_xml_path(
        FREE_BODY_XML,
        mj_timestep=0.01,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    orbits = [_orbit_init(400.0), _orbit_init(500.0)]
    data = model.make_data(orbit=orbits, nworld=2)
    data.qvel[:, 0] = [0.1, -0.2]
    data.upload(fields="state")
    mjo_forward(model, data)
    stale_public_time = data.time.copy()
    stale_host_times = [run.mj_data.time for run in data._host_runs]

    for _ in range(5):
        mjo_step(model, data, sync=False)

    def fail_get_data_into(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("selective pull should not download full host MjData")

    monkeypatch.setattr(mjw, "get_data_into", fail_get_data_into)
    mjo_pull(model, data, fields=("state", "orbit"))

    np.testing.assert_allclose(data.time, stale_public_time + 5.0 * model.opt.timestep)
    for world_id, stale_time in enumerate(stale_host_times):
        assert data.host_data(world_id).time == stale_time
    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))
    assert np.all(np.isfinite(data.orbit.R_eci))


def test_selective_pull_reshapes_matrix_fields():
    model = MjoModel.from_xml_path(
        FREE_BODY_XML,
        mj_timestep=0.01,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    single_orbit = _orbit_init()
    batched_orbits = [_orbit_init(400.0), _orbit_init(500.0)]
    cases = [
        (model.make_data(orbit=single_orbit), model.make_data(orbit=single_orbit)),
        (
            model.make_data(orbit=batched_orbits, nworld=2),
            model.make_data(orbit=batched_orbits, nworld=2),
        ),
    ]
    matrix_fields = ("xmat", "ximat", "geom_xmat", "site_xmat", "cam_xmat")

    for reference, selective in cases:
        for data in (reference, selective):
            data.qpos[..., :3] = 0.1
            data.upload(fields="state")
            mjo_forward(model, data)

        mjo_pull(model, reference)
        selective.pull(fields=matrix_fields)

        for field in matrix_fields:
            np.testing.assert_allclose(getattr(selective, field), getattr(reference, field))
            assert getattr(selective, field).shape[-1] == 9


def test_selective_pull_rejects_unknown_field():
    model = MjoModel.from_xml_path(
        FREE_BODY_XML,
        mj_timestep=0.01,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    data = model.make_data(orbit=_orbit_init())

    with pytest.raises(ValueError, match="not_a_field"):
        data.pull(fields="not_a_field")


def test_default_forward_step_and_passthrough_do_not_transfer_host_buffers(monkeypatch):
    import importlib

    import mujoco_warp as mjw

    step_mod = importlib.import_module("mjorbit_warp.step")

    model = MjoModel.from_xml_path(
        FREE_BODY_XML,
        mj_timestep=0.01,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    data = model.make_data(orbit=_orbit_init())

    def fail_upload(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("default warp execution should not upload host buffers")

    def fail_pull(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("default warp execution should not download host buffers")

    monkeypatch.setattr(step_mod, "_sync_device_from_public", fail_upload)
    monkeypatch.setattr(mjw, "get_data_into", fail_pull)

    mjo_forward(model, data)
    mjo_step(model, data)
    mjow.kinematics(model, data)


def test_explicit_upload_refreshes_device_state_from_public_buffers():
    model = MjoModel.from_xml_path(
        FREE_BODY_XML,
        mj_timestep=0.01,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
    )
    data = model.make_data(orbit=_orbit_init())

    data.qpos[:3] = [0.25, -0.15, 0.05]
    mjo_upload(model, data, fields="state")
    mjo_forward(model, data)
    data.pull(fields="xpos")

    np.testing.assert_allclose(data.xpos[1], data.qpos[:3], atol=1e-6)


def test_forward_and_step_do_not_call_cpu_coupling(monkeypatch):
    model = MjoModel.from_xml_path(
        FREE_BODY_XML,
        mj_timestep=0.01,
        use_j2=True,
        use_drag=True,
        use_srp=True,
        use_magnetic=True,
        surfaces=[
            SurfaceSpec(
                body_name="spacecraft",
                center_of_pressure_body=np.array([0.0, 0.0, 0.1]),
                normal_body=np.array([0.0, 1.0, 0.0]),
                area=1.0,
            )
        ],
        magnetic_bodies=[
            MagneticBodySpec(body_name="spacecraft", dipole_body=np.array([0.01, 0.0, 0.0]))
        ],
        reaction_wheels=[
            ReactionWheelSpec(
                body_name="spacecraft",
                axis_body=np.array([0.0, 0.0, 1.0]),
                inertia=0.01,
                torque_limit=0.02,
            )
        ],
        magnetorquers=[
            MagnetorquerSpec(
                body_name="spacecraft",
                axis_body=np.array([1.0, 0.0, 0.0]),
                dipole_limit=0.1,
            )
        ],
        thrusters=[
            ThrusterSpec(
                body_name="spacecraft",
                position_body=np.array([0.1, 0.0, 0.0]),
                direction_body=np.array([0.0, 1.0, 0.0]),
                force_limit=1.0,
            )
        ],
    )
    data = model.make_data(orbit=_orbit_init())

    data.actuators.rw_speed[0] = 1.0
    data.actuators.rw_torque_cmd[0] = 0.01
    data.actuators.mtq_dipole_cmd[0] = 0.05
    data.actuators.thr_force_cmd[0] = 0.5
    data.upload(fields="core")

    mjo_forward(model, data)
    mjo_step(model, data)
    mjo_forward(model, data)
    data.pull(fields=("state", "orbit", "actuators"))

    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.orbit.R_eci))
    assert data.actuators.rw_speed[0] > 1.0


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
    data.upload(fields="state")
    mjow.forward(model, data)
    data.pull(fields="xpos")
    np.testing.assert_allclose(data.xpos[1], data.qpos[:3], atol=1e-6)

    time0 = data.time
    mjow.step(model, data)
    data.pull(fields="state")
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

    assert mjow.put_model(model) is model
    assert mjow.put_data(model, data) is data
    assert mjow.make_data(model.mj_model, nworld=1).nworld == 1
    assert mjow.Model is not None
    assert mjow.Data is not None
    assert mjow.GeomType is not None

    data.qpos[:3] = [0.4, 0.2, -0.1]
    data.upload(fields="state")
    mjow.kinematics(model, data)
    data.pull(fields="xpos")
    np.testing.assert_allclose(data.xpos[1], data.qpos[:3], atol=1e-6)
    assert data.device_data is data.warp_data
    assert model.device_model is model.warp_model


def test_raw_mujoco_put_model_and_put_data_match_mjwarp_style():
    mjm = mujoco.MjModel.from_xml_path(FREE_BODY_XML)
    mjd = mujoco.MjData(mjm)

    m = mjow.put_model(mjm)
    d = mjow.put_data(mjm, mjd)

    assert not isinstance(m, MjoModel)
    assert not isinstance(d, mjow.MjoData)
    assert d.nworld == 1
    mjow.forward(m, d)


def test_cpu_orbit_put_model_and_put_data_return_warp_wrappers(tmp_path: Path):
    orbit = _orbit_init()
    cpu_xml = _cpu_xml_with_mjorbit(
        FREE_BODY_XML,
        orbit_dt=None,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=False,
        use_gravity_gradient=True,
        surfaces=(),
        magnetic_bodies=(),
        reaction_wheels=(),
        magnetorquers=(),
        thrusters=(),
        cmgs=(),
    )
    cpu_xml_path = tmp_path / "cpu_free_body.xml"
    cpu_xml_path.write_text(cpu_xml)
    cpu_model = mjo_cpu.MjoModel.from_xml_path(str(cpu_xml_path), mj_timestep=0.01)
    cpu_data = cpu_model.make_data(orbit=orbit)
    cpu_data.qpos[:3] = [0.3, -0.2, 0.1]
    cpu_data.qvel[:3] = [0.01, 0.02, -0.03]

    model = mjow.put_model(cpu_model)
    data = mjow.put_data(cpu_model, cpu_data)

    assert isinstance(model, MjoModel)
    assert isinstance(data, mjow.MjoData)
    assert data.model is model
    assert mjow.put_model(cpu_model) is model

    mjow.forward(model, data)
    data.pull(fields=("state", "xpos", "orbit"))

    np.testing.assert_allclose(data.qpos, cpu_data.qpos)
    np.testing.assert_allclose(data.qvel, cpu_data.qvel)
    np.testing.assert_allclose(data.xpos[1], cpu_data.qpos[:3], atol=1e-6)
    np.testing.assert_allclose(data.orbit.R_eci, cpu_data.orbit.R_eci)
    np.testing.assert_allclose(data.orbit.V_eci, cpu_data.orbit.V_eci)
