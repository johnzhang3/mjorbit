"""Sensor discovery, noise injection, and Wahba verification tests."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from mjorbit import mjo_forward
from mjorbit.testdata import FREE_BODY_SENSORS_XML
from tests.mjorbit.reference.sensors import (
    _POS_STAGE,
    _QUATERNION_DATATYPE,
    _USER_SENSOR_TYPE,
    SensorDescriptor,
    _apply_sensor_noise,
    _quat_from_rotvec,
    _quat_mul,
)

from ._helpers import make_model_data


def _make_model_data(xml_path: str = FREE_BODY_SENSORS_XML, *, rng_seed: int | None = None):
    return make_model_data(
        xml_path=xml_path,
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=True,
        rng_seed=rng_seed,
    )


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    half = 0.5 * angle
    return np.array([np.cos(half), *(np.sin(half) * axis)])


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y**2 + z**2), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x**2 + z**2), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x**2 + y**2)],
        ]
    )


def _rotation_matrix_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        return np.eye(3) + np.array(
            [
                [0.0, -rotvec[2], rotvec[1]],
                [rotvec[2], 0.0, -rotvec[0]],
                [-rotvec[1], rotvec[0], 0.0],
            ]
        )

    axis = rotvec / theta
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _set_attitude(
    model,
    data,
    quat_world_body: np.ndarray,
    omega_body: np.ndarray = np.array([0.05, -0.02, 0.03]),
) -> None:
    data.qpos[3:7] = quat_world_body
    data.qvel[3:6] = omega_body
    mjo_forward(model, data)


def _make_sensor_descriptor(
    *, name: str, datatype: int, dim: int, noise: float
) -> SensorDescriptor:
    return SensorDescriptor(
        sensor_id=0,
        name=name,
        sensor_type=_USER_SENSOR_TYPE,
        datatype=datatype,
        objtype=0,
        objid=0,
        adr=0,
        dim=dim,
        noise=noise,
        cutoff=0.0,
        needstage=_POS_STAGE,
        user=np.zeros(0),
    )


def _q_method(
    body_vecs: list[np.ndarray], world_vecs: list[np.ndarray], weights: list[float]
) -> np.ndarray:
    B = np.zeros((3, 3))
    for body_vec, world_vec, weight in zip(body_vecs, world_vecs, weights):
        B += weight * np.outer(body_vec, world_vec)

    S = B + B.T
    sigma = np.trace(B)
    Z = np.array([B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]])

    K = np.zeros((4, 4))
    K[0, 0] = sigma
    K[0, 1:] = Z
    K[1:, 0] = Z
    K[1:, 1:] = S - sigma * np.eye(3)

    eigenvalues, eigenvectors = np.linalg.eigh(K)
    q_bw = eigenvectors[:, np.argmax(eigenvalues)]
    if q_bw[0] < 0.0:
        q_bw = -q_bw
    return q_bw / np.linalg.norm(q_bw)


def _rotation_error_rad(r_est: np.ndarray, r_true: np.ndarray) -> float:
    delta = r_est @ r_true.T
    cos_angle = 0.5 * (np.trace(delta) - 1.0)
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def _wahba_inputs(model, data, *, noisy: bool, rng: np.random.Generator | None = None):
    suite = model.sensors

    sun_world = data.env.sun_vector_eci
    nadir_eci = -data.orbit.R_eci / np.linalg.norm(data.orbit.R_eci)
    horizon_world = nadir_eci
    star_world = suite.by_name["orbit_star_body"].reference_eci
    mag_world = data.env.mag_field_eci

    body_vecs = [
        data.sensors.measure("orbit_star_body", noisy=noisy, rng=rng),
        data.sensors.measure("orbit_sun_body", noisy=noisy, rng=rng),
        data.sensors.measure("orbit_horizon_body", noisy=noisy, rng=rng),
        data.sensors.measure("mag_body", noisy=noisy, rng=rng),
    ]
    world_vecs = [star_world, sun_world, horizon_world, mag_world]

    body_vecs = [vec / np.linalg.norm(vec) for vec in body_vecs]
    world_vecs = [vec / np.linalg.norm(vec) for vec in world_vecs]

    mag_sigma = suite.by_name["mag_body"].noise / np.linalg.norm(mag_world)
    weights = [
        1.0 / suite.by_name["orbit_star_body"].noise**2,
        1.0 / suite.by_name["orbit_sun_body"].noise**2,
        1.0 / suite.by_name["orbit_horizon_body"].noise**2,
        1.0 / mag_sigma**2,
    ]
    return body_vecs, world_vecs, weights


class TestSensorDiscovery:
    def test_compile_discovers_native_and_custom_sensors(self):
        model, data = _make_model_data()

        assert set(model.sensors.by_name) == {
            "acc_body",
            "gyro_body",
            "mag_body",
            "mag_rotated",
            "orbit_sun_body",
            "orbit_horizon_body",
            "orbit_star_body",
        }
        assert model.sensors.by_name["orbit_sun_body"].orbit_kind == "sun"
        assert model.sensors.by_name["orbit_horizon_body"].orbit_kind == "horizon"
        assert model.sensors.by_name["orbit_star_body"].orbit_kind == "star"
        assert data.sensors.bias("gyro_body").shape == (3,)

    def test_invalid_orbit_user_sensor_raises(self, tmp_path: pathlib.Path):
        bad_xml = """\
<mujoco model="bad_sensor">
  <size nuser_sensor="4"/>
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <body name="spacecraft">
      <freejoint/>
      <geom type="box" size="0.5 0.5 0.5" mass="100"/>
      <site name="head" pos="0 0 0"/>
    </body>
  </worldbody>
  <sensor>
    <user
      name="orbit_star_bad"
      objtype="site"
      objname="head"
      datatype="real"
      needstage="pos"
      dim="3"
      user="1 0 0 0"
    />
  </sensor>
</mujoco>
"""
        xml_path = tmp_path / "bad_sensor.xml"
        xml_path.write_text(bad_xml)

        with pytest.raises(ValueError, match="datatype='axis'"):
            make_model_data(
                xml_path=str(xml_path),
                mj_timestep=0.01,
                use_j2=False,
                use_drag=False,
                use_srp=False,
                use_magnetic=True,
            )


class TestSensorMeasurements:
    def test_noisy_measurement_does_not_mutate_sensordata(self):
        model, data = _make_model_data()
        _set_attitude(
            model,
            data,
            _quat_from_axis_angle(np.array([1.0, 2.0, -0.5]), 0.6),
            np.array([0.1, -0.2, 0.3]),
        )

        truth_before = data.sensordata.copy()
        noisy = data.sensors.measure("gyro_body", noisy=True, rng=np.random.default_rng(7))
        truth_after = data.sensordata.copy()

        np.testing.assert_allclose(data.sensors.measure("gyro_body", noisy=False), [0.1, -0.2, 0.3])
        np.testing.assert_allclose(truth_after, truth_before)
        assert not np.allclose(noisy, truth_before[:3])

    def test_measure_uses_supplied_rng_for_noise(self):
        model, data = _make_model_data(rng_seed=123)
        _set_attitude(
            model,
            data,
            _quat_from_axis_angle(np.array([1.0, 2.0, -0.5]), 0.6),
            np.array([0.1, -0.2, 0.3]),
        )

        descriptor = model.sensors.by_name["gyro_body"]
        truth = data.sensors.measure("gyro_body", noisy=False)
        bias = data.sensors.bias("gyro_body")
        seed = 7

        measurement = data.sensors.measure("gyro_body", noisy=True, rng=np.random.default_rng(seed))
        expected = _apply_sensor_noise(
            np.random.default_rng(seed),
            descriptor,
            truth,
            bias=bias,
        )

        np.testing.assert_allclose(measurement, expected)

    def test_gyro_bias_is_constant_per_data_instance(self, tmp_path: pathlib.Path):
        gyro_xml = """\
<mujoco model="gyro_only">
  <size nuser_sensor="1"/>
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <body name="spacecraft">
      <freejoint/>
      <geom type="box" size="0.5 0.5 0.5" mass="100"/>
      <site name="imu" pos="0 0 0"/>
    </body>
  </worldbody>
  <sensor>
    <gyro name="gyro" site="imu" noise="0" user="1e-4"/>
  </sensor>
</mujoco>
"""
        xml_path = tmp_path / "gyro_only.xml"
        xml_path.write_text(gyro_xml)

        model, data = _make_model_data(str(xml_path), rng_seed=123)
        _set_attitude(model, data, np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.03, 0.04, -0.02]))

        truth = data.sensors.measure("gyro", noisy=False)
        meas1 = data.sensors.measure("gyro", noisy=True, rng=np.random.default_rng(1))
        meas2 = data.sensors.measure("gyro", noisy=True, rng=np.random.default_rng(2))

        np.testing.assert_allclose(meas1, meas2)
        np.testing.assert_allclose(meas1 - truth, data.sensors.bias("gyro"))

    def test_accelerometer_and_magnetometer_bias_are_constant_per_data_instance(
        self, tmp_path: pathlib.Path
    ):
        xml = """\
<mujoco model="imu_biases">
  <size nuser_sensor="4"/>
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <body name="spacecraft">
      <freejoint/>
      <geom type="box" size="0.5 0.5 0.5" mass="100"/>
      <site name="imu" pos="0 0 0"/>
    </body>
  </worldbody>
  <sensor>
    <accelerometer name="acc" site="imu" noise="0" user="1e-3"/>
    <magnetometer name="mag" site="imu" noise="0" user="2e-7"/>
  </sensor>
</mujoco>
"""
        xml_path = tmp_path / "imu_biases.xml"
        xml_path.write_text(xml)

        model, data = _make_model_data(str(xml_path), rng_seed=123)
        _set_attitude(model, data, np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))

        for name in ["acc", "mag"]:
            truth = data.sensors.measure(name, noisy=False)
            meas1 = data.sensors.measure(name, noisy=True, rng=np.random.default_rng(1))
            meas2 = data.sensors.measure(name, noisy=True, rng=np.random.default_rng(2))

            np.testing.assert_allclose(meas1, meas2)
            np.testing.assert_allclose(meas1 - truth, data.sensors.bias(name))

    def test_axis_sensor_bias_is_applied_as_fixed_misalignment(self, tmp_path: pathlib.Path):
        xml = """\
<mujoco model="vector_biases">
  <size nuser_sensor="4"/>
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <body name="spacecraft">
      <freejoint/>
      <geom type="box" size="0.5 0.5 0.5" mass="100"/>
      <site name="tracker" pos="0.2 0 0"/>
    </body>
  </worldbody>
  <sensor>
    <user
      name="orbit_star_body"
      objtype="site"
      objname="tracker"
      datatype="axis"
      needstage="pos"
      dim="3"
      noise="0"
      user="0.3 0.8 -0.4 1e-3"
    />
  </sensor>
</mujoco>
"""
        xml_path = tmp_path / "vector_biases.xml"
        xml_path.write_text(xml)

        model, data = _make_model_data(str(xml_path), rng_seed=123)
        _set_attitude(model, data, _quat_from_axis_angle(np.array([0.3, 0.2, -0.5]), 0.4))

        truth = data.sensors.measure("orbit_star_body", noisy=False)
        meas1 = data.sensors.measure("orbit_star_body", noisy=True, rng=np.random.default_rng(1))
        meas2 = data.sensors.measure("orbit_star_body", noisy=True, rng=np.random.default_rng(2))
        bias = data.sensors.bias("orbit_star_body")

        expected = _rotation_matrix_from_rotvec(bias) @ truth
        expected /= np.linalg.norm(expected)

        np.testing.assert_allclose(meas1, meas2)
        np.testing.assert_allclose(meas1, expected)
        assert not np.allclose(meas1, truth)
        assert bias.shape == (3,)

    def test_axis_sensor_bias_is_preserved_when_noise_is_enabled(self, tmp_path: pathlib.Path):
        xml = """\
<mujoco model="vector_biases_with_noise">
  <size nuser_sensor="4"/>
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <body name="spacecraft">
      <freejoint/>
      <geom type="box" size="0.5 0.5 0.5" mass="100"/>
      <site name="tracker" pos="0.2 0 0"/>
    </body>
  </worldbody>
  <sensor>
    <user
      name="orbit_star_body"
      objtype="site"
      objname="tracker"
      datatype="axis"
      needstage="pos"
      dim="3"
      noise="5e-4"
      user="0.3 0.8 -0.4 1e-3"
    />
  </sensor>
</mujoco>
"""
        xml_path = tmp_path / "vector_biases_with_noise.xml"
        xml_path.write_text(xml)

        model, data = _make_model_data(str(xml_path), rng_seed=123)
        _set_attitude(model, data, _quat_from_axis_angle(np.array([0.3, 0.2, -0.5]), 0.4))

        descriptor = model.sensors.by_name["orbit_star_body"]
        truth = data.sensors.measure("orbit_star_body", noisy=False)
        bias = data.sensors.bias("orbit_star_body")
        seed = 7
        meas = data.sensors.measure("orbit_star_body", noisy=True, rng=np.random.default_rng(seed))

        biased = _rotation_matrix_from_rotvec(bias) @ truth
        biased /= np.linalg.norm(biased)

        assert abs(np.linalg.norm(meas) - 1.0) < 1e-12
        angular_error = np.arccos(np.clip(float(np.dot(meas, biased)), -1.0, 1.0))
        assert angular_error < 10.0 * descriptor.noise

    def test_quaternion_sensor_bias_is_preserved_when_noise_is_enabled(self):
        descriptor = _make_sensor_descriptor(
            name="quat_sensor",
            datatype=_QUATERNION_DATATYPE,
            dim=4,
            noise=4e-4,
        )
        truth = _quat_from_axis_angle(np.array([0.4, -0.1, 0.2]), 0.7)
        bias = np.array([0.01, -0.02, 0.03])
        seed = 11

        meas = _apply_sensor_noise(
            np.random.default_rng(seed),
            descriptor,
            truth,
            bias=bias,
        )

        noise_rot = np.random.default_rng(seed).normal(0.0, descriptor.noise, size=3)
        expected = _quat_mul(
            _quat_from_rotvec(noise_rot),
            _quat_mul(_quat_from_rotvec(bias), truth),
        )
        expected /= np.linalg.norm(expected)
        if np.dot(meas, expected) < 0.0:
            expected = -expected

        np.testing.assert_allclose(meas, expected)

    def test_truth_vectors_match_expected_transforms(self):
        model, data = _make_model_data()
        quat_world_body = _quat_from_axis_angle(np.array([1.0, -1.0, 0.5]), 0.7)
        omega_body = np.array([0.08, -0.03, 0.02])
        _set_attitude(model, data, quat_world_body, omega_body)

        body_id = model.body_id("spacecraft")
        r_world_body = data.xmat[body_id].reshape(3, 3)

        sun_world = data.env.sun_vector_eci
        nadir_eci = -data.orbit.R_eci / np.linalg.norm(data.orbit.R_eci)
        horizon_world = nadir_eci
        star_world = model.sensors.by_name["orbit_star_body"].reference_eci
        mag_world = data.env.mag_field_eci
        mag_rot_site = data.site_xmat[model.sensors.by_name["mag_rotated"].objid].reshape(3, 3)

        np.testing.assert_allclose(data.sensors.measure("gyro_body", noisy=False), omega_body)
        np.testing.assert_allclose(
            data.sensors.measure("orbit_sun_body", noisy=False),
            r_world_body.T @ (sun_world / np.linalg.norm(sun_world)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            data.sensors.measure("orbit_horizon_body", noisy=False),
            r_world_body.T @ (horizon_world / np.linalg.norm(horizon_world)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            data.sensors.measure("orbit_star_body", noisy=False),
            r_world_body.T @ (star_world / np.linalg.norm(star_world)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            data.sensors.measure("mag_body", noisy=False),
            r_world_body.T @ mag_world,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            data.sensors.measure("mag_rotated", noisy=False),
            mag_rot_site.T @ mag_world,
            atol=1e-12,
        )

    def test_measure_all_returns_named_measurements(self):
        model, data = _make_model_data()
        _set_attitude(model, data, np.array([1.0, 0.0, 0.0, 0.0]))
        readings = data.sensors.measure_all(noisy=False)
        assert set(readings) == set(model.sensors.by_name)
        assert all(value.ndim == 1 for value in readings.values())


class TestWahbaVerification:
    def test_wahba_recovers_attitude_without_noise(self):
        model, data = _make_model_data()
        _set_attitude(
            model,
            data,
            _quat_from_axis_angle(np.array([0.3, 1.0, -0.7]), 0.9),
            np.array([0.02, 0.01, -0.03]),
        )

        body_vecs, world_vecs, weights = _wahba_inputs(model, data, noisy=False)
        r_wb_est = _quat_to_rotmat(_q_method(body_vecs, world_vecs, weights))
        r_wb_true = data.xmat[model.body_id("spacecraft")].reshape(3, 3)

        assert _rotation_error_rad(r_wb_est, r_wb_true) < 1e-8

    def test_wahba_recovers_attitude_with_sensor_noise(self):
        model, data = _make_model_data()
        _set_attitude(
            model,
            data,
            _quat_from_axis_angle(np.array([1.0, -0.4, 0.2]), 0.8),
            np.array([0.02, -0.04, 0.01]),
        )

        rng = np.random.default_rng(1234)
        body_vecs, world_vecs, weights = _wahba_inputs(model, data, noisy=True, rng=rng)
        r_wb_est = _quat_to_rotmat(_q_method(body_vecs, world_vecs, weights))
        r_wb_true = data.xmat[model.body_id("spacecraft")].reshape(3, 3)

        assert _rotation_error_rad(r_wb_est, r_wb_true) < np.deg2rad(0.5)
