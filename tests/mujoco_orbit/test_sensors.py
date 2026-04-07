"""Sensor discovery, noise injection, and Wahba verification tests."""

from __future__ import annotations

import pathlib

import mujoco
import numpy as np
import pytest

from mujoco_orbit import compile, measure_sensor, measure_sensors
from mujoco_orbit.constants import R_EARTH
from mujoco_orbit.core.config import MuJoCoCfg, OrbitCfg, ScenarioCfg
from mujoco_orbit.orbit.elements import keplerian_to_cartesian
from mujoco_orbit.testdata import FREE_BODY_SENSORS_XML


def _circular_leo_cfg(xml_path: str = FREE_BODY_SENSORS_XML) -> ScenarioCfg:
    a = R_EARTH + 400.0
    R, V = keplerian_to_cartesian(a=a, e=0.0, inc=np.deg2rad(51.6), raan=0.0, argp=0.0, nu=0.0)
    return ScenarioCfg(
        orbit=OrbitCfg(R_eci=R, V_eci=V),
        mujoco=MuJoCoCfg(xml_path=xml_path, dt=0.01),
        use_j2=False,
        use_drag=False,
        use_srp=False,
        use_magnetic=True,
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


def _set_attitude(
    scenario,
    quat_world_body: np.ndarray,
    omega_body: np.ndarray = np.array([0.05, -0.02, 0.03]),
) -> None:
    scenario.mjd.qpos[3:7] = quat_world_body
    scenario.mjd.qvel[3:6] = omega_body
    mujoco.mj_forward(scenario.mjm, scenario.mjd)


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


def _rotation_error_rad(R_est: np.ndarray, R_true: np.ndarray) -> float:
    delta = R_est @ R_true.T
    cos_angle = 0.5 * (np.trace(delta) - 1.0)
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def _wahba_inputs(scenario, *, noisy: bool, rng: np.random.Generator | None = None):
    suite = scenario.sensor_suite
    assert suite is not None

    sun_world = scenario.frame_cache.C_LI @ scenario.env_cache.sun_vector_eci
    nadir_eci = -scenario.orbit.R_eci / np.linalg.norm(scenario.orbit.R_eci)
    horizon_world = scenario.frame_cache.C_LI @ nadir_eci
    star_world = scenario.frame_cache.C_LI @ suite.by_name["orbit_star_body"].reference_eci
    mag_world = scenario.frame_cache.C_LI @ scenario.env_cache.mag_field_eci

    body_vecs = [
        measure_sensor(scenario, "orbit_star_body", noisy=noisy, rng=rng),
        measure_sensor(scenario, "orbit_sun_body", noisy=noisy, rng=rng),
        measure_sensor(scenario, "orbit_horizon_body", noisy=noisy, rng=rng),
        measure_sensor(scenario, "mag_body", noisy=noisy, rng=rng),
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
        scenario = compile(_circular_leo_cfg())
        suite = scenario.sensor_suite
        assert suite is not None

        assert set(suite.by_name) == {
            "gyro_body",
            "mag_body",
            "mag_rotated",
            "orbit_sun_body",
            "orbit_horizon_body",
            "orbit_star_body",
        }
        assert suite.by_name["orbit_sun_body"].orbit_kind == "sun"
        assert suite.by_name["orbit_horizon_body"].orbit_kind == "horizon"
        assert suite.by_name["orbit_star_body"].orbit_kind == "star"
        assert suite.by_name["gyro_body"].gyro_bias.shape == (3,)

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
            compile(_circular_leo_cfg(str(xml_path)))


class TestSensorMeasurements:
    def test_noisy_measurement_does_not_mutate_sensordata(self):
        scenario = compile(_circular_leo_cfg())
        _set_attitude(
            scenario,
            _quat_from_axis_angle(np.array([1.0, 2.0, -0.5]), 0.6),
            np.array([0.1, -0.2, 0.3]),
        )

        truth_before = scenario.mjd.sensordata.copy()
        noisy = measure_sensor(scenario, "gyro_body", noisy=True, rng=np.random.default_rng(7))
        truth_after = scenario.mjd.sensordata.copy()

        np.testing.assert_allclose(
            measure_sensor(scenario, "gyro_body", noisy=False), [0.1, -0.2, 0.3]
        )
        np.testing.assert_allclose(truth_after, truth_before)
        assert not np.allclose(noisy, truth_before[:3])

    def test_gyro_bias_is_constant_per_scenario(self, tmp_path: pathlib.Path):
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

        scenario = compile(_circular_leo_cfg(str(xml_path)))
        _set_attitude(scenario, np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.03, 0.04, -0.02]))
        suite = scenario.sensor_suite
        assert suite is not None

        truth = measure_sensor(scenario, "gyro", noisy=False)
        meas1 = measure_sensor(scenario, "gyro", noisy=True, rng=np.random.default_rng(1))
        meas2 = measure_sensor(scenario, "gyro", noisy=True, rng=np.random.default_rng(2))

        np.testing.assert_allclose(meas1, meas2)
        np.testing.assert_allclose(meas1 - truth, suite.by_name["gyro"].gyro_bias)

    def test_truth_vectors_match_expected_transforms(self):
        scenario = compile(_circular_leo_cfg())
        quat_world_body = _quat_from_axis_angle(np.array([1.0, -1.0, 0.5]), 0.7)
        omega_body = np.array([0.08, -0.03, 0.02])
        _set_attitude(scenario, quat_world_body, omega_body)
        suite = scenario.sensor_suite
        assert suite is not None

        body_id = scenario.body_id("spacecraft")
        R_world_body = scenario.body_com_rotmat(body_id)

        sun_world = scenario.frame_cache.C_LI @ scenario.env_cache.sun_vector_eci
        nadir_eci = -scenario.orbit.R_eci / np.linalg.norm(scenario.orbit.R_eci)
        horizon_world = scenario.frame_cache.C_LI @ nadir_eci
        star_world = scenario.frame_cache.C_LI @ suite.by_name["orbit_star_body"].reference_eci
        mag_world = scenario.frame_cache.C_LI @ scenario.env_cache.mag_field_eci
        mag_rot_site = scenario.mjd.site_xmat[suite.by_name["mag_rotated"].objid].reshape(3, 3)

        np.testing.assert_allclose(measure_sensor(scenario, "gyro_body", noisy=False), omega_body)
        np.testing.assert_allclose(
            measure_sensor(scenario, "orbit_sun_body", noisy=False),
            R_world_body.T @ (sun_world / np.linalg.norm(sun_world)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            measure_sensor(scenario, "orbit_horizon_body", noisy=False),
            R_world_body.T @ (horizon_world / np.linalg.norm(horizon_world)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            measure_sensor(scenario, "orbit_star_body", noisy=False),
            R_world_body.T @ (star_world / np.linalg.norm(star_world)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            measure_sensor(scenario, "mag_body", noisy=False),
            R_world_body.T @ mag_world,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            measure_sensor(scenario, "mag_rotated", noisy=False),
            mag_rot_site.T @ mag_world,
            atol=1e-12,
        )

    def test_measure_sensors_returns_named_measurements(self):
        scenario = compile(_circular_leo_cfg())
        _set_attitude(scenario, np.array([1.0, 0.0, 0.0, 0.0]))
        readings = measure_sensors(scenario, noisy=False)
        assert set(readings) == set(scenario.sensor_suite.by_name)
        assert all(value.ndim == 1 for value in readings.values())


class TestWahbaVerification:
    def test_wahba_recovers_attitude_without_noise(self):
        scenario = compile(_circular_leo_cfg())
        _set_attitude(
            scenario,
            _quat_from_axis_angle(np.array([0.3, 1.0, -0.7]), 0.9),
            np.array([0.02, 0.01, -0.03]),
        )

        body_vecs, world_vecs, weights = _wahba_inputs(scenario, noisy=False)
        R_wb_est = _quat_to_rotmat(_q_method(body_vecs, world_vecs, weights))
        R_wb_true = scenario.body_com_rotmat(scenario.body_id("spacecraft"))

        assert _rotation_error_rad(R_wb_est, R_wb_true) < 1e-8

    def test_wahba_recovers_attitude_with_sensor_noise(self):
        scenario = compile(_circular_leo_cfg())
        _set_attitude(
            scenario,
            _quat_from_axis_angle(np.array([1.0, -0.4, 0.2]), 0.8),
            np.array([0.02, -0.04, 0.01]),
        )

        rng = np.random.default_rng(1234)
        body_vecs, world_vecs, weights = _wahba_inputs(scenario, noisy=True, rng=rng)
        R_wb_est = _quat_to_rotmat(_q_method(body_vecs, world_vecs, weights))
        R_wb_true = scenario.body_com_rotmat(scenario.body_id("spacecraft"))

        assert _rotation_error_rad(R_wb_est, R_wb_true) < np.deg2rad(0.5)
