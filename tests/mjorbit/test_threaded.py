"""Thread-safety checks for shared-model, per-data rollouts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from mjorbit import (
    MjoData,
    MjoModel,
    ReactionWheelSpec,
    SurfaceSpec,
    ThrusterSpec,
    mjo_forward,
    mjo_step,
)
from mjorbit.testdata import FREE_BODY_SENSORS_XML

from ._helpers import circular_leo_orbit_init, make_model_data


@dataclass(frozen=True)
class RolloutSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    orbit_R_eci: np.ndarray
    orbit_V_eci: np.ndarray
    orbit_t: float
    rw_speed: np.ndarray
    sensordata: np.ndarray
    wrench_buffer: np.ndarray
    xfrc_applied: np.ndarray


def _make_thread_model() -> MjoModel:
    model, _ = make_model_data(
        xml_path=FREE_BODY_SENSORS_XML,
        mj_timestep=0.005,
        use_j2=False,
        use_drag=True,
        use_srp=False,
        use_magnetic=True,
        reaction_wheels=[
            ReactionWheelSpec(
                body_name="spacecraft",
                axis_body=np.array([0.0, 0.0, 1.0]),
                inertia=0.01,
                torque_limit=0.05,
            )
        ],
        surfaces=[
            SurfaceSpec(
                body_name="spacecraft",
                center_of_pressure_body=np.array([0.1, 0.0, 0.0]),
                normal_body=np.array([0.0, 1.0, 0.0]),
                area=2.0,
                drag_coeff=2.2,
                use_drag=True,
                use_srp=False,
            )
        ],
        thrusters=[
            ThrusterSpec(
                body_name="spacecraft",
                position_body=np.array([0.0, 0.0, 0.0]),
                direction_body=np.array([0.0, 1.0, 0.0]),
                force_limit=1.0,
            )
        ],
    )
    return model


def _run_rollout(model: MjoModel, steps: int) -> RolloutSnapshot:
    data = MjoData(model, orbit=circular_leo_orbit_init(), rng_seed=123)
    data.qpos[:3] = np.array([10.0, -3.0, 2.0])
    data.qvel[:6] = np.array([0.01, -0.02, 0.005, 0.02, -0.01, 0.03])
    data.actuators.rw_torque_cmd[0] = 0.01
    data.actuators.thr_force_cmd[0] = 0.2
    mjo_forward(model, data)

    for _ in range(steps):
        mjo_step(model, data)

    return RolloutSnapshot(
        qpos=data.qpos.copy(),
        qvel=data.qvel.copy(),
        orbit_R_eci=data.orbit.R_eci.copy(),
        orbit_V_eci=data.orbit.V_eci.copy(),
        orbit_t=data.orbit.t,
        rw_speed=data.actuators.rw_speed.copy(),
        sensordata=data.sensordata.copy(),
        wrench_buffer=data.wrench_buffer.copy(),
        xfrc_applied=data.xfrc_applied.copy(),
    )


def _assert_snapshot_equal(actual: RolloutSnapshot, expected: RolloutSnapshot) -> None:
    for field in (
        "qpos",
        "qvel",
        "orbit_R_eci",
        "orbit_V_eci",
        "rw_speed",
        "sensordata",
        "wrench_buffer",
        "xfrc_applied",
    ):
        actual_value = getattr(actual, field)
        expected_value = getattr(expected, field)
        assert np.array_equal(actual_value, expected_value), field
    assert actual.orbit_t == expected.orbit_t


def test_each_data_owns_distinct_native_plugin_state():
    model = _make_thread_model()
    first = MjoData(model, orbit=circular_leo_orbit_init(), rng_seed=1)
    second = MjoData(model, orbit=circular_leo_orbit_init(), rng_seed=2)

    first.orbit.R_eci[0] += 1.0
    first.actuators.rw_torque_cmd[0] = 0.02
    assert first.orbit.R_eci[0] != second.orbit.R_eci[0]
    assert first.actuators.rw_torque_cmd[0] != second.actuators.rw_torque_cmd[0]


def test_parallel_rollouts_match_sequential_bit_for_bit():
    model = _make_thread_model()
    workers = 8
    steps = 1000

    sequential = [_run_rollout(model, steps) for _ in range(workers)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        threaded = list(pool.map(lambda _: _run_rollout(model, steps), range(workers)))

    for actual, expected in zip(threaded, sequential):
        _assert_snapshot_equal(actual, expected)
