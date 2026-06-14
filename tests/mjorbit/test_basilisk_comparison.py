"""Tests for the Basilisk-MuJoCo comparison harness."""

from __future__ import annotations

import numpy as np

from comparisons.basilisk_mujoco.cases import (
    run_single_body_mjorbit,
    run_two_arm_free_drift_mjorbit,
)
from comparisons.basilisk_mujoco.common import (
    circular_orbit_state_at,
    make_circular_orbit,
    sample_steps,
)
from comparisons.basilisk_mujoco.eci_frame import run_two_arm_eci_frame_check


def test_circular_orbit_samples_close_after_one_period() -> None:
    orbit = make_circular_orbit()
    r_eci, v_eci = circular_orbit_state_at([0.0, orbit.period_s])

    np.testing.assert_allclose(r_eci[1], r_eci[0], atol=1.0e-9)
    np.testing.assert_allclose(v_eci[1], v_eci[0], atol=1.0e-12)


def test_sample_steps_include_endpoints_and_are_unique() -> None:
    steps = sample_steps(10, 5)

    assert steps[0] == 0
    assert steps[-1] == 10
    assert len(steps) == len(set(int(x) for x in steps))


def test_single_body_one_orbit_matches_exact_reference() -> None:
    result = run_single_body_mjorbit(n_steps=1000, orbit_dt=0.1, max_samples=64)

    assert abs(result.summary["period_error_s"]) < 1.0e-8
    assert result.summary["final_position_error_m"] < 1.0
    assert result.summary["final_velocity_error_m_s"] < 1.0e-3
    assert result.r_eci_km.shape == result.r_ref_eci_km.shape


def test_two_arm_free_drift_short_run_stays_finite() -> None:
    result = run_two_arm_free_drift_mjorbit(
        duration_s=2.0,
        dt_s=0.1,
        orbit_dt=0.1,
        max_samples=8,
        basilisk_integrator="rkf45",
    )

    assert result.summary["all_finite"]
    assert result.body_names == ("hub", "arm_1", "arm_2")
    assert result.body_r_eci_km.shape[1:] == (3, 3)
    assert result.body_v_eci_km_s.shape == result.body_r_eci_km.shape
    assert result.hinge_angles_rad.shape[1] == 2


def test_two_arm_raw_eci_and_chief_centered_short_run_match() -> None:
    result = run_two_arm_eci_frame_check(duration_s=1.0, dt_s=0.1, max_samples=8)

    assert result.summary["max_hub_position_error_m"] < 1.0e-2
    assert result.summary["max_hinge_angle_error_rad"] < 1.0e-7
    assert result.summary["max_body_relative_position_error_m"] < 1.0e-6
    assert result.local_body_r_eci_km.shape == result.eci_body_r_eci_km.shape
