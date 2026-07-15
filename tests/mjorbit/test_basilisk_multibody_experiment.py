"""Tests for the Basilisk multibody paper experiment helpers."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from experiments.basilisk_multibody.run import (
    BIMANUAL_POSE,
    make_random_control_batch,
    make_random_control_profile,
    make_smooth_random_controls,
    write_bimanual_assets,
)


def test_bimanual_assets_remove_joint_limits_and_split_backends(tmp_path):
    assets = write_bimanual_assets(tmp_path, dt_s=0.02, orbit_dt_s=0.05, mj_integrator="RK4")

    basilisk_root = ET.fromstring(assets.basilisk_xml.read_text())
    mjorbit_root = ET.fromstring(assets.mjorbit_xml.read_text())

    assert basilisk_root.find("mjorbit") is None
    mjorbit = mjorbit_root.find("mjorbit")
    assert mjorbit is not None
    assert mjorbit.get("plugin_body") == "bus"
    assert np.isclose(float(mjorbit.get("orbit_dt", "nan")), 0.05)
    assert mjorbit.get("use_gravity_gradient") == "false"

    assert assets.body_names[0] == "bus"
    assert assets.actuator_names == (
        "shoulder_a_pos",
        "elbow_a_pos",
        "shoulder_b_pos",
        "elbow_b_pos",
    )
    assert "shoulder_a" in assets.joint_names
    assert "hn_2" in assets.joint_names

    default_joint = basilisk_root.find("./default/joint")
    assert default_joint is not None
    assert default_joint.get("limited") == "false"
    for joint in basilisk_root.iter("joint"):
        assert "range" not in joint.attrib
    for joint in mjorbit_root.iter("joint"):
        assert joint.get("limited") == "false"
        assert "range" not in joint.attrib
    for actuator in basilisk_root.iter("position"):
        assert actuator.get("ctrllimited") == "false"
        assert "ctrlrange" not in actuator.attrib


def test_random_controls_are_deterministic_and_bounded():
    times = np.linspace(0.0, 1.0, 5)
    a = make_smooth_random_controls(times, n_control=4, seed=12)
    b = make_smooth_random_controls(times, n_control=4, seed=12)
    c = make_smooth_random_controls(times, n_control=4, seed=13)

    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c)
    assert a.shape == (5, 4)
    assert np.max(np.abs(a - BIMANUAL_POSE)) < 1.0

    batch = make_random_control_batch(3, 2, 4, seed=99)
    assert batch.shape == (2, 3, 4)
    assert np.max(np.abs(batch - BIMANUAL_POSE)) <= 0.35

    profile = make_random_control_profile(5, 4, seed=99)
    assert profile.shape == (5, 4)
    assert np.max(np.abs(profile - BIMANUAL_POSE)) <= 0.35
