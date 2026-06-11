from __future__ import annotations

import functools
import socket

import numpy as np
import pytest
import trimesh.visual

from mujoco_orbit import mjo_forward
from mujoco_orbit.constants import R_EARTH
from mujoco_orbit.step import mjo_step
from mujoco_orbit.testdata import SPACECRAFT_DUAL_ARM_PANELS_XML
from tests.mujoco_orbit._helpers import make_model_data
from viewer.earth import EARTH_TEXTURE_PATH, create_earth_mesh, create_fallback_earth_mesh
from viewer.framing import (
    default_camera_pose,
    lvlh_basis_eci,
    spacecraft_bounding_radius,
)
from viewer.tasks import available_tasks, get_task_class
from viewer.tasks.free_drift import FreeDriftTask


@functools.lru_cache(maxsize=1)
def _viewer_server_skip_reason() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
    except OSError as exc:
        return f"viewer tests require local socket bind: {exc}"
    return None


def _require_viewer_server() -> None:
    if reason := _viewer_server_skip_reason():
        pytest.skip(reason)


# ----------------------------------------------------------------------
# Earth mesh
# ----------------------------------------------------------------------


def test_earth_texture_asset_is_bundled() -> None:
    assert EARTH_TEXTURE_PATH.is_file()


def test_earth_mesh_is_textured_uv_sphere() -> None:
    radius = R_EARTH * 1000.0
    mesh = create_earth_mesh(radius)

    norms = np.linalg.norm(mesh.vertices, axis=1)
    np.testing.assert_allclose(norms, radius, rtol=1e-9)

    assert isinstance(mesh.visual, trimesh.visual.TextureVisuals)
    uv = mesh.visual.uv
    assert uv is not None
    assert uv.min() >= 0.0 and uv.max() <= 1.0
    assert mesh.visual.material.baseColorTexture is not None

    # Outward-facing triangles: face normals align with face centroids.
    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    alignment = np.einsum("ij,ij->i", mesh.face_normals, centroids)
    assert np.all(alignment > 0.0)


def test_earth_mesh_falls_back_without_texture(tmp_path) -> None:
    mesh = create_earth_mesh(1.0, texture_path=tmp_path / "missing.jpg")
    fallback = create_fallback_earth_mesh(1.0)
    assert len(mesh.vertices) == len(fallback.vertices)


# ----------------------------------------------------------------------
# Framing
# ----------------------------------------------------------------------


def test_lvlh_basis_is_orthonormal_right_handed() -> None:
    R = np.array([7000.0, 0.0, 0.0])
    V = np.array([0.0, 7.5, 0.0])
    basis = lvlh_basis_eci(R, V)
    np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.cross(basis[0], basis[1]), basis[2], atol=1e-12)
    np.testing.assert_allclose(basis[0], [1.0, 0.0, 0.0], atol=1e-12)


def test_default_camera_pose_keeps_earth_in_background() -> None:
    """The view direction must point inside Earth's disk from LEO."""
    R = np.array([R_EARTH + 500.0, 0.0, 0.0])
    V = np.array([0.0, 7.6, 0.0])
    target = 1000.0 * R
    position, look_at = default_camera_pose(
        target, distance=25.0, basis=lvlh_basis_eci(R, V)
    )
    np.testing.assert_allclose(look_at, target)

    view = look_at - position
    view = view / np.linalg.norm(view)
    earth_dir = -position / np.linalg.norm(position)
    angle_to_center = np.arccos(np.clip(view @ earth_dir, -1.0, 1.0))
    earth_angular_radius = np.arcsin(R_EARTH / np.linalg.norm(R))
    assert angle_to_center < earth_angular_radius


def test_spacecraft_bounding_radius_covers_arm() -> None:
    model, data = make_model_data(xml_path=SPACECRAFT_DUAL_ARM_PANELS_XML, mj_timestep=0.005)
    mjo_forward(model, data)
    # The dual-arm bus spans ~1.5 m arms and wings; the bound must cover
    # them but stay model-scale.
    radius = spacecraft_bounding_radius(model, data)
    assert 1.0 < radius < 10.0


# ----------------------------------------------------------------------
# Task registry and tasks
# ----------------------------------------------------------------------


def test_builtin_tasks_registered() -> None:
    names = available_tasks()
    assert {"free_drift", "arm_reach_mppi", "capture_stabilize_mppi"} <= set(names)


def test_free_drift_task_builds_and_steps() -> None:
    task = FreeDriftTask()
    assert task.model.body_id("bus") == task.track_body_id
    for _ in range(5):
        task.pre_step()
        mjo_step(task.model, task.data)
        task.post_step()
    assert np.all(np.isfinite(np.asarray(task.data.qpos)))
    assert task.status()


def test_arm_reach_task_replans_and_steps() -> None:
    task = get_task_class("arm_reach_mppi")()
    task.params.num_rollouts = 8  # keep the test fast
    task.on_param_changed("num_rollouts")
    for _ in range(3):
        task.pre_step()
        mjo_step(task.model, task.data)
        task.post_step()
    assert np.all(np.isfinite(np.asarray(task.data.qpos)))
    assert "distance" in task.status()


def test_capture_task_latch_swaps_model() -> None:
    task = get_task_class("capture_stabilize_mppi")()
    model_before = task.model
    assert task.phase == "capture"
    # Force the latch path without running the (slow) capture phase.
    task._latch()
    assert task.phase == "stabilize"
    assert task.model is not model_before
    mjo_step(task.model, task.data)
    assert np.all(np.isfinite(np.asarray(task.data.qpos)))


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------


def test_app_constructs_switches_tasks_and_tracks() -> None:
    _require_viewer_server()
    from viewer.app import MjOrbitApp

    app = MjOrbitApp(task="free_drift", port=0)
    try:
        assert app.task is not None and app.mj_scene is not None
        target_before = app._body_render_position(app.task.track_body_id)

        # ECI render frame: the spacecraft is at the chief ECI position.
        np.testing.assert_allclose(
            target_before,
            1000.0 * app.task.data.orbit.R_eci + app.task.data.xpos[1],
            atol=1.0e-6,
        )

        # Default framing looks at the spacecraft from the configured ratio.
        camera = np.asarray(app.server.initial_camera.position)
        look_at = np.asarray(app.server.initial_camera.look_at)
        np.testing.assert_allclose(look_at, target_before, atol=1.0e-6)
        assert np.linalg.norm(camera - target_before) > 1.0

        # Step and confirm the default pose follows the spacecraft.
        for _ in range(10):
            app.task.pre_step()
            mjo_step(app.task.model, app.task.data)
            app.task.post_step()
        app._render()
        look_at_after = np.asarray(app.server.initial_camera.look_at)
        assert np.linalg.norm(look_at_after - look_at) > 0.0

        # Task switch rebuilds the scene and task GUI.
        scene_before = app.mj_scene
        app._load_task("arm_reach_mppi")
        assert app.task is not None and type(app.task).name == "arm_reach_mppi"
        assert app.mj_scene is not scene_before
    finally:
        app.server.stop()


def test_app_cli_list_tasks(capsys) -> None:
    from viewer.app import main

    main(["--list-tasks"])
    out = capsys.readouterr().out
    assert "free_drift" in out
