from __future__ import annotations

import functools
import socket

import mujoco
import numpy as np
import pytest
import trimesh.visual
import viser.transforms as vtf

from mjorbit import mjo_forward
from mjorbit.constants import R_EARTH
from mjorbit.step import mjo_step
from mjorbit.testdata import SPACECRAFT_DUAL_ARM_PANELS_XML
from tests.mjorbit._helpers import make_model_data
from viewer.earth import EARTH_TEXTURE_PATH, create_earth_mesh, create_fallback_earth_mesh
from viewer.framing import (
    DEFAULT_CAMERA_FOV,
    DEFAULT_VIEW_FILL,
    default_camera_pose,
    distance_for_fill,
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


def test_default_framing_centers_and_fills_view() -> None:
    """The default pose must center the spacecraft, make it fill a
    significant fraction of the view, and keep Earth in the background —
    all simultaneously, using the real camera intrinsics."""
    model, data = make_model_data(xml_path=SPACECRAFT_DUAL_ARM_PANELS_XML, mj_timestep=0.005)
    mjo_forward(model, data)

    R = np.array([R_EARTH + 500.0, 0.0, 0.0])
    V = np.array([0.0, 7.6, 0.0])
    target = 1000.0 * R
    radius = spacecraft_bounding_radius(model, data)
    distance = distance_for_fill(radius)
    position, look_at = default_camera_pose(
        target, distance=distance, basis=lvlh_basis_eci(R, V)
    )

    # Centered: the camera looks directly at the spacecraft.
    np.testing.assert_allclose(look_at, target)

    # Prominent: the bounding sphere subtends a significant fraction of the
    # vertical field of view, but the camera stays outside the geometry.
    angular_diameter = 2.0 * np.arctan(radius / distance)
    fill = angular_diameter / DEFAULT_CAMERA_FOV
    assert 0.5 <= fill <= 0.95
    assert fill == pytest.approx(DEFAULT_VIEW_FILL, rel=0.15)
    assert np.linalg.norm(position - target) > 1.2 * radius

    # Earth still fills the background: the view axis points inside
    # Earth's disk as seen from the camera.
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


def test_build_rollout_traces_orders_and_subsamples() -> None:
    from viewer.tasks.base import TRACE_BEST_COLOR, TRACE_OTHER_COLOR, build_rollout_traces

    rng = np.random.default_rng(0)
    positions = rng.normal(size=(6, 500, 3))
    costs = np.array([3.0, 0.5, 4.0, 2.0, 1.0, 5.0])

    traces = build_rollout_traces(positions, costs, max_others=3, max_points=50)

    assert len(traces) == 4  # 3 others + best
    # Best rollout (lowest cost, index 1) is drawn last, in orange.
    best = traces[-1]
    assert best.color == TRACE_BEST_COLOR
    np.testing.assert_allclose(best.points[0], positions[1, 0])
    assert all(t.color == TRACE_OTHER_COLOR for t in traces[:-1])
    assert all(len(t.points) <= 50 for t in traces)
    assert all(np.all(np.isfinite(t.points)) for t in traces)


def test_arm_reach_replan_publishes_predicted_traces() -> None:
    task = get_task_class("arm_reach_mppi")()
    task.params.num_rollouts = 8
    task.on_param_changed("num_rollouts")
    assert task.traces == []
    version_before = task.traces_version

    task.pre_step()  # triggers one MPPI replan, which publishes traces

    assert task.traces_version > version_before
    assert task.traces
    # Traces start near the end effector's current position (world frame, m).
    sensordata = np.asarray(task.data.sensordata)
    ee_now = sensordata[0:3]
    for trace in task.traces:
        assert trace.points.shape[1] == 3
        assert np.linalg.norm(trace.points[0] - ee_now) < 0.5


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

    app = MjOrbitApp(port=0)
    try:
        assert app.task is not None and app.mj_scene is not None
        assert app.task.name == "free_drift"
        target_before = app._body_render_position(app.task.track_body_id)

        # ECI render frame is chief-centered (floating origin): the scene
        # stays at float32-safe magnitudes and Earth is translated instead.
        np.testing.assert_allclose(target_before, app.task.data.xpos[1], atol=1.0e-6)
        np.testing.assert_allclose(
            np.asarray(app.earth.handle.position),
            -1000.0 * app.task.data.orbit.R_eci,
            atol=2.0,  # handle positions round-trip through float32
        )

        # Default framing looks at the spacecraft, from outside it, with
        # every render-frame coordinate small enough for float32 rendering.
        camera = np.asarray(app.server.initial_camera.position)
        look_at = np.asarray(app.server.initial_camera.look_at)
        np.testing.assert_allclose(look_at, target_before, atol=1.0e-6)
        assert np.linalg.norm(camera - target_before) > 1.0
        assert np.linalg.norm(camera) < 1.0e4
        assert np.linalg.norm(target_before) < 1.0e4

        # Step and confirm the default pose stays locked on the spacecraft.
        for _ in range(10):
            app.task.pre_step()
            mjo_step(app.task.model, app.task.data)
            app.task.post_step()
        app._render()
        look_at_after = np.asarray(app.server.initial_camera.look_at)
        np.testing.assert_allclose(
            look_at_after,
            app._body_render_position(app.task.track_body_id),
            atol=1.0e-9,
        )

        # Task switch rebuilds the scene and task GUI.
        scene_before = app.mj_scene
        app._load_task("arm_reach_mppi")
        assert app.task is not None and type(app.task).name == "arm_reach_mppi"
        assert app.mj_scene is not scene_before

        # A replan publishes predicted traces, which the app renders.
        assert app._trace_scene._handle is None
        app.task.params.num_rollouts = 8
        app.task.on_param_changed("num_rollouts")
        app.task.pre_step()
        app._render()
        assert app._trace_scene._handle is not None
    finally:
        app.server.stop()


def test_app_cli_list_tasks(capsys) -> None:
    from viewer.app import main

    main(["--list-tasks"])
    out = capsys.readouterr().out
    assert "free_drift" in out


@pytest.mark.parametrize("render_frame", ["eci", "lvlh"])
def test_reset_republishes_browser_geometry_poses(render_frame: str) -> None:
    """A rebuilt scene must not inherit name-keyed poses from before Reset.

    Checking Python handles alone misses this: Viser can retain an old pose in
    its browser message buffer while a replacement handle reports identity.
    """
    _require_viewer_server()
    from viewer.app import MjOrbitApp

    app = MjOrbitApp(port=0, render_frame=render_frame, textured_earth=False, stars=False)
    try:
        task = app.task
        assert task is not None
        task.data.qpos[:3] = [0.2, -0.3, 0.1]
        task.data.qpos[3:7] = vtf.SO3.from_rpy_radians(0.2, -0.3, 0.4).wxyz
        task.data.qpos[7:] = np.linspace(0.1, 0.6, task.model.nq - 7)
        mjo_forward(task.model, task.data)
        app._render()

        # Follow the app's Reset path, including rebuilding nodes at the same
        # paths and a subsequent mesh rebuild when the display scale changes.
        task.reset()
        app._attach_scene()
        assert app.mj_scene is not None
        app._scale = 3.0
        app.mj_scene.set_scale(app._scale)
        app._render()

        # Read the actual retained messages replayed to a connecting browser.
        # Keep this Viser-internal access here, out of production code.
        buffer = app.server._websock_server._broadcast_buffer
        with buffer.buffer_lock:
            messages = list(buffer.message_from_id.values())
        positions, rotations = {}, {}
        for message in messages:
            if type(message).__name__ == "SetPositionMessage":
                positions[message.name] = np.asarray(message.position)
            elif type(message).__name__ == "SetOrientationMessage":
                rotations[message.name] = vtf.SO3(np.asarray(message.wxyz)).as_matrix()

        # Independent reference: plain MuJoCo forward kinematics, including
        # the static capsule rotations and the solar panels' local offsets.
        reference = mujoco.MjModel.from_xml_string(task.model._raw_xml)
        state = mujoco.MjData(reference)
        state.qpos[:] = task.data.qpos
        mujoco.mj_forward(reference, state)
        rotation = task.data.frame.C_LI if render_frame == "lvlh" else np.eye(3)
        for geom_id in range(reference.ngeom):
            body_name = task.model.body_name(int(reference.geom_bodyid[geom_id]))
            body_path = f"/task/spacecraft/{body_name}"
            geom_path = f"{body_path}/{task.model.geom_name(geom_id)}"
            body_rotation = rotations.get(body_path, np.eye(3))
            position = positions.get(body_path, np.zeros(3)) + body_rotation @ positions.get(
                geom_path, np.zeros(3)
            )
            orientation = body_rotation @ rotations.get(geom_path, np.eye(3))
            np.testing.assert_allclose(
                position, app._scale * (rotation @ state.geom_xpos[geom_id]),
                atol=1e-8, err_msg=geom_path,
            )
            np.testing.assert_allclose(
                orientation, rotation @ state.geom_xmat[geom_id].reshape(3, 3),
                atol=1e-8, err_msg=geom_path,
            )
    finally:
        app.close()
