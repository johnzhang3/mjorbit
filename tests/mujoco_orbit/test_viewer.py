from __future__ import annotations

import numpy as np
import pytest

from mujoco_orbit import mjo_forward, mjo_step
from mujoco_orbit.testdata import FREE_BODY_XML, TWO_BODIES_XML
from tests.mujoco_orbit._helpers import make_model_data, set_freejoint_lvlh_state
from viewer import MjOrbitViewer
from viewer.contacts import contact_force_segments
from viewer.earth import BodyTrail


def test_contact_force_segments_for_collision() -> None:
    model, data = make_model_data(
        xml_path=TWO_BODIES_XML,
        mj_timestep=0.002,
    )
    data.qvel[0] += 1.0
    mjo_forward(model, data)

    for _ in range(3000):
        mjo_step(model, data)
        if data.mj_data.ncon > 0:
            break

    assert data.mj_data.ncon > 0, "expected the collision fixture to generate contact"

    payload = contact_force_segments(
        model.mj_model,
        data.mj_data,
        force_scale=1.0e-3,
    )

    assert payload is not None
    segments, colors = payload
    assert segments.shape[1:] == (2, 3)
    assert colors.shape == (segments.shape[0], 2, 3)
    assert np.all(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) > 0.0)


def test_viewer_reset_restores_initial_state() -> None:
    model, data = make_model_data(xml_path=FREE_BODY_XML)
    data.qvel[0] += 1.25
    mjo_forward(model, data)

    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()
    initial_R_eci = data.orbit.R_eci.copy()
    initial_V_eci = data.orbit.V_eci.copy()

    viewer = MjOrbitViewer(
        model,
        data,
        port=0,
        show_earth=False,
        show_axes=False,
        track_body="spacecraft",
    )

    try:
        assert viewer._speed == pytest.approx(1.0)
        assert viewer._local_scene_scale == pytest.approx(1.0)

        for _ in range(3):
            mjo_step(model, data)
            viewer.trails[0].append(data.xipos[1].copy())

        viewer.set_local_scene_scale(10.0)
        np.testing.assert_allclose(
            np.asarray(viewer.mj_scene._body_frames[0].position),
            10.0 * data.lvlh_position_from_eci(data.xpos[1]),
            atol=1e-8,
        )

        set_freejoint_lvlh_state(data, slice(0, 3), slice(0, 3), [4.0, 0.0, 0.0])
        data.orbit.R_eci[:] = initial_R_eci + np.array([1.0, 2.0, 3.0])
        data.orbit.V_eci[:] = initial_V_eci + np.array([0.1, 0.2, 0.3])
        mjo_forward(model, data)

        assert viewer.trails[0]._positions

        viewer.reset_simulation()

        np.testing.assert_allclose(data.qpos, initial_qpos)
        np.testing.assert_allclose(data.qvel, initial_qvel)
        np.testing.assert_allclose(data.orbit.R_eci, initial_R_eci)
        np.testing.assert_allclose(data.orbit.V_eci, initial_V_eci)
        assert viewer._sim_t == pytest.approx(0.0)
        np.testing.assert_allclose(
            np.asarray(viewer.mj_scene._body_frames[0].position),
            viewer._local_scene_scale * data.lvlh_position_from_eci(data.xpos[1]),
            atol=1e-8,
        )
        assert viewer.trails[0]._positions == []
    finally:
        viewer.server.stop()


def test_body_trail_renders_through_current_position() -> None:
    import viser

    server = viser.ViserServer(port=0)
    try:
        trail = BodyTrail(server, "body")
        trail.append(np.array([0.0, 0.0, 0.0]))
        pts = trail._render_points(current_position=np.array([1.0, 0.0, 0.0]))
        assert pts is not None
        np.testing.assert_allclose(pts[-1], [1.0, 0.0, 0.0])
    finally:
        server.stop()


def test_viewer_eci_render_positions_bodies_around_earth() -> None:
    model, data = make_model_data(xml_path=FREE_BODY_XML)
    viewer = MjOrbitViewer(
        model,
        data,
        port=0,
        show_earth=False,
        show_axes=False,
        render_frame="eci",
    )
    try:
        viewer.set_local_scene_scale(10.0)
        origin = 1000.0 * data.orbit.R_eci
        expected = origin + 10.0 * (data.xpos[1] - origin)
        np.testing.assert_allclose(
            np.asarray(viewer.mj_scene._body_frames[0].position),
            expected,
        )
    finally:
        viewer.server.stop()
