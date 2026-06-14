"""Instanced-rendering math checks for viewer.batched.

The batched fleet renderer recomputes geom world poses on the host from body
``xpos``/``xquat``; these tests pin that math against scipy and against
MuJoCo's own ``geom_xpos``/``geom_xmat``.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from mjorbit import MjoModel, OrbitInit, mjo_forward
from mjorbit.testdata import SPACECRAFT_BIMANUAL_PANELS_XML

batched = pytest.importorskip("viewer.batched")


def test_quat_helpers_match_scipy() -> None:
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(0)
    quats = rng.standard_normal((64, 4))
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)

    # scipy uses (x,y,z,w); ours is MuJoCo's (w,x,y,z).
    ref = Rotation.from_quat(np.roll(quats, -1, axis=-1)).as_matrix()
    np.testing.assert_allclose(batched._quat_to_mats(quats), ref, atol=1e-12)

    q2 = rng.standard_normal(4)
    q2 /= np.linalg.norm(q2)
    prod = batched._quat_mul(quats, q2)
    ref_prod = (
        Rotation.from_quat(np.roll(quats, -1, axis=-1))
        * Rotation.from_quat(np.roll(q2, -1))
    ).as_matrix()
    np.testing.assert_allclose(batched._quat_to_mats(prod), ref_prod, atol=1e-12)


def test_instance_math_matches_mujoco_geom_poses() -> None:
    """pos = xpos + R(xquat) @ geom_pos and quat = xquat * geom_quat must
    reproduce MuJoCo's geom_xpos / geom_xmat exactly."""
    model = MjoModel.from_xml_path(SPACECRAFT_BIMANUAL_PANELS_XML)
    data = model.make_data(
        orbit=OrbitInit(R_eci=np.array([7000.0, 0.0, 0.0]), V_eci=np.array([0.0, 7.5, 0.0]))
    )
    rng = np.random.default_rng(1)
    quat = rng.standard_normal(4)
    data.qpos[3:7] = quat / np.linalg.norm(quat)
    data.qpos[7:11] = rng.uniform(-1.0, 1.0, size=4)  # arm joints
    data.qpos[11:17] = rng.uniform(0.0, 0.5, size=6)  # panel hinges
    mjo_forward(model, data)

    # Independent reference: plain MuJoCo forward kinematics on the stripped
    # XML with the same qpos.
    mj_model = mujoco.MjModel.from_xml_string(model._raw_xml)
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = np.asarray(data.qpos)
    mujoco.mj_forward(mj_model, mj_data)

    xpos = np.asarray(data.xpos)[None]  # (1, nbody, 3)
    xquat = np.asarray(data.xquat)[None]  # (1, nbody, 4)
    mats = batched._quat_to_mats(xquat)

    for geom_id in range(int(model.ngeom)):
        body_id = int(model.geom_bodyid[geom_id])
        local_pos = np.asarray(model.geom_pos[geom_id], dtype=float)
        local_quat = np.asarray(model.geom_quat[geom_id], dtype=float)

        pos = xpos[0, body_id] + mats[0, body_id] @ local_pos
        rot = batched._quat_to_mats(batched._quat_mul(xquat[:, body_id], local_quat))[0]

        np.testing.assert_allclose(
            pos, mj_data.geom_xpos[geom_id], atol=1e-10,
            err_msg=f"geom {geom_id} position",
        )
        np.testing.assert_allclose(
            rot, mj_data.geom_xmat[geom_id].reshape(3, 3), atol=1e-10,
            err_msg=f"geom {geom_id} rotation",
        )
