from __future__ import annotations

import mujoco
import numpy as np

# Extra collision-detection range added on top of the penalty margin, so MuJoCo
# always reports a contact slightly before the penalised gap is reached.
_DETECT_PAD = 0.3

def quat_mult(q1, q_ref):
    # q1 shape: (n_samp, M, 4), q_ref shape: (4,)
    # return: (n_samp, M, 4)
    """Hamilton product of two quaternions."""
    w1, x1, y1, z1 = q1[:, :, 0], q1[:, :, 1], q1[:, :, 2], q1[:, :, 3]
    w2, x2, y2, z2 = q_ref[0], q_ref[1], q_ref[2], q_ref[3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack((w, x, y, z), axis=-1)

class DockingCollision:
    """Class for Collision cheker during the ISS Soyuz docking scenario"""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        iss_qadr: int,
        soyuz_qadr: int,
        margin: float = 0.0,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.iss_qadr = iss_qadr
        self.soyuz_qadr = soyuz_qadr
        self.margin = float(margin)
        mujoco.mj_forward(model, self.data)
        # Reference qpos holding the (static) ISS pose; the Soyuz block is
        # overwritten per query. Defaults to the model's initial configuration.
        self._ref_qpos = self.data.qpos.copy()

    @classmethod
    def from_xml(
        cls,
        xml_path: str,
        *,
        iss_body: str = "iss_main",
        soyuz_body: str = "soyuz",
        margin: float = 0.0,
    ) -> DockingCollision:
        """Compile a contact-enabled collision model from the docking XML.

        ``margin`` is the keep-out buffer (m): a contact closer than this counts
        as a collision. The collision geoms are given a slightly larger detection
        margin so near-contacts are reported before that buffer is crossed.
        """
        model = mujoco.MjModel.from_xml_path(xml_path)
        # Re-enable contact (the XML disables it for the guidance sim).
        model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONTACT)

        iss_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, iss_body)
        soyuz_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, soyuz_body)
        if iss_root < 0 or soyuz_root < 0:
            raise ValueError("could not find iss/soyuz root bodies in the model")

        # Rewire collision filtering: ISS mesh geoms (contype 1) only see Soyuz
        # mesh geoms (conaffinity 1) and vice-versa; everything else (visual
        # duplicates, docking-port helper geoms) is disabled. Only the
        # collision-class meshes (group 3) carry geometry, matching the XML.
        detect = max(0.0, float(margin) + _DETECT_PAD)
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        for geom in range(model.ngeom):
            if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            if int(model.geom_group[geom]) != 3:  # use the collision-class meshes
                continue
            root = int(model.body_rootid[model.geom_bodyid[geom]])
            if root == iss_root:
                model.geom_contype[geom], model.geom_conaffinity[geom] = 1, 2
            elif root == soyuz_root:
                model.geom_contype[geom], model.geom_conaffinity[geom] = 2, 1
            else:
                continue
            model.geom_margin[geom] = detect

        iss_qadr = int(model.jnt_qposadr[model.body_jntadr[iss_root]])
        soyuz_qadr = int(model.jnt_qposadr[model.body_jntadr[soyuz_root]])
        return cls(model, iss_qadr=iss_qadr, soyuz_qadr=soyuz_qadr, margin=margin)

    def set_reference(self, qpos: np.ndarray) -> None:
        """Freeze the static state (notably the ISS pose) from a full qpos."""
        self._ref_qpos = np.asarray(qpos, dtype=np.float64).copy()

    # -- collision query -------------------------------------------------------

    def in_collision_pose(self, soyuz_pos: np.ndarray, soyuz_quat: np.ndarray) -> float:
        """1.0 if the Soyuz hull collides with the ISS, else 0.0.

        A collision is any convex-mesh contact closer than ``margin``, so a
        positive margin acts as a keep-out buffer and a negative one tolerates
        grazing contact.
        """
        s = self.soyuz_qadr
        data = self.data
        data.qpos[:] = self._ref_qpos
        data.qpos[s : s + 3] = soyuz_pos
        data.qpos[s + 3 : s + 7] = soyuz_quat
        mujoco.mj_kinematics(self.model, data)
        mujoco.mj_collision(self.model, data)
        for i in range(data.ncon):
            if data.contact[i].dist < self.margin:
                return 1.0
        return 0.0

    def in_collision(self, qpos: np.ndarray) -> float:
        """Collision flag (1.0/0.0) for a full qpos vector (uses its Soyuz block)."""
        s = self.soyuz_qadr
        qpos = np.asarray(qpos)
        return self.in_collision_pose(qpos[s : s + 3], qpos[s + 3 : s + 7])

    def collision_from_states(self, states: np.ndarray) -> np.ndarray:
        """Per-state collision flags (1.0/0.0) over packed rollout states.

        ``states`` is ``[t, qpos, qvel]`` with shape ``(..., state_dim)``;
        returns an array of the leading shape.
        """
        base = 1 + self.soyuz_qadr
        flat = states.reshape(-1, states.shape[-1])
        out = np.empty(flat.shape[0], dtype=np.float64)
        for i in range(flat.shape[0]):
            row = flat[i]
            out[i] = self.in_collision_pose(row[base : base + 3], row[base + 3 : base + 7])
        return out.reshape(states.shape[:-1])
