"""Single source of truth for which MJWarp data fields the orbit overlay mirrors.

``_MIRRORED_ARRAY_FIELDS`` and ``_MIRRORED_SCALAR_FIELDS`` enumerate every
field that the device→host pull copies into the public ``MjoData`` view.
``_HOST_MUTABLE_FIELDS`` lists the subset that callers are allowed to mutate
on the host before re-uploading.
"""

from __future__ import annotations

_MIRRORED_ARRAY_FIELDS = (
    "solver_niter",
    "energy",
    "qpos",
    "qvel",
    "qacc",
    "act_dot",
    "act",
    "ctrl",
    "qacc_warmstart",
    "qfrc_applied",
    "xfrc_applied",
    "sensordata",
    "xpos",
    "xipos",
    "xquat",
    "xmat",
    "ximat",
    "xanchor",
    "xaxis",
    "geom_xpos",
    "geom_xmat",
    "site_xpos",
    "site_xmat",
    "cam_xpos",
    "cam_xmat",
    "light_xpos",
    "light_xdir",
    "subtree_com",
    "cdof",
    "cinert",
    "flexvert_xpos",
    "flexedge_J",
    "flexedge_length",
    "flexedge_velocity",
    "actuator_length",
    "actuator_moment",
    "actuator_velocity",
    "moment_rownnz",
    "moment_rowadr",
    "moment_colind",
    "crb",
    "qM",
    "qLD",
    "qLDiagInv",
    "ten_wrapadr",
    "ten_wrapnum",
    "ten_J",
    "ten_length",
    "ten_velocity",
    "wrap_obj",
    "wrap_xpos",
    "cvel",
    "cdof_dot",
    "qfrc_bias",
    "qfrc_spring",
    "qfrc_damper",
    "qfrc_gravcomp",
    "qfrc_fluid",
    "qfrc_passive",
    "subtree_linvel",
    "subtree_angmom",
    "actuator_force",
    "qfrc_actuator",
    "qfrc_smooth",
    "qacc_smooth",
    "qfrc_constraint",
    "qfrc_inverse",
    "cacc",
    "cfrc_int",
    "cfrc_ext",
    "tree_island",
    "mocap_pos",
    "mocap_quat",
    "eq_active",
)

_MIRRORED_SCALAR_FIELDS = ("ncon", "ne", "nf", "nl", "nefc", "nisland")

_HOST_MUTABLE_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "qacc_warmstart",
    "qfrc_applied",
    "xfrc_applied",
    "mocap_pos",
    "mocap_quat",
    "eq_active",
)


__all__: list[str] = []