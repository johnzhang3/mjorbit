# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Camera framing and tracking helpers for the orbit viewers.

Default framing places the camera on the radial-out side of the tracked
spacecraft, slightly behind along-track and above cross-track, looking at
the spacecraft. The view direction therefore has a nadir component, which
keeps Earth's limb in the background of the default shot. The camera
distance is derived from the spacecraft's geometric bounding radius so
small and large models are both framed sensibly.

``CameraTracker`` keeps connected browser cameras locked onto a moving
target by translating each client camera by the target's per-frame motion,
which preserves user-controlled orbiting/zooming relative to the target.
"""

from __future__ import annotations

import mujoco
import numpy as np
import viser

# Camera offset direction in (radial-out, along-track, cross-track) axes.
# Mostly behind the spacecraft along-track and above it, with a positive
# radial component so the look direction tilts toward Earth.
_VIEW_OFFSET_RSW = np.array([0.45, -0.78, 0.42])
_VIEW_OFFSET_RSW /= np.linalg.norm(_VIEW_OFFSET_RSW)

#: Vertical field of view (radians) assumed when the live camera value is
#: unavailable; matches viser's default client camera.
DEFAULT_CAMERA_FOV = np.radians(75.0)

#: Fraction of the vertical field of view the spacecraft's bounding sphere
#: should fill in the default framing.
DEFAULT_VIEW_FILL = 0.75


def distance_for_fill(
    radius: float,
    *,
    fov: float = DEFAULT_CAMERA_FOV,
    fill: float = DEFAULT_VIEW_FILL,
) -> float:
    """Camera distance at which a bounding sphere fills *fill* of the view.

    Derived from the camera intrinsics: a sphere of *radius* at distance d
    subtends ``2 * asin(radius / d)`` (the tangent lines from the eye touch the
    limb, so the half-angle is ``asin``, not ``atan``), so filling ``fill * fov``
    of the vertical field of view gives ``d = radius / sin(fill * fov / 2)``. The
    distance is clamped to stay outside the bounding sphere itself.
    """
    radius = float(radius)
    distance = radius / np.sin(0.5 * float(fill) * float(fov))
    return max(distance, 1.3 * radius)


def geom_bounding_radius(geom_type: int, size: np.ndarray) -> float:
    """Conservative bounding-sphere radius of a single geom (local frame)."""
    s = np.asarray(size, dtype=float)
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(s[0])
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return float(s[0] + s[1])
    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return float(np.hypot(s[0], s[1]))
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return float(np.linalg.norm(s))
    return float(np.max(np.abs(s))) if s.size else 0.0


def spacecraft_bounding_radius(model, data, center_body_id: int = 1) -> float:
    """Bounding radius (m) of all geoms about *center_body_id*'s origin.

    Uses the current kinematics in *data* (call ``mjo_forward`` first), so
    articulated arms and deployed panels are measured in their actual pose.
    """
    center = np.asarray(data.xpos[center_body_id], dtype=float)
    radius = 0.0
    for geom_id in range(int(model.ngeom)):
        body_id = int(model.geom_bodyid[geom_id])
        if body_id == 0:
            continue
        body_rot = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
        geom_pos = np.asarray(data.xpos[body_id], dtype=float) + body_rot @ np.asarray(
            model.geom_pos[geom_id], dtype=float
        )
        rbound = geom_bounding_radius(int(model.geom_type[geom_id]), model.geom_size[geom_id])
        radius = max(radius, float(np.linalg.norm(geom_pos - center)) + rbound)
    return max(radius, 0.5)


def lvlh_basis_eci(R_eci: np.ndarray, V_eci: np.ndarray) -> np.ndarray:
    """Rows are the LVLH axes (radial-out, along-track, cross-track) in ECI."""
    r = np.asarray(R_eci, dtype=float)
    v = np.asarray(V_eci, dtype=float)
    rhat = r / np.linalg.norm(r)
    w = np.cross(r, v)
    what = w / np.linalg.norm(w)
    shat = np.cross(what, rhat)
    return np.stack([rhat, shat, what])


def default_camera_pose(
    target: np.ndarray,
    *,
    distance: float,
    basis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Camera (position, look_at) framing *target* with Earth behind it.

    *basis* maps (radial-out, along-track, cross-track) offsets into render
    coordinates; ``None`` means the render frame is already LVLH-aligned
    (x radial, y along-track, z cross-track).
    """
    target = np.asarray(target, dtype=float)
    offset = _VIEW_OFFSET_RSW if basis is None else _VIEW_OFFSET_RSW @ np.asarray(basis)
    return target + float(distance) * offset, target


class CameraTracker:
    """Keeps connected client cameras locked to a moving target.

    Each frame, :meth:`update` translates every client camera by the
    target's motion since the previous frame, so user orbit/zoom offsets
    relative to the spacecraft are preserved. :meth:`reframe` snaps all
    clients back to the default framing.
    """

    def __init__(self, server: viser.ViserServer) -> None:
        self._server = server
        self.enabled = True
        self._last_target: np.ndarray | None = None
        self._default_position: np.ndarray | None = None
        self._default_look_at: np.ndarray | None = None

        @server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            if self._default_position is None or self._default_look_at is None:
                return
            client.camera.position = tuple(self._default_position)
            client.camera.look_at = tuple(self._default_look_at)

    def set_default_pose(self, position: np.ndarray, look_at: np.ndarray) -> None:
        """Set the framing applied to newly connecting clients."""
        self._default_position = np.asarray(position, dtype=float).copy()
        self._default_look_at = np.asarray(look_at, dtype=float).copy()
        self._server.initial_camera.position = tuple(self._default_position)
        self._server.initial_camera.look_at = tuple(self._default_look_at)

    def retarget(self, target: np.ndarray | None) -> None:
        """Reset the motion reference without moving any cameras."""
        self._last_target = None if target is None else np.asarray(target, dtype=float).copy()

    def update(self, target: np.ndarray) -> None:
        """Follow *target* (render-frame metres) with all connected clients."""
        target = np.asarray(target, dtype=float)
        last, self._last_target = self._last_target, target.copy()
        # Keep the default pose riding along so late-joining clients frame
        # the spacecraft at its current location.
        if (
            last is not None
            and self._default_position is not None
            and self._default_look_at is not None
        ):
            self.set_default_pose(
                self._default_position + (target - last),
                self._default_look_at + (target - last),
            )
        if not self.enabled or last is None:
            return
        delta = target - last
        if not np.any(delta):
            return
        for client in self._server.get_clients().values():
            with client.atomic():
                client.camera.position = tuple(np.asarray(client.camera.position) + delta)
                client.camera.look_at = tuple(np.asarray(client.camera.look_at) + delta)

    def reframe(self) -> None:
        """Snap every connected client back to the default framing."""
        if self._default_position is None or self._default_look_at is None:
            return
        for client in self._server.get_clients().values():
            with client.atomic():
                client.camera.position = tuple(self._default_position)
                client.camera.look_at = tuple(self._default_look_at)
