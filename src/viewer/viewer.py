# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""MjOrbitViewer — interactive 3D viewer for mujoco_orbit simulations.

MuJoCo simulates in chief-centered inertial coordinates, while the viewer can
render either a local LVLH view or the chief-relative state translated into ECI
so the system visibly orbits Earth. Units are **metres** (MuJoCo SI).
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence

import numpy as np
import viser

from mujoco_orbit.rollout import mjo_get_state, mjo_set_state
from mujoco_orbit.runtime import MjoData, MjoModel
from mujoco_orbit.step import mjo_forward, mjo_step

from .bodies import MuJoCoScene
from .contacts import ContactForceOverlay
from .earth import BodyTrail, EarthVisual
from .framing import (
    DEFAULT_DISTANCE_RATIO,
    CameraTracker,
    default_camera_pose,
    lvlh_basis_eci,
    spacecraft_bounding_radius,
)

# Default trail colour cycle (RGBA)
_TRAIL_COLORS: list[tuple[int, int, int]] = [
    (255, 200, 50),   # gold
    (50, 200, 255),   # cyan
    (255, 100, 100),  # salmon
    (100, 255, 100),  # lime
    (200, 150, 255),  # lavender
]

_SPEED_OPTIONS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
_LOCAL_SCALE_OPTIONS = [
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    2500.0,
    5000.0,
    10000.0,
]
_DEFAULT_LOCAL_SCENE_SCALE = 1.0


def _assert_socket_bindable(host: str, port: int) -> None:
    bind_host = host if host not in ("", "0.0.0.0") else "127.0.0.1"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((bind_host, port))
    except OSError as exc:
        raise RuntimeError(
            f"MjOrbitViewer cannot start a local viser server on {host}:{port}. "
            "Local socket bind is unavailable in this environment."
        ) from exc


@dataclass
class _ViewerSnapshot:
    state: np.ndarray
    ctrl: np.ndarray
    rw_torque_cmd: np.ndarray
    mtq_dipole_cmd: np.ndarray
    thr_force_cmd: np.ndarray
    cmg_gimbal_rate_cmd: np.ndarray


class MjOrbitViewer:
    """Real-time coupled orbital + multibody viewer.

    Parameters
    ----------
    model, data : MjoModel, MjoData
        Compiled model and runtime state.
    host, port : str, int
        Viser server bind address.
    show_earth : bool
        Render Earth in the active visualization frame.
    textured_earth : bool
        Use the photoreal NASA Blue Marble texture (falls back to a flat
        blue sphere when the asset is unavailable).
    show_axes : bool
        Draw LVLH reference axes at the origin.
    track_body : str or None
        If given, record and draw this body's trajectory trail.
        Deprecated in favour of *track_bodies*.
    track_bodies : list of str, optional
        Body names whose trajectories are drawn as coloured trails.
    trail_max_points : int
        Maximum trail history length per body.
    camera_distance : float or None
        Initial camera distance from the tracked body (metres). If *None*,
        it is derived from the spacecraft's bounding radius so the model
        fills the frame with Earth visible in the background.
    track_camera : bool
        Keep the camera locked onto the spacecraft as it moves (toggleable
        from the GUI).
    render_frame : {"lvlh", "eci"}
        World frame used for visualization. ``"lvlh"`` rotates chief-inertial
        MuJoCo positions into the chief-centered LVLH frame. ``"eci"``
        translates the chief-relative MuJoCo world by the chief ECI position.
    """

    def __init__(
        self,
        model: MjoModel,
        data: MjoData,
        host: str = "0.0.0.0",
        port: int = 8080,
        show_earth: bool = True,
        textured_earth: bool = True,
        show_axes: bool = True,
        track_body: Optional[str] = None,
        track_bodies: Optional[Sequence[str]] = None,
        trail_max_points: int = 2000,
        camera_distance: Optional[float] = None,
        track_camera: bool = True,
        render_frame: Literal["lvlh", "eci"] = "lvlh",
    ) -> None:
        self.model = model
        self.data = data

        self._port = port
        self._show_earth = show_earth
        self._show_axes = show_axes
        self._render_frame = render_frame
        self._camera_distance = camera_distance

        # ---- viser server ---------------------------------------------------
        _assert_socket_bindable(host, port)
        self.server = viser.ViserServer(host=host, port=port)
        self.server.scene.set_up_direction("+z")
        self._local_scene = self.server.scene.add_frame("/local_scene", show_axes=False)

        # ---- Body trails -----------------------------------------------------
        # Unify track_body (legacy) and track_bodies into a single list
        body_names: list[str] = []
        if track_bodies is not None:
            body_names.extend(track_bodies)
        elif track_body is not None:
            body_names.append(track_body)

        self._track_ids: list[int] = []
        self.trails: list[BodyTrail] = []
        for i, name in enumerate(body_names):
            self._track_ids.append(self.model.body_id(name))
            color = _TRAIL_COLORS[i % len(_TRAIL_COLORS)]
            self.trails.append(
                BodyTrail(
                    self.server,
                    name,
                    max_points=trail_max_points,
                    color=color,
                    path_prefix="/local_scene/trail",
                )
            )

        # Backward-compat alias
        self.trail: Optional[BodyTrail] = self.trails[0] if self.trails else None

        # ---- Camera ----------------------------------------------------------
        self._camera_body_id = self._track_ids[0] if self._track_ids else 1
        R_orbit_m = float(np.linalg.norm(self.data.orbit.R_eci)) * 1000.0
        self.tracker = CameraTracker(self.server)
        self.tracker.enabled = track_camera
        if show_earth:
            # Far plane must reach Earth's far limb from orbit altitude.
            self.server.initial_camera.far = R_orbit_m * 2.5

        # ---- MuJoCo body geometry -------------------------------------------
        self.mj_scene = MuJoCoScene(
            self.server,
            self.model,
            root_path="/local_scene/spacecraft",
        )

        # ---- Earth -----------------------------------------------------------
        self.earth: EarthVisual | None = None
        if show_earth:
            self.earth = EarthVisual(self.server, textured=textured_earth)
            self._update_earth()

        # ---- GUI controls ----------------------------------------------------
        self._speed_index = _SPEED_OPTIONS.index(1.0)
        self._speed = _SPEED_OPTIONS[self._speed_index]
        self._scale_index = _LOCAL_SCALE_OPTIONS.index(_DEFAULT_LOCAL_SCENE_SCALE)
        self._local_scene_scale = _LOCAL_SCALE_OPTIONS[self._scale_index]
        self._paused = False
        self._reset_requested = False
        self._sim_t = 0.0
        self._budget = 0.0
        self._last_wall = time.time()
        self._axes_handle: viser.SceneNodeHandle | None = None
        self._initial_snapshot = self._capture_snapshot()
        self._contact_force_overlay = ContactForceOverlay(
            self.server,
            path="/local_scene/contact_forces",
        )
        self._apply_local_scene_scale()
        self._setup_gui()
        self._reframe_camera()

        # Initial render
        rotation, translation = self._world_transform()
        self.mj_scene.update(
            self.data,
            rotation=rotation,
            translation=translation,
            scale_origin=self._scale_origin(),
        )

    # ------------------------------------------------------------------
    # Scene helpers
    # ------------------------------------------------------------------

    def _world_transform(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self._render_frame == "lvlh":
            rotation = self.data.frame.C_LI
            return rotation, None
        return None, 1000.0 * self.data.orbit.R_eci

    def _scale_origin(self) -> np.ndarray:
        if self._render_frame == "eci":
            return 1000.0 * self.data.orbit.R_eci
        return np.zeros(3)

    def _body_render_position(self, body_id: int) -> np.ndarray:
        """A body's position in the render frame, in metres, after scaling."""
        pos = np.asarray(self.data.xpos[body_id], dtype=float).copy()
        rotation, translation = self._world_transform()
        if rotation is not None:
            pos = rotation @ pos
        if translation is not None:
            pos = pos + translation
        origin = self._scale_origin()
        return origin + self._local_scene_scale * (pos - origin)

    def _update_earth(self) -> None:
        if self.earth is None:
            return
        if self._render_frame == "eci":
            position = np.zeros(3)
            rotation = None
        else:
            position = self.data.frame.C_LI @ (-1000.0 * self.data.orbit.R_eci)
            rotation = self.data.frame.C_LI
        self.earth.update(
            position=position,
            sim_time=float(self.data.orbit.t),
            rotation=rotation,
        )

    def _reframe_camera(self) -> None:
        """Frame the tracked body with Earth in the background."""
        target = self._body_render_position(self._camera_body_id)
        if self._camera_distance is not None:
            distance = self._camera_distance
        else:
            radius = spacecraft_bounding_radius(self.model, self.data, self._camera_body_id)
            distance = DEFAULT_DISTANCE_RATIO * radius * self._local_scene_scale
        basis = None
        if self._render_frame == "eci":
            basis = lvlh_basis_eci(self.data.orbit.R_eci, self.data.orbit.V_eci)
        position, look_at = default_camera_pose(target, distance=distance, basis=basis)
        self.tracker.set_default_pose(position, look_at)
        self.tracker.retarget(target)
        self.tracker.reframe()

    def _render_lvlh_axes(self) -> None:
        """Draw R (red), S (green), W (blue) axes at the origin."""
        length = 1.0  # m
        origins = np.zeros((3, 3))
        ends = self._local_scene_scale * np.diag([length, length, length])
        segments = np.stack([origins, ends], axis=1)  # (3, 2, 3)
        if self._render_frame == "eci":
            segments = segments @ self.data.frame.C_IL.T
            segments = segments + 1000.0 * self.data.orbit.R_eci
        colors = np.array(
            [
                [[255, 50, 50], [255, 50, 50]],
                [[50, 255, 50], [50, 255, 50]],
                [[50, 50, 255], [50, 50, 255]],
            ],
            dtype=np.uint8,
        )  # (3, 2, 3)
        if self._axes_handle is not None:
            self._axes_handle.remove()
        self._axes_handle = self.server.scene.add_line_segments(
            "/local_scene/lvlh_axes", segments, colors=colors, line_width=3.0,
        )

    def _add_lvlh_axes(self) -> None:
        self._render_lvlh_axes()

    def _capture_snapshot(self) -> _ViewerSnapshot:
        return _ViewerSnapshot(
            state=mjo_get_state(self.model, self.data).copy(),
            ctrl=self.data.ctrl.copy(),
            rw_torque_cmd=self.data.actuators.rw_torque_cmd.copy(),
            mtq_dipole_cmd=self.data.actuators.mtq_dipole_cmd.copy(),
            thr_force_cmd=self.data.actuators.thr_force_cmd.copy(),
            cmg_gimbal_rate_cmd=self.data.actuators.cmg_gimbal_rate_cmd.copy(),
        )

    def _restore_snapshot(self, snapshot: _ViewerSnapshot) -> None:
        mjo_set_state(self.model, self.data, snapshot.state)
        self.data.ctrl[:] = snapshot.ctrl
        self.data.actuators.rw_torque_cmd[:] = snapshot.rw_torque_cmd
        self.data.actuators.mtq_dipole_cmd[:] = snapshot.mtq_dipole_cmd
        self.data.actuators.thr_force_cmd[:] = snapshot.thr_force_cmd
        self.data.actuators.cmg_gimbal_rate_cmd[:] = snapshot.cmg_gimbal_rate_cmd
        mjo_forward(self.model, self.data)

    def _clear_visual_history(self) -> None:
        for trail in self.trails:
            trail.clear()
        self._contact_force_overlay.clear()

    def _apply_local_scene_scale(self) -> None:
        self.mj_scene.set_scale(self._local_scene_scale)
        for trail in self.trails:
            trail.set_scale(self._local_scene_scale)
        self._contact_force_overlay.set_scale(self._local_scene_scale)
        if self._show_axes:
            self._render_lvlh_axes()

    def set_local_scene_scale(self, scale: float) -> None:
        """Set the local-scene visualization scale relative to Earth."""
        self._local_scene_scale = float(scale)
        self._apply_local_scene_scale()
        rotation, translation = self._world_transform()
        self.mj_scene.update(
            self.data,
            rotation=rotation,
            translation=translation,
            scale_origin=self._scale_origin(),
        )
        if hasattr(self, "_scale_md"):
            self._scale_md.content = self._scale_markdown()
        if self._camera_distance is None and hasattr(self, "tracker"):
            self._reframe_camera()

    def reset_simulation(self) -> None:
        """Restore the viewer-managed simulation to its initial state."""
        self._restore_snapshot(self._initial_snapshot)
        self._clear_visual_history()
        self._sim_t = 0.0
        self._budget = 0.0
        self._last_wall = time.time()
        rotation, translation = self._world_transform()
        self.mj_scene.update(
            self.data,
            rotation=rotation,
            translation=translation,
            scale_origin=self._scale_origin(),
        )
        self.tracker.retarget(self._body_render_position(self._camera_body_id))
        self._time_md.content = f"**t** = {self._sim_t:.2f} s"

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------

    def _setup_gui(self) -> None:
        with self.server.gui.add_folder("Playback"):
            self._pause_btn = self.server.gui.add_button("Pause")
            self._reset_btn = self.server.gui.add_button("Reset")
            self._speed_btns = self.server.gui.add_button_group(
                "Speed", options=["Slower", "1x", "Faster"],
            )
            self._speed_md = self.server.gui.add_markdown(self._speed_markdown())

        with self.server.gui.add_folder("Scene"):
            self._scale_btns = self.server.gui.add_button_group(
                "Local Scale", options=["Smaller", "1x", "Larger"],
            )
            self._scale_md = self.server.gui.add_markdown(self._scale_markdown())

        with self.server.gui.add_folder("Camera"):
            self._track_checkbox = self.server.gui.add_checkbox(
                "Track spacecraft",
                initial_value=self.tracker.enabled,
            )
            self._reframe_btn = self.server.gui.add_button("Reframe view")

        with self.server.gui.add_folder("Visualization"):
            self._contact_force_checkbox = self.server.gui.add_checkbox(
                "Contact Forces",
                initial_value=False,
            )

        self._time_md = self.server.gui.add_markdown("**t** = 0.00 s")

        @self._pause_btn.on_click
        def _(_) -> None:  # type: ignore[arg-type]
            self._paused = not self._paused
            self._pause_btn.label = "Resume" if self._paused else "Pause"

        @self._reset_btn.on_click
        def _(_) -> None:  # type: ignore[arg-type]
            self._reset_requested = True

        @self._speed_btns.on_click
        def _(event) -> None:
            if event.target.value == "Slower":
                self._speed_index = max(0, self._speed_index - 1)
            elif event.target.value == "Faster":
                self._speed_index = min(len(_SPEED_OPTIONS) - 1, self._speed_index + 1)
            else:
                self._speed_index = _SPEED_OPTIONS.index(1.0)

            self._speed = _SPEED_OPTIONS[self._speed_index]
            self._speed_md.content = self._speed_markdown()

        @self._scale_btns.on_click
        def _(event) -> None:
            if event.target.value == "Smaller":
                self._scale_index = max(0, self._scale_index - 1)
            elif event.target.value == "Larger":
                self._scale_index = min(len(_LOCAL_SCALE_OPTIONS) - 1, self._scale_index + 1)
            else:
                self._scale_index = _LOCAL_SCALE_OPTIONS.index(1.0)

            self.set_local_scene_scale(_LOCAL_SCALE_OPTIONS[self._scale_index])
            self._scale_md.content = self._scale_markdown()

        @self._track_checkbox.on_update
        def _(_) -> None:
            self.tracker.enabled = self._track_checkbox.value

        @self._reframe_btn.on_click
        def _(_) -> None:
            self._reframe_camera()

    def _speed_markdown(self) -> str:
        return f"**speed** = {self._speed:g}x"

    def _scale_markdown(self) -> str:
        return f"**scale** = {self._local_scene_scale:g}x"

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        duration: Optional[float] = 600.0,
        action_fn: Optional[Callable[[object, float], Optional[np.ndarray]]] = None,
    ) -> None:
        """Run the simulation + viewer loop until *duration* sim-seconds elapse.

        Parameters
        ----------
        duration : float or None
            Maximum simulation time (seconds). If None, the loop runs
            indefinitely until interrupted (Ctrl+C / browser close).
        action_fn : callable, optional
            ``action_fn(target, sim_time) -> ctrl_array | None``
            Called every physics step to supply MuJoCo controls.
        """
        dt = self.model.opt.timestep
        forever = duration is None
        self._last_wall = time.time()

        print(f"Viewer running at http://localhost:{self._port}")

        try:
            while forever or self._sim_t < duration:
                now = time.time()
                wall_dt = min(now - self._last_wall, 0.1)  # cap to avoid spiral
                self._last_wall = now

                if self._reset_requested:
                    self.reset_simulation()
                    self._reset_requested = False

                if not self._paused:
                    self._budget += wall_dt * self._speed

                    steps = 0
                    while self._budget >= dt and (forever or self._sim_t < duration):
                        ctrl = action_fn(self.data, self._sim_t) if action_fn else None
                        if ctrl is not None:
                            np.copyto(self.data.ctrl, ctrl)
                        mjo_step(self.model, self.data)
                        self._sim_t += dt
                        self._budget -= dt
                        steps += 1

                        if self.trails:
                            for tid, trail in zip(self._track_ids, self.trails):
                                trail.append(self.data.xipos[tid].copy())

                        # Cap steps per render frame to stay responsive
                        if steps >= 200:
                            self._budget = 0.0
                            break

                # ---- update visuals ------------------------------------------
                rotation, translation = self._world_transform()
                self.mj_scene.update(
                    self.data,
                    rotation=rotation,
                    translation=translation,
                    scale_origin=self._scale_origin(),
                )
                self._update_earth()
                if self._show_axes and self._render_frame == "eci":
                    self._render_lvlh_axes()
                for tid, trail in zip(self._track_ids, self.trails):
                    trail.render(
                        current_position=self.data.xipos[tid],
                        rotation=rotation,
                        translation=translation,
                    )
                self._contact_force_overlay.render_transformed(
                    self.model,
                    self.data,
                    visible=self._contact_force_checkbox.value,
                    rotation=rotation,
                    translation=translation,
                )
                self.tracker.update(self._body_render_position(self._camera_body_id))
                self._time_md.content = f"**t** = {self._sim_t:.2f} s"

                time.sleep(1.0 / 60.0)

        except KeyboardInterrupt:
            print("\nViewer stopped.")
