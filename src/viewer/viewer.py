# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""MjOrbitViewer — interactive 3D viewer for mujoco_orbit simulations.

MuJoCo simulates in chief-centered inertial coordinates, while the viewer can
render either a local LVLH view or the chief-relative state translated into ECI
so the system visibly orbits Earth. Units are **metres** (MuJoCo SI).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Sequence

import mujoco
import numpy as np
import viser

from mujoco_orbit.core.runtime import MjoData, MjoModel
from mujoco_orbit.core.step import mjo_forward, mjo_step

from .bodies import MuJoCoScene
from .contacts import ContactForceOverlay
from .earth import BodyTrail, add_earth

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


@dataclass
class _ViewerSnapshot:
    mj_state: np.ndarray
    orbit_R_eci: np.ndarray
    orbit_V_eci: np.ndarray
    orbit_t: float
    rw_speed: np.ndarray
    rw_torque_cmd: np.ndarray
    mtq_dipole_cmd: np.ndarray
    thr_force_cmd: np.ndarray
    cmg_gimbal_angle: np.ndarray
    cmg_gimbal_rate_cmd: np.ndarray
    cmg_rotor_momentum: np.ndarray
    sensor_biases: dict[str, np.ndarray]
    sensor_rng_state: Any


class MjOrbitViewer:
    """Real-time coupled orbital + multibody viewer.

    Parameters
    ----------
    model, data : MjoModel, MjoData
        Compiled model and runtime state.
    host, port : str, int
        Viser server bind address.
    show_earth : bool
        Render an Earth icosphere in the active visualization frame.
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
        Initial camera distance from the origin (metres).  If *None*,
        defaults to 10 m for detail view.
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
        show_axes: bool = True,
        track_body: Optional[str] = None,
        track_bodies: Optional[Sequence[str]] = None,
        trail_max_points: int = 2000,
        camera_distance: Optional[float] = None,
        render_frame: Literal["lvlh", "eci"] = "lvlh",
    ) -> None:
        self.model = model
        self.data = data

        self._port = port
        self._show_earth = show_earth
        self._show_axes = show_axes
        self._render_frame = render_frame

        # ---- viser server ---------------------------------------------------
        self.server = viser.ViserServer(host=host, port=port)
        self.server.scene.set_up_direction("+z")
        self._local_scene = self.server.scene.add_frame("/local_scene", show_axes=False)

        # ---- Camera ----------------------------------------------------------
        R_orbit_m = float(np.linalg.norm(self.data.orbit.R_eci)) * 1000.0
        cam_dist = camera_distance if camera_distance is not None else 10.0
        cam_look_at = np.zeros(3)
        cam_position = np.array([0.0, -cam_dist, cam_dist * 0.5])
        if self._render_frame == "eci":
            cam_look_at = 1000.0 * self.data.orbit.R_eci
            cam_position = cam_look_at + cam_position
        self.server.initial_camera.position = tuple(cam_position)
        self.server.initial_camera.look_at = tuple(cam_look_at)
        if show_earth:
            # Far plane must reach Earth surface: orbit radius + Earth radius
            self.server.initial_camera.far = R_orbit_m * 2.5

        # ---- MuJoCo body geometry -------------------------------------------
        self.mj_scene = MuJoCoScene(
            self.server,
            self.model.mj_model,
            root_path="/local_scene/spacecraft",
        )

        # ---- Earth -----------------------------------------------------------
        if show_earth:
            earth_position = (
                (0.0, 0.0, 0.0)
                if self._render_frame == "eci"
                else (-R_orbit_m, 0.0, 0.0)
            )
            add_earth(self.server, position=earth_position)

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

        # Initial render
        rotation, translation = self._world_transform()
        self.mj_scene.update(
            self.data.mj_data,
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
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        mj_state = np.empty(mujoco.mj_stateSize(self.model.mj_model, state_spec))
        mujoco.mj_getState(self.model.mj_model, self.data.mj_data, mj_state, state_spec)

        return _ViewerSnapshot(
            mj_state=mj_state.copy(),
            orbit_R_eci=self.data.orbit.R_eci.copy(),
            orbit_V_eci=self.data.orbit.V_eci.copy(),
            orbit_t=float(self.data.orbit.t),
            rw_speed=self.data.actuators.rw_speed.copy(),
            rw_torque_cmd=self.data.actuators.rw_torque_cmd.copy(),
            mtq_dipole_cmd=self.data.actuators.mtq_dipole_cmd.copy(),
            thr_force_cmd=self.data.actuators.thr_force_cmd.copy(),
            cmg_gimbal_angle=self.data.actuators.cmg_gimbal_angle.copy(),
            cmg_gimbal_rate_cmd=self.data.actuators.cmg_gimbal_rate_cmd.copy(),
            cmg_rotor_momentum=self.data.actuators.cmg_rotor_momentum.copy(),
            sensor_biases={
                name: bias.copy() for name, bias in self.data.sensors.biases.items()
            },
            sensor_rng_state=copy.deepcopy(self.data.sensors.rng.bit_generator.state),
        )

    def _restore_snapshot(self, snapshot: _ViewerSnapshot) -> None:
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        mujoco.mj_setState(self.model.mj_model, self.data.mj_data, snapshot.mj_state, state_spec)

        self.data.orbit.R_eci[:] = snapshot.orbit_R_eci
        self.data.orbit.V_eci[:] = snapshot.orbit_V_eci
        self.data.orbit.t = snapshot.orbit_t

        self.data.actuators.rw_speed[:] = snapshot.rw_speed
        self.data.actuators.rw_torque_cmd[:] = snapshot.rw_torque_cmd
        self.data.actuators.mtq_dipole_cmd[:] = snapshot.mtq_dipole_cmd
        self.data.actuators.thr_force_cmd[:] = snapshot.thr_force_cmd
        self.data.actuators.cmg_gimbal_angle[:] = snapshot.cmg_gimbal_angle
        self.data.actuators.cmg_gimbal_rate_cmd[:] = snapshot.cmg_gimbal_rate_cmd
        self.data.actuators.cmg_rotor_momentum[:] = snapshot.cmg_rotor_momentum
        self.data.actuators.update_rw_momentum(self.model.rw_inertia)

        self.data.sensors.biases = {
            name: bias.copy() for name, bias in snapshot.sensor_biases.items()
        }
        self.data.sensors.rng.bit_generator.state = copy.deepcopy(snapshot.sensor_rng_state)

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
            self.data.mj_data,
            rotation=rotation,
            translation=translation,
            scale_origin=self._scale_origin(),
        )
        if hasattr(self, "_scale_md"):
            self._scale_md.content = self._scale_markdown()

    def reset_simulation(self) -> None:
        """Restore the viewer-managed simulation to its initial state."""
        self._restore_snapshot(self._initial_snapshot)
        self._clear_visual_history()
        self._sim_t = 0.0
        self._budget = 0.0
        self._last_wall = time.time()
        rotation, translation = self._world_transform()
        self.mj_scene.update(
            self.data.mj_data,
            rotation=rotation,
            translation=translation,
            scale_origin=self._scale_origin(),
        )
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

    def _speed_markdown(self) -> str:
        return f"**speed** = {self._speed:g}x"

    def _scale_markdown(self) -> str:
        return f"**scale** = {self._local_scene_scale:g}x"

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        duration: float = 600.0,
        action_fn: Optional[Callable[[object, float], Optional[np.ndarray]]] = None,
    ) -> None:
        """Run the simulation + viewer loop until *duration* sim-seconds elapse.

        Parameters
        ----------
        duration : float
            Maximum simulation time (seconds).
        action_fn : callable, optional
            ``action_fn(target, sim_time) -> ctrl_array | None``
            Called every physics step to supply MuJoCo controls.
        """
        dt = self.model.opt.timestep
        self._last_wall = time.time()

        print(f"Viewer running at http://localhost:{self._port}")

        try:
            while self._sim_t < duration:
                now = time.time()
                wall_dt = min(now - self._last_wall, 0.1)  # cap to avoid spiral
                self._last_wall = now

                if self._reset_requested:
                    self.reset_simulation()
                    self._reset_requested = False

                if not self._paused:
                    self._budget += wall_dt * self._speed

                    steps = 0
                    while self._budget >= dt and self._sim_t < duration:
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
                    self.data.mj_data,
                    rotation=rotation,
                    translation=translation,
                    scale_origin=self._scale_origin(),
                )
                if self._show_axes and self._render_frame == "eci":
                    self._render_lvlh_axes()
                for tid, trail in zip(self._track_ids, self.trails):
                    trail.render(
                        current_position=self.data.xipos[tid],
                        rotation=rotation,
                        translation=translation,
                    )
                self._contact_force_overlay.render_transformed(
                    self.model.mj_model,
                    self.data.mj_data,
                    visible=self._contact_force_checkbox.value,
                    rotation=rotation,
                    translation=translation,
                )
                self._time_md.content = f"**t** = {self._sim_t:.2f} s"

                time.sleep(1.0 / 60.0)

        except KeyboardInterrupt:
            print("\nViewer stopped.")
