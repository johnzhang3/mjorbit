# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""mjo-viewer — judo-style interactive task viewer for mujoco_orbit.

Launch with ``mjo-viewer`` (or ``python -m viewer``) and open the printed
URL. A dropdown switches between registered tasks; each task's numeric
parameters become sliders. The camera auto-frames and tracks the
spacecraft with Earth in the background, and a photoreal textured Earth
spins at the sidereal rate underneath the orbit.

Tasks register via ``viewer.tasks.register_task``; see
``viewer/tasks/free_drift.py`` for a minimal example.
"""

from __future__ import annotations

import argparse
import dataclasses
import socket
import time
from typing import Literal, Optional, Sequence

import numpy as np
import viser

from mujoco_orbit.step import mjo_step

from .bodies import MuJoCoScene
from .earth import BodyTrail, EarthVisual, add_star_field
from .framing import (
    DEFAULT_DISTANCE_RATIO,
    CameraTracker,
    default_camera_pose,
    lvlh_basis_eci,
    spacecraft_bounding_radius,
)
from .tasks import available_tasks, get_task_class
from .tasks.base import UiSlider, ViewerTask

_TRAIL_COLORS: list[tuple[int, int, int]] = [
    (255, 200, 50),   # gold
    (50, 200, 255),   # cyan
    (255, 100, 100),  # salmon
    (100, 255, 100),  # lime
    (200, 150, 255),  # lavender
]

_SPEED_OPTIONS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
_SCALE_OPTIONS = [0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]

RenderFrame = Literal["eci", "lvlh"]


def _assert_socket_bindable(host: str, port: int) -> None:
    bind_host = host if host not in ("", "0.0.0.0") else "127.0.0.1"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((bind_host, port))
    except OSError as exc:
        raise RuntimeError(
            f"MjOrbitApp cannot start a local viser server on {host}:{port}. "
            "Local socket bind is unavailable in this environment."
        ) from exc


class MjOrbitApp:
    """Browser viewer with task switching, auto-framing, and tracking.

    Parameters
    ----------
    task : str or None
        Initial task name (see ``viewer.tasks.available_tasks()``).
    render_frame : {"eci", "lvlh"}
        Initial visualization frame; switchable from the GUI.
    textured_earth, stars : bool
        Photoreal Earth texture and background star field.
    track : bool
        Start with camera tracking enabled.
    """

    def __init__(
        self,
        *,
        task: str | None = None,
        host: str = "0.0.0.0",
        port: int = 8080,
        render_frame: RenderFrame = "eci",
        textured_earth: bool = True,
        stars: bool = True,
        track: bool = True,
    ) -> None:
        names = available_tasks()
        if not names:
            raise RuntimeError("no viewer tasks are registered")
        self._task_name = task if task is not None else names[0]
        if self._task_name not in names:
            raise KeyError(f"unknown task {self._task_name!r}; available: {', '.join(names)}")

        self._port = port
        self._render_frame: RenderFrame = render_frame
        self._scale = 1.0
        self._speed = 1.0
        self._paused = False
        self._sim_t = 0.0
        self._budget = 0.0
        self._last_wall = time.time()
        self._pending_task: str | None = None
        self._pending_reset = False
        self._pending_reframe = False

        _assert_socket_bindable(host, port)
        self.server = viser.ViserServer(host=host, port=port)
        self.server.scene.set_up_direction("+z")
        self.server.scene.set_background_image(np.zeros((2, 2, 3), dtype=np.uint8))

        self.earth = EarthVisual(self.server, textured=textured_earth)
        self._stars_enabled = stars
        self._star_handle: viser.SceneNodeHandle | None = None

        self.tracker = CameraTracker(self.server)
        self.tracker.enabled = track

        self.task: ViewerTask | None = None
        self.mj_scene: MuJoCoScene | None = None
        self._scene_model = None
        self._trails: list[BodyTrail] = []
        self._trail_body_ids: list[int] = []
        self._trail_countdown = 0

        self._task_folder: viser.GuiFolderHandle | None = None
        self._status_md: viser.GuiMarkdownHandle | None = None
        self._setup_gui()
        self._load_task(self._task_name)

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def _load_task(self, name: str) -> None:
        if self.mj_scene is not None:
            self.mj_scene.remove()
            self.mj_scene = None
        for trail in self._trails:
            trail.clear()
        self._trails.clear()
        if self._task_folder is not None:
            self._task_folder.remove()
            self._task_folder = None

        self._task_name = name
        self.task = get_task_class(name)()
        self._sim_t = 0.0
        self._budget = 0.0
        self._attach_scene()
        self._build_task_gui()
        self._reframe_camera()
        self._render()

    def _attach_scene(self) -> None:
        """(Re)build the rendered scene for the task's current model."""
        assert self.task is not None
        if self.mj_scene is not None:
            self.mj_scene.remove()
        self.mj_scene = MuJoCoScene(
            self.server, self.task.model, root_path="/task/spacecraft"
        )
        self.mj_scene.set_scale(self._scale)
        self._scene_model = self.task.model

        for trail in self._trails:
            trail.clear()
        self._trails = []
        self._trail_body_ids = []
        for i, body_name in enumerate(self.task.trail_bodies):
            self._trail_body_ids.append(self.task.model.body_id(body_name))
            self._trails.append(
                BodyTrail(
                    self.server,
                    body_name,
                    max_points=4000,
                    color=_TRAIL_COLORS[i % len(_TRAIL_COLORS)],
                    path_prefix="/task/trail",
                )
            )

        # Far plane must reach past Earth from orbit altitude, and past the
        # star shell.
        R_orbit_m = 1000.0 * float(np.linalg.norm(self.task.data.orbit.R_eci))
        self.server.initial_camera.far = 12.0 * R_orbit_m
        if self._stars_enabled and self._star_handle is None:
            self._star_handle = add_star_field(self.server, radius=8.0 * R_orbit_m)

    # ------------------------------------------------------------------
    # Render-frame transforms
    # ------------------------------------------------------------------

    def _world_transform(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        assert self.task is not None
        if self._render_frame == "lvlh":
            return self.task.data.frame.C_LI, None
        return None, 1000.0 * self.task.data.orbit.R_eci

    def _scale_origin(self) -> np.ndarray:
        assert self.task is not None
        if self._render_frame == "eci":
            return 1000.0 * self.task.data.orbit.R_eci
        return np.zeros(3)

    def _body_render_position(self, body_id: int) -> np.ndarray:
        """A body's position in the render frame, in metres, after scaling."""
        assert self.task is not None
        pos = np.asarray(self.task.data.xpos[body_id], dtype=float).copy()
        rotation, translation = self._world_transform()
        if rotation is not None:
            pos = rotation @ pos
        if translation is not None:
            pos = pos + translation
        origin = self._scale_origin()
        return origin + self._scale * (pos - origin)

    def _camera_basis(self) -> np.ndarray | None:
        assert self.task is not None
        if self._render_frame == "lvlh":
            return None  # render axes are already (radial, along, cross)
        return lvlh_basis_eci(self.task.data.orbit.R_eci, self.task.data.orbit.V_eci)

    def _reframe_camera(self) -> None:
        """Auto-frame the tracked body with Earth in the background."""
        assert self.task is not None
        target = self._body_render_position(self.task.track_body_id)
        radius = spacecraft_bounding_radius(
            self.task.model, self.task.data, self.task.track_body_id
        )
        distance = DEFAULT_DISTANCE_RATIO * radius * self._scale
        position, look_at = default_camera_pose(
            target, distance=distance, basis=self._camera_basis()
        )
        self.tracker.set_default_pose(position, look_at)
        self.tracker.retarget(target)
        self.tracker.reframe()

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------

    def _setup_gui(self) -> None:
        self.server.gui.add_markdown("### mujoco_orbit tasks")
        self._task_dropdown = self.server.gui.add_dropdown(
            "Task", options=available_tasks(), initial_value=self._task_name
        )

        @self._task_dropdown.on_update
        def _(_) -> None:
            if self._task_dropdown.value != self._task_name:
                self._pending_task = self._task_dropdown.value

        with self.server.gui.add_folder("Playback"):
            self._pause_btn = self.server.gui.add_button("Pause")
            self._reset_btn = self.server.gui.add_button("Reset")
            self._speed_dropdown = self.server.gui.add_dropdown(
                "Speed",
                options=[f"{s:g}x" for s in _SPEED_OPTIONS],
                initial_value="1x",
            )

        with self.server.gui.add_folder("Camera"):
            self._track_checkbox = self.server.gui.add_checkbox(
                "Track spacecraft", initial_value=self.tracker.enabled
            )
            self._reframe_btn = self.server.gui.add_button("Reframe view")
            self._frame_dropdown = self.server.gui.add_dropdown(
                "Frame", options=["eci", "lvlh"], initial_value=self._render_frame
            )
            self._scale_dropdown = self.server.gui.add_dropdown(
                "Spacecraft scale",
                options=[f"{s:g}x" for s in _SCALE_OPTIONS],
                initial_value=f"{self._scale:g}x",
            )

        self._time_md = self.server.gui.add_markdown("**t** = 0.00 s")

        @self._pause_btn.on_click
        def _(_) -> None:
            self._paused = not self._paused
            self._pause_btn.label = "Resume" if self._paused else "Pause"

        @self._reset_btn.on_click
        def _(_) -> None:
            self._pending_reset = True

        @self._speed_dropdown.on_update
        def _(_) -> None:
            self._speed = float(self._speed_dropdown.value.rstrip("x"))

        @self._track_checkbox.on_update
        def _(_) -> None:
            self.tracker.enabled = self._track_checkbox.value

        @self._reframe_btn.on_click
        def _(_) -> None:
            self._pending_reframe = True

        @self._frame_dropdown.on_update
        def _(_) -> None:
            if self._frame_dropdown.value != self._render_frame:
                self._render_frame = self._frame_dropdown.value
                self._pending_reframe = True
                for trail in self._trails:
                    trail.clear()

        @self._scale_dropdown.on_update
        def _(_) -> None:
            scale = float(self._scale_dropdown.value.rstrip("x"))
            if scale != self._scale:
                self._scale = scale
                if self.mj_scene is not None:
                    self.mj_scene.set_scale(scale)
                for trail in self._trails:
                    trail.clear()
                self._pending_reframe = True

    def _build_task_gui(self) -> None:
        assert self.task is not None
        cls = type(self.task)
        folder = self.server.gui.add_folder(cls.name)
        self._task_folder = folder
        with folder:
            if cls.description:
                self.server.gui.add_markdown(cls.description)
            self._status_md = self.server.gui.add_markdown("")
            if self.task.params is not None:
                self._build_param_sliders(self.task)

    def _build_param_sliders(self, task: ViewerTask) -> None:
        params = task.params
        if params is None:
            return
        for field in dataclasses.fields(params):
            value = getattr(params, field.name)
            label = field.name
            ui = field.metadata.get("ui")
            if isinstance(ui, UiSlider) and ui.label:
                label = ui.label
            if isinstance(value, bool):
                checkbox = self.server.gui.add_checkbox(label, initial_value=value)
                self._bind_param(checkbox, task, field.name)
            elif isinstance(value, (int, float)):
                if isinstance(ui, UiSlider):
                    low, high = ui.low, ui.high
                    step = ui.step if ui.step is not None else (high - low) / 100.0
                else:
                    # judo-style auto bounds around the default
                    low = 0.0 if value >= 0 else 2.0 * value
                    high = 2.0 * value if value > 0 else (5.0 if value == 0 else 0.0)
                    step = (high - low) / 100.0
                slider = self.server.gui.add_slider(
                    label, min=low, max=high, step=step, initial_value=value
                )
                self._bind_param(slider, task, field.name)

    def _bind_param(self, handle, task: ViewerTask, field_name: str) -> None:
        @handle.on_update
        def _(_) -> None:
            setattr(task.params, field_name, handle.value)
            task.on_param_changed(field_name)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        assert self.task is not None and self.mj_scene is not None
        data = self.task.data
        rotation, translation = self._world_transform()
        self.mj_scene.update(
            data,
            rotation=rotation,
            translation=translation,
            scale_origin=self._scale_origin(),
        )

        if self._render_frame == "eci":
            earth_position = np.zeros(3)
            earth_rotation = None
        else:
            earth_position = data.frame.C_LI @ (-1000.0 * data.orbit.R_eci)
            earth_rotation = data.frame.C_LI
        self.earth.update(
            position=earth_position,
            sim_time=float(data.orbit.t),
            rotation=earth_rotation,
        )

        for trail in self._trails:
            trail.render()

        self.tracker.update(self._body_render_position(self.task.track_body_id))

    def _record_trails(self) -> None:
        if not self._trails or self.task is None:
            return
        if self._trail_countdown > 0:
            self._trail_countdown -= 1
            return
        # Sample roughly every 0.2 s of sim time.
        dt = float(self.task.model.opt.timestep)
        self._trail_countdown = max(1, int(round(0.2 / dt))) - 1
        for body_id, trail in zip(self._trail_body_ids, self._trails):
            trail.append(self._body_render_position(body_id))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, duration: Optional[float] = None) -> None:
        """Serve the viewer until *duration* sim-seconds elapse (None = forever)."""
        limit = np.inf if duration is None else float(duration)
        self._last_wall = time.time()
        print(f"mjo-viewer running at http://localhost:{self._port}")
        print(f"tasks: {', '.join(available_tasks())} (current: {self._task_name})")

        try:
            while self._sim_t < limit:
                now = time.time()
                wall_dt = min(now - self._last_wall, 0.1)
                self._last_wall = now

                if self._pending_task is not None:
                    name, self._pending_task = self._pending_task, None
                    self._load_task(name)
                if self._pending_reset:
                    self._pending_reset = False
                    assert self.task is not None
                    self.task.reset()
                    self._sim_t = 0.0
                    self._budget = 0.0
                    self._attach_scene()
                    self._reframe_camera()
                if self._pending_reframe:
                    self._pending_reframe = False
                    self._reframe_camera()

                task = self.task
                assert task is not None
                if not self._paused:
                    self._budget += wall_dt * self._speed
                    dt = float(task.model.opt.timestep)
                    steps = 0
                    while self._budget >= dt and self._sim_t < limit:
                        task.pre_step()
                        mjo_step(task.model, task.data)
                        task.post_step()
                        self._sim_t += dt
                        self._budget -= dt
                        steps += 1
                        if task.model is not self._scene_model:
                            # The task swapped models mid-run (e.g. weld latch).
                            self._attach_scene()
                        self._record_trails()
                        if steps >= 200:
                            self._budget = 0.0
                            break

                self._render()
                self._time_md.content = f"**t** = {self._sim_t:.2f} s"
                if self._status_md is not None:
                    status = task.status()
                    if status and self._status_md.content != status:
                        self._status_md.content = status

                time.sleep(1.0 / 60.0)
        except KeyboardInterrupt:
            print("\nViewer stopped.")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="mjo-viewer", description="Interactive task viewer for mujoco_orbit."
    )
    parser.add_argument("--task", default=None, help="initial task name")
    parser.add_argument("--list-tasks", action="store_true", help="list tasks and exit")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--frame", choices=["eci", "lvlh"], default="eci", help="initial render frame"
    )
    parser.add_argument(
        "--flat-earth", action="store_true", help="disable the photoreal Earth texture"
    )
    parser.add_argument("--no-stars", action="store_true", help="disable the star field")
    parser.add_argument(
        "--no-track", action="store_true", help="start with camera tracking disabled"
    )
    parser.add_argument(
        "--duration", type=float, default=None, help="stop after this many sim seconds"
    )
    args = parser.parse_args(argv)

    if args.list_tasks:
        for name in available_tasks():
            print(name)
        return

    app = MjOrbitApp(
        task=args.task,
        host=args.host,
        port=args.port,
        render_frame=args.frame,
        textured_earth=not args.flat_earth,
        stars=not args.no_stars,
        track=not args.no_track,
    )
    app.run(duration=args.duration)


if __name__ == "__main__":
    main()
