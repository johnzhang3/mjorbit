# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""MjOrbitMultiViewer — render N independent spacecraft simulations together.

Each entry is its own ``(MjoModel, MjoData)`` pair, so every spacecraft has
its own chief orbit, qpos, qvel, controls, and integrator state. All entries
are visualized in absolute ECI so they appear at distinct positions around
Earth, which is what you want for figures with several spacecraft in one
shot. Units are metres (MuJoCo SI).
"""

from __future__ import annotations

import socket
import time
from typing import Callable, Optional, Sequence

import numpy as np
import viser

from mjorbit.runtime import MjoData, MjoModel
from mjorbit.step import mjo_step

from .bodies import MuJoCoScene
from .earth import add_earth

ActionFn = Callable[[int, MjoModel, MjoData, float], Optional[np.ndarray]]


def _assert_socket_bindable(host: str, port: int) -> None:
    bind_host = host if host not in ("", "0.0.0.0") else "127.0.0.1"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((bind_host, port))
    except OSError as exc:
        raise RuntimeError(
            f"MjOrbitMultiViewer cannot start a local viser server on {host}:{port}. "
            "Local socket bind is unavailable in this environment."
        ) from exc


class MjOrbitMultiViewer:
    """Render and step N independent spacecraft simulations together.

    Parameters
    ----------
    entries : sequence of (MjoModel, MjoData)
        Independent simulations. Each entry carries its own orbit; all entries
        are rendered in absolute ECI so they sit at distinct positions around
        Earth.
    spacecraft_scale : float
        Multiplier applied to each spacecraft's geometry around its own ECI
        location. Earth is ~6400 km across; a 1 m spacecraft is invisible at
        true scale, so the default of 10000x enlarges each spacecraft's
        geometry locally without warping its orbit position.
    """

    def __init__(
        self,
        entries: Sequence[tuple[MjoModel, MjoData]],
        host: str = "0.0.0.0",
        port: int = 8080,
        show_earth: bool = True,
        spacecraft_scale: float = 10000.0,
        camera_distance: Optional[float] = None,
    ) -> None:
        if not entries:
            raise ValueError("MjOrbitMultiViewer requires at least one entry.")
        self.entries: list[tuple[MjoModel, MjoData]] = list(entries)
        self._port = port
        self._spacecraft_scale = float(spacecraft_scale)

        _assert_socket_bindable(host, port)
        self.server = viser.ViserServer(host=host, port=port)
        self.server.scene.set_up_direction("+z")

        # Frame the camera on the cluster of spacecraft + Earth.
        positions_m = np.stack(
            [1000.0 * np.asarray(d.orbit.R_eci, dtype=float) for _, d in self.entries]
        )
        cluster_center = positions_m.mean(axis=0)
        R_orbit_m = float(np.linalg.norm(positions_m, axis=1).max())
        cam_dist = (
            camera_distance if camera_distance is not None else 2.5 * R_orbit_m
        )
        self.server.initial_camera.position = tuple(
            cluster_center + np.array([0.0, -cam_dist, cam_dist * 0.4])
        )
        self.server.initial_camera.look_at = tuple(cluster_center)
        self.server.initial_camera.far = max(R_orbit_m, 1.0) * 4.0

        if show_earth:
            add_earth(self.server, position=(0.0, 0.0, 0.0))

        # One MuJoCoScene per spacecraft, scaled around its own ECI center.
        self.scenes: list[MuJoCoScene] = []
        for i, (model, _) in enumerate(self.entries):
            scene = MuJoCoScene(
                self.server,
                model,
                root_path=f"/scene/spacecraft_{i}",
            )
            scene.set_scale(self._spacecraft_scale)
            self.scenes.append(scene)

        self._paused = False
        self._sim_t = 0.0
        self._budget = 0.0
        self._last_wall = time.time()

        self._setup_gui()
        self._render()

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------

    def _setup_gui(self) -> None:
        with self.server.gui.add_folder("Playback"):
            self._pause_btn = self.server.gui.add_button("Pause")
        self._time_md = self.server.gui.add_markdown("**t** = 0.00 s")
        self._count_md = self.server.gui.add_markdown(
            f"**spacecraft** = {len(self.entries)}"
        )

        @self._pause_btn.on_click
        def _(_) -> None:  # type: ignore[arg-type]
            self._paused = not self._paused
            self._pause_btn.label = "Resume" if self._paused else "Pause"

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _render(self) -> None:
        for scene, (_, data) in zip(self.scenes, self.entries):
            translation = 1000.0 * np.asarray(data.orbit.R_eci, dtype=float)
            scene.update(
                data,
                rotation=None,
                translation=translation,
                scale_origin=translation,
            )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        duration: Optional[float] = None,
        action_fn: Optional[ActionFn] = None,
    ) -> None:
        """Step all spacecraft and render until *duration* sim-seconds elapse.

        If *duration* is None (default), runs indefinitely until interrupted.
        ``action_fn(i, model, data, t)`` returns optional MuJoCo ctrl for
        spacecraft ``i`` (or None to leave ``data.ctrl`` unchanged).
        """
        # All entries are assumed to share a timestep; pace from the first.
        dt = self.entries[0][0].opt.timestep
        forever = duration is None
        self._last_wall = time.time()

        target = "forever" if forever else f"{duration:g}s"
        print(
            f"Multi-viewer ({len(self.entries)} spacecraft) "
            f"running at http://localhost:{self._port} — target {target}"
        )

        try:
            while forever or self._sim_t < duration:
                now = time.time()
                wall_dt = min(now - self._last_wall, 0.1)
                self._last_wall = now

                if not self._paused:
                    self._budget += wall_dt
                    steps = 0
                    while self._budget >= dt and (
                        forever or self._sim_t < duration
                    ):
                        for i, (model, data) in enumerate(self.entries):
                            ctrl = (
                                action_fn(i, model, data, self._sim_t)
                                if action_fn is not None
                                else None
                            )
                            if ctrl is not None:
                                np.copyto(data.ctrl, ctrl)
                            mjo_step(model, data)
                        self._sim_t += dt
                        self._budget -= dt
                        steps += 1
                        if steps >= 200:
                            self._budget = 0.0
                            break

                self._render()
                self._time_md.content = f"**t** = {self._sim_t:.2f} s"
                time.sleep(1.0 / 60.0)
        except KeyboardInterrupt:
            print("\nMulti-viewer stopped.")
