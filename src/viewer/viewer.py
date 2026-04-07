# pyright: reportMissingImports=false

"""MjOrbitViewer — interactive 3D viewer for mujoco_orbit simulations.

Renders MuJoCo multibody geometry in the LVLH frame with an Earth
reference sphere below.  Uses `viser <https://viser.studio>`_ for
browser-based visualisation.

Coordinate system
-----------------
Everything is rendered in the MuJoCo world frame which, in mujoco_orbit, is
the LVLH (RSW) frame centred on the chief spacecraft:

* **+x** radial   (away from Earth)
* **+y** along-track
* **+z** cross-track (orbit normal)

Units are **metres** (MuJoCo SI).
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import viser

from mujoco_orbit.core.runtime import MjoData, MjoModel
from mujoco_orbit.core.step import mjo_step

from .bodies import MuJoCoScene
from .earth import BodyTrail, add_earth


class MjOrbitViewer:
    """Real-time coupled orbital + multibody viewer.

    Parameters
    ----------
    model, data : MjoModel, MjoData
        Compiled model and runtime state.
    host, port : str, int
        Viser server bind address.
    show_earth : bool
        Render an Earth icosphere at the correct LVLH offset.
    show_axes : bool
        Draw LVLH reference axes at the origin.
    track_body : str or None
        If given, record and draw this body's trajectory trail.
    trail_max_points : int
        Maximum trail history length.
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
        trail_max_points: int = 2000,
    ) -> None:
        self.model = model
        self.data = data

        self._port = port
        self._track_body_id: Optional[int] = None

        # ---- viser server ---------------------------------------------------
        self.server = viser.ViserServer(host=host, port=port)
        self.server.scene.set_up_direction("+z")

        # ---- MuJoCo body geometry -------------------------------------------
        self.mj_scene = MuJoCoScene(self.server, self.model.mj_model)

        # ---- Earth -----------------------------------------------------------
        if show_earth:
            R_orbit_m = float(np.linalg.norm(self.data.orbit.R_eci)) * 1000.0
            # In RSW, +x is radial outward -> Earth centre is at -x
            add_earth(self.server, position=(-R_orbit_m, 0.0, 0.0))

        # ---- LVLH reference axes --------------------------------------------
        if show_axes:
            self._add_lvlh_axes()

        # ---- Body trail ------------------------------------------------------
        self.trail: Optional[BodyTrail] = None
        if track_body is not None:
            self._track_body_id = self.model.body_id(track_body)
            self.trail = BodyTrail(
                self.server, track_body, max_points=trail_max_points,
            )

        # ---- GUI controls ----------------------------------------------------
        self._speed = 1.0
        self._paused = False
        self._setup_gui()

        # Initial render
        self.mj_scene.update(self.data.mj_data)

    # ------------------------------------------------------------------
    # Scene helpers
    # ------------------------------------------------------------------

    def _add_lvlh_axes(self) -> None:
        """Draw R (red), S (green), W (blue) axes at the origin."""
        length = 1.0  # m
        origins = np.zeros((3, 3))
        ends = np.diag([length, length, length])
        segments = np.stack([origins, ends], axis=1)  # (3, 2, 3)
        colors = np.array(
            [
                [[255, 50, 50], [255, 50, 50]],
                [[50, 255, 50], [50, 255, 50]],
                [[50, 50, 255], [50, 50, 255]],
            ],
            dtype=np.uint8,
        )  # (3, 2, 3)
        self.server.scene.add_line_segments(
            "/lvlh_axes", segments, colors=colors, line_width=3.0,
        )

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------

    def _setup_gui(self) -> None:
        with self.server.gui.add_folder("Playback"):
            self._speed_slider = self.server.gui.add_slider(
                "Speed", min=0.1, max=100.0, step=0.1, initial_value=1.0,
            )
            self._pause_btn = self.server.gui.add_button("Pause")

        self._time_md = self.server.gui.add_markdown("**t** = 0.00 s")

        @self._pause_btn.on_click
        def _(_) -> None:  # type: ignore[arg-type]
            self._paused = not self._paused
            self._pause_btn.label = "Resume" if self._paused else "Pause"

        @self._speed_slider.on_update
        def _(_) -> None:  # type: ignore[arg-type]
            self._speed = self._speed_slider.value

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
        sim_t = 0.0
        last_wall = time.time()
        budget = 0.0

        # Trail sampling interval — every 10 physics steps or 20 ms, whichever is larger
        trail_interval = max(dt * 10, 0.02)
        last_trail_t = 0.0

        print(f"Viewer running at http://localhost:{self._port}")

        try:
            while sim_t < duration:
                now = time.time()
                wall_dt = min(now - last_wall, 0.1)  # cap to avoid spiral
                last_wall = now

                if not self._paused:
                    budget += wall_dt * self._speed

                    steps = 0
                    while budget >= dt and sim_t < duration:
                        ctrl = action_fn(self.data, sim_t) if action_fn else None
                        if ctrl is not None:
                            np.copyto(self.data.ctrl, ctrl)
                        mjo_step(self.model, self.data)
                        sim_t += dt
                        budget -= dt
                        steps += 1

                        # Sample trail
                        if (
                            self.trail is not None
                            and self._track_body_id is not None
                            and sim_t - last_trail_t >= trail_interval
                        ):
                            pos = self.data.xipos[self._track_body_id]
                            self.trail.append(pos)
                            last_trail_t = sim_t

                        # Cap steps per render frame to stay responsive
                        if steps >= 200:
                            budget = 0.0
                            break

                # ---- update visuals ------------------------------------------
                self.mj_scene.update(self.data.mj_data)
                if self.trail is not None:
                    self.trail.render()
                self._time_md.content = f"**t** = {sim_t:.2f} s"

                time.sleep(1.0 / 60.0)

        except KeyboardInterrupt:
            print("\nViewer stopped.")
