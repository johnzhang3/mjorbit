# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Headless, deterministic video recorder for the mjorbit viewer.

The viewer renders with viser (three.js in the browser), so frames are produced
by viser's ``client.get_render`` — which asks a *connected* WebGL client to
render offscreen and ships the image back. This module connects a **headless
Chrome** client to the viser server, sets the camera pose explicitly per frame
(no dependence on the live client camera, so it is race-free and deterministic),
pulls a frame, and streams frames straight into ``ffmpeg`` to encode an mp4.

Stepping the simulation is the caller's job and happens *offline* (not in
wall-clock realtime), so the captured video is smooth and reproducible even when
a per-step controller (e.g. MPPI) is expensive.

Typical use::

    server = viser.ViserServer(host="127.0.0.1", port=8231)
    ... build the scene ...
    with HeadlessRecorder(server, port=8231) as rec, \
         VideoWriter("out.mp4", fps=30, width=rec.width, height=rec.height) as vid:
        for k in range(num_frames):
            ... advance sim, update scene nodes ...
            frame = rec.render(position=cam_pos, look_at=cam_target, fov_deg=45.0)
            vid.add(frame)
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import viser
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/SFNSDisplay.ttf",
    "/Library/Fonts/Arial.ttf",
]


def _load_font(size: int):
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def annotate(frame: np.ndarray, text: str, *, corner: str = "bl") -> np.ndarray:
    """Burn a small caption into a corner of an (H, W, 3) uint8 frame."""
    im = Image.fromarray(np.asarray(frame, dtype=np.uint8)[..., :3])
    draw = ImageDraw.Draw(im)
    fs = max(16, im.height // 26)
    font = _load_font(fs)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    pad = fs
    x = pad if "l" in corner else im.width - tw - pad
    y = (im.height - pad - (bbox[3] - bbox[1]) - bbox[1]) if "b" in corner else (pad - bbox[1])
    draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return np.asarray(im)


def speedup_label(sim_seconds: float, video_seconds: float) -> str:
    x = float(sim_seconds) / max(float(video_seconds), 1e-9)
    return f"{x:.0f}× real-time" if x >= 2.0 else f"{x:.1f}× real-time"


_CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
]


def _find_chrome() -> str:
    for path in _CHROME_CANDIDATES:
        if Path(path).is_file():
            return path
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError(
        "No Chrome/Chromium binary found for headless rendering. Install Google "
        "Chrome or set one of the paths in recorder._CHROME_CANDIDATES."
    )


class HeadlessRecorder:
    """Connects a headless Chrome client to a viser server and renders frames.

    Parameters
    ----------
    server:
        A running :class:`viser.ViserServer`. The scene must already be built (or
        be built/updated before each :meth:`render` call).
    port:
        The port ``server`` is listening on (used to point the browser at it).
    width, height:
        Render resolution. Must be even for yuv420p mp4 output.
    up:
        World up direction for the camera (matches ``set_up_direction("+z")``).
    connect_timeout:
        Seconds to wait for the headless client to connect and report a camera.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        *,
        port: int,
        width: int = 1280,
        height: int = 720,
        up: Sequence[float] = (0.0, 0.0, 1.0),
        connect_timeout: float = 45.0,
        warmup: float = 2.0,
        gl: str = "gpu",
        headless: bool = False,
    ) -> None:
        if width % 2 or height % 2:
            raise ValueError("width and height must be even for yuv420p mp4")
        self.server = server
        self.width = int(width)
        self.height = int(height)
        self.up = tuple(float(v) for v in up)
        chrome = _find_chrome()
        # A generous window so the requested render size is always <= the browser
        # canvas. On macOS, new headless mode can drive the real GPU through
        # ANGLE/Metal — which the live viewer uses — so large meshes (the
        # 6371 km Earth sphere) render correctly. The software rasterizer
        # (SwiftShader) drops that mesh, so the GPU path is the default; pass
        # gl="swiftshader" only as a no-GPU fallback.
        # Headless macOS Chrome falls back to the SwiftShader software rasterizer,
        # which drops the 6371 km Earth sphere and drops its texture. A *visible*
        # window drives the real Metal GPU (exactly what the live viewer uses), so
        # the Earth renders correctly. Default to a visible window; headless is an
        # opt-in for environments with no display.
        flags = [chrome, "--no-first-run", "--no-default-browser-check",
                 "--disable-extensions", "--mute-audio", "--hide-scrollbars",
                 f"--window-size={max(self.width + 32, 1024)},{max(self.height + 160, 768)}",
                 "--enable-webgl", "--ignore-gpu-blocklist"]
        if headless:
            gl_flags = (
                ["--use-gl=angle", "--use-angle=swiftshader"]
                if gl == "swiftshader"
                else ["--use-gl=angle", "--use-angle=metal", "--enable-gpu"]
            )
            flags = flags[:1] + ["--headless=new", *gl_flags] + flags[1:]
        # Use a throwaway profile so the window opens clean and does not collide
        # with the user's running Chrome.
        self._profile = tempfile.mkdtemp(prefix="mjo-rec-chrome-")
        flags.append(f"--user-data-dir={self._profile}")
        flags.append(f"http://127.0.0.1:{int(port)}")
        self._proc = subprocess.Popen(flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.client = self._await_client(connect_timeout)
        # Let the scene assets (Earth texture, meshes) upload and the first render
        # settle before we start pulling frames.
        time.sleep(float(warmup))

    def _await_client(self, timeout: float) -> viser.ClientHandle:
        deadline = time.time() + float(timeout)
        client: viser.ClientHandle | None = None
        while time.time() < deadline:
            clients = self.server.get_clients()
            if clients:
                client = next(iter(clients.values()))
                break
            time.sleep(0.2)
        if client is None:
            self.close()
            raise RuntimeError("headless Chrome did not connect to the viser server")
        # Wait for the client to report an initial camera (get_render asserts a
        # nonzero update timestamp).
        while time.time() < deadline:
            try:
                _ = client.camera.position
                _ = client.camera.wxyz
                return client
            except Exception:
                time.sleep(0.1)
        return client

    def render(
        self,
        *,
        position: np.ndarray,
        look_at: np.ndarray,
        fov_deg: float = 45.0,
    ) -> np.ndarray:
        """Render one frame from an explicit camera pose; returns (H, W, 3) uint8.

        The camera ``wxyz`` is computed server-side from ``position`` + ``up`` +
        ``look_at`` (viser's ``look_at`` setter is synchronous), then passed
        explicitly to ``get_render`` so the result never depends on the live
        browser camera state.
        """
        cam = self.client.camera
        cam.up_direction = self.up
        cam.position = tuple(np.asarray(position, dtype=float))
        cam.look_at = tuple(np.asarray(look_at, dtype=float))  # updates cam.wxyz
        img = self.client.get_render(
            self.height,
            self.width,
            position=tuple(np.asarray(cam.position, dtype=float)),
            wxyz=tuple(np.asarray(cam.wxyz, dtype=float)),
            fov=float(np.deg2rad(fov_deg)),
            transport_format="jpeg",
        )
        arr = np.asarray(img)
        return arr[..., :3].astype(np.uint8)

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            self._proc = None
        profile = getattr(self, "_profile", None)
        if profile:
            shutil.rmtree(profile, ignore_errors=True)
            self._profile = None

    def __enter__(self) -> "HeadlessRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class VideoWriter:
    """Streams uint8 RGB frames into an ffmpeg libx264 mp4 (no frames held in RAM)."""

    def __init__(
        self, path: str | Path, *, fps: float, width: int, height: int, crf: int = 17
    ) -> None:
        self.path = str(path)
        self.width = int(width)
        self.height = int(height)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{float(fps):g}",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            self.path,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.n_frames = 0

    def add(self, frame: np.ndarray) -> None:
        arr = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8)[..., :3])
        if arr.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"frame shape {arr.shape[:2]} != ({self.height}, {self.width})"
            )
        assert self._proc.stdin is not None
        self._proc.stdin.write(arr.tobytes())
        self.n_frames += 1

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        self._proc = None
        try:
            if proc.stdin is not None:
                proc.stdin.close()  # EOF -> ffmpeg finalizes the mp4
        except Exception:
            pass
        err = b""
        try:
            if proc.stderr is not None:
                err = proc.stderr.read()
        except Exception:
            pass
        proc.wait()
        if proc.returncode not in (0, None):
            msg = err.decode(errors="ignore")[-800:]
            raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {msg}")

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
