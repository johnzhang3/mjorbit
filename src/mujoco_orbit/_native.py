"""Native plugin loading helpers."""

from __future__ import annotations

import ctypes
import os
import pathlib
import site
import sysconfig

import mujoco

_PLUGIN_LIBRARY: ctypes.CDLL | None = None
_PLUGIN_PATH: str | None = None
_MUJOCO_LIBRARY: ctypes.CDLL | None = None


def _candidate_plugin_dirs() -> list[str]:
    source_pkg_dir = pathlib.Path(__file__).resolve().parent
    repo_root = source_pkg_dir.parent.parent
    candidates = [
        str(repo_root / "build" / "cpp"),
        os.path.join(os.path.dirname(__file__), "plugins"),
    ]
    for base in site.getsitepackages() + [sysconfig.get_paths()["purelib"]]:
        candidates.append(os.path.join(base, "mujoco_orbit", "plugins"))
    return candidates


def find_native_plugin_path() -> str | None:
    """Return the first shipped native plugin path, if it has been built."""
    seen: set[str] = set()
    for plugins_dir in _candidate_plugin_dirs():
        if plugins_dir in seen or not os.path.isdir(plugins_dir):
            continue
        seen.add(plugins_dir)
        for fname in os.listdir(plugins_dir):
            if fname.startswith("mujoco_orbit_plugin") and (
                fname.endswith(".so") or fname.endswith(".dylib") or fname.endswith(".dll")
            ):
                return os.path.join(plugins_dir, fname)
    return None


def load_native_plugin() -> str | None:
    """Load/register the MuJoCo plugin and return its library path if present."""
    global _PLUGIN_LIBRARY, _PLUGIN_PATH
    if _PLUGIN_PATH is not None:
        return _PLUGIN_PATH

    _preload_mujoco_library()
    path = find_native_plugin_path()
    if path is None:
        return None

    load_plugin_library = getattr(mujoco, "mj_loadPluginLibrary", None)
    if callable(load_plugin_library):
        load_plugin_library(path)
    else:
        _PLUGIN_LIBRARY = ctypes.CDLL(path)

    _PLUGIN_PATH = path
    return path


def _preload_mujoco_library() -> None:
    """Load MuJoCo's shared library globally before extension imports."""
    global _MUJOCO_LIBRARY
    if _MUJOCO_LIBRARY is not None:
        return

    mujoco_dir = pathlib.Path(mujoco.__file__).resolve().parent
    candidates = sorted(
        list(mujoco_dir.glob("libmujoco*.dylib"))
        + list(mujoco_dir.glob("libmujoco*.so*"))
        + list(mujoco_dir.glob("mujoco.dll"))
    )
    if not candidates:
        return
    _MUJOCO_LIBRARY = ctypes.CDLL(
        str(candidates[0]),
        mode=getattr(ctypes, "RTLD_GLOBAL", 0),
    )


__all__ = ["find_native_plugin_path", "load_native_plugin"]
