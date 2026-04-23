"""MuJoCo-style coupled orbital and multibody dynamics."""

from __future__ import annotations

import os as _os

import mujoco as _mujoco


def _load_native_plugin() -> None:
    """Locate and register the shipped mujoco_orbit.orbit MuJoCo plugin.

    scikit-build-core installs the compiled plugin at
    ``<site-packages>/mujoco_orbit/plugins/mujoco_orbit_plugin.<ext>``. Under an
    editable install ``__file__`` points at the source tree instead, so the
    plugin dir has to be resolved via site-packages lookup. We try both. If
    neither path exists, fall through silently — lets ``pip install -e .``
    succeed before the first CMake build has run.
    """
    import site as _site
    import sysconfig as _sysconfig

    candidates = [_os.path.join(_os.path.dirname(__file__), "plugins")]
    for base in _site.getsitepackages() + [_sysconfig.get_paths()["purelib"]]:
        candidates.append(_os.path.join(base, "mujoco_orbit", "plugins"))

    seen: set[str] = set()
    for plugins_dir in candidates:
        if plugins_dir in seen or not _os.path.isdir(plugins_dir):
            continue
        seen.add(plugins_dir)
        for fname in _os.listdir(plugins_dir):
            if fname.startswith("mujoco_orbit_plugin") and (
                fname.endswith(".so") or fname.endswith(".dylib") or fname.endswith(".dll")
            ):
                _mujoco.mj_loadPluginLibrary(_os.path.join(plugins_dir, fname))
                return


_load_native_plugin()


from mujoco_orbit.core.config import (
    ControlMomentGyroSpec,
    MagneticBodySpec,
    MagnetorquerSpec,
    OrbitInit,
    ReactionWheelSpec,
    SurfaceSpec,
    ThrusterSpec,
)
from mujoco_orbit.core.runtime import MjoData, MjoModel
from mujoco_orbit.core.step import mjo_forward, mjo_step

__all__ = [
    "ControlMomentGyroSpec",
    "MagneticBodySpec",
    "MagnetorquerSpec",
    "MjoData",
    "MjoModel",
    "OrbitInit",
    "ReactionWheelSpec",
    "SurfaceSpec",
    "ThrusterSpec",
    "mjo_forward",
    "mjo_step",
]
