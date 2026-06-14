# pyright: reportMissingImports=false

"""Optional MJWarp dependency helpers."""

from __future__ import annotations

from typing import Any


def require_mjwarp() -> tuple[Any, Any]:
    """Import MJWarp and Warp lazily with a helpful installation error."""
    try:
        import mujoco_warp as mjw
        import warp as wp
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "mjorbit_warp requires the optional `mujoco-warp` extra. "
            "Install it with `pixi install -e warp`."
        ) from exc

    return mjw, wp


__all__ = ["require_mjwarp"]
