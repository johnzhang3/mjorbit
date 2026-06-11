"""Name -> task-class registry for the viewer app (judo-style)."""

from __future__ import annotations

from .base import ViewerTask

_REGISTRY: dict[str, type[ViewerTask]] = {}


def register_task(cls: type[ViewerTask]) -> type[ViewerTask]:
    """Register a :class:`ViewerTask` subclass under ``cls.name``.

    Usable as a class decorator. Re-registering a name overwrites it, so
    user scripts can shadow built-in tasks.
    """
    if not cls.name:
        raise ValueError(f"{cls.__name__} must define a non-empty `name`")
    _REGISTRY[cls.name] = cls
    return cls


def get_task_class(name: str) -> type[ViewerTask]:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"unknown task {name!r}; registered tasks: {known}") from None


def available_tasks() -> list[str]:
    return sorted(_REGISTRY)
