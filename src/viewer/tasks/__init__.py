"""Task registry and built-in tasks for the interactive viewer app."""

# Importing the modules registers the built-in tasks.
from . import arm_reach as _arm_reach  # noqa: E402,F401
from . import capture_stabilize as _capture_stabilize  # noqa: E402,F401
from . import free_drift as _free_drift  # noqa: E402,F401
from .base import UiSlider, ViewerTask, ui_field
from .registry import available_tasks, get_task_class, register_task

__all__ = [
    "UiSlider",
    "ViewerTask",
    "available_tasks",
    "get_task_class",
    "register_task",
    "ui_field",
]
