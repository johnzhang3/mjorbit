"""Step-time orchestration: forward, step, upload, pull.

Public entry points:
- ``mjo_step`` — advance one fully coupled MJWarp simulation step in-place
- ``mjo_forward`` — synchronize derived runtime state after direct mutation
- ``mjo_upload`` — push public NumPy state into MJWarp/device buffers
- ``mjo_pull`` — read MJWarp/device state back into public NumPy buffers

The implementation is split across ``upload``, ``pull``, ``forward``,
``stepping``, and ``field_specs`` (selective-field normalization).
"""

from __future__ import annotations

from .forward import mjo_forward

# These private helpers are re-exported because some tests monkey-patch them
# at ``mjorbit_warp.step._sync_device_from_public`` etc.
from .pull import (
    _copy_device_field_to_public,
    _pull_device_fields_to_public,
    _pull_device_into_host,
    mjo_pull,
)
from .stepping import mjo_step
from .upload import _sync_device_from_public, mjo_upload

__all__ = [
    "_copy_device_field_to_public",
    "_pull_device_fields_to_public",
    "_pull_device_into_host",
    "_sync_device_from_public",
    "mjo_forward",
    "mjo_pull",
    "mjo_step",
    "mjo_upload",
]
