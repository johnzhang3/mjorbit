"""Pluggable physics backend for the PPO examples.

The fidelity study (``experiments/sim_fidelity``) compares training a policy under
bare **MuJoCo Warp** rigid-body dynamics against training it under the full
**mjorbit_warp** orbital coupling, then evaluating BOTH policies in the full
mjorbit_warp environment. For that comparison to be meaningful the two backends
must expose an *identical* observation / action / reward interface -- only the
per-step dynamics may differ. This module is the single seam where they diverge,
so the env code (``truss_env.py``, ``astrobee_env.py``) stays backend-agnostic.

Backends
--------
``"mjorbit"`` (default)
    Full orbital coupling. ``mjo_step`` applies gravity-gradient torque,
    differential (tidal) gravity, drag, J2, the non-inertial chief-frame
    compensation, and propagates the chief orbit. This is also the EVALUATION
    environment for the study.

``"mjwarp"``
    Bare ``mujoco_warp.step`` rigid-body dynamics on the *same* compiled model and
    device buffers: no gravity gradient, no drag/J2, no non-inertial frame, no
    orbit propagation. ``mujoco_warp`` never touches ``xfrc_applied``, so as long
    as we never call ``mjo_step``/``mjo_forward`` on this data the body is in true
    free-float. The chief orbit does not advance; a task that needs a moving
    reference (e.g. nadir) propagates it kinematically itself.

Reusing the same ``MjoModel``/``MjoData`` for both backends (rather than building a
separate raw ``mujoco_warp`` model) guarantees byte-identical compiled dynamics,
DOF layout, integrator, and ``opt.gravity == 0`` -- so a policy trained on
``mjw.step`` drops straight into the ``mjo_step`` evaluator with no obs/action
remapping. The only difference between training and evaluation is which step
function runs.
"""

from __future__ import annotations

import mujoco_warp as mjw

from mjorbit_warp import mjo_forward, mjo_pull, mjo_step, mjo_upload

MJORBIT = "mjorbit"
MJWARP = "mjwarp"
BACKENDS = (MJORBIT, MJWARP)


class SimBackend:
    """Selects the per-step physics for an env without changing its interface."""

    def __init__(self, kind: str = MJORBIT) -> None:
        if kind not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {kind!r}")
        self.kind = kind

    @property
    def is_orbital(self) -> bool:
        return self.kind == MJORBIT

    def advance(self, model, data, decimation: int) -> None:
        """Upload the current ``ctrl`` and advance ``decimation`` physics substeps."""
        mjo_upload(model, data, fields=("ctrl",))
        if self.kind == MJORBIT:
            for _ in range(decimation):
                mjo_step(model, data)
        else:
            wm, wd = model.warp_model, data.warp_data
            for _ in range(decimation):
                mjw.step(wm, wd)

    def pull(self, model, data, fields) -> None:
        """Copy device state back to the public host buffers.

        On the bare backend the chief orbit is not integrated, so ``"orbit"`` is
        dropped from the pull -- the env keeps (and, if it wants a moving target,
        kinematically propagates) ``data.orbit`` on the host instead.
        """
        if self.kind == MJWARP:
            fields = tuple(f for f in fields if f != "orbit")
        mjo_pull(model, data, fields=fields)

    def reset_forward(self, model, data, upload_fields) -> None:
        """Push reset state to the device and run a forward pass."""
        if self.kind == MJORBIT:
            mjo_upload(model, data, fields=tuple(upload_fields))
            mjo_forward(model, data)
        else:
            # Bare rigid body: zero external wrenches on the device so no stale
            # orbital coupling persists (mjw.* never writes xfrc_applied), then a
            # pure forward. xfrc_applied then stays zero for every mjw.step.
            data.xfrc_applied[...] = 0.0
            mjo_upload(model, data, fields=tuple(upload_fields) + ("xfrc_applied",))
            mjw.forward(model.warp_model, data.warp_data)
