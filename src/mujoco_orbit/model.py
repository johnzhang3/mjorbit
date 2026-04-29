# pyright: reportAttributeAccessIssue=false

"""Python-facing model wrapper for the C++-first runtime."""

from __future__ import annotations

from typing import Any

from mujoco_orbit import _bindings


class MjoModel:
    """Compiled MuJoCo Orbit model.

    The underlying ``mjModel`` is owned by C++ and intentionally not exposed as
    a Python MuJoCo object. Public fields mirror the parts of MuJoCo's Python
    API that mujoco_orbit supports directly.
    """

    __slots__ = ("_native",)

    def __init__(self, native: _bindings.MjoModel) -> None:
        self._native = native

    @classmethod
    def from_xml_path(
        cls,
        xml_path: str,
        *,
        mj_timestep: float | None = None,
        **kwargs: Any,
    ) -> "MjoModel":
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(
                "MjoModel.from_xml_path is XML-first; move orbit config into "
                f"<mjorbit> instead of passing keyword(s): {names}"
            )
        from mujoco_orbit.spec import MjoSpec

        return MjoSpec.from_xml_path(xml_path).compile(mj_timestep=mj_timestep)

    def __getattr__(self, name: str) -> Any:
        if name in {"mj_model", "mj_data"}:
            raise AttributeError(name)
        return getattr(self._native, name)

    @property
    def central_body(self):
        from mujoco_orbit.spec import CentralBodySpec

        return CentralBodySpec._from_native(self._native.central_body)

    def body_id(self, name: str) -> int:
        try:
            return int(self._native.body_id(name))
        except RuntimeError as exc:
            if "not found" in str(exc):
                raise ValueError(str(exc)) from exc
            raise

    def sensor(self, name: str):
        return self._native.sensor(name)


__all__ = ["MjoModel"]
