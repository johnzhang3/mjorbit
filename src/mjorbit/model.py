# pyright: reportAttributeAccessIssue=false

"""Python-facing model wrapper for the C++-first runtime."""

from __future__ import annotations

from typing import Any

from mjorbit import _bindings


class MjoModel:
    """Compiled mjorbit model.

    The underlying ``mjModel`` is owned by C++ and intentionally not exposed as
    a Python MuJoCo object. Public fields mirror the parts of MuJoCo's Python
    API that mjorbit supports directly.
    """

    __slots__ = ("_asset_dir", "_mjorbit_warp_model", "_native", "_raw_xml")

    def __init__(self, native: _bindings.MjoModel, *, raw_xml: str | None = None) -> None:
        self._native = native
        self._raw_xml = raw_xml
        self._asset_dir = None
        self._mjorbit_warp_model = None

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
        from mjorbit.spec import MjoSpec

        # MjoSpec.from_xml_path records the source dir; compile() carries it onto
        # the model as _asset_dir so viewers can resolve relatively-referenced
        # mesh/attach/include files.
        return MjoSpec.from_xml_path(xml_path).compile(mj_timestep=mj_timestep)

    def __getattr__(self, name: str) -> Any:
        if name in {"mj_model", "mj_data"}:
            raise AttributeError(name)
        return getattr(self._native, name)

    @property
    def backend(self) -> str:
        return "cpu"

    @property
    def central_body(self):
        from mjorbit.spec import CentralBodySpec

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

    def make_data(self, *, orbit=None, rng_seed: int | None = None):
        from mjorbit.data import MjoData

        return MjoData(self, orbit=orbit, rng_seed=rng_seed)


__all__ = ["MjoModel"]
