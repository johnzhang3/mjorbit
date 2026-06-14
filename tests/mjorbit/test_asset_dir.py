"""Regression tests for ``MjoModel._asset_dir`` propagation.

The browser viewer (``src/viewer/bodies.py``) reads ``model._asset_dir`` to
recompile the model from its source directory so relatively-referenced
mesh/attach/include files resolve; if it is ``None`` the viewer silently falls
back to a flat asset dict that cannot cover sub-model files. The asset dir is
threaded through ``MjoSpec`` (set in ``from_xml_path``, preserved by ``copy``,
applied in ``compile``), so every on-disk construction path must carry it while
string-built specs must not.
"""

from __future__ import annotations

from pathlib import Path

from mjorbit import MjoModel
from mjorbit.spec import MjoSpec
from mjorbit.testdata import FREE_BODY_XML

_MINIMAL_XML = (
    '<mujoco><worldbody><body name="b"><freejoint/>'
    '<geom type="sphere" size="0.1" mass="1"/></body></worldbody></mujoco>'
)


def _expected_dir() -> str:
    return str(Path(str(FREE_BODY_XML)).resolve().parent)


def test_from_xml_path_records_asset_dir() -> None:
    model = MjoModel.from_xml_path(str(FREE_BODY_XML))
    assert model._asset_dir == _expected_dir()


def test_spec_compile_records_asset_dir() -> None:
    model = MjoSpec.from_xml_path(str(FREE_BODY_XML)).compile()
    assert model._asset_dir == _expected_dir()


def test_copied_spec_preserves_asset_dir() -> None:
    model = MjoSpec.from_xml_path(str(FREE_BODY_XML)).copy().compile()
    assert model._asset_dir == _expected_dir()


def test_from_xml_string_has_no_asset_dir() -> None:
    model = MjoSpec.from_xml_string(_MINIMAL_XML).compile()
    assert model._asset_dir is None
