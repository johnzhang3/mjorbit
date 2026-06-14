"""Import and optional-dependency behavior for ``mjorbit_warp``."""

from __future__ import annotations

import importlib.util

import pytest

import mjorbit_warp
from mjorbit.testdata import FREE_BODY_XML


def test_import_exports_model_data_api():
    assert mjorbit_warp.MjoModel is not None
    assert mjorbit_warp.MjoData is not None
    assert mjorbit_warp.mjo_forward is not None
    assert mjorbit_warp.mjo_step is not None


def test_missing_optional_dependency_raises_helpful_error():
    if importlib.util.find_spec("mujoco_warp") is not None:
        pytest.skip("MJWarp is installed; the optional-dependency error path is unavailable.")

    with pytest.raises(ImportError, match="pixi install -e warp"):
        mjorbit_warp.MjoModel.from_xml_path(FREE_BODY_XML)
