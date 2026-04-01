"""Bundled MuJoCo XML assets used by examples and tests."""

from __future__ import annotations

import pathlib

TESTDATA_DIR = pathlib.Path(__file__).parent

FREE_BODY_XML = str(TESTDATA_DIR / "free_body.xml")
SPACECRAFT_ARM_XML = str(TESTDATA_DIR / "spacecraft_arm.xml")

__all__ = ["FREE_BODY_XML", "SPACECRAFT_ARM_XML", "TESTDATA_DIR"]
