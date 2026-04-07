"""Bundled MuJoCo XML assets used by examples and tests."""

from __future__ import annotations

import pathlib

TESTDATA_DIR = pathlib.Path(__file__).parent

FREE_BODY_XML = str(TESTDATA_DIR / "free_body.xml")
FREE_BODY_SENSORS_XML = str(TESTDATA_DIR / "free_body_sensors.xml")
SPACECRAFT_ARM_XML = str(TESTDATA_DIR / "spacecraft_arm.xml")
TWO_BODIES_XML = str(TESTDATA_DIR / "two_bodies.xml")

__all__ = [
    "FREE_BODY_SENSORS_XML",
    "FREE_BODY_XML",
    "SPACECRAFT_ARM_XML",
    "TESTDATA_DIR",
    "TWO_BODIES_XML",
]
