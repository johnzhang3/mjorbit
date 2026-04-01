"""MuJoCo model asset paths and optional builder helpers."""
import pathlib

ASSETS_DIR = pathlib.Path(__file__).parent / "assets"

FREE_BODY_XML = str(ASSETS_DIR / "free_body.xml")
SPACECRAFT_ARM_XML = str(ASSETS_DIR / "spacecraft_arm.xml")
