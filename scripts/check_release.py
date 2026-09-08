"""Build and exercise clean release artifacts, including an sdist contamination check.

Run with ``pixi run package-check``. Requires network access for isolated build
and runtime dependencies; all test environments and artifacts are temporary.
"""

from __future__ import annotations

import email
import shutil
import subprocess
import sys
import tarfile
import tempfile
import venv
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
NATIVE_SUFFIXES = {".so", ".dylib", ".dll", ".pyd", ".a", ".o"}


def run(*args: str, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def check_sdist(path: Path) -> set[str]:
    with tarfile.open(path) as archive:
        names = {str(Path(name).relative_to(Path(name).parts[0])) for name in archive.getnames()}
    unexpected = {
        name for name in names
        if Path(name).suffix in NATIVE_SUFFIXES | {".pyc"}
        or "__pycache__" in Path(name).parts
        or name.startswith((".pixi/", "build/", "dist/", "docs/_build/"))
    }
    assert not unexpected, f"Generated files in source distribution: {sorted(unexpected)}"
    required = {
        "src/cpp/CMakeLists.txt", "src/cpp/bindings/module.cc",
        "src/cpp/include/mjorbit/runtime.h", "src/mjorbit/testdata/free_body.xml",
        "examples/minimal.py", "experiments/README.md", "scripts/README.md",
        "docs/index.md", "docs/_static/viewer.jpg", "LICENSE", "NOTICE", "CITATION.cff",
    }
    assert required <= names, f"Missing source files: {sorted(required - names)}"
    return names


def check_wheel(path: Path) -> None:
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = email.message_from_bytes(archive.read(metadata_name))
    assert metadata["Summary"]
    assert metadata["License-Expression"] == "Apache-2.0"
    assert metadata["Description-Content-Type"] == "text/markdown"
    assert "# mjorbit" in metadata.get_payload()
    assert any(name.startswith("mjorbit/_bindings") for name in names)
    assert any(name.startswith("mjorbit/plugins/mjorbit_plugin") for name in names)
    assert "mjorbit/testdata/free_body.xml" in names
    assert "viewer/assets/earth_day_2k.jpg" in names
    assert "viewer/assets/earth_day_5400.jpg" in names
    assert any(name.endswith("/licenses/LICENSE") for name in names)
    assert any(name.endswith("/licenses/NOTICE") for name in names)
    assert not any(name.startswith(("examples/", "experiments/", "scripts/", "tests/"))
                   for name in names)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mjorbit-release-") as temporary:
        work = Path(temporary)
        artifacts = work / "dist"
        # Build a wheel from the sdist, as python -m build does by default.
        run(sys.executable, "-m", "build", str(ROOT), "--outdir", str(artifacts), cwd=work)
        sdist = next(artifacts.glob("*.tar.gz"))
        wheel = next(artifacts.glob("*.whl"))
        clean_names = check_sdist(sdist)
        check_wheel(wheel)

        # A used checkout must not add ignored binaries to the source archive.
        # Work on a temporary copy so no local build products are overwritten.
        source = work / "source"
        shutil.copytree(ROOT, source, ignore=shutil.ignore_patterns(
            ".git", ".pixi", ".venv", "build", "dist", "_build", "out",
            "__pycache__", ".pytest_cache", ".ruff_cache", "*.so", "*.dylib",
        ))
        (source / "src/cpp/_bindings.release-check.so").write_bytes(b"local build product")
        (source / "src/cpp/mujoco_orbit_plugin.dylib").write_bytes(b"local build product")
        used_artifacts = work / "used-dist"
        run(sys.executable, "-m", "build", "--sdist", str(source),
            "--outdir", str(used_artifacts), cwd=work)
        used_names = check_sdist(next(used_artifacts.glob("*.tar.gz")))
        assert clean_names == used_names, "Source archive file list depends on checkout artifacts"

        for label, artifact in (("wheel", wheel), ("sdist", sdist)):
            environment = work / f"venv-{label}"
            venv.EnvBuilder(with_pip=True).create(environment)
            python = environment / "bin/python"
            run(str(python), "-m", "pip", "install", str(artifact), cwd=work)
            run(str(python), "-m", "pip", "check", cwd=work)
            # -I ignores PYTHONPATH and the script's source directory.
            run(str(python), "-I", str(ROOT / "scripts/smoke_install.py"), cwd=work)
        print("Release checks passed: clean archives and isolated wheel/sdist installs.")


if __name__ == "__main__":
    main()
