"""Check that documentation publishing preserves the existing project site."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

build_pages = runpy.run_path(str(Path(__file__).parents[1] / "scripts/build_pages.py"))[
    "build_pages"
]


@pytest.fixture
def site_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    docs = tmp_path / "html"
    (project / "assets/videos").mkdir(parents=True)
    (project / "assets/videos/demo.mp4").write_bytes(b"existing video")
    (project / "index.html").write_text(
        '<h1>Paper</h1><div class="quick-links"><a href="#examples">Video</a></div>'
    )
    (project / "style.css").write_text("body { color: red; }")
    (project / "README.md").write_text("Project source instructions")
    (project / ".git").mkdir()
    (project / ".git/config").write_text("checkout metadata")
    (docs / "_static").mkdir(parents=True)
    (docs / "index.html").write_text("<h1>Documentation</h1>")
    (docs / "api.html").write_text("<h1>API</h1>")
    (docs / "searchindex.js").write_text("Search.setIndex({});")
    (docs / ".doctrees").mkdir()
    (docs / ".doctrees/environment.pickle").write_bytes(b"build cache")
    return project, docs, tmp_path / "site"


def test_preserves_project_assets_and_mounts_docs(site_sources: tuple[Path, Path, Path]) -> None:
    project, docs, output = site_sources
    original_index = (project / "index.html").read_bytes()
    build_pages(project, docs, output)

    assert (output / "index.html").read_text().count('href="docs/"') == 1
    assert "<h1>Paper</h1>" in (output / "index.html").read_text()
    assert (project / "index.html").read_bytes() == original_index
    assert (output / "assets/videos/demo.mp4").read_bytes() == b"existing video"
    assert (output / "style.css").read_bytes() == (project / "style.css").read_bytes()
    assert (output / "docs/index.html").read_bytes() == (docs / "index.html").read_bytes()
    assert (output / "docs/searchindex.js").is_file()
    assert (output / ".nojekyll").is_file()
    assert not (output / "README.md").exists()
    assert not (output / ".git").exists()
    assert not (output / "docs/.doctrees").exists()


def test_rejects_missing_docs_before_writing(site_sources: tuple[Path, Path, Path]) -> None:
    project, docs, output = site_sources
    (docs / "index.html").unlink()
    with pytest.raises(ValueError, match="Missing built documentation"):
        build_pages(project, docs, output)
    assert not output.exists()


def test_rejects_output_inside_input(site_sources: tuple[Path, Path, Path]) -> None:
    project, docs, _ = site_sources
    with pytest.raises(ValueError, match="Output must be separate"):
        build_pages(project, docs, project / "output")


def test_rejects_existing_output_without_changing_it(
    site_sources: tuple[Path, Path, Path],
) -> None:
    project, docs, output = site_sources
    output.mkdir()
    (output / "index.html").write_text("Existing site")
    with pytest.raises(FileExistsError):
        build_pages(project, docs, output)
    assert (output / "index.html").read_text() == "Existing site"


def test_rejects_changed_link_target_layout(site_sources: tuple[Path, Path, Path]) -> None:
    project, docs, output = site_sources
    (project / "index.html").write_text("<h1>New layout without quick links</h1>")
    with pytest.raises(ValueError, match="quick-links block"):
        build_pages(project, docs, output)
    assert not output.exists()


def test_rejects_symlinked_assets(site_sources: tuple[Path, Path, Path]) -> None:
    project, docs, output = site_sources
    (project / "assets/source-link").symlink_to(project / "README.md")
    with pytest.raises(ValueError, match="symbolic links"):
        build_pages(project, docs, output)
    assert not output.exists()
