"""Combine the existing project page with Sphinx HTML for GitHub Pages.

The output must be a new directory. Inputs are never modified, and only the
project page's public files are copied from its source checkout.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_FILES = ("index.html", "style.css", "assets", "CNAME")
QUICK_LINKS = '<div class="quick-links">'
DOCS_LINK = '<a href="docs/">[documentation]</a>'


def build_pages(project_dir: Path, docs_dir: Path, output_dir: Path) -> None:
    project_dir = project_dir.resolve(strict=True)
    docs_dir = docs_dir.resolve(strict=True)
    output_dir = output_dir.resolve()
    for source in (project_dir, docs_dir):
        if output_dir.is_relative_to(source) or source.is_relative_to(output_dir):
            raise ValueError("Output must be separate from both source directories")
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")

    for required in ("index.html", "style.css", "assets"):
        if not (project_dir / required).exists():
            raise ValueError(f"Missing project page input: {required}")
    for required in ("index.html", "api.html", "searchindex.js", "_static"):
        if not (docs_dir / required).exists():
            raise ValueError(f"Missing built documentation: {required}")

    index = (project_dir / "index.html").read_text(encoding="utf-8")
    if 'href="docs/"' not in index:
        if index.count(QUICK_LINKS) != 1:
            raise ValueError("Expected one project-page quick-links block for the docs link")
        index = index.replace(QUICK_LINKS, f"{QUICK_LINKS}\n      {DOCS_LINK}", 1)

    project_inputs = [project_dir / name for name in PROJECT_FILES
                      if (project_dir / name).exists()]
    for source in (*project_inputs, docs_dir):
        paths = [source, *source.rglob("*")] if source.is_dir() else [source]
        if any(path.is_symlink() for path in paths):
            raise ValueError(f"Pages inputs must not contain symbolic links: {source}")

    output_dir.mkdir(parents=True)
    for source in project_inputs:
        destination = output_dir / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    (output_dir / "index.html").write_text(index, encoding="utf-8")
    (output_dir / ".nojekyll").touch()
    shutil.copytree(
        docs_dir, output_dir / "docs",
        ignore=shutil.ignore_patterns(".doctrees", ".buildinfo"),
    )
    print(f"Pages site: {output_dir} (project page at /, documentation at /docs/)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--docs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_pages(args.project_dir, args.docs_dir, args.output_dir)


if __name__ == "__main__":
    main()
