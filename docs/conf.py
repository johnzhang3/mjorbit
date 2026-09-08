"""Sphinx configuration for the public mjorbit guide."""

from importlib.metadata import version as package_version

project = "mjorbit"
author = "The mjorbit contributors"
copyright = "2026, The mjorbit contributors"
release = package_version("mjorbit")
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build"]
myst_enable_extensions = ["dollarmath"]
myst_heading_anchors = 3
autodoc_typehints = "description"
autodoc_member_order = "bysource"
html_theme = "sphinx_book_theme"
html_title = "mjorbit documentation"
html_baseurl = "https://johnzhang3.github.io/mjorbit/docs/"
html_theme_options = {
    "repository_url": "https://github.com/johnzhang3/mjorbit",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "extra_footer": '<a href="https://johnzhang3.github.io/mjorbit/">Project page</a>',
}
html_show_sphinx = False
