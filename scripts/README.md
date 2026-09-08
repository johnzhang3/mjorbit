# Maintainer and recording tools

These tools remain in the source repository and source distribution. They are
not installed as runtime packages.

| Tool | Purpose | Run |
| --- | --- | --- |
| [build_pages.py](build_pages.py) | Combine the existing project page with built documentation for GitHub Pages | See [publishing](../docs/publishing.md) |
| [check_release.py](check_release.py) | Build and inspect clean archives, then test wheel and source installs in separate temporary environments | `pixi run package-check` |
| [smoke_install.py](smoke_install.py) | Check native loading, bundled assets, simulation, and the installed CLI | Invoked by `check_release.py` inside each clean environment |
| [record/](record/README.md) | Produce trajectories and render paper clips/stills with Chrome and ffmpeg | See the recording guide |

Release checks need network access for build and install dependencies. Their
temporary files are removed on completion. Normal distributable builds use
`pixi run build-dist`, which leaves artifacts in the ignored `dist/` directory.

See [RELEASING.md](../RELEASING.md) for the complete release process.
