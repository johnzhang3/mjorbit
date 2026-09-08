# Development and release checks

Install the development environment and run the checks from the repository root:

```bash
pixi install
pixi run lint
pixi run typecheck
pixi run test
pixi run cpp-test
```

The test and example tasks rebuild and synchronize the native extension.
`pixi run test-api` runs the public model/data tests. Optional frame-conversion
tests use `pixi run -e frames test-frames`; GPU tests use
`pixi run -e warp test-warp` on a compatible machine.

## Documentation

```bash
pixi run -e docs docs-build
pixi run -e docs docs-serve
```

The Sphinx site is generated in `docs/_build/html` and served locally at
`http://localhost:8000`. Warnings fail the build. The minimal tutorials include
the actual example files, and API signatures come from the installed CPU
package. Update examples and their guides together.

The Pages workflow builds the documentation and combines it with the current
`gh-pages` project page. PRs receive a downloadable `pages-preview` artifact;
pushes to `main` publish the validated site. See [publishing](publishing.md)
for the site layout, initial setup, and project-page updates.

## Distribution checks

```bash
pixi run build-dist
pixi run package-check
```

`build-dist` writes a source archive and a wheel under `dist/`. `package-check`
builds into temporary directories, checks metadata/assets, verifies that local
native build products cannot contaminate the source archive, and installs both
artifacts into separate clean environments. It runs a simulation and checks
the installed `mjo-viewer` entry point from outside the source tree.
It requires network access for isolated build and runtime dependencies.

See the repository
[contribution guide](https://github.com/johnzhang3/mjorbit/blob/main/CONTRIBUTING.md)
and [release checklist](https://github.com/johnzhang3/mjorbit/blob/main/RELEASING.md)
for review expectations, provenance checks, and publication steps.
