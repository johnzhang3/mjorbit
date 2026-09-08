# Release checklist

Use a dedicated branch and a reviewed PR. The current version is `0.1.0`;
preparing these files does not publish a package or create a release tag.

## Before the first public release

- Replace the pending paper notice in the README and citation metadata with
  the permanent arXiv identifier, verified title, authors, and publication date.
  A submission identifier is not a public preprint identifier.
- Resolve the remaining ISS/Soyuz source and redistribution details in
  [the asset provenance notes](examples/docking/assets/README.md). Record the
  evidence and required attribution, or replace/remove assets whose terms
  cannot be established. Check paper recordings containing those assets too.
- Review `LICENSE`, `NOTICE`, and `CITATION.cff`, including third-party notices.
- Verify the GPU example and tests on the supported Linux/NVIDIA environment;
  CPU and docs CI do not establish GPU readiness.
- Verify the first combined Pages deployment: the project page stays at
  `/mjorbit/` and documentation is served at `/mjorbit/docs/`. See
  [publishing setup](docs/publishing.md) for the one-time Pages configuration.

## Validate a release candidate

Run from a clean checkout with the committed lockfile:

```bash
pixi install --locked
pixi run lint
pixi run typecheck
pixi run test
pixi run cpp-test
pixi run example-minimal
pixi run example-free-drift
pixi run example-mppi-arm-reach --nthread 2
pixi run -e frames test-frames
pixi run -e docs docs-build
pixi run package-check
```

On a supported GPU host:

```bash
pixi run -e warp example-batched
pixi run -e warp test-warp
pixi run -e warp typecheck-warp
```

Launch `pixi run viewer` and check the default free-drift scene, Pause, Reset,
task switching, and shutdown. Run the documented training/playback workflow
if publishing a policy checkpoint. Paper results have separate reproduction
commands in [experiments](experiments/README.md).

`package-check` tests both a wheel and source-distribution installation in
fresh temporary environments, including native libraries, bundled data,
viewer textures, entry points, and metadata. It also simulates a used checkout
containing ignored native binaries and rejects archive contamination. CI runs
these checks on Linux and macOS. Build wheels separately for each Python ABI
and supported platform; a locally built native wheel is not universal.

## Publish the reviewed revision

1. Update the version in `pyproject.toml`, `pixi.toml`, and `CITATION.cff`;
   refresh `pixi.lock`, and date the entry in `CHANGELOG.md`.
2. Confirm CI passes on the final commit, inspect release archive contents,
   and record platform/Python/GPU validation results.
3. Create the agreed version tag and GitHub release from that commit. Attach
   reviewed artifacts and describe platform support and known limitations.
4. If publishing to a package index, validate the upload in the intended
   staging workflow first, then test installation from the public index.
5. Publish matching documentation and verify all public README, paper, demo,
   citation, and download links.

Tags and index uploads are separate maintainer actions. The Pages workflow
builds a preview for PRs and publishes the combined site on pushes to `main`.
