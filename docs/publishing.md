# Publish the project page and documentation

One GitHub Pages site serves both:

| Content | Public URL | Source |
| --- | --- | --- |
| Paper and project page | `https://johnzhang3.github.io/mjorbit/` | `gh-pages` branch |
| User guide and API reference | `https://johnzhang3.github.io/mjorbit/docs/` | `docs/` on `main` |

## Build and deploy

The `Pages` workflow in `.github/workflows/pages.yml` builds Sphinx with
warnings treated as errors, checks out the latest `gh-pages` source, and
combines the two into a fresh directory with `scripts/build_pages.py`.

The assembly copies the project page's `index.html`, `style.css`, `assets/`,
and optional `CNAME`. It adds a Documentation link to the existing
`quick-links` block in the generated page. The project source is not rewritten,
and its media stays on `gh-pages`. Generated documentation lives under `docs/`;
checkout metadata and Sphinx build caches are excluded.

Every PR builds a downloadable **pages-preview** artifact. PRs do not deploy.
Pushes to `main` and manual runs on `main` publish the combined artifact through
`actions/deploy-pages`, using the repository's `github-pages` environment.
The workflow needs no personal access token.

## Initial GitHub setup

The repository's **Settings → Pages → Build and deployment → Source** must be
**GitHub Actions**. The `github-pages` environment must allow deployments from
`main`. Build and review the PR preview before changing an existing site's
publishing source. The first deployment runs after this workflow reaches `main`.

## Update the project page

Continue editing and pushing project-page content on `gh-pages`. After the
Pages workflow has landed on `main`, publish those edits by running:

```bash
gh workflow run pages.yml --ref main
```

The workflow retrieves the latest project page and rebuilds the documentation
from `main`. A push to `gh-pages` alone does not run the workflow on `main`.
Run it manually after project-only changes; code/documentation changes merged
to `main` publish automatically.

When changing the page layout, preserve its `quick-links` block or update the
assembly script. If adding public files outside `assets/`, also update
`PROJECT_FILES` in `scripts/build_pages.py` so those files are included.

## Preview locally

Build docs, export the current project-page branch, then assemble into a new
directory. Run from the code checkout:

```bash
pixi run -e docs docs-build
git fetch origin gh-pages
mjorbit_preview_dir="$(mktemp -d)"
git archive origin/gh-pages -o "$mjorbit_preview_dir/project.tar"
mkdir "$mjorbit_preview_dir/project"
tar -xf "$mjorbit_preview_dir/project.tar" -C "$mjorbit_preview_dir/project"
pixi run -e docs python scripts/build_pages.py \
  --project-dir "$mjorbit_preview_dir/project" \
  --docs-dir docs/_build/html \
  --output-dir "$mjorbit_preview_dir/site"
pixi run -e docs python -m http.server 8000 --bind 127.0.0.1 \
  --directory "$mjorbit_preview_dir/site"
```

Open `http://localhost:8000/` and follow the Documentation link. Verify the
project page's figures/videos, documentation navigation, search, and API pages.
The output directory must be new: the assembly never deletes or overlays an
existing site. For another preview, choose another temporary directory.

## Versions and recovery

The initial site publishes the latest `main` documentation at `/docs/`.
Versioned documentation can be added under `/docs/vX.Y.Z/` after release tags
exist; the current workflow does not publish separate tag snapshots.

If a build fails, deployment is skipped and the previous published site stays
available. To return to branch publishing, select `gh-pages` and `/ (root)` in
Pages settings and ensure the environment allows that branch. That restores
the project-only site; generated documentation is not committed to `gh-pages`.
