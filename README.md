# mjorbit project page

Source for the [mjorbit](https://github.com/johnzhang3/mjorbit) project website,
combined with the documentation by the `Pages` workflow on `main`.

**Project website:** https://johnzhang3.github.io/mjorbit/

**Documentation:** https://johnzhang3.github.io/mjorbit/docs/

Layout adapted from the [Sumo](https://sumo.rai-inst.com/) project page
(Playfair Display + Roboto Mono, warm-paper background, black tagline pills),
with mjorbit's dark-red accent.

## Files

```
index.html                 # the page
style.css                  # styles
.nojekyll                  # tell GitHub Pages to serve assets/ verbatim
assets/
  js/main.js               # scroll indicator, click-to-fullscreen, lazy autoplay, TOC highlight
  images/                  # banner + figures cropped from the paper PDF
  posters/                 # per-example placeholder posters (shown until real videos land)
  videos/                  # ← drop example .mp4 files here (see below)
```

## Publishing updates

Push project-page edits to this branch, then run the workflow on `main`:

```bash
git push -u origin gh-pages
gh workflow run pages.yml --ref main
```

GitHub Pages uses **GitHub Actions** as its publishing source. The workflow
combines this branch's latest page and assets with Sphinx documentation under
`docs/`. A push to `gh-pages` alone does not deploy; pushes to `main` deploy
automatically. See the [publishing guide](https://johnzhang3.github.io/mjorbit/docs/publishing.html)
for previews and recovery instructions.

## Adding the example videos

Each example slot is already a `<video>` element. Its poster shows
`PREVIEW COMING SOON` until a matching file appears in `assets/videos/`.
Just drop in the MP4s (no HTML edits needed):

| Example                          | File                                      |
|----------------------------------|-------------------------------------------|
| (a) Multibody attitude control   | `assets/videos/attitude_control.mp4`      |
| (b) Autonomous docking           | `assets/videos/autonomous_docking.mp4`    |
| (c) Grasping under gravity grad. | `assets/videos/grasping_gravity_gradient.mp4` |
| (d) RL truss pointing            | `assets/videos/rl_truss_pointing.mp4`     |
| (optional) hero reel             | `assets/videos/hero.mp4`                  |

Once a real video is present, also remove that card's
`<span class="badge">video coming soon</span>` in `index.html` (and add
`controls` to the `<video>` if you want a scrubber). Encode as H.264 MP4
(`-movflags +faststart`), ideally 16:9, muted, looping-friendly.

## Updating figures / text

Page content is drawn from the paper. The analysis figures
(`assets/images/{zero_g_comparison,frame_comparison,pipeline}.png`) were cropped
from the paper PDF; re-export them if the figures change. Copy is in `index.html`.
