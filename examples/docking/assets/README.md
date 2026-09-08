# Docking mesh provenance

The project Apache 2.0 license does not establish the license of these
third-party meshes. Complete the source and redistribution records below
before the first public release. These assets are used by source-tree docking
examples and recordings; they are not included in the runtime wheel.

## Evidence in the repository

- `ISS/` and `Soyuz/` were introduced in commit
  `463acd483dba220fbde71d9a9c24fbce9046f7ca` ("added assets for docking scenario").
- A `models:` comment in [main_mppi.py](../main_mppi.py) points to NASA's
  [International Space Station (ISS) (A) resource](https://science.nasa.gov/3d-resources/international-space-station-iss-a/).
  The page credits NASA/Ames Research Center. This is an ISS source lead;
  the comment alone does not establish which downloaded files were used or
  the source of the Soyuz geometry.
- OBJ/MTL export headers mention Blender 5.1.2 and `www.blender.org`. Those
  identify the export tool, not the mesh creator or redistribution terms.
- The source commit and current mesh files do not include a model-specific
  license, original archive name, or complete transformation history.

## Records to complete

| Asset | Recorded source | Still to confirm |
| --- | --- | --- |
| `ISS/ISS_lowpoly.obj` and split `ISS_lowpoly/` meshes | NASA ISS (A) URL in the docking script | Match to original download; original creator and terms; simplification/splitting history |
| `Soyuz/Soyuz_lowpoly.obj` and split `Soyuz_lowpoly/` meshes | No explicit source identified | Original source/creator, redistribution terms, and modification history |

When the originals are located, record the precise download URL or revision,
required attribution, applicable license/usage terms, and the changes made to
produce these low-poly exports. Preserve any accompanying notices. If the
terms cannot be established, replace or remove the affected assets and update
the docking example and recordings accordingly.
