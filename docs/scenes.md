# Scenes: ids, versions, the cache, the dataset

A spec names a scene by id and never by file path:

```yaml
scene:
  id: bear_loop_house      # a scene folder, or a release of the scene dataset
  version: 1.1.0           # optional: pins a release; a plain folder accepts any pin
  name: house              # optional label for reports and the robot tests' stances (default: the id)
  ground: false            # true adds a ground plane (the file has no floor)
```

`phygs.scene.registry.scene_file(id, version)` looks, in order, under `$PHYGS_SCENES`, under
`~/.cache/phygs/scenes` and under the robot tests' fixtures (`tests/robot_tests/scenes`). In each
root it accepts a folder `<id>/scene.usda|usdc|usd` or a record `<id>.json` written by the fetcher
(`usd_path`, `scene_version`, `sha256`, the dataset commit). The USD file is resolved when first
used, so a spec validates without the scene, and the report's provenance holds the file's hash.

## Where released scenes live

The scene dataset is a private Hugging Face dataset, `mandmlabresearch/ForageBench`
(the name predates the repository split; the scenes are framework data that any task may use):

```
catalog.json                                   # lists every manifest
README.md                                      # the dataset card: scenes, versions, what changed
scenes/<id>/<version>/scene.json               # manifest: archive path, SHA-256, sizes, entry point, physics timestep
scenes/<id>/<version>/<folder>.zip             # the scene: <folder>/scene.usdc, textures/, README.md, ATTRIBUTION.md, SHA256SUMS
```

A release is immutable: a changed scene is a new version of the same id (`bear_loop_house` 1.0.0 is
the first house; 1.1.0 is the storage revision). A benchmark's episode manifest records
`scene_id@version` and the archive's SHA-256; a report's provenance records the USD's hash, so a
result names the exact scene it ran in.

## Fetching

```bash
./isaaclab.sh -p -m huggingface_hub.cli.hf auth login                      # once; the dataset is private
./isaaclab.sh -p -m phygs.scene.fetch list                                 # ids, versions, sizes, the commit
./isaaclab.sh -p -m phygs.scene.fetch fetch --scene_id bear_loop_house --scene_version 1.1.0
./isaaclab.sh -p -m phygs.scene.fetch fetch --all                          # every catalogued release
```

The fetcher resolves the dataset's `main` to a commit (or takes `--revision`), downloads the
manifest and the ZIP, verifies the archive's SHA-256, refuses unsafe paths and symlinks inside it,
extracts into `<cache>/<sha256>/`, checks that every USD dependency resolved inside the extraction,
and writes `<cache>/records/<commit>/<id>-<version>.json` plus `<cache>/<id>.json` for the registry.
The cache is `--cache_dir` (default `~/.cache/phygs/scenes`, which is also `$PHYGS_SCENES`'
default). Rerunning verifies the cached files; nothing is overwritten. Nothing downloads during a
simulation.

## Making a release

Scenes are built, packaged and published outside PhyGS; PhyGS only reads releases. A release must
follow the format above: one scene id at one version, a ZIP with the USD's dependency closure, a
manifest with its SHA-256 and entry point, an entry in the catalog. It is never replaced: a changed
scene is a new version. The scene folder holds `README.md` (what the scene is, how it was validated)
and `ATTRIBUTION.md` (asset provenance); keep the dataset private until the asset licences are
reviewed.

## Fixture scenes

The three small scenes the robot tests use (grid storage, the navigation course, the dynamic
course) are committed as files, `tests/robot_tests/scenes/<id>/scene.usda`; their builders live with the
other scene-building code, outside PhyGS.
