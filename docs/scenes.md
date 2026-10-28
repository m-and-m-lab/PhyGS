# Scenes: ids, versions, the cache, the dataset

A spec names a scene by id and never by file path:

```yaml
scene:
  id: bear_loop_house      # a scene folder, or a release of the scene dataset
  version: 1.1.0           # optional: pins a release; a plain folder accepts any pin
  name: house              # optional label for reports and the robot tests' stances (default: the id)
  ground: false            # true adds a ground plane (the file has no floor)
```

`phygs.episodes.scenes.scene_file(id, version)` looks, in order, under `$PHYGS_SCENES`, under
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
./isaaclab.sh -p -m phygs.scenes.fetch list                                 # ids, versions, sizes, the commit
./isaaclab.sh -p -m phygs.scenes.fetch fetch --scene_id bear_loop_house --scene_version 1.1.0
./isaaclab.sh -p -m phygs.scenes.fetch fetch --all                          # every catalogued release
```

The fetcher resolves the dataset's `main` to a commit (or takes `--revision`), downloads the
manifest and the ZIP, verifies the archive's SHA-256, refuses unsafe paths and symlinks inside it,
extracts into `<cache>/<sha256>/`, checks that every USD dependency resolved inside the extraction,
and writes `<cache>/records/<commit>/<id>-<version>.json` plus `<cache>/<id>.json` for the registry.
The cache is `--cache_dir` (default `~/.cache/phygs/scenes`, which is also `$PHYGS_SCENES`'
default). Rerunning verifies the cached files; nothing is overwritten. Nothing downloads during a
simulation.

## Publishing a new version

From the Isaac Lab root, with `$PHYGS` the checkout:

```bash
# 1. Package: the USD's dependency closure plus README.md and ATTRIBUTION.md, verified after extraction.
./isaaclab.sh -p $PHYGS/scene_generation/usd_tools/package_scene.py \
  --scene_dir /scenes/bear_loop_house_storage_revision --output /scenes/bear_loop_house_storage_revision.zip

# 2. Manifest and catalog entry in a local release folder (a checkout of the dataset's metadata).
./isaaclab.sh -p $PHYGS/scene_generation/usd_tools/publish_scene.py prepare \
  --archive /scenes/bear_loop_house_storage_revision.zip --release_dir /scenes/releases \
  --scene_id bear_loop_house --version 1.1.0

# 3. Edit the dataset card (releases/README.md), then one commit: catalog, manifest, card, ZIP.
./isaaclab.sh -p $PHYGS/scene_generation/usd_tools/publish_scene.py upload \
  --archive /scenes/bear_loop_house_storage_revision.zip --release_dir /scenes/releases \
  --scene_id bear_loop_house --version 1.1.0
```

`prepare` refuses a manifest that exists and `upload` refuses a release the dataset already holds,
so a published scene is never replaced. The archive's top-level folder is the scene's authoring
name (`bear_loop_house_storage_revision`); the id and version are the dataset's. The scene folder
must hold `README.md` (what the scene is, how it was validated) and `ATTRIBUTION.md` (asset
provenance); keep the dataset private until the asset licences are reviewed.

## Fixture scenes

The three small scenes the robot tests use (grid storage, the navigation course, the dynamic
course) are committed as `tests/robot_tests/scenes/<id>/scene.usda`, built by
`scene_generation/usd_tools/build_grid_storage_scene.py` and `build_test_course_scenes.py`.
