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
`~/.cache/phygs/scenes` and under the built-in scenes (`phygs/scene/builtin`). In each
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
isaaclab -p -m huggingface_hub.cli.hf auth login                      # once; the dataset is private
isaaclab -p -m phygs.scene.fetch list --repo_id mandmlabresearch/ForageBench     # ids, versions, sizes, the commit
isaaclab -p -m phygs.scene.fetch fetch --repo_id mandmlabresearch/ForageBench --scene_id bear_loop_house --scene_version 1.1.0
isaaclab -p -m phygs.scene.fetch fetch --repo_id mandmlabresearch/ForageBench --all  # every catalogued release
```

The fetcher resolves the dataset's `main` to a commit (or takes `--revision`), downloads the
manifest and the ZIP, verifies the archive's SHA-256, refuses unsafe paths and symlinks inside it,
extracts into `<cache>/<sha256>/`, checks that every USD dependency resolved inside the extraction,
and writes `<cache>/records/<commit>/<id>-<version>.json` plus `<cache>/<id>.json` for the registry.
`<id>.json` is the last fetch; a spec pinned to another fetched release of the id is found through
that release's own record, so several releases of one id can stay installed side by side.
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

## Articulated furniture

A scene may tie a cabinet and its doors and drawers into one PhysX articulation: an
`ArticulationRootAPI` on the fixed joint that mounts the cabinet to the world, the doors' and drawers'
joints from the cabinet to them (`bear_loop_house` 1.2.0 builds its 32 storage units this way; 1.1.0
joins each door and drawer to the world instead). PhyGS runs both (`phygs/scene/articulations.py`):

- Every link is a scene body, read like any other (pose, velocity, held forces).
- PhysX places a link from its joint positions only and ignores a pose written to it. A world's
  restore and a replayed sample set each joint's position from the written link poses
  (`SceneBodies.write_all`); writing one link alone raises.
- A drawer a spec opens before the start (`parts`) also gets its slide's joint state
  (`PhysicsJointStateAPI`), which PhysX starts it at.

On 1.3.0 a drawer pulled to 0.20 m carries its contents 0.20 m, a pushed door swings as in 1.1.0, the
cabinets do not move, and a restore closes them again. At rest, a physics step of the house costs
0.53 ms on 1.3.0 and 0.43 ms on 1.1.0 (CPU, 250 Hz, scene only; 2026-11-08).

## Floor plans

A release may ship its floor plans in `plan/` beside its scene file: `plan.json` and one image per
floor, seen straight down with everything above the floor's cut height left out. `plan.json`
(`phygs/scene/plan.py`, schema 1) gives each floor's `name`, `image`, `elevation` [m], `cut` [m],
`width` and `height` [px], `center` [m] and `ppm` [px/m]: world (x, y) on the floor plane is pixel
(`width` / 2 + (x - cx) `ppm`, `height` / 2 - (y - cy) `ppm`), +x right and +y up. Replays and pages
that draw a run on the plan read it; a scene without one gets a plan rendered on first use and cached.

## Fixture scenes

The three small scenes the robot tests use (grid storage, the navigation course, the dynamic
course) ship with PhyGS as files, `phygs/scene/builtin/<id>/scene.usda`; their builders live with the
other scene-building code, outside PhyGS.
