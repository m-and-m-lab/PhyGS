# PhyGS Scene Generation

This directory hosts the PhyGS scene-generation pipeline — a fork of [Infinigen-Indoors](https://github.com/princeton-vl/infinigen) extended for articulated assets, semantic object placement, and lighting modifications that survive the export from Blender to USD.

Scene generation runs in its own Python environment (Blender 4.2 / `bpy` / Infinigen). It is **isolated** from `phygs_simulation/`, which depends on IsaacSim/IsaacLab/cuRobo — the dependency pins conflict and the two must not share an interpreter.

## Installation

- **PhyGS-tailored guide:** [`docs/installation/scene_generation.md`](../../docs/installation/scene_generation.md) at the repo root. Recommended starting point — it documents the conda env name, the `[sim]` extra needed for the PhyGS export path, and how scene USDs cross over to `phygs_simulation`.
- **Upstream Infinigen install reference:** [`docs/Installation.md`](docs/Installation.md). Use this for the Blender-UI variant, OpenGL ground-truth, terrain/CUDA install, Docker images, and platform-specific dependency lists.
- **Original Infinigen README:** [`docs/README_original.md`](docs/README_original.md) for upstream feature context and citations.

## Example: indoor scene generation

After activating the scene-generation conda env (see the installation guide), from `scene_generation/infinigen/`:

```bash
# Single-seed coarse indoor scene (~8–15 min CPU depending on room type).
python -m infinigen_examples.generate_indoors \
  --seed 0 --task coarse \
  --output_folder outputs/indoors/coarse/indoor_0 \
  -g fast_solve.gin overhead.gin \
  -p compose_indoors.terrain_enabled=False
```

Room-type variants (replace the `restrict_parent_rooms` value):

```bash
python -m infinigen_examples.generate_indoors \
  --seed 0 --task coarse \
  --output_folder outputs/indoors/coarse-dining \
  -g fast_solve.gin singleroom.gin \
  -p compose_indoors.terrain_enabled=False \
     'restrict_solving.restrict_parent_rooms=["DiningRoom"]'
```

Substitute `DiningRoom` with `Bathroom`, `Bedroom`, `Kitchen`, `LivingRoom`, etc. See [`docs/HelloRoom.md`](docs/HelloRoom.md) for the full menu and rendering options.

### Batch generation

`generate_multi.sh` is a tmux driver that launches multiple seeds in parallel windows. Inspect it before use — the seed range (`0..3` by default) and tmux session name are hardcoded.

```bash
bash generate_multi.sh
```

## Example: USD export for IsaacSim

After a scene has been generated, export it to USD with the Infinigen tool:

```bash
python -m infinigen.tools.export \
  --input_folder outputs/indoors/coarse/indoor_0 \
  --output_folder outputs/my_export \
  -f usdc -r 1024 \
  --omniverse
```

`--omniverse` enables the articulated-asset path: doors (and additional articulated kinds going forward) are stripped from the static scene USD and listed in a sibling `articulated_assets.json` alongside their standalone articulated USD twins.

### Compose articulated assets back into the scene

Inject the articulated USDs into the exported scene as USD reference Xforms at their recorded world poses:

```bash
python scripts/usd_articulated_scene_composer.py \
  --scene-usd outputs/my_export/export_scene.blend/export_scene.usdc \
  --articulated-json outputs/my_export/export_scene.blend/articulated_assets.json
```

By default the composer **bundles** per-asset USDs (and their texture folders) into `<scene_usd_dir>/articulated_assets/<kind>_<idx>/` and rewrites references to be relative, so the whole `export_scene.blend/` directory is self-contained and portable to another machine (e.g. into a PhyGS IsaacSim container). Pass `--no-bundle` to keep absolute references instead.

### Batch export

`export_isaacsim_multi.sh` mirrors `generate_multi.sh` for the export step. Same caveat — seed range is hardcoded; edit before running.

```bash
bash export_isaacsim_multi.sh
```

## Standalone articulated asset export

To export individual articulated objects (doors, dishwashers, lamps, fridges, toasters, etc.) as MJCF/URDF/USD without composing them into a full scene:

```bash
./scripts/spawn_sim_ready_asset.sh <asset_name> <num_instances> <mjcf|urdf|usd>

# Examples
./scripts/spawn_sim_ready_asset.sh door 10 usd
./scripts/spawn_sim_ready_asset.sh dishwasher 10 usd
```

See `infinigen/assets/sim_objects/mapping.py` (`OBJECT_CLASS_MAP`) for the current list of exportable articulated assets, and [`docs/simulation/ExportingToSimulators.md`](docs/simulation/ExportingToSimulators.md) for the full pipeline.

## Further reading

| Topic | Document |
| --- | --- |
| Generating your first room interactively | [`docs/HelloRoom.md`](docs/HelloRoom.md) |
| Exporting to MJCF / URDF / USD (assets and full scenes) | [`docs/simulation/ExportingToSimulators.md`](docs/simulation/ExportingToSimulators.md) |
| Other export targets (OBJ, GLB, FBX, PLY, etc.) | [`docs/ExportingToExternalFileFormats.md`](docs/ExportingToExternalFileFormats.md) |
| Articulation-aware joint nodes | [`docs/simulation/UsingJointNodes.md`](docs/simulation/UsingJointNodes.md) |
| Visualizing articulated assets | [`docs/simulation/VisualizingArticulatedAssets.md`](docs/simulation/VisualizingArticulatedAssets.md) |
| Verifying sim-ready assets | [`docs/simulation/VerifyingSimAssets.md`](docs/simulation/VerifyingSimAssets.md) |
| Gin-config knobs and overrides | [`docs/ConfiguringInfinigen.md`](docs/ConfiguringInfinigen.md) |
| Ground-truth annotations | [`docs/GroundTruthAnnotations.md`](docs/GroundTruthAnnotations.md) |

## Crossing into `phygs_simulation`

A successful export produces a scene USDC at:

```text
outputs/<your_run>/export_scene.blend/export_scene.usdc
```

Pass that absolute path as `--scene_usd` to the IsaacLab launchers in `phygs_simulation` — for example:

```bash
./isaaclab.sh -p scripts/phygs_simulation/scripts/interactive_search.py \
  --enable_cameras \
  --robot_usd scripts/phygs_simulation/spot_model/spot_arm_w_cam.usd \
  --scene_usd /absolute/path/to/export_scene.usdc
```

See [`docs/installation/phygs_simulation.md`](../../docs/installation/phygs_simulation.md) for the simulation-side install and launch details.
