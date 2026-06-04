## Exporting to Robotics Simulators

### Exporting Articulated Objects

To export to robotics simulation, install Infinigen using the command:
```
pip install -e ".[sim]"
```

To export articulated assets to the MJCF/URDF/USD file formats, we provide the following command.

```bash
./scripts/spawn_sim_ready_asset.sh {asset name} {number of instances} {mjcf/urdf/usd}
```

For instance, to generate 10 instances of all available assets in the MJCF file format, run the following scripts.

```bash
./scripts/spawn_sim_ready_asset.sh door 10 mjcf
./scripts/spawn_sim_ready_asset.sh dishwasher 10 mjcf
./scripts/spawn_sim_ready_asset.sh lamp 10 mjcf
./scripts/spawn_sim_ready_asset.sh multifridge 10 mjcf
./scripts/spawn_sim_ready_asset.sh multidoublefridge 10 mjcf
./scripts/spawn_sim_ready_asset.sh toaster 10 mjcf
```

To view the most updated list of included articulated assets, please see `OBJECT_CLASS_MAP` in `infinigen/assets/sim_objects/mapping.py`. Each key in the map corresponds to an articulated asset. Note that `singlefridge` and `doublefridge` are used by `multifridge` and `multidoublefridge` and are not meant to be used on their own.

### Exporting Infinigen-Indoors Scenes

This section details how to run a robotics simulation in an exported Infinigen scene using NVIDIA IsaacSim. For more information on scene generation or export, refer to [Hello Room](HelloRoom.md) or [Exporting to External File Formats](./ExportingToExternalFileFormats.md).

:warning: The whole-scene exporter outputs .usd files with no articulation. For whole indoor scense, we have only tested compatibility for Indoor scenes using IsaacSim, not nature scenes or any other simulator.

First, create and export a scene with the commands below:

```bash
python -m infinigen_examples.generate_indoors --seed 0 --task coarse --output_folder outputs/indoors/coarse -g overhead.gin singleroom.gin -p compose_indoors.terrain_enabled=False  restrict_solving.solve_max_rooms=1 
```

```bash
python -m infinigen.tools.export --input_folder outputs/indoors/coarse --output_folder outputs/my_export -f usdc -r 1024 --omniverse
```

#### Articulated assets in whole-house scenes (doors today)

When `--omniverse` is set, the exporter detects objects tagged with articulation
metadata (currently doors), removes their static geometry from the baked scene
USD, and writes `articulated_assets.json` next to the scene USD. That JSON lists
each instance's final world pose and the absolute path to its standalone
articulated USD twin (joints/drives/axes), already written under
`./sim_exports/usd/door/<idx>/door.usda` during scene generation.

Run the composer to inject those assets back into the scene USD as USD reference
Xforms at their recorded world poses:

```bash
python scripts/usd_articulated_scene_composer.py \
    --scene-usd outputs/my_export/export_scene.blend/export_scene.usdc \
    --articulated-json outputs/my_export/export_scene.blend/articulated_assets.json
```

By default the composer **bundles** each per-asset USD (and its `assets/` folder
of textures) into `<scene_usd_dir>/articulated_assets/<kind>_<idx>/` and uses
relative references in the scene USD, making the scene directory self-contained.
You can copy the whole `export_scene.blend/` directory to another machine
(e.g. an IsaacSim container at `/workspace/.../scenes/my_scene/`) and the
references resolve locally without any path rewriting.

Pass `--no-bundle` if you'd rather keep absolute references to the original
`sim_exports/usd/...` location (faster, no copy, but only works on the same
filesystem). The composer is idempotent — re-running clears the existing
`/World/Articulated` subtree before composing.

After composition, IsaacSim sees doors at their correct in-scene poses with full
articulation.

Download IsaacSim from [NVIDIA Omniverse](https://developer.nvidia.com/isaac/sim) and set up an IsaacSim conda environment by running the following commands in your IsaacSim Directory (typically ` ~/.local/share/ov/pkg/isaac_sim-2023.1.1`) 

```bash
conda env create -f environment.yml
conda activate isaac-sim
source setup_conda_env.sh
```

Import scene and run a simulation

```bash
python {PATH_TO/isaac_sim.py} --scene-path outputs/my_export/export_scene.blend/export_scene.usdc --json-path outputs/my_export/export_scene.blend/solve_state.json 
```

:warning: Physical properties are applied based on object relations specified in `solve_state.json`. Scenes can be imported without a `solve_state.json`, but all objects will be static colliders. 