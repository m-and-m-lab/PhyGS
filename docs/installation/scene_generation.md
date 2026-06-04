# Installing scene_generation (Infinigen-Indoors fork)

PhyGS scene generation is a vendored fork of Infinigen-Indoors located at `scene_generation/infinigen/`. The upstream install procedure applies directly — this guide notes only the PhyGS-specific bits and points you at the upstream guide for everything else.

> **Authoritative upstream guide:** [`scene_generation/infinigen/docs/Installation.md`](../../scene_generation/infinigen/docs/Installation.md). It covers conda dependencies for Ubuntu / Mac, the Blender-script vs. Python-module install options, the optional terrain/OpenGL/fluid feature sets, and the Docker path. Follow it for any platform- or feature-specific question.

## What's different in PhyGS

| Upstream Infinigen | PhyGS scene_generation |
| --- | --- |
| Clone `princeton-vl/infinigen` standalone. | Already vendored under `scene_generation/infinigen/` in this repo. Use that directory as your `infinigen` working copy. |
| Choose any install extra (`terrain`, `vis`, `dev`, `sim`, …). | For PhyGS pipelines that export to IsaacSim, **install the `[sim]` extra**. Other extras follow upstream guidance. |
| Generation scripts under `infinigen_examples/`. | Same — call `python -m infinigen_examples.generate_indoors …` from `scene_generation/infinigen/`. The PhyGS batch drivers (`generate_multi.sh`, `export_isaacsim_multi.sh`) live in the same directory. |

## Recommended install (conda + Python module, `[sim]` extra)

This is the path most PhyGS users want. If you need OpenGL ground truth, terrain, fluid simulation, or the Blender-UI install, read the upstream guide instead and substitute the path below.

```bash
# From the PhyGS repo root, with submodules initialized.
cd scene_generation/infinigen

# Create and activate the conda env. The exact name doesn't matter; we use
# `phygs-scene` to make it obvious which env this is.
conda create -n phygs-scene python=3.11
conda activate phygs-scene

# OS-level deps (Ubuntu/Debian/WSL example — see upstream for Mac/conda variants).
sudo apt-get install wget cmake g++ libgles2-mesa-dev libglew-dev libglfw3-dev libglm-dev zlib1g-dev

# Install Infinigen with the simulation export extra.
pip install -e ".[sim]"
```

For the `[terrain,vis]` extras, OpenGL ground truth, the Docker path, or the Blender-UI variant, see [`scene_generation/infinigen/docs/Installation.md`](../../scene_generation/infinigen/docs/Installation.md).

## Smoke test

From `scene_generation/infinigen/` with the `phygs-scene` environment active:

```bash
python -m infinigen_examples.generate_indoors \
  --seed 0 --task coarse \
  --output_folder outputs/indoors/coarse/indoor_0 \
  -g fast_solve.gin overhead.gin \
  -p compose_indoors.terrain_enabled=False
```

This drives the same code path PhyGS uses for benchmark scene authoring. Outputs land at `scene_generation/infinigen/outputs/indoors/coarse/indoor_0/`. The exported USD lives under `export_scene.blend/export_scene.usdc`, which is the path you pass to `phygs_simulation`'s `--scene_usd` argument when running the agent stack.

## Batch generation

The PhyGS-specific tmux drivers `generate_multi.sh` and `export_isaacsim_multi.sh` in `scene_generation/infinigen/` launch parallel seeds across windows. Inspect them before use — they assume a tmux session name and hardcoded seed range.

## Crossing into PhyGS simulation

Do **not** install `phygs_simulation` into the `phygs-scene` environment. The IsaacSim/IsaacLab dependencies conflict with Infinigen's pins (notably `numpy<2`, `bpy==4.2.0`, and the Torch version). Use the dedicated install in [`phygs_simulation.md`](./phygs_simulation.md), and hand scene USDs between the environments via file paths only.
