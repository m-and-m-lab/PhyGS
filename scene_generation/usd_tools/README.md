# USD scene authoring tools

Scene builders, inspection and packaging tools, copied from the lab's `.scene_usd/tools` folder on 2026-10-28 (without `legacy/` and the authoring textures in `assets/`, which the house builders need; see `assets/README.md`). The framework consumes exported scenes by id (`$PHYGS_SCENES`); it does not own their authoring code. The layout below is the lab's scene folder that these tools write into.

```text
.scene_usd/
├── tools/
│   ├── build_*_scene.py           # House builders; build_grid_storage_scene.py: the robot tests' grid storage scene
│   ├── load_usd_asset_standalone.py
│   ├── package_scene.py
│   ├── legacy/                   # Older Blender/Infinigen and catalog authoring scripts
│   ├── sort_infinigen_usd_scopes.py
│   ├── scene_lib/                # Import, materials, furnishing and physics checks
│   └── assets/                   # Authoring materials; not required by exported scenes
├── tests/
│   ├── test_package_scene.py
│   └── robot_tests/              # Runnable scene simulation checks and pytest entry points
└── bear_loop_house_revised/
│   ├── bear_loop_house_revised.zip         # Uploadable archive
│   ├── bear_loop_house_revised.zip.sha256  # Download checksum
│   ├── scene.usdc                # Stable entry point
│   ├── textures/                 # Relative USD dependencies
│   ├── README.md                 # Download/open instructions
│   ├── ATTRIBUTION.md            # Source provenance
│   ├── previews/                 # Simulator captures, outside the runtime archive
│   ├── metadata/                 # Authoring inventory and source manifests
│   ├── validation/               # Logs and reports, outside the runtime archive
│   └── docs/DEVELOPMENT.md        # Local build and validation details
```

Earlier scenes and reference asset directories remain in place. Moving the tools does not change a scene's physics or geometry.

Run from the Isaac Lab workspace root:

```bash
./isaaclab.sh -p scripts/PhyGS/scene_generation/usd_tools/build_bear_loop_scene.py

CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/PhyGS/scene_generation/usd_tools/load_usd_asset_standalone.py \
  --usd_path $PHYGS_SCENES/bear_loop_house_revised/scene.usdc \
  --device cpu --viz kit --physics_dt 0.004166666666666667 --no_render_every_step

./isaaclab.sh -p scripts/PhyGS/scene_generation/usd_tools/package_scene.py \
  --scene_dir $PHYGS_SCENES/bear_loop_house_revised \
  --output $PHYGS_SCENES/bear_loop_house_revised/bear_loop_house_revised.zip
```

The packager includes only USD's resolved dependency closure, `README.md`, `ATTRIBUTION.md` and a generated `SHA256SUMS`. It extracts and verifies its own archive before publishing the local ZIP. Missing, external and non-portable absolute dependencies cause an error. Metadata, logs, previews, builders and unused textures are excluded. Existing output archives are replaced atomically after verification.

For Hugging Face, upload the ZIP and its `.sha256` sidecar; use the scene's `README.md` as the repository description. No upload is performed by the packaging script.

Archives and checksum sidecars are stored inside each scene folder. Rebuilding an archive excludes the previous ZIP and checksum because neither is a USD dependency.
