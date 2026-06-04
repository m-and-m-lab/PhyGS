# Scripts Directory

This directory contains standalone launchers, asset-generation utilities, Isaac Sim helpers, and the benchmark skills package.

## Main Entrypoints

| File or directory | Purpose |
| --- | --- |
| `interactive_search.py` | General IsaacLab interactive manipulation launcher with Spot, cameras, AO-Grasp, and CuRobo. |
| `interactive_search_scene_robot_test.py` | Scene and robot load test without manipulation logic. |
| `spot_locomotion_policy.py` | Spot locomotion policy test and development script. |
| `spot_walk_in_scene.py` | Scene walking experiment script. |
| `load_usd_asset_standalone.py` | Standalone USD loading utility. |
| `skills/` | Reusable robot skills, SDK facade, and IsaacLab backends. |
| `generate/` | Infinigen and asset-generation wrappers. |
| `blender/` | Blender-side conversion and asset-export helpers. |
| `isaacsim/` | Isaac Sim exploratory scripts for USD and articulation loading. |
| `helpers/` | Camera, point-cloud, USD sorting, and visualization helpers. |
| `docs/` | Scripts used to generate markdown asset catalogs. |

## Recommended Launch Style

Launch IsaacLab scripts from the Isaac Lab root:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/interactive-search/scripts/interactive_search.py --enable_cameras
```

Keep benchmark configuration in YAML when possible. Use CLI arguments for temporary debugging only.

## Development Rules

- Keep standalone scripts thin and move reusable behavior into `scripts/skills`.
- Validate asset paths before launching long simulation loops.
- Prefer dataclass-backed config loading for new smoke tests.
- Do not add hidden fallbacks for missing cameras, target prims, or planners.
- If a script is a benchmark, make it exit non-zero on failed prerequisites or failed task execution.
