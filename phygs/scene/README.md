# scene

Scenes by id, and what is read from a scene's USD file.

| File | Holds |
| --- | --- |
| `registry.py` | `scene_file(id, version)`: finds a scene's USD by id (and pinned version) under `$PHYGS_SCENES`, `~/.cache/phygs/scenes`, then the built-in scenes. |
| `fetch.py` | Scene releases from the dataset: download, verify, install with a record. Command: `phygs scenes list`, `phygs scenes fetch`. |
| `reader.py` | `UsdScene`: loads a scene USD and reads its ground truth by name. |
| `parts.py`, `storage_units.py`, `articulations.py`, `assets.py` | Drawers from prismatic joints; a house's storage units; articulated furniture; registering USD rigid bodies for reset. |
| `plan.py` | A scene's floor plan: an image per floor and the world-to-pixel mapping. |
| `types.py` | Plain types a reader returns (no USD or Isaac imports). |
| `builtin/` | The built-in scenes: `grid_storage`, `nav_course`, `dynamic_course`. |

The registry, the fetcher and the types are pure Python; the reader needs Isaac Sim. Scenes are
made and released outside PhyGS: [docs/scenes.md](../../docs/scenes.md).

Up: [phygs/README.md](../README.md). Tests: `tests/test_scene_registry.py`, `tests/test_scene.py`,
`tests/test_scene_assets.py`, `tests/test_scene_plan.py`, `tests/test_parts.py`,
`tests/test_articulations.py`.
