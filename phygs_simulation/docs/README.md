# Docs and Generated Asset Notes

This directory contains asset notes, generated markdown catalogs, and supporting assets used while building PhyGS simulation scenes.

> **Historical content.** The catalog markdown below was produced by Infinigen-side helpers that have moved to `scene_generation/infinigen/`. Commands referencing `scripts/generate_indoors.py` etc. are kept as historical references; run the equivalent commands from inside `scene_generation/infinigen/` instead. See [`docs/installation/scene_generation.md`](../../docs/installation/scene_generation.md) for the current scene-generation workflow.

## Files

| File | Purpose |
| --- | --- |
| `00_todo.md` | Local notes and TODO items for scene and asset work. |
| `01_indoors.md` | Indoor scene documentation. |
| `02_indoor_objects.md` | Indoor object documentation. |
| `03_articulated_objects.md` | Articulated object documentation. |
| `04_blend_to_isaac.md` | Blender to Isaac conversion notes. |
| `05_simple_cabinets.md` | Simple cabinet asset notes. |
| `06_tiny.md` | Tiny scene collection notes. |
| `07_small.md` | Small scene collection notes. |
| `08_handheld.md` | Handheld object notes. |
| `09_receps.md` | Receptacle notes. |
| `assets/` | Generated or curated assets referenced by these docs. |

## Related Scripts

Scene-generation, Blender, and Infinigen asset-catalog helpers now live in the sibling component at `scene_generation/infinigen/` (separate Python environment).

## Benchmark Use

These docs are useful for selecting deterministic assets and scenes for benchmark tasks. When a generated asset becomes part of a benchmark, move its runtime configuration into a task YAML and document:

- Asset USD path.
- Target prim path.
- Expected articulation joint names.
- Required camera visibility.
- Success metric.

## Rules

- Treat generated docs as orientation, not benchmark truth.
- Benchmark launch configs should not depend on undocumented local output paths.
- If a scene asset is required for a smoke test, document its exact path and expected target prims.
