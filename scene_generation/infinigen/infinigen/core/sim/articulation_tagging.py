# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Tag an in-scene Blender object as articulated and export its standalone
articulated USD twin.

This is the shared backend that whole-house factories (doors, cabinets,
appliances, windows, ...) call from their `create_asset` methods. The end-to-end
flow is documented in [docs/simulation/ExportingToSimulators.md](../../../docs/simulation/ExportingToSimulators.md);
the steps below run during scene *generation*:

  1. Snapshot `bpy.data.objects` so the cleanup step can purge anything created
     during the export pipeline — the spawned `Sim<Kind>Factory` wrappers, the
     deep-copy we hand into `spawn_simready`, etc.
  2. Copy `in_scene_obj` (and any extra `bpy.types.Object` values in
     `extra_parts`) so the export pipeline's `apply_modifiers` and `butil.delete`
     don't mutate the in-scene assets.
  3. Call `export_articulated_asset(..., copy_from_existing=True)`. This routes
     through `spawn_simready` → the registered `Sim<Kind>Factory`. That sim
     factory must short-circuit on `existing_asset` (returning the copy
     directly) — otherwise it regenerates a fresh, different-seed asset and the
     articulated USD twin won't match the in-scene instance. Cabinet's
     non-trivial parts dict is also handled via `extra_parts`.
  4. Snapshot-diff cleanup of `bpy.data.objects`.
  5. Tag `in_scene_obj` with `sim_articulated_kind`, `sim_articulated_idx`, and
     `sim_articulated_usd_path` custom properties. `infinigen.tools.export.extract_articulated_assets`
     reads these during whole-house export to record world pose and remove the
     baked geometry from the scene USD.

Returns the articulated index; the per-asset USD lands at
`./sim_exports/usd/<kind>/<idx>/<kind>.usda`.
"""

from pathlib import Path
from typing import Optional

import bpy


def export_articulated_twin(
    in_scene_obj: bpy.types.Object,
    kind: str,
    *,
    extra_parts: Optional[dict] = None,
) -> int:
    """See module docstring for the contract."""
    from scripts.usd_articulated_asset_exporter import export_articulated_asset

    _objs_before = {o.as_pointer() for o in bpy.data.objects}

    copy_obj = in_scene_obj.copy()
    copy_obj.data = in_scene_obj.data.copy()

    extra_parts_copy = None
    if extra_parts:
        extra_parts_copy = {}
        for k, v in extra_parts.items():
            if isinstance(v, bpy.types.Object):
                vc = v.copy()
                if v.data is not None:
                    vc.data = v.data.copy()
                extra_parts_copy[k] = vc
            else:
                extra_parts_copy[k] = v

    articulated_idx = export_articulated_asset(
        copy_obj,
        kind,
        cabinet_params=extra_parts_copy,
        copy_from_existing=True,
    )

    for _new in [o for o in bpy.data.objects if o.as_pointer() not in _objs_before]:
        bpy.data.objects.remove(_new, do_unlink=True)

    asset_usd_path = (
        Path(f"./sim_exports/usd/{kind}") / str(articulated_idx) / f"{kind}.usda"
    ).resolve()
    in_scene_obj["sim_articulated_kind"] = kind
    in_scene_obj["sim_articulated_idx"] = int(articulated_idx)
    in_scene_obj["sim_articulated_usd_path"] = str(asset_usd_path)

    return int(articulated_idx)
