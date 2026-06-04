# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Repair per-asset articulated USDs whose mesh topology was written incorrectly.

Background: an earlier version of `infinigen/core/sim/exporters/usd_exporter.py`
sized `faceVertexCounts` by the length of the *flattened* face-vertex-index array
instead of by the number of faces. The result is that every triangulated mesh in
those USDs reports `len(faceVertexCounts) == 3 * actual_face_count` while
`faceVertexIndices` has the correct triangulated length. IsaacSim/Hydra then
warns "numVerts and verts are inconsistent" and ignores the geometry.

This utility walks any USD (or directory tree of USDs) and rewrites
`faceVertexCounts` to the correct `[3] * (len(faceVertexIndices) // 3)` whenever
it detects the exact bug pattern: every entry is 3 and the counts/indices length
mismatch matches the bug. Other shapes are left alone with a warning.

Run after a buggy export, or after bundling buggy per-asset USDs into a scene
directory via `usd_articulated_scene_composer.py`. The new exporter writes
correct topology, so newly generated assets do not need this step.

Usage:
    # repair every door.usda under a bundled scene directory
    python scripts/repair_per_asset_usd_topology.py \\
        --root sim_exports/phygs_0507/export_scene.blend/articulated_assets

    # repair a single file
    python scripts/repair_per_asset_usd_topology.py \\
        --usd sim_exports/usd/door/1044/door.usda
"""

import argparse
from pathlib import Path

from pxr import Usd, UsdGeom, Vt


def _repair_stage(stage: Usd.Stage) -> tuple[int, int, int]:
    """Return (n_meshes_seen, n_repaired, n_unrecognized_shape)."""
    seen = 0
    repaired = 0
    skipped = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        seen += 1
        m = UsdGeom.Mesh(prim)
        fvc_attr = m.GetFaceVertexCountsAttr()
        fvi_attr = m.GetFaceVertexIndicesAttr()
        fvc = list(fvc_attr.Get() or [])
        fvi = list(fvi_attr.Get() or [])
        if not fvc or not fvi:
            continue
        if sum(fvc) == len(fvi):
            continue  # already consistent
        # Bug pattern: all-triangle mesh, counts written 3x as long as faces.
        if all(v == 3 for v in fvc) and len(fvi) % 3 == 0 and len(fvc) == len(fvi):
            n_faces = len(fvi) // 3
            fvc_attr.Set(Vt.IntArray([3] * n_faces))
            repaired += 1
        else:
            print(
                f"  unrecognized shape at {prim.GetPath()}: "
                f"len(fvc)={len(fvc)} sum(fvc)={sum(fvc)} len(fvi)={len(fvi)} "
                f"unique(fvc)={sorted(set(fvc))} — skipping"
            )
            skipped += 1
    return seen, repaired, skipped


def repair_file(usd_path: Path) -> tuple[int, int, int]:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        print(f"FAILED to open {usd_path}")
        return 0, 0, 0
    seen, repaired, skipped = _repair_stage(stage)
    if repaired:
        stage.Save()
    return seen, repaired, skipped


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--usd", type=Path, help="single USD file (.usd / .usda / .usdc) to repair"
    )
    g.add_argument(
        "--root",
        type=Path,
        help="directory tree to scan recursively for *.usd / *.usda / *.usdc",
    )
    args = p.parse_args()

    if args.usd is not None:
        targets = [args.usd]
    else:
        targets = sorted(
            list(args.root.rglob("*.usda"))
            + list(args.root.rglob("*.usdc"))
            + list(args.root.rglob("*.usd"))
        )

    if not targets:
        print(f"no USDs found under {args.root}")
        return

    total_seen = total_repaired = total_skipped = 0
    n_files_touched = 0
    for t in targets:
        seen, repaired, skipped = repair_file(t)
        total_seen += seen
        total_repaired += repaired
        total_skipped += skipped
        if repaired:
            n_files_touched += 1
            print(f"  repaired {repaired}/{seen} mesh(es) in {t}")

    print(
        f"\nscanned {len(targets)} USD(s), {n_files_touched} modified — "
        f"{total_repaired} mesh(es) repaired, {total_skipped} skipped (unrecognized), "
        f"{total_seen - total_repaired - total_skipped} already correct"
    )


if __name__ == "__main__":
    main()
