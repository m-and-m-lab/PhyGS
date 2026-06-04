# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Compose articulated assets into an exported scene USD.

After `python -m infinigen.tools.export ... --omniverse` writes the whole-house USD,
it also drops `articulated_assets.json` next to it. That JSON lists every articulated
object (doors today; cabinets, fridges, etc. later) with its final world pose and the
absolute path to its standalone articulated USD twin (joints, drives, axes — written
separately by the factory during scene generation).

This script reads that JSON, opens the scene USD, and for each entry creates an
`Xform` prim under `/World/Articulated/<kind>_<idx>` at the recorded world pose with
a USD `references` arc pointing at the per-asset USD. The scene exporter has already
removed the static geometry of those objects, so the reference is the sole source of
their appearance and articulation in the final USD.

By default the script also **bundles** each per-asset USD (and its `assets/` folder of
baked textures) into `<scene_usd_dir>/articulated_assets/<kind>_<idx>/` and uses a
relative reference path. This makes the scene self-contained and portable — you can
copy the entire scene-USD directory to another machine (e.g. an IsaacSim container)
without losing references. Pass `--no-bundle` to keep absolute paths to the original
`sim_exports/usd/<kind>/<idx>/` location instead.

Re-running the script is idempotent: any existing `/World/Articulated` subtree is
cleared before composition.

Usage:
    python scripts/usd_articulated_scene_composer.py \\
        --scene-usd outputs/my_export/export_scene.blend/export_scene.usdc \\
        --articulated-json outputs/my_export/export_scene.blend/articulated_assets.json
"""

import argparse
import json
import shutil
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom


def _blender_matrix_to_usd(world_matrix_row_major):
    """Blender's `matrix_world` uses column-vector convention (`world = M @ local`);
    USD's `Gf.Matrix4d` for `AddTransformOp` uses row-vector convention
    (`world = local * U`). The two are related by transpose: `U = M^T`.

    Input is a 16-element row-major flattening of Blender's M (i.e. `M[i][j]` at
    index `4*i + j`). We re-pack as USD row-major where USD row i = Blender column i,
    i.e. `flat_usd[4*i + j] = M[j][i]`.
    """
    m = world_matrix_row_major
    flat_usd = [m[4 * j + i] for i in range(4) for j in range(4)]
    return Gf.Matrix4d(*flat_usd)


def _bundle_asset(src_usd: Path, dest_dir: Path) -> str:
    """Copy the per-asset directory tree (USD + assets/ + sidecar metadata) into
    `dest_dir`. Returns the basename of the copied USD so the caller can build a
    relative reference path. Overwrites any existing bundle."""
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(src_usd.parent, dest_dir)
    return src_usd.name


def _clear_articulated_subtree(stage: Usd.Stage):
    """Remove a previously composed `/World/Articulated` subtree so re-runs of the
    composer don't accumulate stale Xforms or references."""
    art_path = Sdf.Path("/World/Articulated")
    if stage.GetPrimAtPath(art_path).IsValid():
        stage.RemovePrim(art_path)


def compose(
    scene_usd: Path,
    articulated_json: Path,
    output_usd: Path | None = None,
    bundle: bool = True,
):
    records = json.loads(articulated_json.read_text())
    if not records:
        print(f"no articulated assets in {articulated_json}; nothing to compose")
        return

    stage = Usd.Stage.Open(str(scene_usd))
    if stage is None:
        raise RuntimeError(f"failed to open {scene_usd}")

    _clear_articulated_subtree(stage)
    UsdGeom.Xform.Define(stage, Sdf.Path("/World/Articulated"))

    bundle_root = scene_usd.parent / "articulated_assets" if bundle else None
    if bundle_root is not None:
        bundle_root.mkdir(exist_ok=True)

    n_bundled = 0
    n_missing = 0
    for r in records:
        kind = r["kind"]
        idx = r.get("idx")
        suffix = f"{kind}_{idx}" if idx is not None else kind
        prim_path = Sdf.Path(f"/World/Articulated/{suffix}")
        xform = UsdGeom.Xform.Define(stage, prim_path)
        xform.AddTransformOp().Set(_blender_matrix_to_usd(r["world_matrix"]))

        src_asset = Path(r["asset_usd_path"])
        if not src_asset.is_file():
            print(
                f"WARNING: missing source asset for {suffix}: {src_asset} "
                "(reference will be unresolved at load time)"
            )
            n_missing += 1
            xform.GetPrim().GetReferences().AddReference(str(src_asset))
            continue

        if bundle:
            dest_dir = bundle_root / suffix
            usd_name = _bundle_asset(src_asset, dest_dir)
            ref_path = f"./articulated_assets/{suffix}/{usd_name}"
            n_bundled += 1
        else:
            ref_path = str(src_asset)
        xform.GetPrim().GetReferences().AddReference(ref_path)

    target = output_usd if output_usd is not None else scene_usd
    if target == scene_usd:
        stage.Save()
    else:
        stage.GetRootLayer().Export(str(target))

    msg = f"composed {len(records)} articulated assets into {target}"
    if bundle:
        msg += f" ({n_bundled} bundled into {bundle_root})"
    if n_missing:
        msg += f" ({n_missing} source asset(s) missing — references will be broken)"
    print(msg)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--scene-usd", type=Path, required=True)
    p.add_argument("--articulated-json", type=Path, required=True)
    p.add_argument(
        "--output-usd",
        type=Path,
        default=None,
        help="defaults to in-place save of --scene-usd",
    )
    p.add_argument(
        "--no-bundle",
        action="store_true",
        help="skip copying per-asset USDs into the scene directory; use absolute "
        "references to the original sim_exports/usd/... paths instead. The scene "
        "USD will only resolve on machines where those paths are accessible.",
    )
    args = p.parse_args()
    compose(
        args.scene_usd,
        args.articulated_json,
        args.output_usd,
        bundle=not args.no_bundle,
    )


if __name__ == "__main__":
    main()
