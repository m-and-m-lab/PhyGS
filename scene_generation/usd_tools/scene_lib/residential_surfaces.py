# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real-scale photographic surface materials and chamfered architectural visuals."""

import itertools
import shutil
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

ASSETS = Path(__file__).parent.parent / "assets/residential"


def beveled_box(stage: Usd.Stage, collider: Usd.Prim, material: UsdShade.Material, tile_width: float = 1.0) -> None:
    """Create a chamfered visual within a box collider, using metric UVs [m]."""
    dimensions = np.array(collider.GetAttribute("xformOp:scale").Get(), dtype=float)
    dimensions *= UsdGeom.Cube(collider).GetSizeAttr().Get()
    center = np.array(collider.GetAttribute("xformOp:translate").Get(), dtype=float)
    radius = min(0.003, float(dimensions.min()) / 4)
    half = dimensions / 2
    vertices = []
    for sign in itertools.product((-1, 1), repeat=3):
        for axis in range(3):
            point = np.array(sign) * (half - radius)
            point[axis] = sign[axis] * half[axis]
            vertices.append(point)
    hull = trimesh.convex.convex_hull(vertices)
    points = np.asarray(hull.vertices) + center
    mesh = UsdGeom.Mesh.Define(stage, str(collider.GetPath()) + "_visual")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3] * len(hull.faces))
    mesh.CreateFaceVertexIndicesAttr(np.asarray(hull.faces).reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.repeat(hull.face_normals, 3, axis=0).astype(np.float32)))
    mesh.SetNormalsInterpolation("faceVarying")
    uv = []
    for face, normal in zip(hull.faces, hull.face_normals):
        axes = [axis for axis in range(3) if axis != int(np.argmax(np.abs(normal)))]
        uv.extend((points[face][:, axes] / tile_width).tolist())
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying").Set(uv)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    UsdGeom.Imageable(collider).MakeInvisible()


def photographic_material(stage: Usd.Stage, material: UsdShade.Material, asset: str) -> None:
    """Bind diffuse, roughness, and OpenGL normal maps with explicit color spaces."""
    root = str(material.GetPath())
    surface = UsdShade.Shader(stage.GetPrimAtPath(root + "/surface"))
    uv = UsdShade.Shader.Define(stage, root + "/metric_uv")
    uv.CreateIdAttr("UsdPrimvarReader_float2")
    uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    for channel, input_name, output, value_type in (
        ("diff", "diffuseColor", "rgb", Sdf.ValueTypeNames.Color3f),
        ("rough", "roughness", "r", Sdf.ValueTypeNames.Float),
        ("nor_gl", "normal", "rgb", Sdf.ValueTypeNames.Normal3f),
    ):
        texture = UsdShade.Shader.Define(stage, root + "/photographic_" + channel)
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f"./textures/{asset}_{channel}_2k.jpg"))
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB" if channel == "diff" else "raw")
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
        if channel == "nor_gl":
            texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
            texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        if asset == "white_plaster_02" and channel == "diff":
            texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(0.25, 0.25, 0.25, 1))
            texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(0.65, 0.65, 0.65, 0))
        surface.CreateInput(input_name, value_type).ConnectToSource(texture.ConnectableAPI(), output)


def install_materials(
    stage: Usd.Stage, materials: dict[str, UsdShade.Material], texture_dir: Path, metadata_dir: Path | None = None
) -> None:
    """Install local photographic maps; no network access occurs during scene builds."""
    for path in ASSETS.iterdir():
        if path.suffix in (".jpg", ".hdr"):
            shutil.copy2(path, texture_dir / path.name)
    for name, source in (("floor", "wood_floor"), ("oak", "oak_veneer_01"), ("plaster", "white_plaster_02")):
        for channel in ("diff", "rough", "nor_gl"):
            if not (texture_dir / f"{source}_{channel}_2k.jpg").is_file():
                raise FileNotFoundError(f"Missing photographic material map: {source}/{channel}")
        photographic_material(stage, materials[name], source)
    shutil.copy2(
        ASSETS / "sources.json",
        (metadata_dir if metadata_dir is not None else texture_dir.parent) / "material_sources.json",
    )
