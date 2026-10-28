#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a textured house using a curated subset of the indoor_2 visual assets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from build_three_room_scene import House

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

SCENES = Path(__file__).resolve().parents[1]
REFERENCE = SCENES / "indoor_2/indoor_2/export_scene.blend/export_scene_sorted.usdc"
OUTPUT = SCENES / "three_room_house_detailed/scene.usdc"


class DetailedHouse(House):
    """Keep render meshes separate from the furniture's solid-panel colliders."""

    def __init__(self, output: Path, reference: Path):
        if not reference.is_file():
            raise FileNotFoundError(reference)
        super().__init__(output)
        self.reference = Usd.Stage.Open(str(reference))
        self.reference_dir = reference.parent
        self.output_dir = output.parent
        self.texture_dir = output.parent / "textures"
        self.texture_dir.mkdir(exist_ok=True)
        self.copied_materials: dict[str, str] = {}
        UsdGeom.Scope.Define(self.stage, "/World/reference_materials")

    def material(self, source: str) -> UsdShade.Material:
        """Copy a material and its texture files, retaining a portable relative layout."""
        if source not in self.copied_materials:
            destination = source.replace("/World/_materials", "/World/reference_materials", 1)
            if not self.reference.GetPrimAtPath(source):
                raise ValueError(f"Missing reference material: {source}")
            Sdf.CopySpec(self.reference.GetRootLayer(), source, self.stage.GetRootLayer(), destination)
            self.copied_materials[source] = destination
            for prim in Usd.PrimRange(self.stage.GetPrimAtPath(destination)):
                for attr in prim.GetAttributes():
                    if attr.HasAuthoredConnections():
                        attr.SetConnections([p.ReplacePrefix(source, destination) for p in attr.GetConnections()])
                    value = attr.Get()
                    if isinstance(value, Sdf.AssetPath):
                        texture = (self.reference_dir / value.path).resolve()
                        if not texture.is_file():
                            raise FileNotFoundError(texture)
                        target = self.texture_dir / texture.name
                        if not target.exists():
                            shutil.copy2(texture, target)
                        attr.Set(Sdf.AssetPath(f"./textures/{texture.name}"))
                shader = UsdShade.Shader(prim)
                if shader and shader.GetIdAttr().Get() == "UsdUVTexture":
                    name = prim.GetName()
                    if name in ("NORMAL_node", "ROUGHNESS_node", "METAL_node"):
                        shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
                    if name == "NORMAL_node":
                        shader.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
                        shader.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        return UsdShade.Material(self.stage.GetPrimAtPath(self.copied_materials[source]))

    def visual(self, source: str, destination: str, center: tuple, size: tuple, turn: bool = False) -> UsdGeom.Mesh:
        """Fit a selected visual mesh to explicitly modeled center/dimensions [m]."""
        prim = self.reference.GetPrimAtPath(source)
        if not prim or not prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"Missing reference mesh: {source}")
        if UsdGeom.Xformable(prim).GetLocalTransformation() != Gf.Matrix4d(1):
            raise ValueError(f"Expected identity mesh transform: {source}")
        Sdf.CopySpec(self.reference.GetRootLayer(), source, self.stage.GetRootLayer(), destination)
        mesh = UsdGeom.Mesh(self.stage.GetPrimAtPath(destination))
        points = np.asarray(mesh.GetPointsAttr().Get()).copy()
        normals = np.asarray(mesh.GetNormalsAttr().Get()).copy()
        if turn:
            points[:, :2] = points[:, [1, 0]] * np.array([-1, 1])
            normals[:, :2] = normals[:, [1, 0]] * np.array([-1, 1])
        low, high = points.min(axis=0), points.max(axis=0)
        scale = np.asarray(size) / (high - low)
        points = (points - (low + high) / 2) * scale + np.asarray(center)
        normals /= scale
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        if np.any(lengths == 0):
            # USD generates surface normals from topology when none are authored.
            mesh.GetNormalsAttr().Clear()
        else:
            normals /= lengths
            mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
        mesh.CreateExtentAttr([Gf.Vec3f(*points.min(axis=0)), Gf.Vec3f(*points.max(axis=0))])
        mesh.CreateSubdivisionSchemeAttr("none")
        for child in Usd.PrimRange(mesh.GetPrim()):
            binding = child.GetRelationship("material:binding")
            if binding:
                targets = binding.GetTargets()
                if targets:
                    binding.SetTargets([self.material(str(targets[0])).GetPath()])
        return mesh

    def textured_box(self, collider: Usd.Prim, material: UsdShade.Material) -> None:
        """Give a box collider an explicit UV-mapped visual surface."""
        UsdGeom.Imageable(collider).CreateVisibilityAttr("invisible")
        visual = UsdGeom.Mesh.Define(self.stage, str(collider.GetPath()) + "_visual")
        transform = UsdGeom.Xformable(collider).GetLocalTransformation()
        faces = [
            [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)],
            [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)],
            [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)],
            [(-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1)],
            [(-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1)],
            [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],
        ]
        points = [transform.Transform(Gf.Vec3d(*p) * 0.5) for face in faces for p in face]
        visual.CreatePointsAttr(points)
        visual.CreateFaceVertexCountsAttr([4] * 6)
        visual.CreateFaceVertexIndicesAttr(list(range(24)))
        visual.CreateSubdivisionSchemeAttr("none")
        uv = UsdGeom.PrimvarsAPI(visual).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying")
        uv.Set([(0, 0), (1, 0), (1, 1), (0, 1)] * 6)
        UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(material)

    def prop(self, name: str, asset: str, position: tuple, size: tuple, mass: float, approximation: str = "box") -> str:
        """Add a textured dynamic prop with pose/dimensions [m] and mass [kg]."""
        path = self.body(name, position, mass)
        source = f"/World/objects/{asset}/{asset}"
        mesh = self.visual(source, path + "/visual", (0, 0, 0), size)
        if approximation in ("convexHull", "sdf"):
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(approximation)
            UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(self.materials["metal"], materialPurpose="physics")
            if approximation == "sdf":
                mesh.GetPrim().AddAppliedSchema("PhysxSDFMeshCollisionAPI")
                mesh.GetPrim().CreateAttribute("physxSDFMeshCollision:sdfResolution", Sdf.ValueTypeNames.Int).Set(32)
        elif approximation == "box":
            collider = self.box(path + "/collider", (0, 0, 0), size, "cream")
            UsdGeom.Imageable(collider).CreateVisibilityAttr("invisible")
        else:
            raise ValueError(f"Unsupported collider approximation: {approximation}")
        return path

    def build(self) -> None:
        """Populate the house with curated detailed furniture and household objects."""
        super().build()
        texture_source = Path(__file__).parent / "assets/oak_albedo.png"
        shutil.copy2(texture_source, self.texture_dir / texture_source.name)
        wood = self.materials["oak"]
        shader = UsdShade.Shader(self.stage.GetPrimAtPath(str(wood.GetPath()) + "/surface"))
        reader = UsdShade.Shader.Define(self.stage, str(wood.GetPath()) + "/uv")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        texture = UsdShade.Shader.Define(self.stage, str(wood.GetPath()) + "/grain")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("./textures/oak_albedo.png"))
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        shader.GetInput("diffuseColor").ConnectToSource(texture.ConnectableAPI(), "rgb")
        for prim in list(self.stage.Traverse()):
            if not prim.IsA(UsdGeom.Cube):
                continue
            binding = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0].GetPath().name
            if binding in ("oak", "floor"):
                self.textured_box(prim, wood)
        # Grout is visual detail; the continuous floor remains the collision surface.
        for axis, positions in (("x", np.arange(0.5, 4, 0.5)), ("y", np.arange(4, 7, 0.5))):
            for i, coordinate in enumerate(positions):
                center = (float(coordinate), 5.25, 0.0002) if axis == "x" else (2, float(coordinate), 0.0002)
                size = (0.003, 3.5, 0.0004) if axis == "x" else (4, 0.003, 0.0004)
                grout = self.box(f"/World/structure/grout_{axis}_{i}", center, size, "cream")
                grout.RemoveAPI(UsdPhysics.CollisionAPI)
        # Detailed upholstery retains the tested compound support/seat/arm collision geometry.
        for child in self.stage.GetPrimAtPath("/World/objects/sofa").GetChildren():
            UsdGeom.Imageable(child).CreateVisibilityAttr("invisible")
        sofa = "SofaFactory_3070061__spawn_asset_2987914_"
        self.visual(
            f"/World/objects/{sofa}/{sofa}",
            "/World/objects/sofa/upholstery",
            (0, 0, 0.51),
            (2.2, 0.85, 0.98),
            True,
        )
        mattress = "MattressFactory_1349120__spawn_asset_3701424_"
        UsdGeom.Imageable(self.stage.GetPrimAtPath("/World/objects/bed/mattress")).CreateVisibilityAttr("invisible")
        self.visual(
            f"/World/structure/{mattress}/{mattress}",
            "/World/objects/bed/bedding",
            (0, 0, 0.48),
            (1.6, 2.05, 0.22),
            True,
        )
        pillow = "PillowFactory_2446410__spawn_asset_5971071_"
        UsdGeom.Imageable(self.stage.GetPrimAtPath("/World/objects/pillow/shape")).CreateVisibilityAttr("invisible")
        self.visual(
            f"/World/structure/{pillow}/{pillow}",
            "/World/objects/pillow/visual",
            (0, 0, 0),
            (0.65, 0.42, 0.12),
        )
        food = "FoodBoxFactory_2301894__spawn_asset_7759540_"
        food_alt = "FoodBoxFactory_9173839__spawn_asset_5983640_"
        can = "CanFactory_8085087__spawn_asset_805320_"
        bottle = "BottleFactory_1804529__spawn_asset_134498_"
        for name in (
            "kitchen_cabinet_contents",
            "living_console_contents",
            "kitchen_drawers_contents",
            "bedside_drawers_contents",
        ):
            shape = self.stage.GetPrimAtPath(f"/World/objects/{name}/shape")
            size = tuple(shape.GetAttribute("xformOp:scale").Get())
            UsdGeom.Imageable(shape).CreateVisibilityAttr("invisible")
            self.visual(f"/World/objects/{food}/{food}", f"/World/objects/{name}/visual", (0, 0, 0), size)
        # Counter supplies, with clear space around the sink and hob.
        for i, x in enumerate((1.78, 2.0, 2.24, 2.45)):
            self.prop(f"counter_tin_{i}", can, (x, 6.85, 1.005), (0.10, 0.10, 0.13), 0.35, approximation="convexHull")
        self.prop("counter_bottle", bottle, (1.79, 6.48, 1.072), (0.085, 0.085, 0.26), 0.6, approximation="sdf")
        for i, (x, y) in enumerate(((2.15, 4.68), (2.7, 5.0), (2.95, 4.68))):
            self.prop(f"dining_package_{i}", food if i % 2 else food_alt, (x, y, 0.842), (0.10, 0.08, 0.18), 0.25)
        # Loose supplies sit directly on drawer bottoms and cabinet shelves.
        for name, x in (("kitchen_drawers", 3.2), ("bedside_drawers", 5.2)):
            for i, dx in enumerate((-0.27, 0.27)):
                self.prop(f"{name}_extra_{i}", food_alt, (x + dx, 6.65, 0.6045), (0.14, 0.10, 0.08), 0.2)
        for name, x, y in (("kitchen_cabinet", 2.1, 6.65), ("living_console", 2.5, 3.05)):
            for i, dx in enumerate((-0.28, 0.28)):
                self.prop(
                    f"{name}_tin_{i}",
                    can,
                    (x + dx, y, 0.527),
                    (0.095, 0.095, 0.12),
                    0.3,
                    approximation="convexHull",
                )
            for i, dx in enumerate((-0.27, 0, 0.27)):
                self.prop(f"{name}_lower_{i}", food, (x + dx, y, 0.182), (0.14, 0.14, 0.18), 0.4)
        for i, x in enumerate((7.30, 7.53, 7.76)):
            self.prop(f"shelf_package_{i}", food_alt, (x, 2.50, 0.1695), (0.12, 0.13, 0.18), 0.3)
        # Reading desk and storage in the otherwise empty east side of the living room.
        self.table("reading_desk", 6.4, 0.65, 1.3, 0.65, 0.75)
        self.table("reading_stool", 6.4, 1.45, 0.38, 0.38, 0.45)
        for name in ("reading_desk", "reading_stool"):
            for child in list(self.stage.GetPrimAtPath(f"/World/objects/{name}").GetChildren()):
                self.textured_box(child, wood)
        for i in range(6):
            x = 5.93 + i * 0.16
            self.loose(f"desk_book_{i}", (x, 0.62, 0.776), (0.13, 0.22, 0.045), 0.35, "blue" if i % 2 else "ceramic")
            body = f"/World/objects/desk_book_{i}"
            UsdGeom.Imageable(self.stage.GetPrimAtPath(f"{body}/shape")).CreateVisibilityAttr("invisible")
            pages = self.box(f"{body}/pages", (0, 0.002, 0), (0.125, 0.21, 0.037), "cream")
            pages.RemoveAPI(UsdPhysics.CollisionAPI)
            for side, z in (("front", 0.021), ("back", -0.021)):
                cover = self.box(f"{body}/{side}", (0, 0, z), (0.13, 0.22, 0.003), "blue" if i % 2 else "ceramic")
                cover.RemoveAPI(UsdPhysics.CollisionAPI)
        art = "WallArtFactory_4564746__spawn_asset_7694909_"
        picture = self.body("living_art", (6.4, 0.035, 1.85), 2.0, anchored=True)
        frame = self.box(picture + "/collider", (0, 0, 0), (1.1, 0.035, 0.62))
        UsdGeom.Imageable(frame).CreateVisibilityAttr("invisible")
        self.visual(f"/World/objects/{art}/{art}", picture + "/visual", (0, 0, 0), (1.1, 0.035, 0.62), True)
        # Kitchen backsplash and skirting use static architectural collision geometry.
        self.box("/World/structure/backsplash", (2.05, 6.985, 1.025), (3.3, 0.025, 0.13), "tile")
        for name, center, size in (
            ("south", (4, 0.015, 0.055), (8, 0.025, 0.11)),
            ("west", (0.015, 3.5, 0.055), (0.025, 7, 0.11)),
            ("east", (7.985, 3.5, 0.055), (0.025, 7, 0.11)),
        ):
            self.box(f"/World/structure/skirting_{name}", center, size, "cream")
        for prim in self.stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.AddAppliedSchema("PhysxRigidBodyAPI")
                prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(16)
                prim.CreateAttribute("physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(16)
        physics_scene = self.stage.GetPrimAtPath("/World/physicsScene")
        physics_scene.AddAppliedSchema("PhysxSceneAPI")
        physics_scene.CreateAttribute("physxScene:solverType", Sdf.ValueTypeNames.Token).Set("PGS")
        self.stage.GetRootLayer().customLayerData = {
            "description": "Detailed three-room house; curated indoor_2 visuals and explicit collision proxies.",
            "source": self.reference.GetRootLayer().identifier,
            "physics_model": "Rigid materials; passive hinges/slides; estimated masses/friction; no fabric dynamics.",
        }
        self.stage.GetRootLayer().Save()
        meshes = [UsdGeom.Mesh(p) for p in self.stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        report = dict(
            rigid_bodies=sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in self.stage.Traverse()),
            triangles=sum(sum(int(n) - 2 for n in m.GetFaceVertexCountsAttr().Get()) for m in meshes),
            materials=len(self.copied_materials),
            textures=len(list(self.texture_dir.glob("*.png"))),
            source_modified=False,
        )
        (self.output_dir / "asset_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    DetailedHouse(args.output.resolve(), args.reference.resolve()).build()
    print(f"Saved {args.output.resolve()}")
