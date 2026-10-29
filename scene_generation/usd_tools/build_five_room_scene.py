#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a five-room house with reference meshes and independently removable contents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from build_detailed_three_room_scene import SCENES, DetailedHouse
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

REFERENCE = SCENES / "indoor_2_highres/indoor_2/export_scene.blend/export_scene_sorted_physics_easy.usdc"
OUTPUT = SCENES / "five_room_house/scene.usdc"
CUP = "CupFactory_9054484__spawn_asset_3849534_"
BOWL = "BowlFactory_974736__spawn_asset_549778_"
BOTTLE = "BottleFactory_1021896__spawn_asset_7412270_"
TIN = "CanFactory_1526963__spawn_asset_5470950_"
BAG = "FoodBagFactory_4033944__spawn_asset_3424189_"
LAMP = "DeskLampFactory_1556182__spawn_asset_7862839_"
CHAIR = "ChairFactory_3088235__spawn_asset_1641463_"


class FiveRoomHouse(DetailedHouse):
    """Reuse tested storage construction while replacing simplified visible objects."""

    def __init__(self, output: Path, reference: Path):
        protected = (reference, SCENES / "three_room_house/scene.usdc", SCENES / "three_room_house_detailed/scene.usdc")
        if output.resolve() in [path.resolve() for path in protected]:
            raise ValueError("Choose a new output path; existing scenes and references are read-only inputs.")
        super().__init__(output, reference)
        self.meshes: dict[str, str] = {}
        for prim in self.reference.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                name = prim.GetName()
                if name in self.meshes:
                    raise ValueError(f"Reference mesh name is ambiguous: {name}")
                self.meshes[name] = str(prim.GetPath())
        self.stored_items: list[str] = []

    def visual(self, source: str, destination: str, center: tuple, size: tuple, turn: bool = False) -> UsdGeom.Mesh:
        """Copy mesh appearance at center/dimensions [m], without inherited source physics."""
        source = self.meshes[source.split("/")[-1]]
        mesh = super().visual(source, destination, center, size, turn)
        for prim in Usd.PrimRange(mesh.GetPrim()):
            schemas = [name for name in prim.GetAppliedSchemas() if not name.startswith(("Physics", "Physx"))]
            prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(schemas))
            for prop in list(prim.GetProperties()):
                if prop.GetName().startswith(("physics:", "physx")):
                    prim.RemoveProperty(prop.GetName())
        mesh.GetPrim().SetCustomDataByKey("reference_mesh", source)
        return mesh

    def mesh(self, asset: str, destination: str, center: tuple, size: tuple, turn: bool = False) -> UsdGeom.Mesh:
        """Place a named reference mesh with center and dimensions [m]."""
        return self.visual(self.meshes[asset], destination, center, size, turn)

    def pose(self, name: str, position: tuple | None = None, yaw: float | None = None) -> None:
        """Set a free body's position [m] and yaw [deg]."""
        prim = self.stage.GetPrimAtPath(f"/World/objects/{name}")
        if position is not None:
            prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*position))
        if yaw is not None:
            prim.GetAttribute("xformOp:rotateZ").Set(yaw)

    def replace_visual(self, name: str, asset: str, center: tuple, size: tuple, turn: bool = False) -> None:
        """Replace a body's visible surfaces; preserve its explicit collision panels [m]."""
        body = self.stage.GetPrimAtPath(f"/World/objects/{name}")
        for child in body.GetChildren():
            if child.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(child).CreateVisibilityAttr("invisible")
        self.mesh(asset, str(body.GetPath()) + "/reference_visual", center, size, turn)

    def item(
        self,
        name: str,
        asset: str,
        x: float,
        y: float,
        support: float,
        size: tuple,
        mass: float,
        yaw: float = 0,
        approximation: str = "sdf",
    ) -> str:
        """Place a separate loose body on a support [m], with mass [kg] and yaw [deg]."""
        if asset == BAG:
            approximation = "box"
        path = self.prop(name, asset, (x, y, support + size[2] / 2 + 0.002), size, mass, approximation)
        self.pose(name, yaw=yaw)
        if asset == BAG:
            # Choose a true support face nearest a side-lying pose, assuming a filled rigid packet.
            body = self.stage.GetPrimAtPath(path)
            points = np.asarray(UsdGeom.Mesh(self.stage.GetPrimAtPath(path + "/visual")).GetPointsAttr().Get())
            hull = trimesh.convex.convex_hull(points)
            poses, _ = hull.compute_stable_poses()
            intended = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            pose = max(poses, key=lambda candidate: np.trace(candidate[:3, :3] @ intended.T))
            UsdGeom.Xformable(body).AddTransformOp().Set(Gf.Matrix4d(pose.T.tolist()))
            body.GetAttribute("xformOp:translate").Set(Gf.Vec3d(x, y, support + 0.002))
        self.stage.GetPrimAtPath(path).SetCustomDataByKey("independently_pickable", True)
        return path

    def room_door(self, name: str, pivot: tuple, yaw: float) -> None:
        """Create a paneled door and passive hinge at a world pivot [m], yaw [deg]."""
        rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw)
        center = Gf.Vec3d(*pivot) + rotation.TransformDir(Gf.Vec3d(0.45, 0, 0))
        path = self.body(name, tuple(center), 18, yaw=yaw)
        self.box(path + "/panel", (0, 0, 0), (0.9, 0.04, 2.05), "cream")
        self.box(path + "/handle", (0.32, -0.055, 0), (0.12, 0.07, 0.025), "metal")
        self.joint(name + "_hinge", path, pivot, (-0.45, 0, 0), yaw=yaw)

    def architecture(self) -> None:
        """Extend the house to 12 × 7 m, adding a connected study and bathroom."""
        for name in ("east", "skirting_east"):
            self.stage.RemovePrim(f"/World/structure/{name}")
        for name, center, size, material in (
            ("extension_foundation", (10.1, 3.5, -0.11), (4, 7.2, 0.2), "plaster"),
            ("study_floor", (10, 1.75, -0.005), (4, 3.5, 0.01), "floor"),
            ("bathroom_floor", (10, 5.25, -0.005), (4, 3.5, 0.01), "tile"),
            ("extension_east", (12.08, 3.5, 1.35), (0.16, 7.3, 2.7), "plaster"),
            ("extension_south", (10, -0.08, 1.35), (4, 0.16, 2.7), "plaster"),
            ("extension_north", (10, 7.08, 1.35), (4, 0.16, 2.7), "plaster"),
            ("study_entry_south", (8.08, 0.5, 1.35), (0.14, 1, 2.7), "plaster"),
            ("study_entry_north", (8.08, 4.5, 1.35), (0.14, 5, 2.7), "plaster"),
            ("study_entry_lintel", (8.08, 1.5, 2.44), (0.14, 1, 0.52), "plaster"),
            ("bath_partition_left", (8.5, 3.5, 1.35), (1, 0.14, 2.7), "plaster"),
            ("bath_partition_right", (11, 3.5, 1.35), (2, 0.14, 2.7), "plaster"),
            ("bath_entry_lintel", (9.5, 3.5, 2.44), (1, 0.14, 0.52), "plaster"),
        ):
            prim = self.box(f"/World/structure/{name}", center, size, material)
            if material == "floor":
                self.textured_box(prim, self.materials["oak"])
        self.room_door("study_door", (8.08, 1.03, 1.045), 90)
        self.room_door("bathroom_door", (9.03, 3.5, 1.045), 0)
        for room, center in (("study", (10, 1.75, 2.6)), ("bathroom", (10, 5.25, 2.6))):
            light = UsdLux.RectLight.Define(self.stage, f"/World/lights/{room}")
            light.AddTranslateOp().Set(Gf.Vec3d(*center))
            light.CreateWidthAttr(1.5)
            light.CreateHeightAttr(1.5)
            light.CreateIntensityAttr(220)
            light.CreateEnableColorTemperatureAttr(True)
            light.CreateColorTemperatureAttr(4300)
        # Thin visual seams do not fragment the floor's contact surface.
        for i, y in enumerate(np.arange(0.2, 3.5, 0.2)):
            seam = self.box(f"/World/structure/plank_seam_{i}", (6, float(y), 0.0001), (12, 0.001, 0.0002), "oak")
            seam.RemoveAPI(UsdPhysics.CollisionAPI)
        for i, x in enumerate(np.arange(8.5, 12, 0.5)):
            seam = self.box(
                f"/World/structure/bath_grout_x_{i}", (float(x), 5.25, 0.0001), (0.002, 3.5, 0.0002), "cream"
            )
            seam.RemoveAPI(UsdPhysics.CollisionAPI)
        for i, y in enumerate(np.arange(4, 7, 0.5)):
            seam = self.box(f"/World/structure/bath_grout_y_{i}", (10, float(y), 0.0001), (4, 0.002, 0.0002), "cream")
            seam.RemoveAPI(UsdPhysics.CollisionAPI)

    def furnishings(self) -> None:
        """Use recognizable reference furniture and preserve open circulation paths."""
        for child in self.stage.GetPrimAtPath("/World/objects/sink_unit").GetChildren():
            if child.GetName() != "plinth":
                UsdGeom.Imageable(child).MakeInvisible()
        self.mesh(
            "SinkFactory_2296784__spawn_asset_2418135_",
            "/World/objects/sink_unit/basin_visual",
            (0, 0, 0.87),
            (0.86, 0.58, 0.16),
            True,
        )
        self.mesh(
            "TapFactory_2296784__spawn_asset_2418135_",
            "/World/objects/sink_unit/tap_visual",
            (0, 0.22, 1.075),
            (0.16, 0.20, 0.27),
            True,
        )
        plant = self.body("living_plant", (0.48, 2.5, 0.002), 8)
        pot = UsdGeom.Cylinder.Define(self.stage, plant + "/pot_collider")
        pot.CreateRadiusAttr(0.19)
        pot.CreateHeightAttr(0.28)
        pot.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.14))
        UsdPhysics.CollisionAPI.Apply(pot.GetPrim())
        pot.MakeInvisible()
        UsdShade.MaterialBindingAPI.Apply(pot.GetPrim()).Bind(self.materials["ceramic"], materialPurpose="physics")
        foliage = self.box(plant + "/foliage_collider", (0, 0, 0.68), (0.18, 0.28, 0.80), "linen")
        UsdGeom.Imageable(foliage).MakeInvisible()
        self.mesh(
            "LargePlantContainerFactory_9370618__spawn_asset_316801__001_SPL",
            plant + "/pot",
            (0, 0, 0.14),
            (0.38, 0.38, 0.28),
        )
        self.mesh(
            "LargePlantContainerFactory_9370618__spawn_asset_316801_",
            plant + "/foliage",
            (0, 0, 0.68),
            (0.18, 0.28, 0.80),
        )
        self.replace_visual(
            "dining_table", "TableDiningFactory_5760614__spawn_asset_5974757_", (0, 0, 0.375), (1.25, 0.75, 0.75)
        )
        for i, yaw in enumerate((-8, 12)):
            name = f"dining_stool_{i}"
            self.replace_visual(name, CHAIR, (0, 0, 0.47), (0.42, 0.42, 0.94))
            self.box(f"/World/objects/{name}/back_support", (0, -0.18, 0.70), (0.36, 0.05, 0.45), "oak")
            UsdGeom.Imageable(self.stage.GetPrimAtPath(f"/World/objects/{name}/back_support")).MakeInvisible()
            self.pose(name, yaw=yaw)
        self.replace_visual("bed", "BedFactory_1349120__spawn_asset_3701424_", (0, 0, 0.56), (1.70, 2.20, 1.12), True)
        self.mesh(
            "MattressFactory_1349120__spawn_asset_3701424_",
            "/World/objects/bed/mattress_visual",
            (0, 0, 0.48),
            (1.6, 2.05, 0.22),
            True,
        )
        self.mesh(
            "ComforterFactory_1349120__spawn_asset_3701424_",
            "/World/objects/bed/comforter",
            (0, -0.22, 0.48),
            (1.68, 1.62, 0.30),
        )
        self.replace_visual(
            "television", "TVFactory_3451393__spawn_asset_3406142_", (0, 0, 0.36), (0.95, 0.25, 0.72), True
        )
        self.replace_visual(
            "coffee_table", "SideTableFactory_7091040__spawn_asset_1097056_", (0, 0, 0.215), (1.25, 0.65, 0.43)
        )
        # The office desk replaces the living room's conspicuous row of books.
        for name in ("reading_desk", "reading_stool", "living_art", *(f"desk_book_{i}" for i in range(6))):
            self.stage.RemovePrim(f"/World/objects/{name}")
        self.stage.RemovePrim("/World/joints/living_art_mount")
        self.table("study_desk", 10.7, 0.6, 1.65, 0.75, 0.76)
        self.table("study_chair", 10.5, 1.55, 0.44, 0.44, 0.45)
        self.replace_visual("study_chair", CHAIR, (0, 0, 0.47), (0.44, 0.44, 0.94))
        self.pose("study_chair", yaw=167)
        self.cabinet("study_drawers", 11.35, 2.6, drawer=True)
        self.cabinet("bathroom_cabinet", 10.25, 6.65)
        self.item("desk_lamp", LAMP, 11.22, 0.70, 0.762, (0.23, 0.23, 0.49), 1.8)
        self.item("bedside_lamp", LAMP, 5.15, 6.65, 0.94, (0.20, 0.20, 0.43), 1.5)
        self.item("study_cup", CUP, 10.28, 0.46, 0.762, (0.09, 0.09, 0.10), 0.2, 17)
        self.item("study_bottle", BOTTLE, 10.56, 0.77, 0.762, (0.065, 0.065, 0.19), 0.35, -21)
        # Rigid rug with one continuous contact surface, clear of furniture feet.
        rug = self.body("living_rug", (5.35, 1.7, 0.008), 5)
        collider = self.box(rug + "/collider", (0, 0, 0), (2.3, 2.35, 0.012), "linen")
        UsdGeom.Imageable(collider).MakeInvisible()
        self.mesh("RugFactory_153477__spawn_asset_4812438_", rug + "/visual", (0, 0, 0), (2.3, 2.35, 0.012))
        # Bathroom fittings use cavity-preserving SDF meshes, fixed to the building.
        for name, asset, position, size, turn in (
            ("bathtub", "BathtubFactory_9458014__spawn_asset_2435735_", (11.35, 5.65, 0.32), (0.84, 1.55, 0.62), False),
            ("toilet", "ToiletFactory_3350275__spawn_asset_8343411_", (8.65, 6.45, 0.49), (0.48, 0.78, 0.98), True),
            (
                "washbasin",
                "StandingSinkFactory_7768945__spawn_asset_3642553_",
                (8.65, 5.15, 0.5),
                (0.60, 0.50, 1.0),
                True,
            ),
        ):
            path = self.body(name, position, 40, anchored=True)
            mesh = self.mesh(asset, path + "/visual", (0, 0, 0), size, turn)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("sdf")
            mesh.GetPrim().AddAppliedSchema("PhysxSDFMeshCollisionAPI")
            mesh.GetPrim().CreateAttribute("physxSDFMeshCollision:sdfResolution", Sdf.ValueTypeNames.Int).Set(32)
            UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(self.materials["tile"], materialPurpose="physics")
        self.item("bathroom_bottle", BOTTLE, 10.4, 6.65, 0.94, (0.065, 0.065, 0.19), 0.3, 35)
        mirror = self.body("bathroom_mirror", (8.165, 5.15, 1.65), 3, anchored=True)
        self.mesh("MirrorFactory_7320119__spawn_asset_7037920_", mirror + "/visual", (0, 0, 0), (0.015, 0.55, 0.85))
        collider = self.box(mirror + "/collider", (0, 0, 0), (0.015, 0.55, 0.85), "metal")
        UsdGeom.Imageable(collider).MakeInvisible()

    def contents(self) -> None:
        """Replace repeated packages with varied, separately graspable household objects."""
        for name in (
            "counter_tin_0",
            "counter_tin_1",
            "counter_tin_2",
            "counter_tin_3",
            "dining_package_0",
            "dining_package_1",
            "dining_package_2",
            "kitchen_counter_box",
            "dining_book",
            "shelf_box",
            "shelf_package_0",
            "shelf_package_1",
            "shelf_package_2",
        ):
            self.stage.RemovePrim(f"/World/objects/{name}")
        for name, asset, x, y, z, size, mass, yaw in (
            ("breakfast_bowl", BOWL, 2.15, 4.79, 0.752, (0.17, 0.17, 0.12), 0.35, 14),
            ("breakfast_cup", CUP, 2.33, 5.00, 0.752, (0.09, 0.09, 0.10), 0.2, -23),
            ("breakfast_bag", BAG, 2.87, 4.96, 0.752, (0.12, 0.09, 0.18), 0.25, -17),
            ("coffee_cup", CUP, 2.88, 1.50, 0.432, (0.09, 0.09, 0.10), 0.2, 28),
            ("counter_bowl", BOWL, 2.46, 6.65, 0.94, (0.17, 0.17, 0.12), 0.35, -12),
            ("counter_tin", TIN, 2.30, 6.87, 0.94, (0.10, 0.10, 0.08), 0.3, 21),
            ("counter_bag", BAG, 1.82, 6.84, 0.94, (0.12, 0.09, 0.18), 0.3, 8),
            ("shelf_bowl", BOWL, 7.57, 2.5, 1.1975, (0.17, 0.17, 0.12), 0.35, 0),
        ):
            self.item(name, asset, x, y, z, size, mass, yaw)
        self.pose("coffee_book", (2.30, 1.32, 0.452), -17)
        # Books have individual covers/pages; no multi-book mesh is welded into a single prop.
        for name, dimensions in (("coffee_book", (0.26, 0.19, 0.04)), ("shelf_book", (0.25, 0.18, 0.06))):
            path = f"/World/objects/{name}"
            UsdGeom.Imageable(self.stage.GetPrimAtPath(path + "/shape")).MakeInvisible()
            w, d, h = dimensions
            for suffix, center, size, material in (
                ("pages", (0, 0, 0), (w - 0.008, d - 0.008, h - 0.006), "cream"),
                ("cover_top", (0, 0, h / 2 - 0.0015), (w, d, 0.003), "blue"),
                ("cover_bottom", (0, 0, -h / 2 + 0.0015), (w, d, 0.003), "blue"),
            ):
                panel = self.box(path + "/" + suffix, center, size, material)
                panel.RemoveAPI(UsdPhysics.CollisionAPI)
        for name, x in (("kitchen_drawers", 3.2), ("bedside_drawers", 5.2), ("study_drawers", 11.35)):
            y = 2.6 if name == "study_drawers" else 6.65
            for suffix in ("contents", "extra_0", "extra_1"):
                self.stage.RemovePrim(f"/World/objects/{name}_{suffix}")
            for suffix, asset, dx, dy, size, yaw in (
                ("contents", TIN, -0.19, -0.09, (0.10, 0.10, 0.08), -13),
                ("extra_0", CUP, 0.17, -0.12, (0.08, 0.08, 0.09), 29),
                ("extra_1", BOWL, 0.04, 0.08, (0.12, 0.12, 0.075), 8),
            ):
                path = self.item(f"{name}_{suffix}", asset, x + dx, y + dy, 0.5625, size, 0.2, yaw)
                self.stored_items.append(path)
        for name, x, y in (
            ("kitchen_cabinet", 2.1, 6.65),
            ("living_console", 2.5, 3.05),
            ("bathroom_cabinet", 10.25, 6.65),
        ):
            for suffix in ("contents", "tin_0", "tin_1", "lower_0", "lower_1", "lower_2"):
                self.stage.RemovePrim(f"/World/objects/{name}_{suffix}")
            for suffix, asset, dx, dy, support, size, yaw in (
                ("contents", BOWL, -0.19, -0.09, 0.465, (0.15, 0.15, 0.10), -11),
                ("tin_0", CUP, 0.16, -0.12, 0.465, (0.09, 0.09, 0.10), 34),
                ("tin_1", TIN, 0.03, 0.12, 0.465, (0.10, 0.10, 0.08), 7),
                ("lower_0", BOTTLE, -0.25, 0.05, 0.09, (0.065, 0.065, 0.19), -19),
                ("lower_1", BAG, 0.06, -0.10, 0.09, (0.12, 0.09, 0.18), 16),
                ("lower_2", TIN, 0.28, 0.07, 0.09, (0.10, 0.10, 0.08), -4),
            ):
                path = self.item(f"{name}_{suffix}", asset, x + dx, y + dy, support, size, 0.25, yaw)
                self.stored_items.append(path)

    def build(self) -> None:
        """Build and save the house, including an explicit independent-storage manifest."""
        super().build()
        self.architecture()
        self.furnishings()
        self.contents()
        # Decorative panel moldings give the existing doors depth without changing contact geometry.
        for name in ("kitchen_door", "bedroom_door", "study_door", "bathroom_door"):
            for i, z in enumerate((-0.52, 0.40)):
                for side in (-1, 1):
                    panel = self.box(
                        f"/World/objects/{name}/inset_{i}_{side + 1}",
                        (0, side * 0.023, z),
                        (0.67, 0.006, 0.65),
                        "plaster",
                    )
                    panel.RemoveAPI(UsdPhysics.CollisionAPI)
        for prim in list(self.stage.Traverse()):
            if prim.IsA(UsdGeom.Cube) and prim.GetName().startswith(("leg_", "top")):
                if UsdGeom.Imageable(prim).ComputeVisibility() != "invisible":
                    self.textured_box(prim, self.materials["oak"])
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.AddAppliedSchema("PhysxRigidBodyAPI")
                prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(16)
                prim.CreateAttribute("physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(16)
        self.stage.GetRootLayer().customLayerData = {
            "description": "Five-room house: living room, kitchen, bedroom, study, bathroom; 84 square meters.",
            "physics_model": "Independent loose props; passive joints; rigid upholstery; estimated masses/friction.",
            "reference": str(REFERENCE),
        }
        self.stage.GetRootLayer().Save()
        report = dict(
            rooms=["living_room", "kitchen", "bedroom", "study", "bathroom"],
            floor_area_m2=84,
            rigid_bodies=sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in self.stage.Traverse()),
            movable_joints=sum(
                p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint) for p in self.stage.Traverse()
            ),
            triangles=sum(
                sum(n - 2 for n in UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get())
                for p in self.stage.Traverse()
                if p.IsA(UsdGeom.Mesh)
            ),
            independent_stored_items=self.stored_items,
        )
        (self.output_dir / "asset_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    FiveRoomHouse(args.output.resolve(), args.reference.resolve()).build()
    print(f"Saved {args.output.resolve()}")
