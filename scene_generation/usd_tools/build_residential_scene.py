#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a 336 m² residential benchmark with SDF64, CCD, and photographic materials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from build_detailed_three_room_scene import SCENES
from build_five_room_scene import BAG, BOTTLE, BOWL, CUP, LAMP, REFERENCE, TIN, FiveRoomHouse
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt
from scene_lib.residential_surfaces import beveled_box, install_materials

OUTPUT = SCENES / "residential_house_336/scene.usdc"
ROOMS = {
    "family_lounge": (0, 0, 8, 3.5),
    "kitchen": (0, 3.5, 4, 7),
    "bedroom": (4, 3.5, 8, 7),
    "study": (8, 0, 12, 3.5),
    "bathroom": (8, 3.5, 12, 7),
    "great_room_dining": (12, 0, 24, 7),
    "hall": (0, 7, 24, 9),
    "guest_bedroom": (0, 9, 6, 14),
    "main_bedroom": (6, 9, 12, 14),
    "ensuite": (10, 11, 12, 14),
    "library": (12, 9, 18, 14),
    "pantry_utility": (18, 9, 24, 14),
}


class ResidentialHouse(FiveRoomHouse):
    """Extend the tested house with full-scale rooms, independent props, and robust contacts."""

    def __init__(self, output: Path, reference: Path):
        if output.resolve() == (SCENES / "five_room_house/scene.usdc").resolve():
            raise ValueError("The five-room scene is preserved; choose a new output path.")
        super().__init__(output, reference)
        self.connections = [
            dict(door=door, rooms=rooms, clear_width_m=0.94)
            for door, rooms in (
                ("kitchen_door", ("family_lounge", "kitchen")),
                ("bedroom_door", ("family_lounge", "bedroom")),
                ("study_door", ("family_lounge", "study")),
                ("bathroom_door", ("study", "bathroom")),
            )
        ]

    def textured_box(self, collider: Usd.Prim, material: UsdShade.Material) -> None:
        """Author chamfered visual edges and metric texture coordinates [m]."""
        beveled_box(self.stage, collider, material, 1.7 if material.GetPath().name == "floor" else 1.0)

    def prop(self, name: str, asset: str, position: tuple, size: tuple, mass: float, approximation: str = "box") -> str:
        """Author a loose prop [m, kg], using a flat-base convex profile for sealed bottles."""
        if asset not in (BOTTLE, "BottleFactory_1804529__spawn_asset_134498_"):
            return super().prop(name, asset, position, size, mass, approximation)
        path = super().prop(name, asset, position, size, mass, "convexHull")
        visual = self.stage.GetPrimAtPath(path + "/visual")
        visual.RemoveAPI(UsdPhysics.CollisionAPI)
        visual.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        # A sealed bottle has a flat supporting base, cylindrical body, shoulder, and neck.
        # Keep the authored convex proxy below PhysX's 64-vertex cooking limit.
        angles = np.arange(12) * 2 * np.pi / 12
        points = [
            (radius * size[0] * np.cos(angle), radius * size[1] * np.sin(angle), z * size[2])
            for z, radius in ((-0.5, 0.45), (-0.48, 0.5), (0.18, 0.5), (0.32, 0.18), (0.5, 0.18))
            for angle in angles
        ]
        hull = trimesh.convex.convex_hull(points)
        collider = UsdGeom.Mesh.Define(self.stage, path + "/sealed_bottle_collider")
        collider.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(hull.vertices, dtype=np.float32)))
        collider.CreateFaceVertexCountsAttr([3] * len(hull.faces))
        collider.CreateFaceVertexIndicesAttr(np.asarray(hull.faces).reshape(-1).tolist())
        collider.CreateSubdivisionSchemeAttr("none")
        collider.MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim()).CreateApproximationAttr("convexHull")
        UsdShade.MaterialBindingAPI.Apply(collider.GetPrim()).Bind(self.materials["metal"], materialPurpose="physics")
        self.stage.GetPrimAtPath(path).SetCustomDataByKey("collision_model", "Sealed bottle: flat-base convex profile.")
        return path

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
        """Use identical solid packet geometry for stable-pose and contact calculations [m, kg, deg]."""
        if asset != BAG:
            return super().item(name, asset, x, y, support, size, mass, yaw, approximation)
        path = self.prop(name, asset, (x, y, support + 0.002), size, mass, "box")
        proxy = trimesh.creation.box(extents=size)
        poses, _ = proxy.compute_stable_poses()
        intended = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        pose = max(poses, key=lambda candidate: np.trace(candidate[:3, :3] @ intended.T))
        body = self.stage.GetPrimAtPath(path)
        self.pose(name, yaw=yaw)
        UsdGeom.Xformable(body).AddTransformOp().Set(Gf.Matrix4d(pose.T.tolist()))
        body.SetCustomDataByKey("independently_pickable", True)
        body.SetCustomDataByKey("collision_model", "Filled rigid packet: box proxy; matching stable-pose geometry.")
        return path

    def clone(self, source: str, name: str, position: tuple, yaw: float = 0, anchored: bool = False) -> str:
        """Copy a furniture assembly to position [m] and yaw [deg], preserving independent bodies."""
        source_path = f"/World/objects/{source}"
        original = self.stage.GetPrimAtPath(source_path)
        if not original:
            raise ValueError(f"Missing furniture assembly: {source}")
        mass = UsdPhysics.MassAPI(original).GetMassAttr().Get()
        path = self.body(name, position, mass, anchored=anchored, yaw=yaw)
        Sdf.CopySpec(self.stage.GetRootLayer(), source_path, self.stage.GetRootLayer(), path)
        self.pose(name, position, yaw)
        return path

    def wall(self, name: str, a: tuple, b: tuple, openings: list[tuple[float, float, float, float]]) -> None:
        """Build an axis-aligned wall with explicitly sized openings [m]."""
        vertical = a[0] == b[0]
        start, stop = (a[1], b[1]) if vertical else (a[0], b[0])
        fixed = a[0] if vertical else a[1]
        pieces = []
        cursor = start
        for lo, hi, bottom, top in openings:
            if not cursor <= lo < hi <= stop or not 0 <= bottom < top <= 2.7:
                raise ValueError(f"Invalid or overlapping wall opening: {name}, {(lo, hi, bottom, top)}")
            if lo > cursor:
                pieces.append((cursor, lo, 0, 2.7))
            cursor = hi
        if cursor < stop:
            pieces.append((cursor, stop, 0, 2.7))
        for lo, hi, bottom, top in openings:
            if bottom > 0:
                pieces.append((lo, hi, 0, bottom))
            if top < 2.7:
                pieces.append((lo, hi, top, 2.7))
        for i, (lo, hi, bottom, top) in enumerate(pieces):
            center = (
                (fixed, (lo + hi) / 2, (bottom + top) / 2) if vertical else ((lo + hi) / 2, fixed, (bottom + top) / 2)
            )
            size = (0.16, hi - lo, top - bottom) if vertical else (hi - lo, 0.16, top - bottom)
            self.box(f"/World/structure/{name}_{i}", center, size, "plaster")
        for i, (lo, hi, bottom, top) in enumerate(openings):
            if bottom == 0:
                continue
            for suffix, horizontal, z, width, height in (
                ("glass", (lo + hi) / 2, (bottom + top) / 2, hi - lo - 0.06, top - bottom - 0.06),
                ("sill", (lo + hi) / 2, bottom + 0.02, hi - lo, 0.04),
                ("top", (lo + hi) / 2, top - 0.02, hi - lo, 0.04),
                ("left", lo + 0.02, (bottom + top) / 2, 0.04, top - bottom),
                ("right", hi - 0.02, (bottom + top) / 2, 0.04, top - bottom),
                ("mullion", (lo + hi) / 2, (bottom + top) / 2, 0.035, top - bottom),
            ):
                center = (fixed, horizontal, z) if vertical else (horizontal, fixed, z)
                depth = 0.012 if suffix == "glass" else 0.10
                size = (depth, width, height) if vertical else (width, depth, height)
                self.box(
                    f"/World/structure/{name}_window_{i}_{suffix}",
                    center,
                    size,
                    "glass" if suffix == "glass" else "cream",
                )

    def connected_door(self, name: str, pivot: tuple, yaw: float, rooms: tuple[str, str]) -> None:
        """Add a passive doorway at pivot [m] and record its room connectivity."""
        self.room_door(name, pivot, yaw)
        self.connections.append(dict(door=name, rooms=rooms, clear_width_m=0.94))

    def extension(self) -> None:
        """Expand the footprint from 12 × 7 m to 24 × 14 m without scaling household objects."""
        UsdGeom.Xformable(self.stage.GetPrimAtPath("/World/objects/bed/reference_visual")).AddRotateZOp().Set(180)
        for path, center_z, height in (("sofa/base", 0.21, 0.42), ("bed/frame", 0.185, 0.37)):
            collider = self.stage.GetPrimAtPath("/World/objects/" + path)
            center = collider.GetAttribute("xformOp:translate").Get()
            scale = collider.GetAttribute("xformOp:scale").Get()
            collider.GetAttribute("xformOp:translate").Set(Gf.Vec3d(center[0], center[1], center_z))
            collider.GetAttribute("xformOp:scale").Set(Gf.Vec3d(scale[0], scale[1], height))
        # Keep the study drawer's fully extended front outside the new doorway swing.
        for prim in self.stage.GetPrimAtPath("/World/objects").GetChildren():
            if prim.GetName().startswith("study_drawers"):
                position = prim.GetAttribute("xformOp:translate").Get()
                prim.GetAttribute("xformOp:translate").Set(position + Gf.Vec3d(0, 0.30, 0))
        for name in ("study_drawers_mount", "study_drawers_slide"):
            joint = UsdPhysics.Joint(self.stage.GetPrimAtPath("/World/joints/" + name))
            joint.GetLocalPos0Attr().Set(joint.GetLocalPos0Attr().Get() + Gf.Vec3f(0, 0.30, 0))
        structure = self.stage.GetPrimAtPath("/World/structure")
        for prim in list(structure.GetChildren()):
            name = prim.GetName()
            exterior = name in ("extension_north", "extension_east", "west")
            if name.startswith(("north_", "window_", "plank_seam_")) or exterior:
                self.stage.RemovePrim(prim.GetPath())
        for name, center, size, material in (
            ("east_foundation", (18, 3.5, -0.11), (12, 7.2, 0.2), "plaster"),
            ("north_foundation", (12, 10.55, -0.11), (24.2, 7.1, 0.2), "plaster"),
            ("great_floor", (18, 3.5, -0.005), (12, 7, 0.01), "floor"),
            ("hall_floor", (12, 8, -0.005), (24, 2, 0.01), "floor"),
            ("north_floor", (12, 11.5, -0.005), (24, 5, 0.01), "floor"),
            ("ensuite_floor", (11, 12.5, 0.001), (1.84, 2.84, 0.002), "tile"),
            ("ensuite_west", (10, 12.5, 1.35), (0.14, 3, 2.7), "plaster"),
            ("ceiling", (12, 7, 2.79), (24.16, 14.16, 0.18), "plaster"),
        ):
            self.box(f"/World/structure/{name}", center, size, material)
        self.wall(
            "west_exterior",
            (-0.08, 0),
            (-0.08, 14),
            [(0.6, 2.6, 1.05, 2.3), (4.3, 6.3, 1.1, 2.3), (10.7, 12.7, 1.05, 2.3)],
        )
        self.wall("east_exterior", (24.08, 0), (24.08, 14), [(1.0, 5.5, 0.8, 2.35), (10.2, 12.2, 1.1, 2.3)])
        self.wall(
            "north_exterior",
            (0, 14.08),
            (24, 14.08),
            [(1.8, 4.2, 1.1, 2.3), (7.0, 9.4, 1.1, 2.3), (13.0, 15.5, 1.1, 2.3), (20, 22, 1.1, 2.3)],
        )
        self.wall("south_extension", (12, -0.08), (24, -0.08), [(13, 16, 1.1, 2.3), (18, 19, 0, 2.18)])
        self.wall("old_north_partition", (0, 7.08), (12, 7.08), [])
        self.wall("great_west", (12.08, 0), (12.08, 7), [(1, 2, 0, 2.18)])
        self.wall("hall_south", (12, 7.08), (24, 7.08), [(15, 16, 0, 2.18), (22, 23, 0, 2.18)])
        self.wall(
            "hall_north", (0, 9), (24, 9), [(1, 2, 0, 2.18), (7, 8, 0, 2.18), (13, 14, 0, 2.18), (19, 20, 0, 2.18)]
        )
        for x in (6, 12, 18):
            self.wall(f"north_partition_{x}", (x, 9), (x, 14), [])
        self.wall("ensuite_south", (10, 11), (12, 11), [(10.4, 11.4, 0, 2.18)])
        for name, pivot, yaw, rooms in (
            ("great_room_door", (12.08, 1.03, 1.045), 90, ("study", "great_room_dining")),
            ("entrance_door", (18.03, -0.08, 1.045), 0, ("outside", "great_room_dining")),
            ("hall_west_door", (15.03, 7.08, 1.045), 0, ("great_room_dining", "hall")),
            ("hall_east_door", (22.03, 7.08, 1.045), 0, ("great_room_dining", "hall")),
            ("guest_door", (1.03, 9, 1.045), 0, ("hall", "guest_bedroom")),
            ("main_door", (7.03, 9, 1.045), 0, ("hall", "main_bedroom")),
            ("library_door", (13.03, 9, 1.045), 0, ("hall", "library")),
            ("utility_door", (19.03, 9, 1.045), 0, ("hall", "pantry_utility")),
            ("ensuite_door", (10.43, 11, 1.045), 0, ("main_bedroom", "ensuite")),
        ):
            self.connected_door(name, pivot, yaw, rooms)

    def storage(self, name: str, x: float, y: float, drawer: bool = False) -> None:
        """Add accessible storage with separate loose items on its support surfaces [m]."""
        self.cabinet(name, x, y, drawer=drawer)
        self.stage.RemovePrim(f"/World/objects/{name}_contents")
        support = 0.5625 if drawer else 0.465
        for suffix, asset, dx, dy, size, mass, yaw in (
            ("contents", TIN, -0.23, -0.09, (0.10, 0.10, 0.08), 0.3, 11),
            ("extra_cup", CUP, 0.15, -0.07, (0.09, 0.09, 0.10), 0.2, -22),
            ("extra_bowl", BOWL, 0.02, 0.13, (0.13, 0.13, 0.085), 0.3, 7),
        ):
            path = self.item(name + "_" + suffix, asset, x + dx, y + dy, support, size, mass, yaw)
            self.stored_items.append(path)
        if not drawer:
            path = self.item(name + "_lower_bottle", BOTTLE, x - 0.21, y - 0.1, 0.09, (0.065, 0.065, 0.19), 0.3, 18)
            self.stored_items.append(path)

    def chair(self, name: str, position: tuple, yaw: float) -> None:
        """Place a chair with seat, back, and leg contact surfaces [m, deg]."""
        self.clone("dining_stool_0", name, position, yaw)

    def furnish_extension(self) -> None:
        """Furnish distinct living, sleeping, working, and utility areas with usable clearances."""
        self.clone("sofa", "great_sofa", (14.4, 0.85, 0.002))
        self.clone("sofa", "great_sofa_side", (17, 2.6, 0.002), 90)
        self.clone("living_rug", "great_rug", (14.4, 2.55, 0.008))
        self.clone("coffee_table", "great_coffee_table", (14.4, 2.55, 0.016))
        self.item("great_coffee_cup", CUP, 14.72, 2.65, 0.444, (0.09, 0.09, 0.10), 0.2, 23)
        self.clone("coffee_book", "great_coffee_book", (14.17, 2.47, 0.466), -12)
        self.storage("great_media", 14.4, 6.65)
        self.clone("television", "great_television", (14.4, 6.65, 0.942))
        self.clone("living_plant", "great_plant", (12.6, 5.7, 0.002))
        self.table("formal_dining_table", 21, 3.6, 2.4, 1.1, 0.76)
        self.replace_visual(
            "formal_dining_table", "TableDiningFactory_5760614__spawn_asset_5974757_", (0, 0, 0.38), (2.4, 1.1, 0.76)
        )
        for i, (x, y, yaw) in enumerate(
            (
                (20.2, 2.60, -4),
                (21.0, 2.58, 3),
                (21.8, 2.62, -7),
                (20.2, 4.6, 177),
                (21.0, 4.64, 185),
                (21.8, 4.59, 174),
            )
        ):
            self.chair(f"dining_chair_{i}", (x, y, 0.002), yaw)
        for name, asset, x, y, size, mass, yaw in (
            ("dining_water", BOTTLE, 21.12, 3.71, (0.08, 0.08, 0.23), 0.6, 14),
            ("dining_bowl", BOWL, 20.40, 3.43, (0.18, 0.18, 0.13), 0.4, -7),
            ("dining_cup_a", CUP, 20.65, 3.28, (0.09, 0.09, 0.1), 0.2, 17),
            ("dining_cup_b", CUP, 21.69, 3.91, (0.09, 0.09, 0.1), 0.2, -31),
        ):
            self.item(name, asset, x, y, 0.762, size, mass, yaw)
        self.storage("dining_sideboard", 20.25, 6.65)
        self.storage("dining_drawers", 21.35, 6.65, drawer=True)
        for room, x in (("guest", 2.9), ("main", 8.45)):
            self.clone("bed", room + "_bed", (x, 12.2, 0.002))
            self.clone("pillow", room + "_pillow", (x, 12.83, 0.654))
            self.storage(room + "_nightstand", x - 1.55, 13.25, drawer=True)
            self.item(room + "_lamp", LAMP, x - 1.6, 13.24, 0.94, (0.20, 0.20, 0.43), 1.4)
            self.item(room + "_water", BOTTLE, x - 1.27, 13.12, 0.94, (0.065, 0.065, 0.19), 0.3, 11)
        self.clone("toilet", "ensuite_toilet", (11.35, 13.35, 0.492), anchored=True)
        self.clone("washbasin", "ensuite_sink", (10.5, 12.25, 0.502), anchored=True)
        self.clone("bathroom_mirror", "ensuite_mirror", (10.09, 12.25, 1.65), anchored=True)
        self.clone("study_desk", "library_desk", (14.25, 10.45, 0.002))
        self.chair("library_chair", (14.25, 11.5, 0.002), 173)
        self.item("library_lamp", LAMP, 14.75, 10.58, 0.762, (0.23, 0.23, 0.49), 1.8)
        self.item("library_cup", CUP, 13.77, 10.35, 0.762, (0.09, 0.09, 0.1), 0.2, -21)
        self.clone("coffee_book", "library_book", (14.17, 10.43, 0.784), 8)
        self.clone("sofa", "library_sofa", (16.9, 11.5, 0.002), 90)
        for i, x in enumerate((13.25, 14.2, 15.15, 16.10, 17.05)):
            self.clone("bookcase", f"library_bookcase_{i}", (x, 13.62, 0), yaw=180, anchored=True)
            for j, (z, dx, yaw) in enumerate(((0.6495, -0.08, -7), (1.2295, 0.12, 8))):
                self.clone("shelf_book", f"library_shelf_book_{i}_{j}", (x + dx, 13.62, z), yaw)
            self.item(f"library_shelf_tin_{i}", TIN, x + 0.19, 13.58, 0.0775, (0.1, 0.1, 0.08), 0.3, i * 13)
        self.table("utility_table", 21.15, 11.1, 2, 0.85, 0.9)
        self.clone("washbasin", "utility_sink", (23.4, 10.5, 0.5), anchored=True)
        for i, x in enumerate((19.0, 20.15, 21.3, 22.45)):
            self.storage(f"pantry_{i}", x, 13.6, drawer=i == 3)
            self.item(
                f"pantry_counter_bottle_{i}", BOTTLE, x - 0.22, 13.48, 0.94, (0.08, 0.08, 0.22), 0.45, i * 19 - 12
            )
        self.item("utility_bowl", BOWL, 20.65, 11.16, 0.902, (0.18, 0.18, 0.13), 0.4, -6)
        self.item("utility_bottle", BOTTLE, 21.62, 11.37, 0.902, (0.08, 0.08, 0.22), 0.45, 23)
        self.table("hall_console", 4.25, 7.45, 1.4, 0.42, 0.78)
        self.item("hall_bowl", BOWL, 4.55, 7.46, 0.782, (0.17, 0.17, 0.12), 0.35, 11)
        self.clone("living_plant", "hall_plant", (11, 7.55, 0.002))

    def lighting_and_surfaces(self) -> None:
        """Add photographic PBR maps, daylight, interior lighting, and a real ceiling."""
        install_materials(self.stage, self.materials, self.texture_dir)
        for prim in list(self.stage.Traverse()):
            if not prim.IsA(UsdGeom.Cube):
                continue
            visual_path = str(prim.GetPath()) + "_visual"
            if UsdGeom.Imageable(prim).ComputeVisibility() == "invisible" and not self.stage.GetPrimAtPath(visual_path):
                continue
            name = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0].GetPath().name
            material = self.materials["floor"] if "floor" in prim.GetName() and name != "tile" else self.materials[name]
            if name != "glass":
                self.textured_box(prim, material)
        sky = UsdLux.DomeLight(self.stage.GetPrimAtPath("/World/lights/sky"))
        sky.CreateTextureFileAttr(Sdf.AssetPath())
        sky.CreateIntensityAttr(350)
        sun = UsdLux.DistantLight.Define(self.stage, "/World/lights/sun")
        sun.CreateIntensityAttr(20000)
        sun.CreateAngleAttr(0.7)
        sun.AddRotateXYZOp().Set(Gf.Vec3f(-35, -25, -35))
        for name, x, y, width in (
            ("great_west", 14.6, 3.7, 2.5),
            ("dining", 21, 3.6, 2.2),
            ("hall_west", 5, 8, 1.2),
            ("hall_east", 17, 8, 1.2),
            ("guest", 3, 11.5, 1.5),
            ("main", 8.7, 11.6, 1.5),
            ("library", 15, 11.5, 1.5),
            ("utility", 21, 11.5, 1.5),
            ("ensuite", 11, 12, 0.7),
        ):
            lamp = UsdLux.RectLight.Define(self.stage, "/World/lights/" + name)
            lamp.AddTranslateOp().Set(Gf.Vec3d(x, y, 2.62))
            lamp.CreateWidthAttr(width)
            lamp.CreateHeightAttr(width)
            lamp.CreateIntensityAttr(4000)
            lamp.CreateEnableColorTemperatureAttr(True)
            lamp.CreateColorTemperatureAttr(3500 if name != "utility" else 4200)
        for prim in self.stage.Traverse():
            if prim.IsA(UsdLux.RectLight):
                lamp = UsdLux.RectLight(prim)
                lamp.CreateIntensityAttr(4000)
                lamp.CreateEnableColorTemperatureAttr(True)
                lamp.CreateColorTemperatureAttr(4500)
        roof = self.stage.GetPrimAtPath("/World/structure/ceiling")
        roof.SetCustomDataByKey("overview_hide_with_visual", "/World/structure/ceiling_visual")

    def contact_settings(self) -> None:
        """Author SDF64 and velocity-aware speculative contact generation on dynamic bodies."""
        scene = self.stage.GetPrimAtPath("/World/physicsScene")
        scene.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(240)
        constrained = {
            path
            for prim in self.stage.Traverse()
            if prim.IsA(UsdPhysics.Joint)
            for relationship in (UsdPhysics.Joint(prim).GetBody0Rel(), UsdPhysics.Joint(prim).GetBody1Rel())
            for path in relationship.GetTargets()
        }
        for prim in self.stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.AddAppliedSchema("PhysxRigidBodyAPI")
                prim.CreateAttribute("physxRigidBody:enableSpeculativeCCD", Sdf.ValueTypeNames.Bool).Set(
                    prim.GetPath() not in constrained
                )
                prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(16)
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "sdf":
                    prim.AddAppliedSchema("PhysxSDFMeshCollisionAPI")
                    prim.CreateAttribute("physxSDFMeshCollision:sdfResolution", Sdf.ValueTypeNames.Int).Set(64)
                    prim.CreateAttribute(
                        "physxSDFMeshCollision:sdfTriangleCountReductionFactor", Sdf.ValueTypeNames.Float
                    ).Set(0.1)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                prim.AddAppliedSchema("PhysxCollisionAPI")
                is_sdf = (
                    prim.HasAPI(UsdPhysics.MeshCollisionAPI)
                    and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "sdf"
                )
                # SDF64 contact tolerances validated with TGS at 240 and 500 Hz.
                # The negative rest offset permits 1 mm overlap per SDF surface.
                prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.03 if is_sdf else 0.01)
                prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(-0.001 if is_sdf else 0.0)

    def build(self) -> None:
        """Save the expanded, independently interactive house and its validation manifest."""
        super().build()
        self.extension()
        self.furnish_extension()
        self.lighting_and_surfaces()
        self.contact_settings()
        self.stage.GetRootLayer().customLayerData = dict(
            description="336 m² connected residential benchmark; physical 24 × 14 m footprint; photographic PBR.",
            physics_model="SDF64; speculative CCD; rigid materials; estimated masses/friction; passive storage joints.",
            reference=str(REFERENCE),
            physics_timestep_seconds=1 / 240,
        )
        self.stage.GetRootLayer().Save()
        report = dict(
            floor_area_m2=336,
            footprint_m=[24, 14],
            rooms=ROOMS,
            connections=self.connections,
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
            sdf_resolution=64,
            speculative_ccd=True,
            render_mode="Real-time RTX; closed ceiling; overview images hide ceiling in memory only.",
        )
        (self.output_dir / "asset_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    ResidentialHouse(args.output.resolve(), args.reference.resolve()).build()
    print(f"Saved {args.output.resolve()}")
