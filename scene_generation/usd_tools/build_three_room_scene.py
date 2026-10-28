#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a self-contained, meter-scale house with passive furniture joints."""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "three_room_house/scene.usdc"


class House:
    """Author furniture from solid panels so collision geometry retains cavities."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(self.stage, 1.0)
        world = UsdGeom.Xform.Define(self.stage, "/World")
        self.stage.SetDefaultPrim(world.GetPrim())
        for group in ("structure", "objects", "joints", "materials", "lights"):
            UsdGeom.Scope.Define(self.stage, f"/World/{group}")
        scene = UsdPhysics.Scene.Define(self.stage, "/World/physicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(9.81)
        self.materials = {}
        for name, color, roughness, friction in (
            ("plaster", (0.78, 0.76, 0.70), 0.85, 0.6),
            ("oak", (0.33, 0.17, 0.075), 0.55, 0.6),
            ("floor", (0.48, 0.31, 0.17), 0.6, 0.6),
            ("tile", (0.62, 0.65, 0.62), 0.45, 0.6),
            ("sage", (0.22, 0.34, 0.29), 0.55, 0.6),
            ("linen", (0.58, 0.63, 0.68), 0.95, 0.8),
            ("cream", (0.84, 0.79, 0.64), 0.85, 0.7),
            ("metal", (0.25, 0.27, 0.29), 0.25, 0.4),
            ("ceramic", (0.75, 0.30, 0.15), 0.3, 0.5),
            ("glass", (0.60, 0.75, 0.80), 0.1, 0.4),
            ("screen", (0.012, 0.018, 0.025), 0.2, 0.4),
            ("blue", (0.08, 0.20, 0.35), 0.7, 0.6),
        ):
            material = UsdShade.Material.Define(self.stage, f"/World/materials/{name}")
            shader = UsdShade.Shader.Define(self.stage, f"{material.GetPath()}/surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.8 if name == "metal" else 0.0)
            if name == "glass":
                shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.2)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            physics.CreateStaticFrictionAttr(friction)
            physics.CreateDynamicFrictionAttr(friction * 0.8)
            physics.CreateRestitutionAttr(0.0)
            self.materials[name] = material

    def box(self, path: str, center: tuple, size: tuple, material: str = "oak") -> Usd.Prim:
        """Create a solid panel with dimensions and center [m]."""
        cube = UsdGeom.Cube.Define(self.stage, path)
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*center))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
        binding = UsdShade.MaterialBindingAPI.Apply(cube.GetPrim())
        binding.Bind(self.materials[material])
        binding.Bind(self.materials[material], materialPurpose="physics")
        return cube.GetPrim()

    def body(self, name: str, position: tuple, mass: float, anchored: bool = False, yaw: float = 0) -> str:
        """Create a dynamic furniture body at position [m] with mass [kg]."""
        path = f"/World/objects/{name}"
        body = UsdGeom.Xform.Define(self.stage, path)
        body.AddTranslateOp().Set(Gf.Vec3d(*position))
        body.AddRotateZOp().Set(yaw)
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(mass)
        if anchored:
            joint = UsdPhysics.FixedJoint.Define(self.stage, f"/World/joints/{name}_mount")
            joint.CreateBody1Rel().SetTargets([path])
            joint.CreateLocalPos0Attr(Gf.Vec3f(*position))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0))
            joint.CreateLocalRot0Attr(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw).GetQuat()))
        return path

    def joint(
        self, name: str, body: str, pivot: tuple, local_pivot: tuple, sliding: bool = False, yaw: float = 0
    ) -> None:
        """Attach a passive world joint with pivot positions [m]."""
        schema = UsdPhysics.PrismaticJoint if sliding else UsdPhysics.RevoluteJoint
        joint = schema.Define(self.stage, f"/World/joints/{name}")
        joint.CreateBody1Rel().SetTargets([body])
        joint.CreateAxisAttr("Y" if sliding else "Z")
        joint.CreateLocalPos0Attr(Gf.Vec3f(*pivot))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*local_pivot))
        joint.CreateLocalRot0Attr(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw).GetQuat()))
        joint.CreateLowerLimitAttr(0.0)
        joint.CreateUpperLimitAttr(0.38 if sliding else 100.0)
        # Viscous resistance models hinge/runner losses without a position servo.
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear" if sliding else "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(8.0 if sliding else 0.08)
        drive.CreateTargetVelocityAttr(0.0)

    def door(self, name: str, x: float) -> None:
        """Build an inward-opening room door at doorway x [m]."""
        path = self.body(name, (x + 0.48, 3.5, 1.045), 18.0)
        self.box(f"{path}/panel", (0, 0, 0), (0.9, 0.04, 2.05), "cream")
        for y in (-0.055, 0.055):
            self.box(f"{path}/handle_{'front' if y > 0 else 'back'}", (0.31, y, -0.05), (0.12, 0.07, 0.025), "metal")
        self.joint(name + "_hinge", path, (x + 0.03, 3.5, 1.045), (-0.45, 0, 0))
        for suffix, center, size in (
            ("left", (x - 0.035, 3.5, 1.07), (0.06, 0.19, 2.14)),
            ("right", (x + 0.995, 3.5, 1.07), (0.06, 0.19, 2.14)),
            ("top", (x + 0.48, 3.5, 2.14), (1.09, 0.19, 0.06)),
        ):
            self.box(f"/World/structure/{name}_trim_{suffix}", center, size, "cream")

    def cabinet(self, name: str, x: float, y: float, drawer: bool = False, yaw: float = 180) -> None:
        """Build a wall-mounted cabinet with an open cavity at x/y [m]."""
        sign = 1 if yaw == 0 else -1
        base = self.body(name, (x, y, 0), 35.0, anchored=True, yaw=yaw)
        for suffix, center, size in (
            ("left", (-0.48, 0, 0.45), (0.04, 0.6, 0.9)),
            ("right", (0.48, 0, 0.45), (0.04, 0.6, 0.9)),
            ("back", (0, -0.28, 0.45), (0.92, 0.04, 0.9)),
            ("bottom", (0, 0, 0.07), (0.92, 0.52, 0.04)),
            ("counter", (0, 0, 0.92), (1.04, 0.65, 0.04)),
        ):
            self.box(f"{base}/{suffix}", center, size, "tile" if suffix == "counter" else "sage")
        if drawer:
            moving = self.body(name + "_drawer", (x, y, 0.55), 5.0, yaw=yaw)
            for suffix, center, size in (
                ("bottom", (0, 0.035, 0), (0.86, 0.54, 0.025)),
                ("left", (-0.4175, 0.035, 0.105), (0.025, 0.54, 0.185)),
                ("right", (0.4175, 0.035, 0.105), (0.025, 0.54, 0.185)),
                ("back", (0, -0.2225, 0.105), (0.81, 0.025, 0.185)),
                ("front", (0, 0.3225, 0.12), (0.94, 0.035, 0.28)),
            ):
                self.box(f"{moving}/{suffix}", center, size, "oak")
            self.box(f"{moving}/handle", (0, 0.37, 0.12), (0.22, 0.07, 0.025), "metal")
            self.joint(name + "_slide", moving, (x, y, 0.55), (0, 0, 0), sliding=True, yaw=yaw)
            self.loose(name + "_contents", (x, y, 0.6045), (0.18, 0.12, 0.08), 0.35, "blue")
        else:
            self.box(f"{base}/shelf", (0, 0, 0.45), (0.92, 0.52, 0.03))
            moving = self.body(name + "_door", (x, y + sign * 0.325, 0.46), 4.0, yaw=yaw)
            self.box(f"{moving}/panel", (0, 0, 0), (0.94, 0.025, 0.82), "sage")
            self.box(f"{moving}/handle", (0.34, 0.04, 0.12), (0.025, 0.07, 0.16), "metal")
            self.joint(name + "_hinge", moving, (x - sign * 0.47, y + sign * 0.325, 0.46), (-0.47, 0, 0), yaw=yaw)
            self.loose(name + "_contents", (x, y, 0.527), (0.18, 0.16, 0.12), 0.5, "ceramic")

    def loose(self, name: str, position: tuple, size: tuple, mass: float, material: str) -> None:
        """Place a free object with size/position [m] and mass [kg]."""
        path = self.body(name, position, mass)
        self.box(f"{path}/shape", (0, 0, 0), size, material)

    def table(self, name: str, x: float, y: float, width: float, depth: float, height: float) -> None:
        """Create a free-standing table with dimensions and location [m]."""
        path = self.body(name, (x, y, 0.002), 18.0)
        self.box(f"{path}/top", (0, 0, height - 0.025), (width, depth, 0.05))
        for i, (dx, dy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
            center = (dx * (width / 2 - 0.07), dy * (depth / 2 - 0.07), (height - 0.05) / 2)
            self.box(f"{path}/leg_{i}", center, (0.06, 0.06, height - 0.05))

    def build(self) -> None:
        """Populate three connected rooms with furniture and supported loose objects."""
        self.box("/World/structure/foundation", (4, 3.5, -0.11), (8.2, 7.2, 0.2), "plaster")
        for name, center, size, material in (
            ("living_floor", (4, 1.75, -0.005), (8, 3.5, 0.01), "floor"),
            ("kitchen_floor", (2, 5.25, -0.005), (4, 3.5, 0.01), "tile"),
            ("bedroom_floor", (6, 5.25, -0.005), (4, 3.5, 0.01), "floor"),
            ("west", (-0.08, 3.5, 1.35), (0.16, 7.3, 2.7), "plaster"),
            ("east", (8.08, 3.5, 1.35), (0.16, 7.3, 2.7), "plaster"),
            ("south", (4, -0.08, 1.35), (8, 0.16, 2.7), "plaster"),
            ("bedroom_partition", (4, 5.25, 1.35), (0.14, 3.5, 2.7), "plaster"),
        ):
            self.box(f"/World/structure/{name}", center, size, material)
        for i, (lo, hi) in enumerate(((0, 1.2), (2.8, 5.2), (6.8, 8))):
            self.box(f"/World/structure/north_{i}", ((lo + hi) / 2, 7.08, 1.35), (hi - lo, 0.16, 2.7), "plaster")
        for i, x in enumerate((2, 6)):
            for suffix, center, size, material in (
                ("below", (x, 7.08, 0.55), (1.6, 0.16, 1.1), "plaster"),
                ("above", (x, 7.08, 2.45), (1.6, 0.16, 0.5), "plaster"),
                ("glass", (x, 7.08, 1.65), (1.52, 0.015, 1.02), "glass"),
                ("sill", (x, 7.02, 1.12), (1.7, 0.28, 0.04), "cream"),
                ("top", (x, 7.08, 2.18), (1.6, 0.07, 0.04), "cream"),
                ("left", (x - 0.78, 7.08, 1.65), (0.04, 0.07, 1.06), "cream"),
                ("right", (x + 0.78, 7.08, 1.65), (0.04, 0.07, 1.06), "cream"),
                ("mullion", (x, 7.08, 1.65), (0.035, 0.06, 1.06), "cream"),
            ):
                self.box(f"/World/structure/window_{i}_{suffix}", center, size, material)
        for i, (lo, hi) in enumerate(((0, 0.6), (1.56, 4.6), (5.56, 8))):
            self.box(f"/World/structure/partition_{i}", ((lo + hi) / 2, 3.5, 1.35), (hi - lo, 0.14, 2.7), "plaster")
        for name, x in (("kitchen_door", 0.6), ("bedroom_door", 4.6)):
            self.box(f"/World/structure/{name}_lintel", (x + 0.48, 3.5, 2.44), (0.96, 0.14, 0.52), "plaster")
            self.door(name, x)
        # Kitchen units face the clear aisle. Wall fixings are explicit constraints.
        self.cabinet("kitchen_cabinet", 2.1, 6.65)
        self.cabinet("kitchen_drawers", 3.2, 6.65, drawer=True)
        sink = self.body("sink_unit", (0.95, 6.65, 0), 30, anchored=True)
        for suffix, center, size in (
            ("plinth", (0, 0, 0.4), (0.9, 0.6, 0.8)),
            ("basin_floor", (0, 0, 0.82), (0.54, 0.4, 0.03)),
            ("left_rim", (-0.35, 0, 0.89), (0.16, 0.6, 0.12)),
            ("right_rim", (0.35, 0, 0.89), (0.16, 0.6, 0.12)),
            ("back_rim", (0, 0.25, 0.89), (0.54, 0.1, 0.12)),
            ("front_rim", (0, -0.25, 0.89), (0.54, 0.1, 0.12)),
            ("tap_stem", (0, 0.25, 1.07), (0.035, 0.035, 0.24)),
            ("tap_spout", (0, 0.16, 1.18), (0.035, 0.21, 0.035)),
        ):
            self.box(f"{sink}/{suffix}", center, size, "sage" if suffix == "plinth" else "metal")
        self.box("/World/objects/kitchen_drawers/hob", (0, 0, 0.946), (0.65, 0.45, 0.012), "screen")
        for i, (x, y) in enumerate(((-0.16, -0.11), (-0.16, 0.11), (0.16, -0.11), (0.16, 0.11))):
            self.box(f"/World/objects/kitchen_drawers/burner_{i}", (x, y, 0.955), (0.13, 0.13, 0.006), "metal")
        self.table("dining_table", 2.5, 4.85, 1.25, 0.75, 0.75)
        for i, x in enumerate((2.1, 2.9)):
            self.table(f"dining_stool_{i}", x, 4.1, 0.36, 0.36, 0.45)
        self.loose("kitchen_counter_box", (2.15, 6.65, 1.017), (0.2, 0.18, 0.15), 0.6, "cream")
        self.loose("dining_book", (2.5, 4.85, 0.772), (0.24, 0.18, 0.04), 0.4, "blue")
        # Living room seating faces a low media console; door swing areas stay clear.
        sofa = self.body("sofa", (2.4, 0.5, 0.002), 65)
        self.box(f"{sofa}/base", (0, 0, 0.22), (2.2, 0.85, 0.4), "oak")
        self.box(f"{sofa}/seat", (0, 0.04, 0.48), (1.9, 0.72, 0.12), "linen")
        self.box(f"{sofa}/back", (0, -0.34, 0.71), (2.2, 0.18, 0.58), "linen")
        for i, x in enumerate((-1.025, 1.025)):
            self.box(f"{sofa}/arm_{i}", (x, 0, 0.5), (0.15, 0.85, 0.36), "linen")
        self.table("coffee_table", 2.5, 1.4, 1.25, 0.65, 0.43)
        self.loose("coffee_book", (2.4, 1.4, 0.452), (0.26, 0.19, 0.04), 0.5, "ceramic")
        self.cabinet("living_console", 2.5, 3.05)
        tv = self.body("television", (2.5, 3.05, 0.942), 8)
        self.box(f"{tv}/foot", (0, 0, 0.015), (0.55, 0.25, 0.03), "metal")
        self.box(f"{tv}/stand", (0, 0, 0.12), (0.08, 0.08, 0.2), "metal")
        self.box(f"{tv}/display", (0, 0, 0.44), (0.95, 0.045, 0.56), "screen")
        shelf = self.body("bookcase", (7.55, 2.5, 0), 35, anchored=True)
        for i, x in enumerate((-0.38, 0.38)):
            self.box(f"{shelf}/side_{i}", (x, 0, 0.9), (0.035, 0.36, 1.8))
        self.box(f"{shelf}/back", (0, -0.1625, 0.9), (0.725, 0.035, 1.8))
        for i, z in enumerate((0.06, 0.6, 1.18, 1.78)):
            self.box(f"{shelf}/shelf_{i}", (0, 0, z), (0.725, 0.29, 0.035))
        self.loose("shelf_book", (7.5, 2.5, 0.6495), (0.25, 0.18, 0.06), 0.7, "blue")
        self.loose("shelf_box", (7.55, 2.5, 1.2795), (0.24, 0.22, 0.16), 0.6, "cream")
        # Bedroom: bed along the outer wall, bedside storage and an unobstructed entrance.
        bed = self.body("bed", (6.7, 5.65, 0.002), 75)
        self.box(f"{bed}/frame", (0, 0, 0.19), (1.65, 2.15, 0.36))
        self.box(f"{bed}/mattress", (0, 0, 0.48), (1.6, 2.05, 0.22), "cream")
        self.box(f"{bed}/headboard", (0, 1.06, 0.62), (1.7, 0.08, 1.0))
        self.loose("pillow", (6.7, 6.28, 0.654), (0.65, 0.42, 0.12), 1.0, "linen")
        self.cabinet("bedside_drawers", 5.2, 6.65, drawer=True)
        dome = UsdLux.DomeLight.Define(self.stage, "/World/lights/sky")
        dome.CreateIntensityAttr(650)
        for name, pos in (("living", (4, 1.75, 2.6)), ("kitchen", (2, 5.25, 2.6)), ("bedroom", (6, 5.25, 2.6))):
            light = UsdLux.RectLight.Define(self.stage, f"/World/lights/{name}")
            light.AddTranslateOp().Set(Gf.Vec3d(*pos))
            light.CreateWidthAttr(1.5)
            light.CreateHeightAttr(1.5)
            light.CreateIntensityAttr(350)
        self.stage.GetRootLayer().customLayerData = {
            "description": "Three-room rigid-body house; open roof for inspection; SI units; passive joints.",
            "physics_model": (
                "Rigid panels, Coulomb friction, inelastic contacts, viscous hinges/runners. No latches or soft bodies."
            ),
        }
        self.stage.GetRootLayer().Save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    House(args.output.resolve()).build()
    print(f"Saved {args.output.resolve()}")
