# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the grid storage scene (``grid_storage/scene.usda``): a table with a target and a clutter
box, and the two-drawer corner nightstand copied from the bear-loop house, on no floor of its own
(the episode adds Isaac Lab's ground plane). The layout of the robot tests' grid episode.

    ./isaaclab.sh -p scripts/scene_generation/usd_tools/build_grid_storage_scene.py

Objects are rigid bodies under ``/World/objects/<name>``: the table and the clutter kinematic,
the target dynamic (0.2 kg); one physics material (friction 1.0) on the three. Initial states other
than the target on the table are authored by the episode (``phygs/configs/scenes/grid_storage.yaml``).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/
sys.path.insert(0, str(ROOT / "interactive-search" / "src"))

from phygs.simulation import storage_units  # noqa: E402

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

OUTPUT = ROOT / ".scene_usd" / "grid_storage" / "scene.usda"
SOURCE = ROOT / ".scene_usd" / "bear_loop_house_storage_revision" / "scene.usdc"
UNIT, FRONT, YAW_DEG = "corner_nightstand", (0.0, 2.6), 0.0  # Drawer-front centre [m]; yaw 0: front faces -y.
DRAWER_FRONT_DEPTH_M = 0.3125  # Carcass origin to the drawer front's outer face [m].
TABLE = dict(center=(0.85, 0.0, 0.2), size=(0.5, 1.0, 0.4), color=(0.55, 0.4, 0.25))
OBJECTS = dict(  # On the table top [m].
    target=dict(xy=(0.63, 0.0), size=(0.04, 0.04, 0.12), color=(0.1, 0.6, 0.2), mass_kg=0.2),
    clutter=dict(xy=(0.85, 0.25), size=(0.10, 0.10, 0.20), color=(0.8, 0.15, 0.1)),
)


def box_body(stage, material, name, center, size, color, *, mass_kg=None):
    """A rigid body at center [m] with one box collider of size [m]; kinematic without a mass."""
    body = UsdGeom.Xform.Define(stage, f"{storage_units.OBJECTS}/{name}")
    body.AddTranslateOp().Set(Gf.Vec3d(*center))
    rigid = UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    rigid.CreateKinematicEnabledAttr(mass_kg is None)
    if mass_kg is not None:
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(mass_kg)
    cube = UsdGeom.Cube.Define(stage, f"{storage_units.OBJECTS}/{name}/box")
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(OUTPUT))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    Sdf.CreatePrimInLayer(stage.GetRootLayer(), storage_units.OBJECTS).specifier = Sdf.SpecifierDef
    material = UsdShade.Material.Define(stage, "/World/Looks/rubber")
    friction = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    friction.CreateStaticFrictionAttr(1.0)
    friction.CreateDynamicFrictionAttr(1.0)
    yaw = math.radians(YAW_DEG)
    origin = (FRONT[0] - math.sin(yaw) * DRAWER_FRONT_DEPTH_M, FRONT[1] + math.cos(yaw) * DRAWER_FRONT_DEPTH_M)
    storage_units.copy_unit(stage, str(SOURCE), UNIT, *origin, YAW_DEG)
    box_body(stage, material, "table", TABLE["center"], TABLE["size"], TABLE["color"])
    top = TABLE["center"][2] + TABLE["size"][2] / 2
    for name, settings in OBJECTS.items():
        center = (*settings["xy"], top + settings["size"][2] / 2)
        box_body(stage, material, name, center, settings["size"], settings["color"], mass_kg=settings.get("mass_kg"))
    stage.GetRootLayer().Save()
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
