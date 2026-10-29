# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the robot tests' navigation courses, on no floor of their own (the episode adds Isaac Lab's
ground plane):

- ``nav_course/scene.usda``: a wall to walk around and a closed pen whose inside no route reaches
  (the navigation rung's goals, ``phygs_robot_tests/configs/navigation.yaml``).
- ``dynamic_course/scene.usda``: a far wall, a movable box and a room with a ceiling around them
  (the obstacle-avoidance rung). Nav2's voxel layer clears only along rays that return, so the room
  gives every ray a surface.

    ./isaaclab.sh -p scripts/scene_generation/usd_tools/build_test_course_scenes.py

Every piece is a kinematic rigid body under ``/World/objects/<name>`` with one box collider, so the
navigation map sees it and a robot test can move it (the movable box is moved mid-episode).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_grid_storage_scene import ROOT, box_body, storage_units  # noqa: E402
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

WALL, OBSTACLE, ROOM = (0.7, 0.25, 0.1), (0.9, 0.15, 0.05), (0.6, 0.6, 0.62)  # Colors.
# name -> (center [m], size [m], color).
NAV_COURSE = dict(
    wall=((1.6, 0.0, 0.75), (0.3, 1.2, 1.5), WALL),
    # A pen around (3.8, -2.6) with a 2 m square inside and no way in. The padded standing footprint
    # (1.3 x 0.62 m, 0.72 m to a corner) fits inside at any heading, so a goal there is valid but
    # unreachable; in a 1.4 m pen the 5 cm map left no free cell and the goal was invalid (2026-10-28).
    pen_north=((3.8, -1.55, 0.5), (2.2, 0.1, 1.0), WALL),
    pen_south=((3.8, -3.65, 0.5), (2.2, 0.1, 1.0), WALL),
    pen_west=((2.75, -2.6, 0.5), (0.1, 2.0, 1.0), WALL),
    pen_east=((4.85, -2.6, 0.5), (0.1, 2.0, 1.0), WALL),
)


def room(bounds: tuple[float, float, float, float], height: float) -> dict:
    """Four walls just outside ``bounds`` [m] (xmin, ymin, xmax, ymax) and a ceiling ``height`` [m] up."""
    xmin, ymin, xmax, ymax = bounds
    x, y, w, d = (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin + 1.0, ymax - ymin + 1.0
    return dict(
        ceiling=((x, y, height + 0.05), (w, d, 0.1), ROOM),
        room_west=((xmin - 0.45, y, height / 2), (0.1, d, height), ROOM),
        room_east=((xmax + 0.45, y, height / 2), (0.1, d, height), ROOM),
        room_south=((x, ymin - 0.45, height / 2), (w, 0.1, height), ROOM),
        room_north=((x, ymax + 0.45, height / 2), (w, 0.1, height), ROOM),
    )


DYNAMIC_COURSE = dict(
    wall=((5.0, 0.0, 1.5), (0.2, 5.0, 3.0), WALL),
    # Parked out of the way; the test moves it into the route (0.4 x 1.2 x 1.2 m, as before 2026-10-28).
    obstacle=((1.5, 3.3, 0.6), (0.4, 1.2, 1.2), OBSTACLE),
    **room((-3.0, -4.0, 6.0, 4.0), 2.8),
)


def build(name: str, pieces: dict) -> None:
    output = ROOT / ".scene_usd" / name / "scene.usda"
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    Sdf.CreatePrimInLayer(stage.GetRootLayer(), storage_units.OBJECTS).specifier = Sdf.SpecifierDef
    material = UsdShade.Material.Define(stage, "/World/Looks/rubber")
    friction = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    friction.CreateStaticFrictionAttr(1.0)
    friction.CreateDynamicFrictionAttr(1.0)
    for piece, (center, size, color) in pieces.items():
        box_body(stage, material, piece, center, size, color)
    stage.GetRootLayer().Save()
    print(f"wrote {output}")


def main() -> None:
    build("nav_course", NAV_COURSE)
    build("dynamic_course", DYNAMIC_COURSE)


if __name__ == "__main__":
    main()
