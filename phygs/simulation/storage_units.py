# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Storage units of a built house scene: in place, or copied into a check scene at another pose.

The house builder (``scene_generation/usd_tools/build_bear_loop_scene.py``, ``storage()``) authors a
unit as an anchored carcass body ``/World/objects/<unit>`` and one drawer body per tier
(``<unit>_drawer``, ``<unit>_tier_<n>_drawer``, bottom first), each with its handle
cylinder, on prismatic joints anchored to the world (``/World/joints/<unit>_slide``,
``<unit>_tier_<n>_slide``). Copying the specs keeps the geometry, masses, limits, damping and
solver iterations as built; the bodies and the world joint anchors are moved by one planar transform.
Drawer contents and materials are not copied. :func:`map_boxes` gives a navigation map's boxes
from the house's colliders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

OBJECTS = "/World/objects"
JOINTS = "/World/joints"


@dataclass(frozen=True)
class StorageUnit:
    """Paths of a copied unit; every list is bottom tier first."""

    carcass: str
    drawers: list[str]
    slides: list[str]
    handles: list[str]


def _unit_paths(has_prim, unit: str, where: str) -> tuple[str, list[str], list[str]]:
    """The carcass, drawer and slide paths of ``unit``; ``has_prim(path)`` says whether a path exists."""
    carcass = f"{OBJECTS}/{unit}"
    if not has_prim(carcass):
        raise ValueError(f"No storage unit {unit!r} in {where}")
    drawers, slides = [f"{OBJECTS}/{unit}_drawer"], [f"{JOINTS}/{unit}_slide"]
    while has_prim(f"{OBJECTS}/{unit}_tier_{len(drawers) + 1}_drawer"):
        drawers.append(f"{OBJECTS}/{unit}_tier_{len(drawers) + 1}_drawer")
        slides.append(f"{JOINTS}/{unit}_tier_{len(slides) + 1}_slide")
    if not all(has_prim(path) for path in [*drawers, *slides]):
        raise ValueError(f"{unit!r} is not a drawer unit")
    return carcass, drawers, slides


def unit_in_place(stage: Usd.Stage, unit: str) -> StorageUnit:
    """Paths of ``unit`` where the house scene on ``stage`` has it."""
    carcass, drawers, slides = _unit_paths(lambda path: stage.GetPrimAtPath(path).IsValid(), unit, "the stage")
    return StorageUnit(carcass, drawers, slides, [f"{path}/handle" for path in drawers])


def copy_unit(stage: Usd.Stage, source: str, unit: str, x: float, y: float, yaw_deg: float) -> StorageUnit:
    """Copy ``unit`` from the house layer at source; its carcass origin goes to (x, y) [m] on the floor, at yaw [deg].

    The yaw is the unit's own rotation about z: 0 keeps the house builder's frame, where
    the front faces -y.
    """
    layer = Sdf.Layer.FindOrOpen(source)
    if layer is None:
        raise FileNotFoundError(source)
    carcass, drawers, slides = _unit_paths(lambda path: bool(layer.GetPrimAtPath(path)), unit, source)
    target = stage.GetRootLayer()
    for parent in ("/World", OBJECTS, JOINTS):  # Defined, or the copies are not on the stage.
        Sdf.CreatePrimInLayer(target, parent).specifier = Sdf.SpecifierDef
    for path in (carcass, *drawers, f"{JOINTS}/{unit}_mount", *slides):
        if not Sdf.CopySpec(layer, path, target, path):
            raise RuntimeError(f"Could not copy {path}")
    origin = Gf.Vec3d(stage.GetPrimAtPath(carcass).GetAttribute("xformOp:translate").Get())
    yaw = math.radians(yaw_deg)
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])

    def moved(point) -> Gf.Vec3d:
        planar = rotation @ np.array([point[0] - origin[0], point[1] - origin[1]]) + [x, y]
        return Gf.Vec3d(float(planar[0]), float(planar[1]), float(point[2] - origin[2]))

    for path in (carcass, *drawers):
        prim = stage.GetPrimAtPath(path)
        prim.GetAttribute("xformOp:translate").Set(moved(prim.GetAttribute("xformOp:translate").Get()))
        turn = prim.GetAttribute("xformOp:rotateZ")
        turn.Set(float(turn.Get()) + yaw_deg)
    turn = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw_deg).GetQuat()
    for path in (f"{JOINTS}/{unit}_mount", *slides):
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(path))
        if joint.GetBody0Rel().GetTargets():
            raise ValueError(f"{path} is not anchored to the world")
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(moved(joint.GetLocalPos0Attr().Get())))
        joint.GetLocalRot0Attr().Set(Gf.Quatf(turn * Gf.Quatd(joint.GetLocalRot0Attr().Get())))
    return StorageUnit(carcass, drawers, slides, [f"{path}/handle" for path in drawers])


def collider_boxes(
    stage: Usd.Stage, roots: list[str], frame: np.ndarray | None = None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Axis-aligned boxes {path: (center, size)} [m] of the colliding cubes and cylinders under roots.

    In the world frame, or in ``frame`` (4x4 world pose of a body whose axes are parallel to
    the world's up to quarter turns, as the builder places them).
    """
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
    inverse = np.eye(4) if frame is None else np.linalg.inv(frame)
    boxes = {}
    for root in roots:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if (prim.IsA(UsdGeom.Cube) or prim.IsA(UsdGeom.Cylinder)) and prim.HasAPI(UsdPhysics.CollisionAPI):
                box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                low, high = np.array(box.GetMin()), np.array(box.GetMax())
                center = (inverse @ [*((low + high) / 2), 1.0])[:3]
                boxes[str(prim.GetPath())] = (center, np.abs(inverse[:3, :3]) @ (high - low))
    return boxes


def map_boxes(
    stage: Usd.Stage,
    bounds: tuple[float, float, float, float],
    height: tuple[float, float],
    exclude: tuple[str, ...] = (),
) -> list[tuple[np.ndarray, np.ndarray]]:
    """World axis-aligned boxes (center, size) [m] for a navigation map of the house.

    One box per structure collider, and one per object under ``/World/objects`` (the bounds of
    all its colliders, at their authored poses), that overlaps the xy ``bounds`` (xmin, ymin,
    xmax, ymax) and the ``height`` band (low, high) [m]. Objects named in ``exclude`` are left out.
    """
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
    groups: dict[str, list[np.ndarray]] = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World")):
        path = str(prim.GetPath())
        if path.startswith("/World/envs") or not prim.HasAPI(UsdPhysics.CollisionAPI):  # Not the robot.
            continue
        parts = path.split("/")
        key = "/".join(parts[:4]) if path.startswith(OBJECTS + "/") else path  # The object, or the collider.
        if key.rsplit("/", 1)[-1] in exclude:
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        groups.setdefault(key, []).extend([np.array(box.GetMin()), np.array(box.GetMax())])
    boxes = []
    for corners in groups.values():
        low, high = np.min(corners, axis=0), np.max(corners, axis=0)
        inside = low[0] < bounds[2] and high[0] > bounds[0] and low[1] < bounds[3] and high[1] > bounds[1]
        if inside and low[2] < height[1] and high[2] > height[0]:
            boxes.append(((low + high) / 2, high - low))
    return boxes
