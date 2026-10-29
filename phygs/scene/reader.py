# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load any scene USD and read its ground truth by name.

A scene is a USD file; an episode spec names it and says where objects start. Nothing
scene-specific is listed: objects are the stage's rigid bodies, named by the last element of their
path; drawers are bodies on a prismatic joint with a ``handle`` cylinder; obstacles are the stage's
colliders; the navigation map covers the colliders (and the robot's start) with a margin; where an
object starts (inside a drawer, or on a support) is found from its authored pose.

:func:`drawers`, :func:`collider_boxes`, :func:`placement` and :func:`apply_initial_state` read a stage
only (testable without Isaac Sim). :class:`UsdScene` uses them in a running app: ``open_stage``
before the simulation context exists, ``define`` before the InteractiveScene exists, ``bind``
after the simulation reset, then ground-truth reads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

from phygs.scene.parts import sliding_part
from phygs.scene.types import ArticulatedPart

ROBOT_ROOT = "/World/envs"
PURPOSES = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide]
# House storage units (``storage()`` in build_bear_loop_scene.py), drawer frame (front -y) [m]: the
# top of the drawer's bottom panel, the drawer front's inner face and the carcass front edge.
DRAWER_FLOOR_Z_M = 0.0125
DRAWER_FRONT_INNER_Y_M = -0.2875
CARCASS_FRONT_Y_M = -0.28
SUPPORT_GAP_M = 0.03  # An object rests on the body whose top is at most this far below its bottom [m].


def pose_matrix(pose) -> np.ndarray:
    """4x4 transform from a pose [x, y, z, qx, qy, qz, qw] [m]."""
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def world_pose(stage: Usd.Stage, path: str) -> np.ndarray:
    """Authored world pose (4x4) [m] of a prim."""
    return np.array(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T


def name_of(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def rigid_bodies(stage: Usd.Stage) -> list[str]:
    """Paths of the scene's own rigid bodies (not the robot's), in stage order."""
    return [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and not str(prim.GetPath()).startswith(ROBOT_ROOT)
    ]


@dataclass(frozen=True)
class Drawer:
    """A body on a prismatic joint with a ``handle`` cylinder."""

    body: str
    slide: str
    handle: str


def drawers(stage: Usd.Stage) -> dict[str, Drawer]:
    """Every drawer of the stage by name: a prismatic joint's moving body with a ``handle`` cylinder."""
    found = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.PrismaticJoint):
            continue
        targets = UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
        if not targets:
            continue
        body = str(targets[0])
        handle = stage.GetPrimAtPath(f"{body}/handle")
        if handle.IsValid() and handle.IsA(UsdGeom.Cylinder):
            found[name_of(body)] = Drawer(body, str(prim.GetPath()), str(handle.GetPath()))
    return found


def collider_boxes(stage: Usd.Stage) -> list[tuple[str, str | None, np.ndarray, np.ndarray]]:
    """Every collider outside the robot: (collider path, owning body or None, center, size) [m].

    Centre and size are the collider's axis-aligned box in its body's authored frame (in the
    world for colliders of no body); the box moves with the body as a body-frame box.
    """
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), PURPOSES)
    bodies = set(rigid_bodies(stage))
    boxes = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(ROBOT_ROOT) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        owner = prim
        while owner.IsValid() and str(owner.GetPath()) not in bodies:
            owner = owner.GetParent()
        body = str(owner.GetPath()) if owner.IsValid() and str(owner.GetPath()) in bodies else None
        frame = np.eye(4) if body is None else world_pose(stage, body)
        bound = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        low, high = np.array(bound.GetMin()), np.array(bound.GetMax())
        if not np.all(np.isfinite([*low, *high])) or np.any(high < low):
            continue
        inverse = np.linalg.inv(frame)
        center = (inverse @ [*((low + high) / 2), 1.0])[:3]
        boxes.append((path, body, center, np.abs(inverse[:3, :3]) @ (high - low)))
    return boxes


def body_box(stage: Usd.Stage, body: str) -> tuple[np.ndarray, np.ndarray]:
    """A body's authored world box (center, size) [m], from its colliders."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), PURPOSES)
    corners = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(body)):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            bound = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            corners += [np.array(bound.GetMin()), np.array(bound.GetMax())]
    if not corners:
        raise ValueError(f"{body} has no colliders")
    low, high = np.min(corners, axis=0), np.max(corners, axis=0)
    return (low + high) / 2, high - low


def drawer_interior(stage: Usd.Stage, drawer: Drawer) -> tuple[float, ...]:
    """The drawer's inside in its frame [m], between the inner faces of its panels (children named
    left, right, front, back and bottom, as the house builder authors them) and below the side
    panels' top: (x low, x high, y low, y high, z low, z high)."""
    frame = np.linalg.inv(world_pose(stage, drawer.body))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), PURPOSES)
    panels = {}
    for name in ("left", "right", "front", "back", "bottom"):
        prim = stage.GetPrimAtPath(f"{drawer.body}/{name}")
        if not prim.IsValid():
            raise KeyError(f"{drawer.body} has no {name} panel")
        bound = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        low, high = np.array(bound.GetMin()), np.array(bound.GetMax())
        corners = np.array(
            [[x, y, z, 1.0] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]
        )
        local = (frame @ corners.T)[:3]
        panels[name] = (local.min(axis=1), local.max(axis=1))
    sides = sorted((panels["left"], panels["right"]), key=lambda box: box[0][0])
    ends = sorted((panels["front"], panels["back"]), key=lambda box: box[0][1])
    top = min(panels["left"][1][2], panels["right"][1][2])
    return sides[0][1][0], sides[1][0][0], ends[0][1][1], ends[1][0][1], panels["bottom"][1][2], top


def inside_interior(interior, local_point) -> bool:
    x_low, x_high, y_low, y_high, z_low, z_high = interior
    x, y, z = local_point
    return bool(x_low < x < x_high and y_low < y < y_high and z_low < z < z_high)


def placement(stage: Usd.Stage, body: str, found: dict[str, Drawer]) -> tuple[str | None, str | None]:
    """Where a body starts: (the drawer it is inside, or None; the body it rests on, or None)."""
    center, size = body_box(stage, body)
    for name, drawer in found.items():
        if drawer.body == body:
            continue
        local = (np.linalg.inv(world_pose(stage, drawer.body)) @ [*center, 1.0])[:3]
        if inside_interior(drawer_interior(stage, drawer), local):
            return name, None
    bottom, best = center[2] - size[2] / 2, (None, -np.inf)
    for other in rigid_bodies(stage):
        if other == body:
            continue
        other_center, other_size = body_box(stage, other) if _has_colliders(stage, other) else (None, None)
        if other_center is None:
            continue
        top = other_center[2] + other_size[2] / 2
        overlap = np.all(np.abs(center[:2] - other_center[:2]) < (size[:2] + other_size[:2]) / 2)
        if overlap and bottom - SUPPORT_GAP_M <= top <= bottom + 1e-3 and top > best[1]:
            best = (name_of(other), top)
    return None, best[0]


def _has_colliders(stage: Usd.Stage, body: str) -> bool:
    return any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in Usd.PrimRange(stage.GetPrimAtPath(body)))


def apply_initial_state(stage: Usd.Stage, bodies: dict[str, str], state: dict) -> None:
    """Move bodies to an initial state's positions [m]: ``{name: {position: [x, y, z], frame: name?}}``,
    in the world or in another body's authored frame. Only the translation is authored."""
    for name, where in state.items():
        path = bodies[name]
        position = np.asarray(where["position"], dtype=float)
        if "frame" in where:
            position = (world_pose(stage, bodies[where["frame"]]) @ [*position, 1.0])[:3]
        parent = world_pose(stage, str(stage.GetPrimAtPath(path).GetParent().GetPath()))
        local = (np.linalg.inv(parent) @ [*position, 1.0])[:3]
        attribute = stage.GetPrimAtPath(path).GetAttribute("xformOp:translate")
        if not attribute.IsValid():
            raise ValueError(f"{path} has no xformOp:translate to move")
        attribute.Set(Gf.Vec3d(*map(float, local)))


def map_bounds(
    stage: Usd.Stage, height: tuple[float, float], resolution: float, margin: float, keep: tuple[float, float]
) -> tuple[float, float, float, float]:
    """The navigation map's bounds [xmin, ymin, xmax, ymax] [m]: the colliders in the ``height``
    band [m] and the point ``keep`` (the robot's start), grown by ``margin`` and out to whole cells."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), PURPOSES)
    low, high = np.array(keep, dtype=float), np.array(keep, dtype=float)
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(ROBOT_ROOT) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        bound = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = np.array(bound.GetMin()), np.array(bound.GetMax())
        if not np.all(np.isfinite([*lo, *hi])) or hi[2] < height[0] or lo[2] > height[1]:
            continue
        low, high = np.minimum(low, lo[:2]), np.maximum(high, hi[:2])
    low = np.floor((low - margin) / resolution) * resolution
    high = np.ceil((high + margin) / resolution) * resolution
    return tuple(round(float(value), 6) for value in (*low, *high))


class UsdScene:
    """A scene USD with an episode's object placements; ground truth by name once the simulation runs.

    ``objects``: bodies moved before the episode (:func:`apply_initial_state`). ``navigation_map``:
    the map's ``resolution`` [m], ``clearance`` [m], collider ``height`` band [m] and ``margin`` [m]
    around the colliders and ``robot_start`` (x, y, ...) [m]; its bounds are read from the stage
    (:attr:`map`, after :meth:`define`).
    """

    def __init__(self, name: str, usd: Path, *, ground: bool, objects: dict, navigation_map: dict, robot_start) -> None:
        self.name, self.ground, self.objects = name, ground, dict(objects)
        self.usd = str(Path(usd).resolve(strict=True))
        self._map_settings, self._robot_start = dict(navigation_map), tuple(robot_start[:2])
        self.map_height = tuple(navigation_map["height"])
        self.map: dict | None = None  # bounds, resolution, clearance; set by define().
        self._stage = self._scene = self._bodies = None

    # Building.
    def open_stage(self) -> None:
        """The scene file is the stage (before the simulation context exists)."""
        import isaaclab.sim as sim_utils

        sim_utils.open_stage(self.usd)

    def define(self, stage: Usd.Stage, cfg, dt: float, *, contact_links: tuple[str, ...]) -> None:
        """Apply the initial state, register the scene's bodies, read what is where, add the contact
        sensor on the robot's ``contact_links`` (its arm and gripper links)."""
        from isaaclab.sensors import ContactSensorCfg

        from phygs.scene.assets import register_scene_bodies

        self._stage = stage
        self._load(stage)
        if not self.ground:
            cfg.ground = None  # The scene has its own floors.
        paths = rigid_bodies(stage)
        self.bodies = {}
        for path in paths:
            if name_of(path) in self.bodies:
                raise ValueError(f"Two scene bodies are named {name_of(path)!r}: {self.bodies[name_of(path)]}, {path}")
            self.bodies[name_of(path)] = path
        apply_initial_state(stage, self.bodies, self.objects)
        settings = self._map_settings
        self.map = dict(
            bounds=map_bounds(stage, self.map_height, settings["resolution"], settings["margin"], self._robot_start),
            resolution=settings["resolution"],
            clearance=settings["clearance"],
        )
        register_scene_bodies(stage, cfg)
        self.drawers = drawers(stage)
        self.part_names = tuple(self.drawers)
        self._interiors = {
            name: drawer_interior(stage, drawer) for name, drawer in self.drawers.items() if _panelled(stage, drawer)
        }
        self._colliders = collider_boxes(stage)
        self._initial = {
            name: body_box(stage, path) for name, path in self.bodies.items() if _has_colliders(stage, path)
        }
        self._center_local = {
            name: (np.linalg.inv(world_pose(stage, self.bodies[name])) @ [*center, 1.0])[:3]
            for name, (center, _) in self._initial.items()
        }
        self._placement = {}
        # Arm contact is recorded against the bodies in the map's region (the robot's work area).
        low, high = self.map["bounds"][:2], self.map["bounds"][2:]
        height = self.map_height
        self.contact_bodies = tuple(
            name
            for name, (center, size) in self._initial.items()
            if np.all(center[:2] + size[:2] / 2 > low)
            and np.all(center[:2] - size[:2] / 2 < high)
            and center[2] - size[2] / 2 < height[1]
            and center[2] + size[2] / 2 > height[0]
        )
        cfg.arm_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(" + "|".join(map(re.escape, contact_links)) + ")",
            update_period=dt,
            filter_prim_paths_expr=[self.bodies[name] for name in self.contact_bodies],
        )

    def _load(self, stage: Usd.Stage) -> None:
        """Make sure the scene file is on the stage: opened as the stage (:meth:`open_stage`), or
        added now as a sublayer when the file has no physics scene of its own (one with a physics
        scene must be the stage: two would conflict)."""
        if (
            Path(stage.GetRootLayer().realPath or "") == Path(self.usd)
            or self.usd in stage.GetRootLayer().subLayerPaths
        ):
            return
        source = Usd.Stage.Open(self.usd)
        if any(prim.IsA(UsdPhysics.Scene) for prim in source.Traverse()):
            raise RuntimeError(f"{self.usd} has its own physics scene: open it as the stage first (open_stage).")
        stage.GetRootLayer().subLayerPaths.append(self.usd)

    def bind(self, scene) -> None:
        """Hold the InteractiveScene for ground-truth reads."""
        from phygs.scene.assets import SceneBodies

        self._scene, self._bodies = scene, SceneBodies(scene)

    def _path(self, name: str) -> str:
        if name not in self.bodies:
            raise KeyError(f"No body {name!r} in scene {self.name!r}.")
        return self.bodies[name]

    # Ground truth.
    def object_pose(self, name: str) -> np.ndarray:
        """World pose (4x4) [m] of a body now."""
        return pose_matrix(self._bodies.pose(self._path(name)).cpu().tolist())

    def object_box(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """World box (center, size) [m] of a body now: its authored box carried by its pose (the
        size is the authored one; objects in these scenes stay upright)."""
        if name not in self._initial:
            raise KeyError(f"{name!r} has no colliders.")
        return (self.object_pose(name) @ [*self._center_local[name], 1.0])[:3], self._initial[name][1]

    def initial_box(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """World box (center, size) [m] a body started at."""
        return self._initial[name]

    def part(self, name: str) -> ArticulatedPart:
        """A drawer now: handle, opening direction, travel limits, opening [m]."""
        if name not in self.drawers:
            raise KeyError(f"No drawer {name!r} in scene {self.name!r}; it has {self.part_names}.")
        drawer = self.drawers[name]
        part = sliding_part(self._stage, drawer.slide, drawer.handle, lambda path: self.object_pose(name))
        if name not in self._interiors:  # Not a panelled house-unit drawer: no known floor.
            return part
        return replace(part, floor_center=tuple(map(float, self.drawer_floor(name, part.opening))))

    def placement(self, name: str) -> tuple[str | None, str | None]:
        """Where a body started: (the drawer it was inside, the body it rested on)."""
        if name not in self._placement:
            self._placement[name] = placement(self._stage, self._path(name), self.drawers)
        return self._placement[name]

    def inside(self, point, part: str) -> bool:
        """Whether a world point [m] is inside a drawer now."""
        if part not in self._interiors:
            raise KeyError(f"No panelled drawer {part!r} in scene {self.name!r}.")
        local = (np.linalg.inv(self.object_pose(part)) @ [*point, 1.0])[:3]
        return inside_interior(self._interiors[part], local)

    def drawer_floor(self, part: str, opening: float) -> np.ndarray:
        """World centre [m] of a house-unit drawer's floor in front of its carcass, open by ``opening`` [m]."""
        y = (DRAWER_FRONT_INNER_Y_M + CARCASS_FRONT_Y_M + opening) / 2
        return (self.object_pose(part) @ [0.0, y, DRAWER_FLOOR_Z_M, 1.0])[:3]

    def obstacle_boxes(
        self, center_xy, radius: float, height: tuple[float, float], exclude: frozenset[str] = frozenset()
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """The colliders now within ``radius`` [m] of a point (box edge to point, xy) and in the
        ``height`` band [m], without the bodies named in ``exclude``: collider path -> (center, size)."""
        poses = {}
        boxes = {}
        center_xy = np.asarray(center_xy, dtype=float)
        for path, body, center, size in self._colliders:
            if body is not None and name_of(body) in exclude:
                continue
            if body is not None:
                pose = poses.setdefault(body, pose_matrix(self._bodies.pose(body).cpu().tolist()))
                center, size = (pose @ [*center, 1.0])[:3], np.abs(pose[:3, :3]) @ size
            if center[2] + size[2] / 2 < height[0] or center[2] - size[2] / 2 > height[1]:
                continue
            gap = np.maximum(np.abs(center[:2] - center_xy) - size[:2] / 2, 0.0)
            if np.hypot(*gap) <= radius:
                boxes[path] = (center, size)
        return boxes

    def map_boxes(self, exclude: tuple[str, ...] = ()) -> list[tuple[np.ndarray, np.ndarray]]:
        """Navigation map boxes from the stage's colliders, without what moves along: the drawers
        (their closed carcasses still cover them), what lies inside a drawer, and ``exclude``."""
        from phygs.scene import storage_units

        riding = []
        for name, (center, _) in self._initial.items():
            for drawer, interior in self._interiors.items():
                local = (np.linalg.inv(world_pose(self._stage, self.drawers[drawer].body)) @ [*center, 1.0])[:3]
                if inside_interior(interior, local):
                    riding.append(name)
        moving = (*self.drawers, *riding, *exclude)
        return storage_units.map_boxes(self._stage, self.map["bounds"], self.map_height, exclude=moving)

    def contacts(self, threshold_n: float) -> list[tuple[str, str, float]]:
        """Arm contacts above ``threshold_n`` [N] now: (arm link, body, force [N])."""
        import torch

        sensor = self._scene["arm_contact"]
        forces = torch.linalg.vector_norm(sensor.data.force_matrix_w.torch, dim=-1)[0]  # [link, body]
        hits = torch.nonzero(forces > threshold_n)
        return [
            (sensor.body_names[int(link)], self.contact_bodies[int(body)], float(forces[link, body]))
            for link, body in hits
        ]


def _panelled(stage: Usd.Stage, drawer: Drawer) -> bool:
    return all(
        stage.GetPrimAtPath(f"{drawer.body}/{n}").IsValid() for n in ("left", "right", "front", "back", "bottom")
    )
