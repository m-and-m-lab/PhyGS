#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reconstruct the furnished Bear Loop reference as an independently interactive two-level house."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from build_residential_scene import REFERENCE, SCENES, ResidentialHouse
from scene_lib.bear_loop_furnishings import furnish_reference_details, install_reference_materials, native
from scene_lib.bear_loop_layout import BATHROOMS, WALKING_STRIPS
from scene_lib.house_asset_library import HouseAssetLibrary
from scene_lib.collision_geometry import box_surface_mesh, convex_collider_to_sdf
from scene_lib.residential_surfaces import install_materials

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

OUTPUT = SCENES / "bear_loop_house_revised/scene.usdc"
REFERENCE_URL = "https://media.zillowmediaexperts.com/sites/42413-bear-loop-big-bear-city-ca-92314-15249997"
FLOOR_HEIGHT = 3.2
# Traced from the published combined plan, in image pixels. The second plan is vertically registered to the first.
OUTLINE = [
    (532, 105),
    (560, 76),
    (630, 76),
    (660, 106),
    (660, 130),
    (733, 130),
    (818, 106),
    (888, 130),
    (980, 130),
    (980, 105),
    (1009, 76),
    (1041, 76),
    (1074, 104),
    (1074, 136),
    (1041, 164),
    (1041, 232),
    (1134, 232),
    (1198, 90),
    (1386, 178),
    (1330, 292),
    (1349, 301),
    (1290, 417),
    (1318, 430),
    (1240, 590),
    (1010, 480),
    (888, 480),
    (888, 464),
    (775, 464),
    (775, 480),
    (660, 480),
    (631, 499),
    (562, 499),
    (532, 478),
]
ATRIUM = [
    (695, 130),
    (745, 130),
    (818, 106),
    (888, 130),
    (917, 130),
    (917, 221.8181818182),
    (695, 221.8181818182),
]
STAIR_VOID = [(820, 418.1818181818), (924, 418.1818181818), (924, 274), (820, 274)]
GARAGE = [(1120, 278), (1325, 351), (1290, 417), (1318, 430), (1240, 590), (1010, 480)]
SCALE = 0.0275


def xy(point: tuple[float, float]) -> tuple[float, float]:
    """Convert registered floor-plan coordinates to world coordinates [m]."""
    return ((point[0] - 532) * SCALE, (480 - point[1]) * SCALE)


def polygon_area(points: list[tuple]) -> float:
    """Return polygon area [m²]."""
    a = np.asarray(points)
    return abs(float(np.sum(a[:, 0] * np.roll(a[:, 1], -1) - a[:, 1] * np.roll(a[:, 0], -1)))) / 2


def floor_triangles(rings: list[list[tuple]]) -> tuple[list, list]:
    """Triangulate a polygon with holes using exact horizontal trapezoid decomposition [m]."""
    edges = [(a, b) for ring in rings for a, b in zip(ring, ring[1:] + ring[:1]) if a[1] != b[1]]
    ys = sorted({p[1] for ring in rings for p in ring})
    vertices, faces = [], []

    def x_at(edge, y):
        a, b = edge
        return a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])

    for low, high in zip(ys, ys[1:]):
        middle = (low + high) / 2
        active = [e for e in edges if min(e[0][1], e[1][1]) < middle < max(e[0][1], e[1][1])]
        active.sort(key=lambda e: x_at(e, middle))
        if len(active) % 2:
            raise ValueError("Invalid floor polygon: odd scanline crossing count.")
        for left, right in zip(active[::2], active[1::2]):
            i = len(vertices)
            vertices.extend(
                [(x_at(left, low), low), (x_at(right, low), low), (x_at(right, high), high), (x_at(left, high), high)]
            )
            faces.extend([(i, i + 1, i + 2), (i, i + 2, i + 3)])
    area = sum(
        abs(np.cross(np.subtract(vertices[b], vertices[a]), np.subtract(vertices[c], vertices[a]))) / 2
        for a, b, c in faces
    )
    expected = polygon_area(rings[0]) - sum(polygon_area(r) for r in rings[1:])
    if not math.isclose(area, expected, abs_tol=1e-6):
        raise ValueError(f"Floor holes overlap or escape the outline: {area} != {expected}")
    return vertices, faces


class BearLoopHouse(ResidentialHouse):
    """Reference-led architecture, varied local meshes, passive storage, and fixture-mounted lights."""

    def __init__(self, output: Path):
        super().__init__(output, REFERENCE)
        self.library = HouseAssetLibrary(
            self.stage,
            output.parent,
            {
                "h47": SCENES / "House_047/usd/scene_scene_047.usda",
                "h48": SCENES / "test_set_5/mujoco/usd/scene_scene_048.usda",
                "h98": SCENES / "test_set_6/mujoco/usd/scene_scene_098.usda",
            },
        )
        for key, name in (("ph_chair", "dining_chair_02"), ("ph_sofa", "sofa_02")):
            self.library.add_asset(key, SCENES / "bear_loop_reference/models" / name / (name + ".usdc"), "/" + name)
        self.metadata_dir = self.output_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        self.inventory = []
        self.fixtures = []
        self.stairs = []
        self.connections = []
        self.stored_items = []
        self.ceilings = []
        self.drawer_stores = []
        self.storage_compartments = []

    def mesh_solid(self, path: str, points: list, faces: list, material: str, collision: bool = True) -> UsdGeom.Mesh:
        """Create a triangulated architectural mesh with points [m]."""
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(points, dtype=np.float32)))
        mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
        mesh.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self.materials[material])
        uv = [tuple(np.asarray(points[i])[:2] / 1.7) for face in faces for i in face]
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying").Set(uv)
        if collision:
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        return mesh

    def slab(self, name: str, rings: list[list[tuple]], top: float, material: str = "floor") -> None:
        """Build a closed floor slab with genuine atrium/stair holes [m]."""
        vertices, triangles = floor_triangles(rings)
        n = len(vertices)
        points = [(x, y, top) for x, y in vertices] + [(x, y, top - 0.2) for x, y in vertices]
        faces = triangles + [tuple(i + n for i in reversed(face)) for face in triangles]
        for ring in rings:
            for a, b in zip(ring, ring[1:] + ring[:1]):
                i = len(points)
                points.extend([(*a, top), (*b, top), (*b, top - 0.2), (*a, top - 0.2)])
                faces.append((i, i + 1, i + 2, i + 3))
        self.mesh_solid("/World/structure/" + name, points, faces, material)

    def segment(
        self,
        name: str,
        a: tuple,
        b: tuple,
        bottom: float,
        height: float,
        thickness: float = 0.14,
        material: str = "plaster",
    ) -> None:
        """Build a straight wall or rail with arbitrary plan orientation [m]."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        path = "/World/structure/" + name
        root = UsdGeom.Xform.Define(self.stage, path)
        root.AddTranslateOp().Set(Gf.Vec3d((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, bottom + height / 2))
        root.AddRotateZOp().Set(math.degrees(math.atan2(dy, dx)))
        self.box(path + "/panel", (0, 0, 0), (length, thickness, height), material)

    def partition(self, name: str, a: tuple, b: tuple, level: int, door_at: float | None = None) -> None:
        """Build a wall and optional 1 m doorway; door_at is distance from a [m]."""
        a, b = np.asarray(a), np.asarray(b)
        length = float(np.linalg.norm(b - a))
        direction = (b - a) / length
        z = level * FLOOR_HEIGHT
        if door_at is None:
            self.segment(name, a, b, z, 3)
            return
        if not 0.15 <= door_at <= length - 1.15:
            raise ValueError(f"Door outside wall: {name}")
        left, right = a + direction * door_at, a + direction * (door_at + 1)
        self.segment(name + "_left", a, left, z, 3)
        self.segment(name + "_right", right, b, z, 3)
        self.segment(name + "_lintel", left, right, z + 2.18, 0.82)
        pivot = (*tuple(left + direction * 0.03), z + 1.045)
        yaw = math.degrees(math.atan2(direction[1], direction[0]))
        self.room_door(name + "_door", pivot, yaw)
        if name == "garage":
            # Stop the access door before its handle reaches the adjacent pantry wall.
            hinge = UsdPhysics.RevoluteJoint(self.stage.GetPrimAtPath("/World/joints/garage_door_hinge"))
            hinge.GetUpperLimitAttr().Set(80)
        self.segment(name + "_door_stop", right - direction * 0.06, right, z, 2.18, 0.12, "cream")
        if name in ("bedroom_south", "primary_bath", "primary_bath_west", "corner_bedroom"):
            hinge = UsdPhysics.Joint(self.stage.GetPrimAtPath("/World/joints/" + name + "_door_hinge"))
            flip = Gf.Rotation(Gf.Vec3d(1, 0, 0), 180).GetQuat()
            hinge.CreateLocalRot0Attr(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw).GetQuat() * flip))
            hinge.CreateLocalRot1Attr(Gf.Quatf(flip))
        self.inventory.append(dict(name=name + "_door", category="interior_door", evidence="floor_plan", level=level))

    def fixture(self, name: str, x: float, y: float, ceiling: float, intensity: float = 16000) -> None:
        """Place a recessed disk emitter in a visible ceiling fitting [m], at 5000 K."""
        path = "/World/structure/fixtures/" + name
        root = UsdGeom.Xform.Define(self.stage, path)
        root.AddTranslateOp().Set(Gf.Vec3d(x, y, ceiling - 0.0125))
        rim = UsdGeom.Cylinder.Define(self.stage, path + "/trim")
        rim.CreateRadiusAttr(0.115)
        rim.CreateHeightAttr(0.025)
        UsdShade.MaterialBindingAPI.Apply(rim.GetPrim()).Bind(self.materials["cream"])
        light = UsdLux.DiskLight.Define(self.stage, path + "/bulb")
        light.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.016))
        light.CreateRadiusAttr(0.088)
        light.CreateNormalizeAttr(True)
        light.CreateIntensityAttr(intensity)
        light.CreateEnableColorTemperatureAttr(True)
        light.CreateColorTemperatureAttr(5000)
        self.fixtures.append(str(light.GetPath()))

    def architecture(self) -> None:
        """Trace the published plan, keeping the central void and a straight-flight switchback connection."""
        outline = [xy(p) for p in OUTLINE]
        atrium, stair_void = [xy(p) for p in ATRIUM], [xy(p) for p in STAIR_VOID]
        self.slab("ground_floor", [outline], 0, "tile")
        bathrooms = []
        for room in BATHROOMS:
            if room["level"] == 1:
                x0, y0, x1, y1 = room["bounds"]
                bathrooms.append((room["name"], [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
        self.slab(
            "upper_floor", [outline, atrium, stair_void, *(ring for _, ring in bathrooms)], FLOOR_HEIGHT, "carpet"
        )
        for name, ring in bathrooms:
            self.slab("bathroom_floor_" + name, [ring], FLOOR_HEIGHT, "tile")
        self.slab("lower_ceiling", [outline, atrium, stair_void], 3.18, "plaster")
        self.slab("roof", [outline], 6.4, "plaster")
        self.ceilings = ["/World/structure/lower_ceiling", "/World/structure/roof"]
        for level in (0, 1):
            z = level * FLOOR_HEIGHT
            for i, (a, b) in enumerate(zip(outline, outline[1:] + outline[:1])):
                length = float(np.linalg.norm(np.subtract(b, a)))
                if abs(a[1] - 0.44) < 1e-6 and abs(b[1] - 0.44) < 1e-6 and length > 3 and level == 0:
                    self.partition("front_entry", a, b, 0, 0.40)
                    continue
                if abs(a[0]) < 1e-6 and abs(b[0]) < 1e-6 and length > 8:
                    self.west_wall(level, min(a[1], b[1]), max(a[1], b[1]))
                    continue
                if length < 1.7:
                    self.segment(f"exterior_{level}_{i}", a, b, z, 3)
                else:
                    direction = np.subtract(b, a) / length
                    margin = min(0.65, length / 4)
                    left, right = np.asarray(a) + direction * margin, np.asarray(b) - direction * margin
                    self.segment(f"exterior_{level}_{i}_a", a, left, z, 3)
                    self.segment(f"exterior_{level}_{i}_b", right, b, z, 3)
                    self.segment(f"exterior_{level}_{i}_sill", left, right, z, 0.85)
                    self.segment(f"exterior_{level}_{i}_top", left, right, z + 2.45, 0.55)
                    self.segment(f"window_{level}_{i}", left, right, z + 0.85, 1.6, 0.012, "glass")
                    self.segment(f"window_trim_{level}_{i}", left, right, z + 0.82, 0.07, 0.20, "cream")
                self.segment(f"baseboard_{level}_{i}", a, b, z + 0.02, 0.13, 0.18, "cream")
        # Room partitions follow the plan's west bedrooms, front foyer, and angled east wing.
        for name, a, b, level, door in (
            ("bar_foyer", (6.68, 0.45), (6.68, 4.7), 0, 2.2),
            ("pantry", (11.7, 3.5), (13.8, 3.5), 0, 0.2),
            ("pantry_west", (11.7, 0.5), (11.7, 3.5), 0, None),
            ("pantry_south", (11.7, 0.5), (13.8, 0.5), 0, None),
            ("pantry_east", (13.8, 0.5), (13.8, 3.5), 0, None),
            ("garage", (13.8, 0.5), (16.15, 5.55), 0, 2.1),
            ("kitchen_back", (9.5, 9.49), (13.3, 9.49), 0, None),
            ("ground_bath_west", (18.7, 5.7), (18.7, 4.1), 0, 0.2),
            ("ground_bath_north", (18.7, 5.7), (21.65, 5.7), 0, None),
            ("ground_bath_east", (21.65, 4.1), (21.65, 5.7), 0, None),
            ("ground_bath_south", (18.7, 4.1), (21.65, 4.1), 0, None),
            ("guest_wing", (16.1, 5.55), (21.8, 2.75), 0, 0.5),
            ("ground_guest_entry", (16.35, 5.55), (16.35, 7.0), 0, 0.2),
            ("ground_guest_return", (16.1, 5.55), (16.35, 5.55), 0, None),
            ("primary_bath", (0, 4.65), (6.90, 4.65), 1, 5.7),
            ("primary_bath_west", (3.6, 4.65), (3.6, 1.7), 1, 0.9),
            ("primary_bath_south", (3.6, 1.7), (6.9, 1.7), 1, None),
            ("guest_bath_south", (12.35, 3.1), (14.45, 3.1), 1, None),
            ("primary_east", (4.5, 6.55), (4.5, 9.6), 1, None),
            ("primary_entry", (4.5, 4.65), (4.5, 6.55), 1, 0.25),
            ("bedroom_south", (6.90, 0), (6.90, 4.65), 1, 0.55),
            ("upper_bath", (12.35, 4.85), (14.45, 4.85), 1, None),
            ("guest_bath_west", (12.35, 3.1), (12.35, 4.85), 1, 0.2),
            ("guest_bath_east", (14.45, 3.1), (14.45, 4.85), 1, None),
            ("upper_guest", (12.35, 0), (12.35, 3.1), 1, 0.2),
            ("guest_bedroom_east", (14.6, 0), (14.6, 3.1), 1, None),
            ("east_bedroom_north", (17.7, 4.0), (19.2, 4.0), 1, None),
            ("east_bedroom_return", (19.2, 3.9), (19.2, 4.0), 1, None),
            ("wing_bedroom", (15.1, -1.2), (17.7, 4.0), 1, 2.6),
            ("corner_bedroom", (17.84, -1.845), (19.686, 1.95), 1, 2.7),
            ("wing_bath", (19.2, 1.95), (19.2, 3.9), 1, 0.2),
            ("wing_bath_north", (19.2, 3.9), (20.9, 3.9), 1, None),
            ("wing_bath_east", (20.9, 1.95), (20.9, 3.9), 1, None),
            ("wing_bath_south", (19.2, 1.95), (20.9, 1.95), 1, None),
        ):
            self.partition(name, a, b, level, door)
        self.switchback_stair()
        for ring_id, ring in enumerate((atrium, stair_void)):
            for i, (a, b) in enumerate(zip(ring, ring[1:] + ring[:1])):
                if ring_id == 1 and i == 0:
                    continue
                self.railing(f"gallery_{ring_id}_{i}", a, b, FLOOR_HEIGHT)

        # Lighting is solely from ceiling fittings; no panorama, sun, or invisible room-sized emitters.
        def inside_polygon(px, py, polygon):
            return (
                sum(
                    (a[1] > py) != (b[1] > py) and px < (b[0] - a[0]) * (py - a[1]) / (b[1] - a[1]) + a[0]
                    for a, b in zip(polygon, polygon[1:] + polygon[:1])
                )
                % 2
                == 1
            )

        for level, centers in (
            (
                0,
                [
                    (2.2, 8.5),
                    (5, 8.5),
                    (2.2, 5.8),
                    (5, 5.8),
                    (12.4, 8.0),
                    (2, 2.1),
                    (4.5, 2.1),
                    (8, 1.0),
                    (12.3, 2.4),
                    (17.8, 7.5),
                    (19.8, 6.6),
                    (17.8, 0.5),
                ],
            ),
            (
                1,
                [
                    (2, 8.4),
                    (3.4, 6.4),
                    (2, 2.4),
                    (5.4, 2.4),
                    (12, 7.7),
                    (13.8, 7.7),
                    (12, 2.5),
                    (16.5, 0.3),
                    (19.3, 1.8),
                    (18.8, 6.3),
                    (21, 6.3),
                    (17.6, 8.3),
                ],
            ),
        ):
            for i, (x, y) in enumerate(centers):
                if level == 0 and any(inside_polygon(x, y, hole) for hole in (atrium, stair_void)):
                    continue
                self.fixture(f"level_{level}_{i}", x, y, 2.98 if level == 0 else 6.2)
        for i, (x, y) in enumerate(((6, 8.5), (8, 8.5), (10, 8.5))):
            self.fixture(f"atrium_{i}", x, y, 6.2, 16000)
        for i, x in enumerate((8.625, 10.035)):
            self.fixture(f"stair_{i}", x, 3.3, 6.2, 16000)

        for room in BATHROOMS:
            x0, y0, x1, y1 = room["bounds"]
            self.fixture("bathroom_" + room["name"], (x0 + x1) / 2, (y0 + y1) / 2, 6.2 if room["level"] else 2.98, 9500)
        self.fixture("primary_shower", 6.1, 2.35, 6.2, 9500)

    def west_wall(self, level: int, low: float, high: float) -> None:
        """Place windows around the photographed fireplace and solid display walls [m]."""
        windows = [(0.8, 2.7), (4.9, 6.0), (8.95, 9.85)] if level == 0 else [(0.8, 3.5)]
        cursor = low
        for i, (start, end) in enumerate(windows):
            self.segment(f"west_{level}_{i}_solid", (0, cursor), (0, start), level * FLOOR_HEIGHT, 3)
            self.segment(f"west_{level}_{i}_sill", (0, start), (0, end), level * FLOOR_HEIGHT, 0.85)
            self.segment(f"west_{level}_{i}_header", (0, start), (0, end), level * FLOOR_HEIGHT + 2.45, 0.55)
            self.segment(
                f"west_{level}_{i}_glass", (0, start), (0, end), level * FLOOR_HEIGHT + 0.85, 1.6, 0.012, "glass"
            )
            self.segment(
                f"west_{level}_{i}_trim", (0, start), (0, end), level * FLOOR_HEIGHT + 0.82, 0.07, 0.20, "cream"
            )
            cursor = end
        self.segment(f"west_{level}_end", (0, cursor), (0, high), level * FLOOR_HEIGHT, 3)

    def railing(self, name: str, a: tuple, b: tuple, z: float) -> None:
        """Model a capped solid guard wall above a floor [m]."""
        self.segment(name + "_handrail", a, b, z + 0.96, 0.055, 0.065, "oak")
        self.segment(name + "_solid_wall", a, b, z, 0.96, 0.065, "plaster")

    def switchback_stair(self) -> None:
        """Create two straight flights with a 1.35 m landing and 0.16 m risers for robot access."""
        width, step_depth, risers = 1.25, 0.28, 20
        rise = FLOOR_HEIGHT / risers
        x0, y0 = 8.0, 1.7
        gap = 0.16
        landing_y = y0 + 9 * step_depth
        for flight in (0, 1):
            x = x0 + width / 2 + flight * (width + gap)
            for i in range(9):
                y = y0 + (i + 0.5) * step_depth if flight == 0 else landing_y - (i + 0.5) * step_depth
                top = (i + 1) * rise if flight == 0 else (11 + i) * rise
                self.box(f"/World/structure/stair_{flight}_{i}", (x, y, top - 0.09), (width, step_depth, 0.18), "oak")
            for side_index, side in enumerate((-1, 1)):
                edge = x + side * (width / 2 + 0.04)
                a = (edge, y0, 0.98 + (rise if flight == 0 else FLOOR_HEIGHT))
                b = (edge, landing_y, FLOOR_HEIGHT / 2 + 0.98)
                self.rod(f"/World/structure/stair_rail_{flight}_{side_index}", a, b, 0.03, "oak")
                points = [
                    (edge + dx, y, z)
                    for dx in (-0.0325, 0.0325)
                    for y, low, high in ((y0, max(0, a[2] - 1.16), a[2] - 0.03), (landing_y, b[2] - 1.16, b[2] - 0.03))
                    for z in (low, high)
                ]
                faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (1, 3, 7, 5), (0, 4, 6, 2)]
                self.mesh_solid(f"/World/structure/stair_side_wall_{flight}_{side_index}", points, faces, "plaster")
        self.box(
            "/World/structure/stair_mid_landing",
            (x0 + width + gap / 2, landing_y + 0.675, FLOOR_HEIGHT / 2 - 0.09),
            (2 * width + gap, 1.35, 0.18),
            "oak",
        )
        # The upper floor itself is the final landing; its opening starts exactly at the top riser.
        self.box(
            "/World/structure/stair_top_landing",
            (x0 + width + gap + width / 2, y0 - 0.60, FLOOR_HEIGHT - 0.10),
            (width, 1.20, 0.20),
            "oak",
        )
        self.railing(
            "stair_landing_back",
            (x0 - 0.04, landing_y + 1.39),
            (x0 + 2 * width + gap + 0.04, landing_y + 1.39),
            FLOOR_HEIGHT / 2,
        )
        for side, edge in enumerate((x0 - 0.04, x0 + 2 * width + gap + 0.04)):
            self.railing(f"stair_landing_side_{side}", (edge, landing_y), (edge, landing_y + 1.39), FLOOR_HEIGHT / 2)
        self.stairs.append(
            dict(
                name="main_switchback",
                risers=risers,
                rise_m=rise,
                tread_m=step_depth,
                clear_width_m=width,
                landing_depth_m=1.35,
                landing_width_m=2 * width + gap,
                total_rise_m=FLOOR_HEIGHT,
                bottom_m=[x0 + width / 2, y0, 0],
                top_m=[x0 + 1.5 * width + gap, y0 - 0.6, FLOOR_HEIGHT],
                reference_departure="Curved foyer stair replaced at user request for Spot access.",
            )
        )

    def rod(self, path: str, a: tuple, b: tuple, radius: float, material: str) -> None:
        """Create a cylindrical handrail/handle between endpoints [m]."""
        delta = Gf.Vec3d(*b) - Gf.Vec3d(*a)
        rod = UsdGeom.Cylinder.Define(self.stage, path)
        rod.CreateRadiusAttr(radius)
        rod.CreateHeightAttr(delta.GetLength())
        rod.AddTranslateOp().Set((Gf.Vec3d(*a) + Gf.Vec3d(*b)) / 2)
        rod.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), delta).GetQuat()))
        UsdShade.MaterialBindingAPI.Apply(rod.GetPrim()).Bind(self.materials[material])
        UsdPhysics.CollisionAPI.Apply(rod.GetPrim())

    def asset(
        self,
        name: str,
        key: str,
        position: tuple,
        size: tuple,
        mass: float,
        yaw: float = 0,
        support: str = "floor",
        kind: str = "box",
        evidence: str = "inferred",
    ) -> str:
        """Place an independently movable textured object at a support-relative base [m, kg, deg]."""
        path = self.body(name, (position[0], position[1], position[2] + 0.002), mass, yaw=yaw)
        record = self.library.visual(key, path + "/visual", size)
        w, d, h = size
        if kind == "sdf":
            for prim in Usd.PrimRange(self.stage.GetPrimAtPath(path + "/visual")):
                if prim.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(prim)
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("sdf")
                    prim.AddAppliedSchema("PhysxSDFMeshCollisionAPI")
                    prim.CreateAttribute("physxSDFMeshCollision:sdfResolution", Sdf.ValueTypeNames.Int).Set(64)
            panels = []
        elif kind == "table":
            panels = [("top", (0, 0, h - 0.025), (w, d, 0.05))]
            panels += [
                (f"leg_{i}", (x, y, (h - 0.05) / 2), (0.055, 0.055, h - 0.05))
                for i, (x, y) in enumerate(
                    (
                        (w / 2 - 0.06, d / 2 - 0.06),
                        (-w / 2 + 0.06, d / 2 - 0.06),
                        (w / 2 - 0.06, -d / 2 + 0.06),
                        (-w / 2 + 0.06, -d / 2 + 0.06),
                    )
                )
            ]
        elif kind == "chair":
            panels = [
                ("seat", (0, 0, 0.43), (w * 0.85, d * 0.8, 0.08)),
                ("back", (0, d * 0.42, (h + 0.47) / 2), (w * 0.9, 0.07, h - 0.47)),
            ]
            panels += [
                (f"leg_{i}", (x, y, 0.195), (0.05, 0.05, 0.39))
                for i, (x, y) in enumerate(
                    ((w * 0.35, d * 0.3), (-w * 0.35, d * 0.3), (w * 0.35, -d * 0.3), (-w * 0.35, -d * 0.3))
                )
            ]
        elif kind == "sofa":
            panels = [
                ("seat", (0, 0, 0.235), (w, d, 0.47)),
                ("back", (0, d * 0.4, (h + 0.47) / 2), (w, d * 0.2, h - 0.47)),
            ]
        else:
            panels = [("shape", (0, 0, h / 2), size)]
        for suffix, center, dims in panels:
            collider = self.box(path + "/" + suffix, center, dims, "cream")
            UsdGeom.Imageable(collider).MakeInvisible()
        record.update(name=name, path=path, evidence=evidence, support=support, mass_kg=mass)
        self.inventory.append(record)
        self.stage.GetPrimAtPath(path).SetCustomDataByKey("independently_pickable", True)
        if support != "floor":
            self.stage.GetPrimAtPath(path).SetCustomDataByKey("support", support)
        return path

    def storage(
        self,
        name: str,
        x: float,
        y: float,
        z: float,
        width: float = 0.9,
        height: float = 0.9,
        yaw: float = 0,
        drawer: bool = False,
        material: str = "oak",
        rows: int = 1,
        drawer_over_cabinet: bool = False,
    ) -> dict:
        """Build hollow panel storage with an independently hinged door or sliding drawer [m, deg]."""
        if rows > 1 or drawer_over_cabinet:
            fractions = [0.72, 0.28] if drawer_over_cabinet else [1 / rows] * rows
            modules = []
            offset = 0.0
            for i, fraction in enumerate(fractions):
                module = self.storage(
                    name if i == 0 else f"{name}_tier_{i + 1}", x, y, z + offset,
                    width, height * fraction, yaw,
                    drawer=(i > 0 if drawer_over_cabinet else True), material=material,
                )
                if i > 0:
                    # One anchored carcass: adjacent tiers must not collide as separate fixed bodies.
                    tier_path = "/World/objects/" + f"{name}_tier_{i + 1}"
                    destination = "/World/objects/" + name + f"/tier_{i + 1}"
                    Sdf.CopySpec(self.stage.GetRootLayer(), tier_path, self.stage.GetRootLayer(), destination)
                    tier = self.stage.GetPrimAtPath(destination)
                    tier.RemoveAPI(UsdPhysics.RigidBodyAPI)
                    tier.RemoveAPI(UsdPhysics.MassAPI)
                    transform = UsdGeom.Xformable(tier)
                    transform.ClearXformOpOrder()
                    transform.AddTranslateOp().Set(Gf.Vec3d(0, 0, offset))
                    self.stage.RemovePrim(tier_path)
                    self.stage.RemovePrim("/World/joints/" + f"{name}_tier_{i + 1}_mount")
                modules.append(module)
                offset += height * fraction
            # A continuous countertop belongs directly to the shared carcass.
            self.stage.RemovePrim("/World/objects/" + name + "/top")
            self.stage.RemovePrim("/World/objects/" + name + f"/tier_{len(fractions)}/top")
            self.box("/World/objects/" + name + "/top", (0, 0, height - 0.0125),
                     (width, 0.56, 0.025), "cream")
            modules[0]["top"] = z + height
            return modules[0]
        depth, t = 0.56, 0.025
        path = self.body(name, (x, y, z), 35, anchored=True, yaw=yaw)
        for suffix, center, size in (
            ("left", (-width / 2 + t / 2, 0, height / 2), (t, depth, height)),
            ("right", (width / 2 - t / 2, 0, height / 2), (t, depth, height)),
            ("back", (0, depth / 2 - t / 2, height / 2), (width - 2 * t, t, height)),
            ("bottom", (0, 0, t / 2), (width - 2 * t, depth - 2 * t, t)),
            ("top", (0, 0, height - t / 2), (width, depth, t)),
        ):
            self.box(path + "/" + suffix, center, size, material if suffix != "top" else "cream")
        rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw)

        def world(p):
            return tuple(Gf.Vec3d(x, y, z) + rotation.TransformDir(Gf.Vec3d(*p)))

        if drawer:
            local = (0, 0, 0.055)
            moving = self.body(name + "_drawer", world(local), 3, yaw=yaw)
            for suffix, center, size in (
                ("bottom", (0, -0.02125, 0), (width - 0.08, 0.5325, 0.025)),
                ("front", (0, -0.30, height / 2 - 0.055), (width - 0.025, 0.025, height - 0.06)),
                ("back", (0, 0.2325, (height - 0.11) / 2), (width - 0.08, 0.025, height - 0.11)),
                ("left", (-width / 2 + 0.045, -0.02125, (height - 0.11) / 2), (0.025, 0.5325, height - 0.11)),
                ("right", (width / 2 - 0.045, -0.02125, (height - 0.11) / 2), (0.025, 0.5325, height - 0.11)),
            ):
                self.box(moving + "/" + suffix, center, size, material if suffix == "front" else "oak")
            self.joint(name + "_slide", moving, world(local), (0, 0, 0), sliding=True, yaw=yaw + 180)
            joint = UsdPhysics.Joint(self.stage.GetPrimAtPath("/World/joints/" + name + "_slide"))
            joint.CreateLocalRot1Attr(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), 180).GetQuat()))
            support = 0.0675
            handle_z = height / 2 - 0.055
        else:
            self.box(path + "/shelf", (0, 0, height * 0.48), (width - 0.05, 0.50, 0.025), material)
            pivot = (-width / 2 + 0.025, -0.31, height / 2)
            moving = self.body(name + "_door", world((0, -0.31, height / 2)), 4, yaw=yaw)
            self.box(moving + "/frame", (0, 0, 0), (width - 0.05, 0.025, height - 0.06), material)
            self.joint(name + "_hinge", moving, world(pivot), (-width / 2 + 0.025, 0, 0), yaw=yaw)
            hinge = UsdPhysics.Joint(self.stage.GetPrimAtPath("/World/joints/" + name + "_hinge"))
            outward = Gf.Rotation(Gf.Vec3d(1, 0, 0), 180).GetQuat()
            hinge.CreateLocalRot0Attr(Gf.Quatf(rotation.GetQuat() * outward))
            hinge.CreateLocalRot1Attr(Gf.Quatf(outward))
            support = height * 0.48 + 0.0125
            handle_z = 0.0
        # Both mounts intersect the front and pull, and inherit its rigid body.
        front_y = -0.3125 if drawer else -0.025
        grip_y = front_y - 0.045
        if drawer:
            endpoints = [(-min(width * 0.28, 0.22), grip_y, handle_z),
                         (min(width * 0.28, 0.22), grip_y, handle_z)]
        else:
            endpoints = [(width / 2 - 0.09, grip_y, -0.07),
                         (width / 2 - 0.09, grip_y, 0.07)]
        self.rod(moving + "/handle", *endpoints, 0.009, "metal")
        for i, point in enumerate(endpoints):
            self.rod(moving + f"/handle_mount_{i}",
                     (point[0], front_y + 0.008, point[2]), point, 0.012, "metal")
        self.inventory.append(
            dict(
                name=name,
                path=path,
                category="drawer" if drawer else "cabinet",
                evidence="photo-guided dimensions and panel construction",
                mass_kg=35,
            )
        )
        result = dict(
            name=name,
            world=world,
            support=support,
            top=z + height,
            moving=moving,
            support_path=moving if drawer else path,
        )
        result.update(width=width, height=height, drawer=drawer)
        self.storage_compartments.append(result)
        if drawer:
            self.drawer_stores.append(result)
        return result

    def stored(self, storage: dict, specs: list[tuple]) -> None:
        """Place distinct loose contents on a shelf or drawer base, never parented to storage."""
        for i, (key, size, mass, dx, dy) in enumerate(specs):
            name = storage["name"] + f"_contents_{i}"
            p = storage["world"]((dx, dy, storage["support"]))
            path = self.asset(
                name, key, p, size, mass, support=storage["support_path"], evidence="inferred concealed contents"
            )
            self.stored_items.append(path)

    def furnish(self) -> None:
        """Match observed furnishing groups and use distinct meshes for unobserved small possessions."""
        # Photo 7: two brown seating groups and a rectangular coffee table by the stone fireplace.
        self.asset(
            "living_sofa",
            "h98:living_room_couch_0",
            (3.8, 6.9, 0),
            (2.45, 0.98, 1.02),
            65,
            yaw=90,
            kind="sofa",
            evidence="photo_007",
        )
        self.asset(
            "living_armchair",
            "h98:living_room_reclining_armchair_0",
            (2.1, 9.3, 0),
            (1.05, 1.3, 0.98),
            28,
            yaw=180,
            kind="sofa",
            evidence="photo_007",
        )
        self.asset(
            "living_coffee_table",
            "h98:living_room_coffee_table_0",
            (2.1, 7.55, 0),
            (1.3, 0.7, 0.43),
            15,
            kind="table",
            evidence="photo_007",
        )
        self.asset(
            "living_remote",
            "h98:living_room_tv_remote_control_1",
            (2.35, 7.65, 0.43),
            (0.19, 0.055, 0.022),
            0.12,
            yaw=-17,
            support="living_coffee_table",
        )
        self.asset(
            "living_magazine",
            "h98:living_room_home_decor_magazine_0",
            (1.82, 7.51, 0.43),
            (0.23, 0.29, 0.009),
            0.2,
            yaw=11,
            support="living_coffee_table",
        )
        self.asset(
            "living_plant", "h47:bedroom_potted_plant_0", (0.6, 9.7, 0), (0.55, 0.55, 1.35), 4, evidence="photo_006"
        )
        self.asset(
            "dining_table",
            "h47:bedroom_writing_desk_0",
            (8, 7.6, 0),
            (1.2, 2.8, 0.77),
            40,
            kind="table",
            evidence="photo_017; table geometry approximation",
        )
        for i, (x, y, yaw) in enumerate(
            (
                (6.95, 6.7, 90),
                (6.95, 7.65, 90),
                (6.95, 8.55, 90),
                (9.05, 6.7, -90),
                (9.05, 7.65, -90),
                (9.05, 8.55, -90),
            )
        ):
            self.asset(
                f"dining_chair_{i}",
                "ph_chair:dining_chair_02",
                (x, y, 0),
                (0.5, 0.55, 0.9),
                5,
                yaw,
                kind="chair",
                evidence="photo_017; matching set",
            )
        # Photo 20–25: kitchen cabinets, granite island, oven tower, breakfast seating.
        stores = []
        for i, x in enumerate((10.9, 11.85, 12.8)):
            stores.append(self.storage(f"kitchen_base_{i}", x, 9.1, 0, drawer=i == 1, rows=3 if i == 1 else 1, drawer_over_cabinet=i == 0, material="cream"))
        for i, x in enumerate((10.9, 11.85, 12.8)):
            stores.append(self.storage(f"kitchen_wall_{i}", x, 9.1, 1.5, height=0.85))
        island = self.storage("kitchen_island", 11.8, 6.9, 0, width=1.6, drawer=True, rows=2)
        stores.append(island)
        for i, y in enumerate((1.2, 2.4)):
            stores.append(self.storage(f"pantry_{i}", 13.4, y, 0, height=1.8, yaw=-90))
        self.kitchen_stores = stores
        self.asset(
            "kitchen_notebook",
            "h47:bedroom_lined_notebook_0",
            (11.5, 6.85, 0.9),
            (0.20, 0.25, 0.012),
            0.25,
            yaw=-8,
            support="kitchen_island",
        )
        self.asset(
            "kitchen_pen",
            "h47:bedroom_ballpoint_pen_0",
            (12.05, 6.82, 0.9),
            (0.014, 0.15, 0.014),
            0.012,
            yaw=24,
            support="kitchen_island",
        )
        # Foyer, wet bar, and household entry items.
        entry = self.storage("entry_console", 4.6, 0.4, 0, width=1.1, drawer=True, yaw=180, rows=2, material="walnut")
        self.stored(
            entry,
            [
                ("h48:hallway_key_fob_0_f1_0", (0.07, 0.035, 0.018), 0.04, -0.2, 0),
                ("h48:bedroom_leather_bifold_wallet_0", (0.11, 0.09, 0.024), 0.15, 0.16, 0),
            ],
        )
        self.asset(
            "entry_tray",
            "h47:hallway_catchall_tray_0",
            (4.6, 0.4, 0.9),
            (0.28, 0.16, 0.04),
            0.3,
            support="entry_console",
        )
        self.asset(
            "entry_envelope",
            "h47:hallway_white_envelope_0",
            (4.96, 0.4, 0.9),
            (0.22, 0.10, 0.006),
            0.02,
            yaw=9,
            support="entry_console",
        )
        self.asset("entry_shoes_left", "h47:hallway_left_loafer_0", (5.8, 2.1, 0), (0.11, 0.29, 0.11), 0.4, yaw=11)
        self.asset("entry_shoes_right", "h47:hallway_right_loafer_0", (5.8, 2.5, 0), (0.11, 0.29, 0.11), 0.4, yaw=-7)
        for i, x in enumerate((1.2, 2.15, 3.1)):
            s = self.storage(f"wet_bar_{i}", x, 1.0, 0, material="oak")
            self.kitchen_stores.append(s)
        self.asset(
            "wet_bar_stool", "h48:bathroom_bathroom_stool_0", (2.0, 2.4, 0), (0.42, 0.42, 0.62), 4, evidence="photo_002"
        )
        # Different source beds and side furniture avoid cloning one bedroom throughout the house.
        bedrooms = [
            ("ground_guest", 0, (20.0, 8.5), "h48:bedroom_bed_0", 154, "photo_033"),
            ("primary", 1, (1.3, 9.45), "h47:bedroom_bed_0", -90, "photo_041"),
            ("south_suite", 1, (2.0, 0.73), "h98:bedroom_bed_0", 0, "photo_043"),
            ("upper_guest", 1, (13.1, 1.96), "h48:bedroom_bed_0", 180, "photo_058"),
            ("east_guest", 1, (17.35, -0.55), "h98:bedroom_bed_0", -26, "photo_060"),
        ]
        for i, (name, level, (x, y), key, yaw, evidence) in enumerate(bedrooms):
            z = level * FLOOR_HEIGHT
            self.asset(name + "_bed", key, (x, y, z), (1.4 if name == "upper_guest" else 1.7, 2.1, 0.90), 70, yaw, evidence=evidence)
            if name == "east_guest":
                continue
            night_x, night_y, night_yaw = {
                "ground_guest": (19.1, 9.55, -26),
                "primary": (1.15, 10.65, 90),
                "south_suite": (0.55, 0.75, 180),
                "upper_guest": (14.16, 2.7, 0),
            }[name]
            s = self.storage(
                name + "_nightstand", night_x, night_y, z, width=0.65, height=0.62, drawer=True, yaw=night_yaw, rows=2, material="cream"
            )
            self.stored(
                s,
                [
                    (
                        (
                            "h98:bedroom_smartphone_0",
                            "h98:bedroom_digital_alarm_clock_0",
                            "h48:closet_cologne_bottle_0",
                            "h98:bedroom_perfume_bottle_0",
                            "h47:walk_in_closet_clutch_purse_0",
                        )[i],
                        (0.13, 0.08, 0.05),
                        0.2,
                        0,
                        0,
                    )
                ],
            )
            self.asset(
                name + "_book",
                (
                    "h98:bedroom_paperback_novel_0",
                    "h98:living_room_art_book_0",
                    "h98:living_room_dictionary_0",
                    "h98:living_room_comic_book_0",
                    "h98:living_room_textbook_0",
                )[i],
                (night_x, night_y, z + 0.62),
                (0.18, 0.23, 0.028),
                0.4,
                yaw=13 - i * 7,
                support=s["name"],
            )
        # Upstairs landing: separate white/blue seating groups and a reading desk.
        self.asset(
            "upper_sofa",
            "ph_sofa:sofa_02",
            (12.55, 7.4, 3.2),
            (2.1, 0.85, 0.92),
            55,
            yaw=180,
            kind="sofa",
            evidence="photo_012",
        )
        self.asset(
            "upper_armchair",
            "h48:bedroom_lounge_chair_0",
            (12, 8.95, 3.2),
            (0.72, 0.78, 0.91),
            18,
            yaw=180,
            kind="sofa",
            evidence="photo_012",
        )
        self.asset(
            "upper_side_table", "h98:living_room_side_table_0", (13.65, 6.65, 3.2), (0.5, 0.5, 0.55), 8, kind="table"
        )
        self.asset(
            "upper_laptop",
            "h47:bedroom_laptop_computer_0",
            (13.65, 6.65, 3.75),
            (0.30, 0.23, 0.18),
            1.4,
            support="upper_side_table",
        )
        # Closets and bathroom contents are inferred; every item is separately graspable.
        self.closet_stores = []
        for i, (x, y) in enumerate(((4.15, 0.4), (5.15, 0.4), (6.15, 0.4))):
            s = self.storage(f"closet_{i}", x, y, 3.2, width=0.8, height=1.8, yaw=180)
            self.closet_stores.append(s)
            keys = ["h48:closet_knit_sweater_1_0", "h48:closet_denim_jeans_0", "h48:closet_wide_brim_hat_0"]
            self.stored(s, [(keys[i], (0.32, 0.24, 0.09), 0.5, 0, 0)])
        for i, room in enumerate(BATHROOMS):
            x, y = room["vanity"]
            z = room["level"] * FLOOR_HEIGHT
            s = self.storage(f"bath_vanity_{i}", x, y, z, width=0.75 if i == 3 else 0.95, drawer=i % 2 == 0)
            items = [
                ("h48:bathroom_bar_of_soap_0_s0_1", (0.075, 0.05, 0.03), 0.1, -0.2, 0),
                ("h48:bathroom_shower_bottle_0", (0.08, 0.06, 0.15), 0.3, 0.15, 0),
            ]
            self.stored(s, items)
            self.asset(
                f"bath_tray_{i}",
                "h48:bathroom_soap_dish_0_s0_0",
                (x - 0.25, y, z + 0.9),
                (0.12, 0.09, 0.03),
                0.2,
                support=s["name"],
            )
        # Recreation wing: distinct consumer electronics and loose games rather than duplicated tins.
        self.asset(
            "recreation_bench", "h98:gaming_room_bench_0", (19.2, 7.8, 3.2), (1.8, 0.6, 0.48), 20, evidence="photo_048"
        )
        s = self.storage("recreation_console", 20.05, 4.32, 3.2, width=1.1, yaw=180)
        self.stored(
            s,
            [
                ("h98:gaming_room_wireless_game_controller_0", (0.18, 0.13, 0.065), 0.25, -0.35, 0),
                ("h98:gaming_room_gaming_headset_0", (0.20, 0.17, 0.10), 0.3, 0.30, 0),
            ],
        )
        self.asset(
            "game_console",
            "h98:gaming_room_gaming_console_0",
            (20.05, 4.32, 4.1),
            (0.32, 0.20, 0.07),
            2,
            support="recreation_console",
        )
        self.asset(
            "game_case",
            "h98:gaming_room_video_game_case_green_0",
            (19.69, 4.29, 4.1),
            (0.14, 0.19, 0.016),
            0.12,
            yaw=-13,
            support="recreation_console",
        )
        self.asset("recreation_pouf", "h98:gaming_room_pouf_0", (19.3, 6.2, 3.2), (0.65, 0.65, 0.42), 4)
        self.asset(
            "recreation_table", "h98:gaming_room_side_table_0", (18.6, 5.45, 3.2), (0.55, 0.55, 0.5), 5, kind="table"
        )
        self.asset(
            "board_game",
            "h98:living_room_board_game_box_0",
            (18.6, 5.45, 3.7),
            (0.28, 0.28, 0.055),
            0.7,
            support="recreation_table",
        )

    def sdf_storage_contacts(self) -> None:
        """Use exact box supports and retain independent SDF contents."""
        supports = set()
        for body in self.stage.GetPrimAtPath("/World/objects").GetChildren():
            has_sdf = any(
                prim.HasAPI(UsdPhysics.MeshCollisionAPI)
                and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "sdf"
                for prim in Usd.PrimRange(body)
            )
            support = body.GetCustomDataByKey("support")
            if has_sdf and support and support != "floor":
                supports.add(support if support.startswith("/") else "/World/objects/" + support)
        transforms = UsdGeom.XformCache()
        for path in sorted(supports):
            body = self.stage.GetPrimAtPath(path)
            if not body:
                raise ValueError(f"Missing SDF object's support: {path}")
            for prim in Usd.PrimRange(body):
                if not prim.IsA(UsdGeom.Cube) or not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                if prim.GetName() not in ("top", "bottom", "shelf"):
                    continue
                size = UsdGeom.Cube(prim).GetSizeAttr().Get()
                vertices, faces = box_surface_mesh(
                    size, np.asarray(transforms.GetLocalToWorldTransform(prim)), max_edge=0.05
                )
                # Preserve transforms, bindings and offsets; only the collision representation changes.
                prim.SetTypeName("Mesh")
                prim.RemoveProperty("size")
                mesh = UsdGeom.Mesh(prim)
                mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices, dtype=np.float32)))
                mesh.CreateFaceVertexCountsAttr([3] * len(faces))
                mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
                mesh.CreateSubdivisionSchemeAttr("none")
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("boundingCube")
                prim.SetCustomDataByKey("collision_source_shape", "box")
        for body in self.stage.GetPrimAtPath("/World/objects").GetChildren():
            support = body.GetCustomDataByKey("support")
            if not support:
                continue
            support = support if support.startswith("/") else "/World/objects/" + support
            if support not in supports:
                continue
            for prim in list(Usd.PrimRange(body)):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                if (
                    prim.HasAPI(UsdPhysics.MeshCollisionAPI)
                    and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "sdf"
                ):
                    continue
                convex_collider_to_sdf(prim)

    def build(self) -> None:
        """Save the standalone scene and an explicit observed/inferred inventory."""
        install_materials(self.stage, self.materials, self.texture_dir, self.metadata_dir)
        install_reference_materials(self)
        self.architecture()
        self.furnish()
        furnish_reference_details(self)
        for index, storage in enumerate(self.drawer_stores):
            support = storage["support_path"]
            occupied = any(self.stage.GetPrimAtPath(p).GetCustomDataByKey("support") == support for p in self.stored_items)
            if not occupied:
                if storage["name"].startswith("kitchen"):
                    path = native(self, storage["name"] + "_contents_cutlery",
                                  "KnifeFactory_9767186__spawn_asset_883869_",
                                  storage["world"]((0, 0, storage["support"])),
                                  (0.24, 0.032, 0.014), 0.08, support)
                    self.stored_items.append(path)
                else:
                    self.stored(storage, [("h47:bedroom_lined_notebook_0", (0.17, 0.22, 0.018), 0.18, 0, 0)])
        from scene_lib.kitchen_appliances import kitchen_additions

        kitchen_additions(self)
        from scene_lib.storage_contents import populate_compartments

        populate_compartments(self)
        self.contact_settings()
        self.sdf_storage_contacts()
        for prim in list(self.stage.Traverse()):
            if prim.IsA(UsdGeom.Cube) and UsdGeom.Imageable(prim).ComputeVisibility() != "invisible":
                material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
                self.textured_box(prim, material)
        self.stage.GetRootLayer().customLayerData = dict(
            reference=REFERENCE_URL,
            physics_timestep_seconds=1 / 240,
            fidelity="Photo-guided furniture approximations; concealed contents inferred.",
            source_area_sqft="Listings disagree: 3986 vs 4242; plan dimensions are approximate.",
        )
        self.stage.GetRootLayer().Save()
        report = dict(
            reference=REFERENCE_URL,
            levels=2,
            floor_to_floor_m=FLOOR_HEIGHT,
            rigid_bodies=sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in self.stage.Traverse()),
            movable_joints=sum(
                p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint) for p in self.stage.Traverse()
            ),
            fixtures=self.fixtures,
            stairs=self.stairs,
            ceilings=self.ceilings,
            bathrooms=BATHROOMS,
            walking_strips=WALKING_STRIPS,
            independent_stored_items=self.stored_items,
            inventory=self.inventory,
            unique_source_meshes=len({r["geometry_sha256"] for r in self.inventory if "geometry_sha256" in r}),
            modeled_floor_area_m2=2 * polygon_area([xy(p) for p in OUTLINE])
            - polygon_area([xy(p) for p in ATRIUM])
            - polygon_area([xy(p) for p in STAIR_VOID])
            - polygon_area([xy(p) for p in GARAGE]),
        )
        (self.metadata_dir / "asset_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ("inventory", "fixtures", "independent_stored_items")},
                indent=2,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    temporary = output.with_name(output.stem + ".building.usdc")
    BearLoopHouse(temporary).build()
    optimized = output.with_name(output.stem + ".optimized.usdc")
    subprocess.run([
        sys.executable, str(Path(__file__).with_name("simplify_sdf_colliders.py")),
        "--usd_path", str(temporary), "--output", str(optimized),
        "--report", str(output.parent / "metadata" / "collision_simplification.json"),
        "--device", "cpu", "--headless",
    ], check=True)
    optimized.replace(output)
    temporary.unlink()
