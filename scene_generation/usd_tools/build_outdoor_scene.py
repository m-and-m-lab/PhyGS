#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the scrub trail observed in IMG_3152/3153 with passive elastic stems."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

OUTPUT = Path(__file__).resolve().parents[1] / "outdoor_scene"


def trail_center(y: float) -> float:
    """Return the estimated trail center x [m] at longitudinal position y [m]."""
    return 0.22 * math.sin(y * 0.65)


def ground_height(x: float, y: float) -> float:
    """Return terrain elevation [m] at x, y [m]."""
    edge = abs(x - trail_center(y))
    bank = 0.38 * (1 - math.exp(-(max(0, edge - 0.48) ** 2) * 2))
    return 0.025 * y + bank + 0.018 * math.sin(4 * x + y) * math.sin(3 * y) + 0.008 * math.sin(19 * x + 9 * y)


class OutdoorScene:
    """Author portable visual geometry and joint-based vegetation in SI units."""

    def __init__(self, output: Path):
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(3153)
        self.stage = Usd.Stage.CreateNew(str(output / "scene.usdc"))
        UsdGeom.SetStageUpAxis(self.stage, "Z")
        UsdGeom.SetStageMetersPerUnit(self.stage, 1)
        UsdPhysics.SetStageKilogramsPerUnit(self.stage, 1)
        self.stage.SetDefaultPrim(UsdGeom.Xform.Define(self.stage, "/World").GetPrim())
        scene = UsdPhysics.Scene.Define(self.stage, "/World/physicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(9.81)
        self.materials = {}
        self.stems = []
        self.rocks = []

    def material(self, name: str, color: tuple, friction: float, texture: str | None = None) -> None:
        """Create a rough surface with Coulomb friction and optional relative texture."""
        mat = UsdShade.Material.Define(self.stage, f"/World/materials/{name}")
        shader = UsdShade.Shader.Define(self.stage, f"{mat.GetPath()}/surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.92)
        if texture:
            reader = UsdShade.Shader.Define(self.stage, f"{mat.GetPath()}/uv")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            for channel, filename in (
                ("diffuseColor", texture),
                ("normal", texture.replace(".png", "_normal.png")),
                ("roughness", texture.replace(".png", "_roughness.png")),
            ):
                tex = UsdShade.Shader.Define(self.stage, f"{mat.GetPath()}/{channel}")
                tex.CreateIdAttr("UsdUVTexture")
                tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(filename))
                tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(
                    "sRGB" if channel == "diffuseColor" else "raw"
                )
                tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
                tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
                if channel == "normal":
                    tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
                    tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
                    shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f)
                shader.GetInput(channel).ConnectToSource(tex.ConnectableAPI(), "r" if channel == "roughness" else "rgb")
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        phys = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        phys.CreateStaticFrictionAttr(friction)
        phys.CreateDynamicFrictionAttr(friction * 0.8)
        phys.CreateRestitutionAttr(0)
        self.materials[name] = mat

    def bind(self, prim: Usd.Prim, name: str) -> None:
        """Bind visual and physical material to geometry."""
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(self.materials[name])
        binding.Bind(self.materials[name], materialPurpose="physics")

    def mesh(self, path: str, points: list, faces: list, material: str, uv: list | None = None) -> Usd.Prim:
        """Create a mesh from points [m] and polygon vertex indices."""
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
        mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        vertices = np.asarray(points)
        normals = np.zeros_like(vertices)
        for face in faces:
            normal = np.cross(vertices[face[1]] - vertices[face[0]], vertices[face[2]] - vertices[face[0]])
            for vertex in face:
                normals[vertex] += normal
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths == 0):
            raise ValueError(f"Degenerate mesh normals: {path}")
        mesh.CreateNormalsAttr((normals / lengths[:, None]).tolist())
        mesh.SetNormalsInterpolation("vertex")
        if uv is not None:
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(uv)
        self.bind(mesh.GetPrim(), material)
        return mesh.GetPrim()

    def capsule(self, path: str, a: np.ndarray, b: np.ndarray, radius: float, material: str) -> Usd.Prim:
        """Create a capsule between endpoints [m] with radius [m]."""
        delta = b - a
        shape = UsdGeom.Capsule.Define(self.stage, path)
        shape.CreateAxisAttr("Z")
        shape.CreateHeightAttr(float(np.linalg.norm(delta)))
        shape.CreateRadiusAttr(radius)
        shape.AddTranslateOp().Set(Gf.Vec3d(*((a + b) / 2)))
        shape.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*delta)).GetQuat()))
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        self.bind(shape.GetPrim(), material)
        return shape.GetPrim()

    def textures(self) -> None:
        """Extract source-video surface patches and estimate micro-normal detail."""
        import cv2

        folder = self.output / "textures"
        folder.mkdir(exist_ok=True)
        patches = {
            "bark": ("IMG_3152.mp4", 11.0, (1130, 476, 1400, 529)),
            "soil": ("IMG_3153.mp4", 0.0, (790, 843, 1080, 1020)),
        }
        for name, (video, seconds, box) in patches.items():
            capture = cv2.VideoCapture(str(OUTPUT / video))
            capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            ok, frame = capture.read()
            capture.release()
            if not ok:
                raise RuntimeError(f"Cannot read {video} at {seconds} s")
            patch = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).crop(box)
            if name == "bark":
                patch = patch.transpose(Image.Transpose.ROTATE_90)
            pixels = np.asarray(patch).astype(float)
            # Remove broad illumination variation; these remain estimated, not measured albedo maps.
            illumination = np.asarray(patch.filter(ImageFilter.GaussianBlur(16))).astype(float).mean(axis=2)
            pixels *= (illumination.mean() / np.maximum(illumination, 1))[:, :, None]
            reflectance = {"soil": (130, 119, 98), "bark": (100, 88, 73), "rock": (112, 117, 116)}[name]
            pixels *= np.array(reflectance) / pixels.mean(axis=(0, 1))
            pixels = np.clip(pixels, 0, 255).astype(np.uint8)
            # Mirrored repeat has matching tile boundaries without a painted seam.
            tile = np.concatenate((pixels, pixels[:, ::-1]), axis=1)
            tile = np.concatenate((tile, tile[::-1]), axis=0)
            Image.fromarray(tile).save(folder / f"{name}.png")
            height = tile.astype(float).mean(axis=2) / 255
            dy, dx = np.gradient(height)
            normals = np.stack((-dx * 2, dy * 2, np.ones_like(dx)), axis=2)
            normals /= np.linalg.norm(normals, axis=2, keepdims=True)
            Image.fromarray(np.uint8((normals * 0.5 + 0.5) * 255)).save(folder / f"{name}_normal.png")
            Image.fromarray(np.uint8(np.clip(0.93 - height * 0.12, 0, 1) * 255)).save(folder / f"{name}_roughness.png")
        granite = Path(__file__).parent / "assets/bear_loop"
        crop = (30, 30, 630, 900)
        for source, target in (("diff", "rock"), ("nor_gl", "rock_normal"), ("rough", "rock_roughness")):
            patch = Image.open(granite / f"granite_tile_03_{source}_2k.jpg").crop(crop)
            if source == "diff":
                luminance = np.asarray(patch).astype(float).mean(axis=2)
                patch = Image.fromarray(np.uint8(np.clip(luminance[:, :, None] * np.array([0.96, 1.0, 1.02]), 0, 255)))
            elif source == "rough":
                patch = Image.fromarray(np.uint8(210 + np.asarray(patch).astype(float) * (45 / 255)))
            patch.save(folder / f"{target}.png")
        sources = dict(
            video_patches=patches,
            rock=dict(
                asset="Poly Haven granite_tile_03",
                license="CC0",
                local_source=str(granite.relative_to(Path(__file__).parent)),
                crop=crop,
                changes="Cropped inside one tile; neutralized color; raised roughness for weathered stone.",
            ),
        )
        (folder / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")

    def terrain(self) -> None:
        """Build an uneven 10 by 16 m trail and dynamic rocks with density [kg/m³]."""
        points, faces, uv = [], [], []
        nx, ny = 151, 241
        for j, y in enumerate(np.linspace(-3, 13, ny)):
            for i, x in enumerate(np.linspace(-5, 5, nx)):
                points.append((x, y, ground_height(x, y)))
                uv.append((x / 0.38, y / 0.38))
                if i and j:
                    k = j * nx + i
                    faces.extend([(k - nx - 1, k - nx, k), (k - nx - 1, k, k - 1)])
        terrain = self.mesh("/World/terrain", points, faces, "soil", uv)
        UsdPhysics.CollisionAPI.Apply(terrain)
        for i in range(180):
            y = float(self.rng.uniform(-2, 12))
            side = self.rng.choice([-1, 1])
            x = trail_center(y) + side * self.rng.uniform(0.6, 3.4)
            radius = self.rng.uniform(0.06, 0.24)
            if i < 6:
                x, y, radius = [
                    (1.15, 4.2, 0.34),
                    (-1.15, 5.4, 0.3),
                    (1.3, 5.5, 0.26),
                    (-1.15, 8, 0.32),
                    (1.1, 2.1, 0.22),
                    (-1.4, 3, 0.24),
                ][i]
            else:
                for _ in range(1000):
                    separated = all(math.hypot(x - rx, y - ry) > 1.4 * (radius + rr) for rx, ry, rr in self.rocks)
                    if separated and not (abs(y) < 0.9 and abs(x) < 1.3):
                        break
                    y = float(self.rng.uniform(-2, 12))
                    x = trail_center(y) + side * self.rng.uniform(0.6, 3.4)
                else:
                    raise RuntimeError("Unable to place non-overlapping rocks within the scene")
            self.rocks.append((float(x), float(y), float(radius)))
            points, faces, uv = [], [], []
            for ring in range(9):
                latitude = (ring + 0.03) / 8.06 * math.pi
                for k in range(16):
                    angle = k / 16 * 2 * math.pi
                    r = radius * (
                        1
                        + 0.12 * math.sin(3 * angle + i) * math.sin(latitude * 2)
                        + 0.07 * math.cos(5 * angle - latitude * 3)
                    )
                    sphere_point = np.array(
                        [math.sin(latitude) * math.cos(angle), math.sin(latitude) * math.sin(angle), math.cos(latitude)]
                    )
                    point = r * np.sign(sphere_point) * np.abs(sphere_point) ** 0.7 * (1, 0.8, 0.85)
                    # A fractured stone has a broad base rather than an ellipsoid's rolling contact.
                    point[2] = max(point[2], -radius * 0.45)
                    points.append(tuple(point))
                    uv.append((k / 16 * 3, ring / 8 * 3))
                    if ring:
                        a, b = ring * 16 + k, ring * 16 + (k + 1) % 16
                        faces.append((a, b, b - 16, a - 16))
            faces.extend([tuple(range(16)), tuple(reversed(range(128, 144)))])
            path = f"/World/rocks/rock_{i:03d}"
            body = UsdGeom.Xform.Define(self.stage, path)
            # Place the entire surface above the terrain, rather than starting with buried colliders.
            elevation = max(ground_height(x + px, y + py) - pz for px, py, pz in points) + 0.008
            body.AddTranslateOp().Set(Gf.Vec3d(x, y, elevation))
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateDensityAttr(2600)
            rock = self.mesh(path + "/shape", points, faces, "rock", uv)
            UsdPhysics.CollisionAPI.Apply(rock)
            UsdPhysics.MeshCollisionAPI.Apply(rock).CreateApproximationAttr("convexHull")
        points, faces = [], []
        for _ in range(14000):
            x, y = self.rng.uniform(-4.5, 4.5), self.rng.uniform(-2.8, 12.8)
            size = self.rng.uniform(0.008, 0.035)
            angle = self.rng.uniform(0, 2 * math.pi)
            center = np.array([x, y, ground_height(x, y) + 0.003])
            u = np.array([math.cos(angle), math.sin(angle), 0]) * size
            v = np.array([-math.sin(angle), math.cos(angle), 0]) * size * 0.45
            k = len(points)
            points.extend(
                [
                    tuple(center - u),
                    tuple(center + v),
                    tuple(center + u),
                    tuple(center - v),
                    tuple(center + [0, 0, size * 0.2]),
                ]
            )
            faces.extend([(k, k + 1, k + 4), (k + 1, k + 2, k + 4), (k + 2, k + 3, k + 4), (k + 3, k, k + 4)])
        self.mesh("/World/ground_litter", points, faces, "straw")

    def stem(self, index: int, x: float, y: float, height: float, radius: float, leafy: bool) -> None:
        """Create a three-link elastic stem; dimensions [m], beam modulus [Pa]."""
        root = np.array([x, y, ground_height(x, y) + radius])
        length = height / 3
        previous = None
        tilt = Gf.Rotation(
            Gf.Vec3d(0, 0, 1),
            Gf.Vec3d(
                float(self.rng.uniform(-0.42, 0.42)) if index > 1 else 0,
                float(self.rng.uniform(-0.42, 0.42)) if index > 1 else 0,
                1,
            ),
        )
        direction = np.array(tilt.TransformDir(Gf.Vec3d(0, 0, 1)))
        modulus = 2e9 if leafy else 3e8
        density = 650 if leafy else 350
        paths = []
        for segment in range(3):
            path = f"/World/vegetation/stem_{index:04d}/link_{segment}"
            origin = root + direction * segment * length
            body = UsdGeom.Xform.Define(self.stage, path)
            body.AddTranslateOp().Set(Gf.Vec3d(*origin))
            body.AddOrientOp().Set(Gf.Quatf(tilt.GetQuat()))
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            r = radius * (1 - segment * 0.19)
            mass = density * math.pi * r * r * length
            UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(mass)
            self.capsule(path + "/stem", np.zeros(3), np.array([0, 0, length]), r, "twig" if leafy else "straw")
            joint = UsdPhysics.Joint.Define(self.stage, f"/World/joints/stem_{index:04d}_{segment}")
            joint.CreateBody1Rel().SetTargets([path])
            if previous:
                joint.CreateBody0Rel().SetTargets([previous])
                joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, length))
            else:
                joint.CreateLocalPos0Attr(Gf.Vec3f(*root))
                joint.CreateLocalRot0Attr(Gf.Quatf(tilt.GetQuat()))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0))
            joint.CreateCollisionEnabledAttr(False)
            for axis in ("transX", "transY", "transZ", "rotZ"):
                limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
                limit.CreateLowAttr(1)
                limit.CreateHighAttr(-1)
            stiffness = modulus * math.pi * r**4 / (4 * length)
            for axis in ("rotX", "rotY"):
                limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
                limit.CreateLowAttr(-85)
                limit.CreateHighAttr(85)
                drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
                drive.CreateTypeAttr("force")
                drive.CreateStiffnessAttr(stiffness)
                drive.CreateDampingAttr(0.3 * 2 * math.sqrt(stiffness * mass * length**2 / 3))
                drive.CreateTargetPositionAttr(0)
            # Twigs and leaves move with their parent stem link; no billboard bushes.
            points, faces = [], []
            twig_count = (3, 7, 10)[segment] if leafy else 3
            for twig in range(twig_count):
                angle = float(self.rng.uniform(0, 2 * math.pi))
                reach = float(self.rng.uniform(0.12, 0.38) if leafy else self.rng.uniform(0.035, 0.11))
                a = np.array([0, 0, length * self.rng.uniform(0.05, 0.95)])
                b = a + np.array(
                    [math.cos(angle) * reach, math.sin(angle) * reach, reach * self.rng.uniform(-0.2, 1.1)]
                )
                self.capsule(path + f"/twig_{twig}", a, b, r * 0.35, "twig" if leafy else "straw")
                mass += density * math.pi * (r * 0.35) ** 2 * np.linalg.norm(b - a)
                leaf_count = (12 if index % 5 else 2) if leafy else 1
                for leaf in range(leaf_count):
                    t = float(self.rng.uniform(0.3, 1))
                    center = a + t * (b - a)
                    theta = angle + float(self.rng.uniform(-2.4, 2.4))
                    size = float(self.rng.uniform(0.025, 0.065) if leafy else self.rng.uniform(0.09, 0.18))
                    leaf_direction = np.array([math.cos(theta), math.sin(theta), self.rng.uniform(-0.7, 0.9)]) * size
                    width = np.array([-math.sin(theta), math.cos(theta), 0]) * size * (0.38 if leafy else 0.025)
                    k = len(points)
                    points.append(tuple(center))
                    for row in range(1, 8):
                        t_leaf = row / 8
                        half_width = math.sin(math.pi * t_leaf) ** 0.8 * width
                        midpoint = center + leaf_direction * t_leaf + np.array([0, 0, -size * 0.25 * t_leaf**2])
                        for side in (-1, 0, 1):
                            point = midpoint + side * half_width
                            point = point + (0, 0, size * 0.055 * math.sin(math.pi * t_leaf) * (1 - abs(side)))
                            points.append(tuple(point))
                        a_leaf = k + 1 + (row - 1) * 3
                        if row == 1:
                            faces.extend([(k, a_leaf, a_leaf + 1), (k, a_leaf + 1, a_leaf + 2)])
                        else:
                            faces.extend(
                                [
                                    (a_leaf - 3, a_leaf, a_leaf + 1, a_leaf - 2),
                                    (a_leaf - 2, a_leaf + 1, a_leaf + 2, a_leaf - 1),
                                ]
                            )
                    tip = len(points)
                    points.append(tuple(center + leaf_direction + (0, 0, -size * 0.25)))
                    faces.extend([(tip - 3, tip, tip - 2), (tip - 2, tip, tip - 1)])
                    mass += 0.12 * size * np.linalg.norm(width) * math.pi / 2
            self.mesh(path + "/leaves", points, faces, f"leaf_{index % 5}" if leafy else "straw")
            UsdPhysics.MassAPI(body.GetPrim()).GetMassAttr().Set(float(mass))
            for axis in ("rotX", "rotY"):
                UsdPhysics.DriveAPI(joint.GetPrim(), axis).GetDampingAttr().Set(
                    0.6 * math.sqrt(stiffness * mass * length**2 / 3)
                )
            previous = path
            paths.append(path)
        self.stems.append(
            dict(root=root.tolist(), height=height, radius=radius, leafy=leafy, paths=paths, modulus_Pa=modulus)
        )

    def log(self) -> None:
        """Create one 8 kg forked rigid log with collision on the visible bark."""
        body = UsdGeom.Xform.Define(self.stage, "/World/log")
        body.AddTranslateOp().Set(Gf.Vec3d(0, 4.6, ground_height(0, 4.6) + 0.15))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(8)
        branches = [
            (
                [(-1.45, 0.2, 0.1), (-0.85, 0, 0.35), (-0.2, 0.04, 0.52), (0.45, 0.23, 0.48), (1.3, 0.32, 0.64)],
                [0.12, 0.12, 0.10, 0.075, 0.045],
            ),
            (
                [(-1.1, 0.15, 0.19), (-0.6, -0.25, 0.14), (0.1, -0.38, 0.13), (0.9, -0.12, 0.25), (1.5, 0.3, 0.56)],
                [0.065, 0.062, 0.046, 0.032, 0.016],
            ),
            ([(-0.2, 0.04, 0.52), (0, 0.42, 0.7), (0.32, 0.62, 0.72)], [0.055, 0.032, 0.012]),
        ]
        for branch, (nodes, radii) in enumerate(branches):
            points, faces, uv = [], [], []
            nodes = np.asarray(nodes, dtype=float)
            rings = []
            for j in range(len(nodes) - 1):
                for step in range(10):
                    t = step / 10
                    p0, p1, p2, p3 = nodes[max(0, j - 1)], nodes[j], nodes[j + 1], nodes[min(len(nodes) - 1, j + 2)]
                    center = 0.5 * (
                        (2 * p1)
                        + (-p0 + p2) * t
                        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                        + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3
                    )
                    rings.append((center, radii[j] * (1 - t) + radii[j + 1] * t))
            rings.append((nodes[-1], radii[-1]))
            for j, (center, r) in enumerate(rings):
                tangent = rings[min(j + 1, len(rings) - 1)][0] - rings[max(0, j - 1)][0]
                tangent /= np.linalg.norm(tangent)
                u = np.cross(tangent, [0, 0, 1])
                u /= np.linalg.norm(u)
                v = np.cross(tangent, u)
                for k in range(25):
                    angle = k / 24 * 2 * math.pi
                    ridge = 1 + 0.065 * math.sin(11 * angle + j * 0.25) + 0.025 * math.sin(19 * angle - j * 0.4)
                    points.append(tuple(center + r * ridge * (u * math.cos(angle) + v * math.sin(angle))))
                    uv.append((k / 24 * 3, j * 0.12))
                    if j and k:
                        a = j * 25 + k
                        faces.append((a - 26, a - 25, a, a - 1))
            faces.extend([tuple(reversed(range(24))), tuple((len(rings) - 1) * 25 + k for k in range(24))])
            bark = self.mesh(f"/World/log/bark_{branch}", points, faces, "bark", uv)
            UsdPhysics.CollisionAPI.Apply(bark)
            UsdPhysics.MeshCollisionAPI.Apply(bark).CreateApproximationAttr("convexDecomposition")

    def build(self) -> None:
        """Write the scene, local textures and estimated physical parameters."""
        self.textures()
        for name in ("soil", "bark", "rock"):
            self.material(name, (0.5, 0.5, 0.5), 0.8 if name == "soil" else 0.65, f"textures/{name}.png")
        self.material("twig", (0.23, 0.18, 0.11), 0.55)
        self.material("straw", (0.47, 0.35, 0.18), 0.45)
        for i, color in enumerate(
            ((0.12, 0.19, 0.055), (0.22, 0.28, 0.09), (0.28, 0.30, 0.12), (0.16, 0.22, 0.12), (0.36, 0.29, 0.11))
        ):
            self.material(f"leaf_{i}", color, 0.5)
        self.terrain()
        # Deterministic isolated stems also provide reproducible contact-test targets.
        self.stem(0, -0.72, 0, 0.8, 0.004, True)
        self.stem(1, 0.72, 0, 0.55, 0.0018, False)
        for i in range(2, 362):
            if (i - 2) % 6 == 0:
                cluster_y = float(self.rng.uniform(1.2, 12))
                side = self.rng.choice([-1, 1])
                cluster_x = trail_center(cluster_y) + side * self.rng.uniform(1.0, 3.3)
                cluster_height = float(self.rng.uniform(1.0, 2.1))
            x = cluster_x + float(self.rng.uniform(-0.22, 0.22))
            y = cluster_y + float(self.rng.uniform(-0.22, 0.22))
            leafy = i % 3 != 0
            self.stem(
                i,
                x,
                y,
                float(cluster_height * self.rng.uniform(0.8, 1.1) if leafy else self.rng.uniform(0.35, 1.1)),
                float(self.rng.uniform(0.004, 0.008) if leafy else self.rng.uniform(0.0013, 0.0023)),
                leafy,
            )
        self.log()
        sun = UsdLux.DistantLight.Define(self.stage, "/World/sun")
        sun.CreateIntensityAttr(3500)
        sun.CreateAngleAttr(0.53)
        sun.CreateColorAttr(Gf.Vec3f(1, 0.94, 0.82))
        sun.AddRotateXYZOp().Set(Gf.Vec3f(28, -35, -25))
        sky = UsdLux.DomeLight.Define(self.stage, "/World/sky")
        sky.CreateIntensityAttr(350)
        sky.CreateColorAttr(Gf.Vec3f(0.68, 0.79, 1))
        camera = UsdGeom.Camera.Define(self.stage, "/World/cameras/trail")
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(0, -1.8, 1.6), Gf.Vec3d(0, 5, 0.85), Gf.Vec3d(0, 0, 1))
        camera.AddTransformOp().Set(view.GetInverse())
        camera.CreateFocalLengthAttr(22)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.03, 300))
        self.stage.GetRootLayer().Save()
        (self.output / "parameters.json").write_text(
            json.dumps(
                dict(
                    seed=3153,
                    log_mass_kg=8,
                    rock_density_kg_m3=2600,
                    stems=self.stems,
                    note="Estimated dimensions and material properties; not measured from calibrated video.",
                ),
                indent=2,
            )
            + "\n"
        )
        print(f"Saved {self.output / 'scene.usdc'}; {len(self.stems)} elastic stems; 8 kg log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT)
    OutdoorScene(parser.parse_args().output_dir).build()
