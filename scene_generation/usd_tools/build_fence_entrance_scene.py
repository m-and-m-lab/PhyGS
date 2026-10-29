#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reconstruct the chain-link gate entrance in IMG_3132.mp4."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from build_low_branches_scene import LowBranchesScene
from build_outdoor_scene import OUTPUT, ground_height
from PIL import Image, ImageFilter
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

DEFAULT_OUTPUT = OUTPUT / "fence_entrance"
HINGE = (-0.07, 3.0, 0.45)
GATE_WIDTH = 1.06
GATE_HEIGHT = 1.95


def elevation(x: float, y: float) -> float:
    """Return ground elevation [m] beneath the entrance and dry embankment."""
    if 2.1 <= y <= 5.7:
        return 0.45
    if y > 5.7:
        return 0.45 + 0.44 * (y - 5.7) + 0.06 * math.sin(x * 1.5) * min(y - 5.7, 1)
    return 0.45 - 0.025 * (2.1 - y) + 0.014 * math.sin(x * 3 + y) * min(2.1 - y, 1)


class FenceEntranceScene(LowBranchesScene):
    """Create passive, outward-opening gates and surrounding terrain."""

    def __init__(self, output: Path):
        super().__init__(output)
        self.rng = np.random.default_rng(3132)
        self.gate_mass = 0.0
        self.vehicle_gates = []

    def reference_textures(self) -> None:
        """Estimate soil and dry-grass surface maps from the fence reference video."""
        import cv2

        self.textures()
        capture = cv2.VideoCapture(str(OUTPUT / "IMG_3132.mp4"))
        capture.set(cv2.CAP_PROP_POS_MSEC, 750)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError("Cannot decode IMG_3132.mp4 at 0.75 s")
        reference = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        crops = {"soil": ((120, 610, 400, 820), (160, 137, 108)), "dry_soil": ((795, 145, 1030, 245), (170, 139, 84))}
        for name, (box, reflectance) in crops.items():
            patch = reference.crop(box)
            pixels = np.array(patch, dtype=float)
            illumination = np.asarray(patch.filter(ImageFilter.GaussianBlur(20))).mean(axis=2)
            pixels *= (illumination.mean() / np.maximum(illumination, 1))[:, :, None]
            pixels *= np.array(reflectance) / pixels.mean(axis=(0, 1))
            pixels = np.clip(pixels, 0, 255).astype(np.uint8)
            tile = np.concatenate((pixels, pixels[:, ::-1]), axis=1)
            tile = np.concatenate((tile, tile[::-1]), axis=0)
            Image.fromarray(tile).save(self.output / f"textures/{name}.png")
            height = tile.mean(axis=2) / 255
            dy, dx = np.gradient(height)
            normal = np.stack((-dx, dy, np.ones_like(dx)), axis=2)
            normal /= np.linalg.norm(normal, axis=2, keepdims=True)
            Image.fromarray(np.uint8((normal * 0.5 + 0.5) * 255)).save(self.output / f"textures/{name}_normal.png")
            Image.fromarray(np.full(height.shape, 235, dtype=np.uint8)).save(
                self.output / f"textures/{name}_roughness.png"
            )
        # A sun-free sky gradient provides background and diffuse illumination.
        sky = np.zeros((256, 512, 3), dtype=np.float32)
        for row in range(256):
            t = min(row / 128, 1)
            color = np.array([0.10, 0.30, 0.72]) * (1 - t**3) + np.array([0.70, 0.80, 0.95]) * t**3
            sky[row] = color if row <= 128 else (0.20, 0.17, 0.12)
        if not cv2.imwrite(str(self.output / "textures/sky.hdr"), sky[:, :, ::-1]):
            raise RuntimeError("Cannot write sky texture")
        source_path = self.output / "textures/sources.json"
        sources = json.loads(source_path.read_text())
        sources["video_patches"]["soil"] = ["IMG_3132.mp4", 0.75, list(crops["soil"][0])]
        sources["video_patches"]["dry_soil"] = ["IMG_3132.mp4", 0.75, list(crops["dry_soil"][0])]
        sources["sky"] = "Procedural sun-free clear-sky gradient; no external image"
        source_path.write_text(json.dumps(sources, indent=2) + "\n")

    def bar(
        self,
        path: str,
        a: tuple[float, float, float],
        b: tuple[float, float, float],
        radius: float,
        material: str = "steel",
    ) -> None:
        """Create a cylindrical bar with endpoints and radius [m], including collision."""
        start, end = np.array(a, dtype=float), np.array(b, dtype=float)
        delta = end - start
        length = float(np.linalg.norm(delta))
        if path.startswith("/World/vehicle_gate/"):
            # Convex cylinder meshes avoid the analytic-cylinder contact stall on long gate rails.
            sides = 24
            points = [
                (radius * math.cos(k * math.tau / sides), radius * math.sin(k * math.tau / sides), z)
                for z in (-length / 2, length / 2)
                for k in range(sides)
            ]
            faces = [(k, (k + 1) % sides, (k + 1) % sides + sides, k + sides) for k in range(sides)]
            faces.extend((tuple(reversed(range(sides))), tuple(range(sides, 2 * sides))))
            prim = self.mesh(path, points, faces, material)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
            bar = UsdGeom.Xformable(prim)
        else:
            bar = UsdGeom.Cylinder.Define(self.stage, path)
            bar.CreateRadiusAttr(radius)
            bar.CreateHeightAttr(length)
            bar.CreateAxisAttr("Z")
        bar.AddTranslateOp().Set(Gf.Vec3d(*((start + end) / 2)))
        bar.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*delta)).GetQuat()))
        prim = bar.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        prim.AddAppliedSchema("PhysxCollisionAPI")
        prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.001)
        prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0)
        self.bind(prim, material)
        if path.startswith(("/World/gate/", "/World/vehicle_gate/")):
            area = math.pi * (radius**2 - max(0, radius - 0.002) ** 2)
            self.gate_mass += 7850 * area * length

    def box(
        self, path: str, center: tuple[float, float, float], size: tuple[float, float, float], material: str
    ) -> None:
        """Create a fixed collision box with center and dimensions [m]."""
        box = UsdGeom.Cube.Define(self.stage, path)
        box.CreateSizeAttr(1)
        box.AddTranslateOp().Set(Gf.Vec3d(*center))
        box.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        self.bind(box.GetPrim(), material)

    def panel(self, path: str, left: float, right: float, bottom: float, top: float, y: float, brace: bool) -> None:
        """Create a framed chain-link panel; all coordinates [m] are parent-local."""
        for name, a, b in (
            ("left", (left, y, bottom), (left, y, top)),
            ("right", (right, y, bottom), (right, y, top)),
            ("top", (left, y, top), (right, y, top)),
            ("bottom", (left, y, bottom), (right, y, bottom)),
            ("middle", (left, y, (bottom + top) / 2), (right, y, (bottom + top) / 2)),
        ):
            self.bar(f"{path}/frame_{name}", a, b, 0.019)
        if brace:
            self.bar(path + "/brace", (left, y, top), (right, y, bottom), 0.007)
            self.bar(path + "/vertical", ((left + right) / 2, y, bottom), ((left + right) / 2, y, top), 0.014)
        width, height = right - left, top - bottom
        wire = 0
        # Two offset families preserve open diamonds; there is no collision sheet across the holes.
        for slope in (-1, 1):
            for intercept in np.arange(-width, height + width, 0.075):
                hits = []
                for x in (0, width):
                    z = slope * x + intercept
                    if 0 <= z <= height:
                        hits.append((x, z))
                for z in (0, height):
                    x = (z - intercept) / slope
                    if 0 < x < width:
                        hits.append((x, z))
                if len(hits) != 2 or np.linalg.norm(np.subtract(hits[0], hits[1])) < 0.008:
                    continue
                a, b = [(left + x, y + slope * 0.0017, bottom + z) for x, z in hits]
                self.bar(f"{path}/wire_{wire:04d}", a, b, 0.0016, "wire")
                wire += 1

    def fence(self) -> None:
        """Build the fixed fence and three passive outward-only gate leaves."""
        z = HINGE[2]
        for i, x in enumerate((-8, -5.7, -3.4, -1.22, 0, 5.9, 8.4, 10.9)):
            self.bar(f"/World/fence/posts/post_{i}", (x, 3, z - 0.3), (x, 3, z + 2.15), 0.045)
            self.bar(f"/World/fence/posts/cap_{i}", (x, 3, z + 2.13), (x, 3, z + 2.18), 0.052)
        for i, (left, right) in enumerate(((-8, -5.7), (-5.7, -3.4), (-3.4, -1.22), (5.9, 8.4), (8.4, 10.9))):
            self.panel(f"/World/fence/panel_{i}", left + 0.04, right - 0.04, z + 0.07, z + 2.05, 3, False)
        for i, (pivot, left, right, limits) in enumerate(((0.08, 0.0, 2.86, (0, 105)), (5.82, -2.82, 0.0, (-105, 0)))):
            path = f"/World/vehicle_gate/panel_{i}"
            anchor = (pivot, 3.0, z)
            body = UsdGeom.Xform.Define(self.stage, path)
            body.AddTranslateOp().Set(Gf.Vec3d(*anchor))
            body.AddOrientOp().Set(Gf.Quatf(1))
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            self.gate_mass = 0.0
            self.panel(path + "/mesh", left, right, 0.12, 2.03, 0, True)
            for j, height in enumerate((0.28, 1.78)):
                self.bar(f"{path}/hinge_sleeve_{j}", (0, 0, height - 0.04), (0, 0, height + 0.04), 0.018)
            if i == 0:
                # Raised bolt travels with the leaf and does not lock it to the ground.
                self.bar(path + "/drop_bolt", (right, -0.04, 0.18), (right, -0.04, 0.80), 0.011)
            UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(self.gate_mass)
            joint = UsdPhysics.RevoluteJoint.Define(self.stage, f"/World/joints/vehicle_hinge_{i}")
            joint.CreateBody1Rel().SetTargets([path])
            joint.CreateAxisAttr("Z")
            joint.CreateLocalPos0Attr(Gf.Vec3f(*anchor))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0))
            joint.CreateLowerLimitAttr(limits[0])
            joint.CreateUpperLimitAttr(limits[1])
            joint.CreateCollisionEnabledAttr(False)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(0)
            drive.CreateDampingAttr(2.0 * math.pi / 180)
            drive.CreateTargetVelocityAttr(0)
            self.vehicle_gates.append(
                dict(
                    path=path,
                    hinge_m=anchor,
                    mass_kg=self.gate_mass,
                    opening_limits_deg=limits,
                    hinge_damping_Nm_s_rad=2.0,
                )
            )
        self.gate_mass = 0.0
        body = UsdGeom.Xform.Define(self.stage, "/World/gate")
        body.AddTranslateOp().Set(Gf.Vec3d(*HINGE))
        body.AddOrientOp().Set(Gf.Quatf(1))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        self.panel("/World/gate/panel", -1.06, 0, 0.10, GATE_HEIGHT + 0.10, 0, False)
        for i, height in enumerate((0.28, 1.78)):
            self.bar(f"/World/gate/hinge_arm_{i}", (-0.025, 0, height), (0, 0, height), 0.013)
            self.bar(f"/World/gate/hinge_sleeve_{i}", (0, 0, height - 0.04), (0, 0, height + 0.04), 0.018)
        self.bar("/World/gate/push_handle", (-0.98, -0.075, 0.88), (-0.98, -0.075, 1.17), 0.011)
        for i, height in enumerate((0.88, 1.17)):
            self.bar(f"/World/gate/handle_mount_{i}", (-0.98, 0, height), (-0.98, -0.075, height), 0.008)
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(self.gate_mass)
        # The hinge pin sits outside the post, with clearance for the rotating bearing.
        joint = UsdPhysics.RevoluteJoint.Define(self.stage, "/World/joints/gate_hinge")
        joint.CreateBody1Rel().SetTargets(["/World/gate"])
        joint.CreateAxisAttr("Z")
        joint.CreateLocalPos0Attr(Gf.Vec3f(*HINGE))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLowerLimitAttr(-105)
        joint.CreateUpperLimitAttr(0)
        joint.CreateCollisionEnabledAttr(False)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0)
        drive.CreateDampingAttr(1.0 * math.pi / 180)
        drive.CreateTargetVelocityAttr(0)
        self.box("/World/fence/closed_stop", (-1.15, 2.951, z + 1.01), (0.14, 0.04, 0.10), "steel")
        # An explicit stop and a joint limit define the permitted side without a controller.
        self.box("/World/fence/stop_mount", (-1.225, 2.97, z + 1.01), (0.04, 0.09, 0.16), "steel")

    def ground(self) -> None:
        """Author sandy ground, level paved lane, hill and scattered loose stones [m]."""
        points, faces, uv, hill_faces = [], [], [], []
        nx, ny = 151, 151
        for j, y in enumerate(np.linspace(-5, 13, ny)):
            for i, x in enumerate(np.linspace(-10, 12, nx)):
                points.append((float(x), float(y), elevation(float(x), float(y))))
                uv.append((x * 1.1, y * 1.1))
                if i and j:
                    a = j * nx + i
                    faces.append((a - nx - 1, a - nx, a, a - 1))
                    if y > 5.82:
                        hill_faces.append(len(faces) - 1)
        prim = self.mesh("/World/terrain", points, faces, "soil", uv)
        UsdPhysics.CollisionAPI.Apply(prim)
        subset = UsdGeom.Subset.Define(self.stage, "/World/terrain/hill_material")
        subset.CreateElementTypeAttr("face")
        subset.CreateFamilyNameAttr("materialBind")
        subset.CreateIndicesAttr(hill_faces)
        self.bind(subset.GetPrim(), "dry_soil")
        self.box("/World/paved_lane", (1, 4.15, 0.405), (22, 3.05, 0.10), "concrete")
        self.box("/World/curb_1", (1, 5.72, 0.445), (22, 0.12, 0.08), "curb")
        # Keep the pedestrian threshold flush and clear of the curb.
        for i, (left, right) in enumerate(((-10, -1.28), (0.1, 12))):
            self.box(f"/World/front_curb_{i}", ((left + right) / 2, 2.58, 0.445), (right - left, 0.12, 0.08), "curb")
        for i in range(95):
            x, y = float(self.rng.uniform(-8, 10)), float(self.rng.uniform(-3, 11))
            if 1.8 < y < 6 or (-1.5 < x < 0.3 and y < 3):
                continue
            size = float(self.rng.uniform(0.035, 0.13))
            vertices, polygons, coords = [], [], []
            for ring in range(7):
                latitude = (ring + 0.04) / 6.08 * math.pi
                for k in range(12):
                    theta = k * math.tau / 12
                    r = size * (1 + 0.12 * math.sin(3 * theta + i) * math.sin(latitude))
                    vertices.append(
                        (
                            r * math.sin(latitude) * math.cos(theta),
                            r * 0.8 * math.sin(latitude) * math.sin(theta),
                            max(-size * 0.22, r * 0.55 * math.cos(latitude)),
                        )
                    )
                    coords.append((k / 12, ring / 6))
                    if ring:
                        a, b = ring * 12 + k, ring * 12 + (k + 1) % 12
                        polygons.append((a, b, b - 12, a - 12))
            polygons.extend([tuple(range(12)), tuple(reversed(range(72, 84)))])
            path = f"/World/rocks/rock_{i:03d}"
            body = UsdGeom.Xform.Define(self.stage, path)
            z = max(elevation(x + px, y + py) - pz for px, py, pz in vertices) + 0.008
            body.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateDensityAttr(2600)
            prim = self.mesh(path + "/shape", vertices, polygons, "rock", coords)
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")

    def planting(self) -> None:
        """Place flexible scrub and straw on the hill, with rooted shade trees."""
        cluster_x, cluster_y = 0.0, 0.0
        for i in range(650):
            leafy = i < 200
            if leafy:
                if i % 5 == 0:
                    cluster_x, cluster_y = self.rng.uniform(-8, 10), self.rng.uniform(6.4, 12.2)
                x, y = (
                    float(cluster_x + self.rng.uniform(-0.25, 0.25)),
                    float(cluster_y + self.rng.uniform(-0.25, 0.25)),
                )
            else:
                x, y = float(self.rng.uniform(-8, 10)), float(self.rng.uniform(6.1, 12.5))
            height = float(self.rng.uniform(0.75, 1.7) if leafy else self.rng.uniform(0.35, 1.05))
            self.stem(i, x, y, height, 0.005 if leafy else 0.0017, leafy)
            dz = elevation(x, y) - ground_height(x, y)
            for path in self.stems[-1]["paths"]:
                attr = self.stage.GetPrimAtPath(path).GetAttribute("xformOp:translate")
                attr.Set(attr.Get() + Gf.Vec3d(0, 0, dz))
            joint = UsdPhysics.Joint(self.stage.GetPrimAtPath(f"/World/joints/stem_{i:04d}_0"))
            joint.GetLocalPos0Attr().Set(joint.GetLocalPos0Attr().Get() + Gf.Vec3f(0, 0, dz))
            self.stems[-1]["root"][2] += dz
        for tree, (x, y) in enumerate(((-4.5, 6.8), (7.2, 1.2), (5.5, 9.5))):
            root = np.array([x, y, elevation(x, y)])
            fork = root + [0.2, 0.1, 3.1]
            self.tube(
                f"/World/trees/tree_{tree}/trunk", np.array([root, root + [0.05, 0, 1.4], fork]), [0.19, 0.14, 0.085]
            )
            for branch in range(9):
                angle = branch * math.tau / 9
                tip = fork + [math.cos(angle) * 2.1, math.sin(angle) * 1.6, self.rng.uniform(0.15, 1.1)]
                self.tube(f"/World/trees/tree_{tree}/wood_{branch}", np.array([fork, tip]), [0.06, 0.008])
                points, faces, uv = [], [], []
                for _ in range(450):
                    center = (
                        fork + (tip - fork) * self.rng.uniform(0.3, 1.15) + self.rng.normal(0, (0.25, 0.25, 0.5), 3)
                    )
                    vertices, polygons, coords = self.leaf_geometry(center, float(self.rng.uniform(0.07, 0.14)))
                    faces.extend(tuple(k + len(points) for k in polygon) for polygon in polygons)
                    points.extend(vertices)
                    uv.extend(coords)
                self.mesh(f"/World/trees/tree_{tree}/leaves_{branch}", points, faces, f"oak_{branch % 4}", uv)

    def build(self) -> None:
        """Save the gate scene, physical parameters and portable materials."""
        self.reference_textures()
        for name in ("soil", "dry_soil", "bark", "rock"):
            self.material(name, (0.5, 0.5, 0.5), 0.75, f"textures/{name}.png")
        self.material("steel", (0.58, 0.61, 0.64), 0.45)
        shader = UsdShade.Shader(self.stage.GetPrimAtPath("/World/materials/steel/surface"))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1)
        shader.GetInput("roughness").Set(0.34)
        self.material("wire", (0.28, 0.30, 0.32), 0.45)
        wire_shader = UsdShade.Shader(self.stage.GetPrimAtPath("/World/materials/wire/surface"))
        wire_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.9)
        wire_shader.GetInput("roughness").Set(0.5)
        self.material("concrete", (0.46, 0.44, 0.39), 0.8)
        self.material("curb", (0.57, 0.56, 0.51), 0.8)
        self.material("twig", (0.19, 0.15, 0.09), 0.55)
        self.material("straw", (0.54, 0.37, 0.13), 0.45)
        self.leaf_materials()
        for i in range(5):
            self.material(f"leaf_{i}", (0.055 + i * 0.015, 0.10 + i * 0.015, 0.025), 0.45)
        self.ground()
        self.fence()
        self.planting()
        self.material("building", (0.65, 0.66, 0.63), 0.7)
        self.material("window", (0.16, 0.23, 0.27), 0.5)
        self.box("/World/background_building/facade", (1.8, 13.1, 4.0), (12, 0.35, 1.0), "building")
        self.box("/World/background_building/roof", (1.8, 13.4, 4.55), (12.3, 1.8, 0.15), "building")
        for i in range(8):
            x = -3.3 + i * 1.45
            self.box(f"/World/background_building/window_{i}", (x, 12.9, 4.22), (1.18, 0.03, 0.40), "window")
        sun = UsdLux.DistantLight.Define(self.stage, "/World/sun")
        sun.CreateIntensityAttr(3000)
        sun.CreateAngleAttr(0.6)
        sun.CreateColorAttr(Gf.Vec3f(1, 0.97, 0.90))
        sun.AddRotateXYZOp().Set(Gf.Vec3f(28, -30, -30))
        sky = UsdLux.DomeLight.Define(self.stage, "/World/sky")
        sky.CreateIntensityAttr(450)
        sky.CreateColorAttr(Gf.Vec3f(1))
        sky.CreateTextureFileAttr(Sdf.AssetPath("textures/sky.hdr"))
        camera = UsdGeom.Camera.Define(self.stage, "/World/cameras/entrance")
        camera.AddTransformOp().Set(
            Gf.Matrix4d().SetLookAt(Gf.Vec3d(3, -4.8, 2.0), Gf.Vec3d(2.6, 3, 1.45), Gf.Vec3d(0, 0, 1)).GetInverse()
        )
        camera.CreateFocalLengthAttr(24)
        self.stage.GetRootLayer().Save()
        metadata = dict(
            reference_video="IMG_3132.mp4",
            hinge_m=HINGE,
            width_m=GATE_WIDTH,
            height_m=GATE_HEIGHT,
            gate_mass_kg=self.gate_mass,
            vehicle_gates=self.vehicle_gates,
            opening_limits_deg=[-105, 0],
            opening_direction="From dirt side (y < 3) toward paved lane (+Y)",
            hinge_damping_Nm_s_rad=1.0,
            stems=self.stems,
            note=(
                "Dimensions and material properties are estimates. All three gate leaves are passive; "
                "vehicle drop bolt is raised. Wire is rigid with its frame; "
                "no wire deformation or latch release is simulated."
            ),
        )
        (self.output / "parameters.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved {self.output / 'scene.usdc'}; gate mass {self.gate_mass:.2f} kg; range [-105, 0] degrees.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    FenceEntranceScene(parser.parse_args().output_dir).build()
