#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recreate the leafy, low-hanging tree canopy in IMG_3154 and IMG_3155."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from build_outdoor_scene import OUTPUT, OutdoorScene, ground_height, trail_center
from PIL import Image
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics

DEFAULT_OUTPUT = OUTPUT / "low_branches"


class LowBranchesScene(OutdoorScene):
    """Build a rooted tree with six-link elastic pendant branches and leaf contact."""

    def __init__(self, output: Path):
        super().__init__(output)
        self.rng = np.random.default_rng(3155)
        self.branches = []

    def stem(self, index: int, x: float, y: float, height: float, radius: float, leafy: bool) -> None:
        """Create a ground stem with dimensions [m] and degree-based USD angular drives."""
        super().stem(index, x, y, height, radius, leafy)
        for segment in range(3):
            joint = self.stage.GetPrimAtPath(f"/World/joints/stem_{index:04d}_{segment}")
            for axis in ("rotX", "rotY"):
                drive = UsdPhysics.DriveAPI(joint, axis)
                for attribute in (drive.GetStiffnessAttr(), drive.GetDampingAttr()):
                    attribute.Set(attribute.Get() * math.pi / 180)

    def tube(self, path: str, nodes: np.ndarray, radii: list[float], dynamic: bool = False) -> None:
        """Create a textured wood surface and matching collision from centers/radii [m]."""
        points, faces, uv = [], [], []
        for j, center in enumerate(nodes):
            tangent = nodes[min(j + 1, len(nodes) - 1)] - nodes[max(0, j - 1)]
            tangent /= np.linalg.norm(tangent)
            axis = np.eye(3)[np.argmin(np.abs(tangent))]
            u = np.cross(tangent, axis)
            u /= np.linalg.norm(u)
            v = np.cross(tangent, u)
            for k in range(17):
                angle = k * math.tau / 16
                radius = radii[j] * (1 + 0.035 * math.sin(angle * 7 + j))
                points.append(tuple(center + radius * (u * math.cos(angle) + v * math.sin(angle))))
                uv.append((k / 16, j * 0.45))
                if j and k:
                    a = j * 17 + k
                    faces.append((a - 18, a - 17, a, a - 1))
        faces.extend([tuple(reversed(range(16))), tuple((len(nodes) - 1) * 17 + k for k in range(16))])
        prim = self.mesh(path, points, faces, "bark", uv)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull" if dynamic else "none")

    def leaf_geometry(self, center: np.ndarray, length: float, ground: bool = False) -> tuple[list, list, list]:
        """Return a thin curved leaf surface with length [m], polygon indices and UVs."""
        axis = Gf.Vec3d(0, 0, 1) if ground else Gf.Vec3d(*self.rng.normal(size=3))
        rotation = Gf.Rotation(axis, float(self.rng.uniform(0, 360)))
        points, faces, uv = [], [], []
        # Separate longitudinal rows preserve the midrib without balloon-shaped shading.
        for row in range(9):
            t = (row + 0.04) / 8.08
            half_width = length * 0.30 * math.sin(math.pi * t) ** 0.75
            half_width *= 1 + 0.05 * math.cos(row * math.pi)
            for side in (-1, 0, 1):
                z = length * (0.025 * math.sin(math.pi * t) * (1 - abs(side)) - 0.05 * (t - 0.5) ** 2)
                point = Gf.Vec3d((t - 0.5) * length, side * half_width, z)
                points.append(tuple(center + np.array(rotation.TransformDir(point))))
                uv.append((t, (side + 1) / 2))
            if row:
                k = row * 3
                faces.extend([(k - 3, k, k + 1, k - 2), (k - 2, k + 1, k + 2, k - 1)])
        return points, faces, uv

    def leaf(self, path: str, center: np.ndarray, length: float, material: str, collision: bool) -> None:
        """Create a thin leaf with length [m] and convex contact on its curved surface."""
        points, faces, uv = self.leaf_geometry(center, length)
        prim = self.mesh(path, points, faces, material, uv)
        if collision:
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.001)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0)

    def leaf_materials(self) -> None:
        """Author leaf albedo and fine vein normals; these are estimated botanical textures."""
        u, v = np.meshgrid(np.linspace(0, 1, 256), np.linspace(-1, 1, 256))
        midrib = np.exp(-((v / 0.022) ** 2))
        veins = np.exp(-((np.sin((u - abs(v) * 0.22) * math.pi * 10) / 0.065) ** 2))
        height = 0.55 * midrib + 0.16 * veins
        dy, dx = np.gradient(height)
        normals = np.stack((-dx * 1.5, -dy * 1.5, np.ones_like(u)), axis=2)
        normals /= np.linalg.norm(normals, axis=2, keepdims=True)
        for i, color in enumerate(((64, 80, 38), (77, 91, 49), (91, 99, 61), (52, 72, 40))):
            variation = 0.90 + 0.10 * np.sin(u * math.pi) + 0.12 * height + self.rng.normal(0, 0.018, u.shape)
            pixels = np.clip(variation[:, :, None] * np.array(color), 0, 255).astype(np.uint8)
            name = f"oak_{i}"
            Image.fromarray(pixels).save(self.output / f"textures/{name}.png")
            Image.fromarray(np.uint8((normals * 0.5 + 0.5) * 255)).save(self.output / f"textures/{name}_normal.png")
            Image.fromarray(np.uint8(np.full(u.shape, 150))).save(self.output / f"textures/{name}_roughness.png")
            self.material(name, (0.1, 0.15, 0.05), 0.45, f"textures/{name}.png")
        manifest_path = self.output / "textures/sources.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["leaves"] = {
            "source": "Procedural oval-leaf albedo and vein normals",
            "reference": ["IMG_3154.mp4", "IMG_3155.mp4"],
            "measured": False,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        for i, color in enumerate(((0.27, 0.20, 0.12), (0.40, 0.32, 0.21), (0.19, 0.15, 0.10))):
            self.material(f"litter_{i}", color, 0.5)

    def litter(self) -> None:
        """Scatter dry leaf surfaces on terrain; litter is visual ground detail."""
        for material in range(3):
            points, faces, uv = [], [], []
            for _ in range(2400):
                x, y = self.rng.uniform(-4.8, 4.8), self.rng.uniform(-1, 12.8)
                if abs(x - trail_center(y)) < 0.40 and self.rng.random() < 0.75:
                    continue
                center = np.array([x, y, ground_height(x, y) + 0.008])
                vertices, polygons, coords = self.leaf_geometry(center, self.rng.uniform(0.045, 0.11), True)
                faces.extend(tuple(k + len(points) for k in polygon) for polygon in polygons)
                points.extend(vertices)
                uv.extend(coords)
            self.mesh(f"/World/dry_leaves/leaves_{material}", points, faces, f"litter_{material}", uv)

    def branch(self, index: int, anchor: np.ndarray, clearance: float, support: str) -> None:
        """Attach an elastic branch with a tip clearance [m] above the local terrain."""
        drift = self.rng.uniform(-0.45, 0.45, 2)
        tip_xy = anchor[:2] + drift
        height = anchor[2] - ground_height(*tip_xy) - clearance
        nodes = []
        bend = self.rng.uniform(-0.40, 0.40, 2)
        for segment in range(7):
            t = segment / 6
            xy = anchor[:2] + drift * t + bend * math.sin(t * math.pi)
            nodes.append(np.array([*xy, anchor[2] - height * t]))
        paths, rotations, lengths, masses = [], [], [], []
        for segment in range(6):
            start, end = nodes[segment : segment + 2]
            length = float(np.linalg.norm(end - start))
            rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*(end - start)))
            path = f"/World/tree/branches/branch_{index:02d}/link_{segment}"
            body = UsdGeom.Xform.Define(self.stage, path)
            body.AddTranslateOp().Set(Gf.Vec3d(*start))
            body.AddOrientOp().Set(Gf.Quatf(rotation.GetQuat()))
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            radius = 0.014 * (1 - segment * 0.12)
            self.tube(path + "/wood", np.array([[0.0, 0.0, 0.0], [0.0, 0.0, length]]), [radius, radius * 0.85], True)
            mass = 650 * math.pi * radius**2 * length
            for twig in range(6):
                azimuth = float(self.rng.uniform(0, math.tau))
                origin = np.array([0, 0, length * (twig + 0.5) / 6])
                reach = float(self.rng.uniform(0.12, 0.22) if segment == 5 else self.rng.uniform(0.14, 0.38))
                tip = origin + np.array([math.cos(azimuth) * reach, math.sin(azimuth) * reach, 0.05])
                self.tube(path + f"/twig_{twig}", np.array([origin, tip]), [radius * 0.3, radius * 0.12], True)
                mass += 650 * math.pi * (radius * 0.22) ** 2 * np.linalg.norm(tip - origin)
                for leaf in range(10):
                    center = origin + (tip - origin) * (leaf + 1) / 10
                    center += self.rng.normal(0, 0.015, 3)
                    size = float(self.rng.uniform(0.045, 0.085))
                    self.leaf(path + f"/leaf_{twig}_{leaf}", center, size, f"oak_{(index + twig + leaf) % 4}", True)
                    mass += 0.12 * math.pi * size**2 * 0.165
            UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(float(mass))
            joint = UsdPhysics.Joint.Define(self.stage, f"/World/tree/joints/branch_{index:02d}_{segment}")
            joint.CreateBody1Rel().SetTargets([path])
            joint.CreateLocalPos1Attr(Gf.Vec3f(0))
            if segment:
                joint.CreateBody0Rel().SetTargets([paths[-1]])
                joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, lengths[-1]))
                relative = rotations[-1].GetQuat().GetInverse() * rotation.GetQuat()
                joint.CreateLocalRot0Attr(Gf.Quatf(relative))
            else:
                joint.CreateLocalPos0Attr(Gf.Vec3f(*anchor))
                joint.CreateLocalRot0Attr(Gf.Quatf(rotation.GetQuat()))
                UsdPhysics.FilteredPairsAPI.Apply(body.GetPrim()).CreateFilteredPairsRel().SetTargets([support])
            joint.CreateCollisionEnabledAttr(False)
            for axis in ("transX", "transY", "transZ"):
                limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
                limit.CreateLowAttr(1)
                limit.CreateHighAttr(-1)
            stiffness = 1.2e9 * math.pi * radius**4 / (4 * length)
            for axis in ("rotX", "rotY", "rotZ"):
                limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
                limit.CreateLowAttr(-70 if axis != "rotZ" else -45)
                limit.CreateHighAttr(70 if axis != "rotZ" else 45)
                drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
                drive.CreateTypeAttr("force")
                # USD angular drives use torque per degree, whereas E I / L is per radian.
                drive.CreateStiffnessAttr((stiffness if axis != "rotZ" else stiffness / 1.3) * math.pi / 180)
                drive.CreateTargetPositionAttr(0)
            paths.append(path)
            rotations.append(rotation)
            lengths.append(length)
            masses.append(float(mass))
        self.branch_damping(index, np.array(nodes), masses)
        self.branches.append(
            dict(
                paths=paths,
                anchor_m=anchor.tolist(),
                lengths_m=lengths,
                masses_kg=masses,
                modulus_Pa=1.2e9,
                tip_m=nodes[-1].tolist(),
            )
        )

    def branch_damping(self, index: int, nodes: np.ndarray, masses: list[float]) -> None:
        """Set 0.35-ratio angular damping using downstream rod inertia [kg m²]."""
        for segment, pivot in enumerate(nodes[:-1]):
            rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*(nodes[segment + 1] - pivot)))
            joint = self.stage.GetPrimAtPath(f"/World/tree/joints/branch_{index:02d}_{segment}")
            for axis, local in zip(("rotX", "rotY", "rotZ"), np.eye(3)):
                direction = np.array(rotation.TransformDir(Gf.Vec3d(*local)))
                inertia = 0.0
                for child in range(segment, len(masses)):
                    delta = nodes[child + 1] - nodes[child]
                    length = np.linalg.norm(delta)
                    offset = (nodes[child + 1] + nodes[child]) / 2 - pivot
                    alignment = float(np.dot(delta / length, direction)) ** 2
                    radius = 0.014 * (1 - child * 0.12)
                    central = (length**2 / 12 + radius**2 / 4) * (1 - alignment) + radius**2 / 2 * alignment
                    inertia += masses[child] * (central + np.dot(offset, offset) - np.dot(offset, direction) ** 2)
                drive = UsdPhysics.DriveAPI(joint, axis)
                stiffness = drive.GetStiffnessAttr().Get() * 180 / math.pi
                drive.CreateDampingAttr(float(0.7 * math.sqrt(stiffness * inertia) * math.pi / 180))

    def tree(self) -> None:
        """Build an irregular branch curtain, rooted off the left side of the trail."""
        root = np.array([-2.9, 4.8, ground_height(-2.9, 4.8)])
        fork = np.array([-2.15, 4.5, 3.6])
        self.tube(
            "/World/tree/trunk",
            np.array([root, root + [0.10, -0.12, 1.0], [-2.6, 4.7, 2.6], fork]),
            [0.16, 0.14, 0.10, 0.065],
        )
        anchors = []
        while len(anchors) < 20:
            anchor = np.array([self.rng.uniform(-1.65, 0.95), self.rng.uniform(1.8, 6.6), self.rng.uniform(2.6, 3.35)])
            if all(np.linalg.norm(anchor[:2] - other[:2]) > 0.58 for other in anchors):
                anchors.append(anchor)
        for index, anchor in enumerate(anchors):
            support = f"/World/tree/bough_{index}"
            elbow = fork * 0.45 + anchor * 0.55 + [0, 0, 0.25]
            self.tube(support, np.array([fork, elbow, anchor]), [0.035, 0.022, 0.012])
            self.branch(index, anchor, float(self.rng.uniform(0.26, 0.28)), support)
            # Upper twigs point away from the hanging link; only their common attachment is filtered.
            for crown in range(4):
                start = fork * 0.25 + anchor * 0.75 + [0, 0, 0.25]
                tip = start + np.array(
                    [self.rng.uniform(-0.8, 0.8), self.rng.uniform(-0.7, 0.7), self.rng.uniform(0.2, 0.65)]
                )
                self.tube(f"/World/tree/canopy/twig_{index}_{crown}", np.array([start, tip]), [0.006, 0.0015])
                for leaf in range(24):
                    position = start + (tip - start) * self.rng.uniform(0.15, 1)
                    position += self.rng.normal(0, 0.035, 3)
                    self.leaf(
                        f"/World/tree/canopy/leaf_{index}_{crown}_{leaf}",
                        position,
                        float(self.rng.uniform(0.045, 0.085)),
                        f"oak_{leaf % 4}",
                        True,
                    )

    def build(self) -> None:
        """Save a portable low-canopy scene and its uncalibrated physical parameters."""
        self.textures()
        for name in ("soil", "bark", "rock"):
            self.material(name, (0.5, 0.5, 0.5), 0.8 if name == "soil" else 0.65, f"textures/{name}.png")
        self.material("twig", (0.20, 0.17, 0.11), 0.55)
        self.material("straw", (0.34, 0.25, 0.13), 0.45)
        self.leaf_materials()
        for i in range(5):
            self.material(f"leaf_{i}", (0.045 + i * 0.012, 0.075 + i * 0.012, 0.024 + i * 0.008), 0.45)
        self.terrain()
        self.litter()
        for i in range(440):
            y = float(self.rng.uniform(0.7, 12.7))
            side = -1 if i % 2 else 1
            x = trail_center(y) + side * self.rng.uniform(1.35, 4.7)
            leafy = i % 3 != 0
            height = float(self.rng.uniform(1.5, 3.2) if y > 6 and leafy else self.rng.uniform(0.4, 1.6))
            self.stem(
                i,
                x,
                y,
                height,
                float(self.rng.uniform(0.004, 0.008) if leafy else self.rng.uniform(0.0015, 0.003)),
                leafy,
            )
        for i in range(440, 540):
            x = float(self.rng.uniform(-4.8, 4.8))
            y = float(self.rng.uniform(7.5, 12.7))
            self.stem(i, x, y, float(self.rng.uniform(3.2, 5.2)), float(self.rng.uniform(0.014, 0.024)), True)
        self.tree()
        sun = UsdLux.DistantLight.Define(self.stage, "/World/sun")
        sun.CreateIntensityAttr(2600)
        sun.CreateAngleAttr(0.75)
        sun.CreateColorAttr(Gf.Vec3f(1, 0.98, 0.93))
        sun.AddRotateXYZOp().Set(Gf.Vec3f(24, -32, -30))
        sky = UsdLux.DomeLight.Define(self.stage, "/World/sky")
        sky.CreateIntensityAttr(550)
        sky.CreateColorAttr(Gf.Vec3f(1, 1, 1))
        camera = UsdGeom.Camera.Define(self.stage, "/World/cameras/trail")
        camera.AddTransformOp().Set(
            Gf.Matrix4d().SetLookAt(Gf.Vec3d(0, 0.6, 1.45), Gf.Vec3d(0.25, 4.5, 0.8), Gf.Vec3d(0, 0, 1)).GetInverse()
        )
        camera.CreateFocalLengthAttr(24)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.03, 300))
        self.stage.GetRootLayer().Save()
        metadata = dict(
            reference_videos=["IMG_3154.mp4", "IMG_3155.mp4"],
            seed=3155,
            branches=self.branches,
            stems=self.stems,
            rock_density_kg_m3=2600,
            note=(
                "Dimensions, stiffness and friction are estimates. "
                "Rooted canopy is static; pendant branches are elastic."
            ),
        )
        (self.output / "parameters.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved {self.output / 'scene.usdc'}: 20 attached branches, 120 dynamic branch links; no fallen log.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    LowBranchesScene(parser.parse_args().output_dir).build()
