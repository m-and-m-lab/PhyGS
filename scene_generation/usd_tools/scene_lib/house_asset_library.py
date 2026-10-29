# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Import textured local household meshes independently of their original simulation."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt


class HouseAssetLibrary:
    """Bake source transforms and copy resolved materials into a portable scene."""

    def __init__(self, stage: Usd.Stage, output: Path, sources: dict[str, Path]):
        self.stage = stage
        self.output = output
        self.sources = {name: Usd.Stage.Open(str(path)) for name, path in sources.items()}
        self.layers = {name: source.Flatten() for name, source in self.sources.items()}
        self.assets: dict[str, Usd.Prim] = {}
        self.materials: dict[tuple[str, str], str] = {}
        for name, source in self.sources.items():
            geometry = next(p for p in source.Traverse() if p.GetName() == "Geometry")
            for prim in geometry.GetChildren():
                if "room_geometry" not in prim.GetName():
                    self.assets[name + ":" + prim.GetName()] = prim

    def add_asset(self, library: str, source_path: Path, root_path: str) -> None:
        """Register an explicit single-model USD root, retaining its materials and textures."""
        if library in self.sources:
            raise ValueError(f"Duplicate asset library: {library}")
        source = Usd.Stage.Open(str(source_path))
        root = source.GetPrimAtPath(root_path)
        if not root:
            raise ValueError(f"Missing asset root: {source_path}:{root_path}")
        self.sources[library] = source
        self.layers[library] = source.Flatten()
        self.assets[library + ":" + root.GetName()] = root

    def material(self, library: str, material: UsdShade.Material) -> UsdShade.Material:
        """Copy a bound material with collision-free texture filenames."""
        path = str(material.GetPath())
        key = (library, path)
        if key not in self.materials:
            destination = "/World/reference_materials/" + library + "_" + material.GetPrim().GetName()
            Sdf.CopySpec(self.layers[library], path, self.stage.GetRootLayer(), destination)
            for prim in Usd.PrimRange(self.stage.GetPrimAtPath(destination)):
                for attr in prim.GetAttributes():
                    if attr.HasAuthoredConnections():
                        attr.SetConnections([p.ReplacePrefix(path, destination) for p in attr.GetConnections()])
                    value = attr.Get()
                    if isinstance(value, Sdf.AssetPath) and value.path:
                        source_attr = self.sources[library].GetAttributeAtPath(
                            attr.GetPath().ReplacePrefix(destination, path)
                        )
                        resolved = source_attr.Get().resolvedPath
                        source = Path(resolved)
                        if not resolved or not source.is_file():
                            raise FileNotFoundError(f"Unresolved texture: {source_attr.GetPath()}")
                        filename = library + "_" + source.name
                        target = self.output / "textures" / filename
                        if not target.exists():
                            shutil.copy2(source, target)
                        attr.Set(Sdf.AssetPath("./textures/" + filename))
            self.materials[key] = destination
        return UsdShade.Material(self.stage.GetPrimAtPath(self.materials[key]))

    def visual(self, key: str, destination: str, size: tuple[float, float, float]) -> dict:
        """Fit visual geometry into dimensions [m], centered in XY with its base at Z=0."""
        root = self.assets[key]
        library = key.split(":", 1)[0]
        inverse = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(0).GetInverse()
        meshes = [
            p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(p).ComputePurpose() != "guide"
        ]
        if not meshes:
            raise ValueError(f"No visible geometry: {key}")
        points, transforms = [], []
        digest = hashlib.sha256()
        for prim in meshes:
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0) * inverse
            transform = np.array(matrix)
            raw = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            digest.update(raw.tobytes())
            points.append(raw @ transform[:3, :3] + transform[3, :3])
            transforms.append(transform)
        all_points = np.concatenate(points)
        low, high = all_points.min(0), all_points.max(0)
        scale = np.array(size) / (high - low)
        origin = np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]])
        UsdGeom.Xform.Define(self.stage, destination)
        triangles = 0
        for i, (prim, vertices, transform) in enumerate(zip(meshes, points, transforms)):
            path = destination + f"/mesh_{i}"
            Sdf.CopySpec(self.layers[library], prim.GetPath(), self.stage.GetRootLayer(), path)
            copied = self.stage.GetPrimAtPath(path)
            copied.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit([]))
            for prop in list(copied.GetProperties()):
                if prop.GetName().startswith(("physics:", "physx", "xformOp")):
                    copied.RemoveProperty(prop.GetName())
            mesh = UsdGeom.Mesh(copied)
            fitted = ((vertices - origin) * scale).astype(np.float32)
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(fitted))
            mesh.CreateExtentAttr([Gf.Vec3f(*map(float, fitted.min(0))), Gf.Vec3f(*map(float, fitted.max(0)))])
            mesh.CreateVisibilityAttr("inherited")
            mesh.CreatePurposeAttr("default")
            mesh.CreateSubdivisionSchemeAttr("none")
            normals = UsdGeom.Mesh(prim).GetNormalsAttr().Get()
            if normals:
                normals = np.asarray(normals) @ np.linalg.inv(transform[:3, :3]).T / scale
                lengths = np.linalg.norm(normals, axis=1, keepdims=True)
                if np.any(lengths == 0):
                    mesh.GetNormalsAttr().Clear()
                else:
                    mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy((normals / lengths).astype(np.float32)))
            material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            if not material:
                raise ValueError(f"Missing visual material: {prim.GetPath()}")
            UsdShade.MaterialBindingAPI.Apply(copied).Bind(self.material(library, material))
            triangles += sum(n - 2 for n in mesh.GetFaceVertexCountsAttr().Get())
        return dict(asset=key, geometry_sha256=digest.hexdigest(), triangles=triangles, dimensions_m=list(size))
