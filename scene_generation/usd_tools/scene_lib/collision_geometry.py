# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Watertight collision surfaces with bounded triangle edge lengths."""

import numpy as np
import trimesh
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Vt


def box_surface_mesh(size: float, transform: np.ndarray, max_edge: float) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a watertight box with bounded world-space edges.

    Args:
        size: Untransformed cube edge length [m].
        transform: Local-to-world row-vector transform, with translation [m].
        max_edge: Maximum world-space triangle edge length [m].

    Returns:
        Local vertices [m] and outward-wound triangle vertex indices.
    """
    if not np.isfinite([size, max_edge]).all() or size <= 0 or max_edge <= 0:
        raise ValueError("Box size and maximum edge length must be positive")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Box transform must be a finite 4 by 4 matrix")
    axes = transform[:3, :3]
    lengths = np.linalg.norm(axes, axis=1)
    if np.any(lengths == 0):
        raise ValueError("Box transform must have nonzero axes")
    directions = axes / lengths[:, None]
    if not np.allclose(directions @ directions.T, np.eye(3)):
        raise ValueError("Box transform must have orthogonal, nonzero axes")
    divisions = np.maximum(1, np.ceil(size * lengths * np.sqrt(2) / max_edge).astype(int))
    vertices = []
    faces = []
    indices = {}

    def vertex(key: tuple[int, int, int]) -> int:
        if key not in indices:
            indices[key] = len(vertices)
            vertices.append(size * (np.asarray(key) / divisions - 0.5))
        return indices[key]

    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        for side in (0, 1):
            for i in range(divisions[u]):
                for j in range(divisions[v]):
                    corners = []
                    for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        key = [0, 0, 0]
                        key[axis], key[u], key[v] = side * divisions[axis], i + du, j + dv
                        corners.append(vertex(tuple(key)))
                    for triangle in ((0, 1, 2), (0, 2, 3)):
                        face = [corners[k] for k in triangle]
                        faces.append(face if side else face[::-1])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def convex_collider_to_sdf(prim: Usd.Prim) -> None:
    """Represent an existing convex collider with SDF64 while preserving its visual mesh.

    Args:
        prim: A cube collider or a mesh using the convexHull collision approximation.
    """
    stage = prim.GetStage()
    if prim.IsA(UsdGeom.Cube):
        vertices, faces = box_surface_mesh(
            UsdGeom.Cube(prim).GetSizeAttr().Get(),
            np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)),
            max_edge=0.02,
        )
    elif (
        prim.IsA(UsdGeom.Mesh)
        and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "convexHull"
    ):
        hull = trimesh.convex.convex_hull(np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get()))
        vertices, faces = np.asarray(hull.vertices, dtype=np.float32), hull.faces
    else:
        raise ValueError(f"Expected a cube or convex-hull mesh collider: {prim.GetPath()}")
    target = str(prim.GetPath()) + "_sdf_collision"
    if stage.GetPrimAtPath(target):
        raise ValueError(f"Collider already exists: {target}")
    if not Sdf.CopySpec(stage.GetRootLayer(), prim.GetPath(), stage.GetRootLayer(), target):
        raise RuntimeError(f"Could not copy collider: {prim.GetPath()}")
    collider = stage.GetPrimAtPath(target)
    collider.SetTypeName("Mesh")
    collider.RemoveProperty("size")
    collider.RemoveProperty("normals")
    for attribute in list(collider.GetAttributes()):
        if attribute.GetName().startswith("primvars:"):
            collider.RemoveProperty(attribute.GetName())
    for child in list(collider.GetChildren()):
        if not child.IsA(UsdGeom.Subset):
            raise ValueError(f"Unexpected collider child: {child.GetPath()}")
        stage.RemovePrim(child.GetPath())
    mesh = UsdGeom.Mesh(collider)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateVisibilityAttr("invisible")
    mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.asarray([vertices.min(0), vertices.max(0)], dtype=np.float32)))
    UsdPhysics.MeshCollisionAPI.Apply(collider).CreateApproximationAttr("sdf")
    collider.AddAppliedSchema("PhysxSDFMeshCollisionAPI")
    collider.CreateAttribute("physxSDFMeshCollision:sdfResolution", Sdf.ValueTypeNames.Int).Set(64)
    collider.SetCustomDataByKey("collision_source_shape", "convex_contents")
    collider.SetCustomDataByKey("collision_source_path", str(prim.GetPath()))
    prim.RemoveAPI(UsdPhysics.CollisionAPI)
    prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
