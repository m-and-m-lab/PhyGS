# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Separate a solid convex contact base from a hollow SDF container."""

import numpy as np
import trimesh
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Vt


def _clip(mesh: trimesh.Trimesh, height: float, above: bool) -> trimesh.Trimesh:
    vertices, faces = [], []
    for triangle in mesh.triangles:
        polygon = []
        for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
            inside_a = a[2] >= height if above else a[2] <= height
            inside_b = b[2] >= height if above else b[2] <= height
            if inside_a:
                polygon.append(a)
            if inside_a != inside_b:
                polygon.append(a + (b - a) * ((height - a[2]) / (b[2] - a[2])))
        start = len(vertices)
        vertices.extend(polygon)
        faces.extend([start, start + i, start + i + 1] for i in range(1, len(polygon) - 1))
    return trimesh.Trimesh(vertices, faces, process=True)


def _close_cut(mesh: trimesh.Trimesh, height: float, up: bool) -> trimesh.Trimesh:
    edges, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    boundary = edges[counts == 1]
    vertices = np.unique(boundary)
    if len(vertices) < 3 or not np.allclose(mesh.vertices[vertices, 2], height, atol=1e-8, rtol=0):
        raise ValueError('Expected a planar cut in a closed source mesh')
    _, degree = np.unique(boundary, return_counts=True)
    if not np.all(degree == 2):
        raise ValueError('Non-manifold cut boundary')
    center = mesh.vertices[vertices].mean(axis=0)
    faces = []
    for a, b in boundary:
        normal = np.cross(mesh.vertices[b] - mesh.vertices[a], center - mesh.vertices[a])[2]
        if (normal > 0) != up:
            a, b = b, a
        faces.append([a, b, len(mesh.vertices)])
    result = trimesh.Trimesh(np.vstack([mesh.vertices, center]), np.vstack([mesh.faces, faces]), process=True)
    if not result.is_watertight or not result.is_winding_consistent:
        raise ValueError('Cut must bound one solid convex base, below the interior cavity')
    return result


def _geometry(prim: Usd.Prim, transform: np.ndarray) -> trimesh.Trimesh:
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get())
    if not np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3):
        raise ValueError(f'Expected triangles: {prim.GetPath()}')
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    return trimesh.Trimesh(points @ transform[:3, :3] + transform[3, :3], faces, process=True)


def _set_geometry(prim: Usd.Prim, geometry: trimesh.Trimesh, inverse: np.ndarray) -> None:
    points = geometry.vertices @ inverse[:3, :3] + inverse[3, :3]
    mesh = UsdGeom.Mesh(prim)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.GetFaceVertexCountsAttr().Set([3] * len(geometry.faces))
    mesh.GetFaceVertexIndicesAttr().Set(geometry.faces.flatten().tolist())
    mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([points.min(0), points.max(0)], dtype=np.float32)))


def split_convex_base(collider: Usd.Prim, depth: float = 0.001) -> dict[str, str | float | int]:
    """Split a solid base [m] into an exact hull, retaining the hollow upper SDF.

    The source lower slab must match its convex hull in volume within 0.01%.
    Reject open meshes or cuts through a cavity; never fill a hollow lower slab.

    Args:
        collider: Simplified SDF mesh with a collision_source_path pointing to its visual mesh.
        depth: Height of the solid base above the source bottom [m].
    """
    stage = collider.GetStage()
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError('Convex base splitting requires a metre-scale, Z-up stage')
    source = stage.GetPrimAtPath(collider.GetCustomDataByKey('collision_source_path'))
    transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(collider))
    original = _geometry(source, transform)
    reduced = _geometry(collider, transform)
    if not original.is_watertight or depth <= 0:
        raise ValueError(f'Expected a closed source and positive base depth: {source.GetPath()}')
    height = original.bounds[0, 2] + depth
    lower = _close_cut(_clip(original, height, False), height, True)
    hull = lower.convex_hull
    volume_error = abs(hull.volume / lower.volume - 1)
    if volume_error > 1e-4:
        raise ValueError(f'Base is not convex: {source.GetPath()}: {volume_error}')
    upper = _close_cut(_clip(reduced, height, True), height, False)
    target = str(collider.GetPath()) + '_base'
    if stage.GetPrimAtPath(target):
        raise ValueError(f'Convex base already exists: {target}')
    if not Sdf.CopySpec(stage.GetRootLayer(), collider.GetPath(), stage.GetRootLayer(), target):
        raise ValueError(f'Cannot create convex base: {target}')
    base = stage.GetPrimAtPath(target)
    inverse = np.linalg.inv(transform)
    _set_geometry(collider, upper, inverse)
    _set_geometry(base, hull, inverse)
    UsdPhysics.MeshCollisionAPI(base).GetApproximationAttr().Set('convexHull')
    base.RemoveAppliedSchema('PhysxSDFMeshCollisionAPI')
    for attr in list(base.GetAttributes()):
        if attr.GetName().startswith('physxSDFMeshCollision:'):
            base.RemoveProperty(attr.GetName())
    base.SetCustomDataByKey('collision_source_shape', 'convex_base')
    base.SetCustomDataByKey('base_depth_m', depth)
    return dict(path=target, depth_m=depth, volume_relative_error=float(volume_error),
                points=len(hull.vertices), faces=len(hull.faces))
