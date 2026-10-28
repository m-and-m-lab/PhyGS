# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Create separate SDF64 collision meshes using Isaac Sim's CPU mesh decimator."""

import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--usd_path', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--report', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.usd_path.is_file():
    parser.error(f"Scene does not exist: {args.usd_path}")
app = AppLauncher(args).app

def main() -> None:
    """Validate reduced collider surfaces before exporting the candidate asset."""

    import numpy as np
    import omni.kit.app
    import omni.kit.commands
    import trimesh
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils
    from scene_lib.convex_contact_base import split_convex_base

    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('omni.scene.optimizer.core', True)
    from omni.scene.optimizer.impl.core import ExecutionContext

    stage = Usd.Stage.Open(str(args.usd_path.resolve()))
    context = ExecutionContext()
    context.usdStageId = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
    transforms = UsdGeom.XformCache()
    records = []
    for prim in list(stage.Traverse()):
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            continue
        if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != 'sdf':
            continue
        if prim.GetCustomDataByKey('collision_source_shape'):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get())
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        if not np.all(counts == 3):
            raise ValueError(f'Expected triangulated collider: {prim.GetPath()}')
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        target = str(prim.GetPath()) + '_sdf_collision'
        if stage.GetPrimAtPath(target):
            raise ValueError(f'Collider already exists: {target}')
        if not Sdf.CopySpec(stage.GetRootLayer(), prim.GetPath(), stage.GetRootLayer(), target):
            raise RuntimeError(f'Could not copy {prim.GetPath()}')
        collider = stage.GetPrimAtPath(target)
        for child in list(collider.GetChildren()):
            if not child.IsA(UsdGeom.Subset):
                raise ValueError(f'Unexpected child: {child.GetPath()}')
            stage.RemovePrim(child.GetPath())
        for attr in list(collider.GetAttributes()):
            if attr.GetName().startswith('primvars:') or attr.GetName() == 'normals':
                collider.RemoveProperty(attr.GetName())
        # Keep 10% of unique vertices, with a 2,000-vertex floor for cavity detail.
        retained_percent = max(10.0, min(100.0, 200000.0 / len(np.unique(points, axis=0))))
        success, result = omni.kit.commands.execute(
            'SceneOptimizerOperation', operation='decimateMeshes', context=context,
            args=json.dumps(dict(paths=[target], reductionFactor=retained_percent, maxMeanError=0.0,
                                 cpuVertexCountThreshold=1000000, gpuVertexCountThreshold=1000000,
                                 pinBoundaries=True)))
        if not success or not result[0]:
            raise RuntimeError(f'Decimation failed: {target}: {result}')
        reduced = UsdGeom.Mesh(collider)
        reduced_points = np.asarray(reduced.GetPointsAttr().Get())
        reduced_faces = np.asarray(reduced.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        transform = np.asarray(transforms.GetLocalToWorldTransform(prim))
        original_geometry = trimesh.Trimesh(points @ transform[:3, :3] + transform[3, :3], faces, process=False)
        geometry = trimesh.Trimesh(reduced_points @ transform[:3, :3] + transform[3, :3], reduced_faces, process=False)
        # Test vertices and triangle centres in both directions, in metres.
        errors = []
        for source, destination in ((original_geometry, geometry), (geometry, original_geometry)):
            samples = np.vstack((source.vertices, source.triangles_center))
            distance = []
            for offset in range(0, len(samples), 2048):
                distance.extend(trimesh.proximity.closest_point(destination, samples[offset:offset + 2048])[1])
            errors.append(float(np.max(distance)))
        if (
            (original_geometry.is_watertight and not geometry.is_watertight)
            or not geometry.is_winding_consistent
            or not np.isfinite(errors).all()
            or max(errors) > 0.002
        ):
            raise ValueError(
                f'Invalid collision simplification: {target}, watertight={geometry.is_watertight}, errors={errors}'
            )
        reduced.CreateVisibilityAttr('invisible')
        collider.SetCustomDataByKey('collision_source_shape', 'simplified_concave')
        collider.SetCustomDataByKey('collision_source_path', str(prim.GetPath()))
        collider.GetAttribute('physxSDFMeshCollision:sdfTriangleCountReductionFactor').Set(1.0)
        prim.RemoveAPI(UsdPhysics.CollisionAPI)
        prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        records.append(dict(path=target, original_points=len(points), points=len(reduced_points),
                            original_faces=len(faces), faces=len(reduced_faces), sampled_error_m=max(errors)))
        print('COLLIDER', json.dumps(records[-1]), flush=True)
    for prim in stage.Traverse():
        if (prim.HasAPI(UsdPhysics.MeshCollisionAPI)
                and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "sdf"):
            prim.GetAttribute("physxSDFMeshCollision:sdfResolution").Set(64)
            prim.GetAttribute("physxCollision:contactOffset").Set(0.003)
            prim.GetAttribute("physxCollision:restOffset").Set(0.0)
            if prim.GetCustomDataByKey("collision_source_shape") == "convex_contents":
                # These meshes already describe convex solids exactly; SDF sampling adds contact noise.
                UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Set("convexHull")
                prim.RemoveAppliedSchema("PhysxSDFMeshCollisionAPI")
                for attr in list(prim.GetAttributes()):
                    if attr.GetName().startswith("physxSDFMeshCollision:"):
                        prim.RemoveProperty(attr.GetName())

    constrained = {
        path for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)
        for rel in (UsdPhysics.Joint(prim).GetBody0Rel(), UsdPhysics.Joint(prim).GetBody1Rel())
        for path in rel.GetTargets()
    }
    for record in records:
        collider = stage.GetPrimAtPath(record['path'])
        source = stage.GetPrimAtPath(collider.GetCustomDataByKey('collision_source_path'))
        if 'BowlFactory_' not in str(source.GetCustomDataByKey('reference_mesh')):
            continue
        body = collider.GetParent()
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body:
            raise ValueError(f'Missing rigid body: {collider.GetPath()}')
        if body.GetPath() not in constrained:
            record['convex_base'] = split_convex_base(collider)
            mesh = UsdGeom.Mesh(collider)
            record['points'] = len(mesh.GetPointsAttr().Get())
            record['faces'] = len(mesh.GetFaceVertexCountsAttr().Get())

    if not records:
        raise ValueError('No source SDF colliders found')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(records, indent=2) + '\n')
    if not stage.GetRootLayer().Export(str(args.output.resolve())):
        raise RuntimeError(f"Could not export scene: {args.output}")


try:
    main()
except Exception:
    traceback.print_exc()
    app.close(exit_code=1)
    raise
app.close()
