from collections import defaultdict
import os
import time
import isaacsim
import numpy as np
import omni
from scipy.spatial.transform.rotation import Rotation
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp(launch_config={"renderer": "RayTracedLighting", "headless": False})
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.prims as prims_utils
from pxr import Usd, UsdLux, UsdGeom, Sdf, Gf, Tf, UsdPhysics
from omni.isaac.core.prims import XFormPrim


if __name__ == '__main__':
    input_path = 'outputs/apartment-5-seq-tiny-25views-articulated_cabinets/0/export_scene.blend/export_scene.usdc'
    input_path = os.path.abspath(input_path)
    output_path = input_path.replace('.blend/export_', '.blend/articulated_')
    stage_utils.open_stage(input_path)
    stage = stage_utils.get_current_stage()

    # save prims to dict of list
    prims = defaultdict(list)
    for prim in stage_utils.traverse_stage():
        prim_path = prims_utils.get_prim_path(prim)
        prim_type_name = prims_utils.get_prim_type_name(prim_path)
        prims[prim_type_name].append(prim_path)
    # input(prims)
    print([k for k in prims['Mesh'] if 'SingleCabinet' in k])

    cabinet_mesh_paths = [
        k for k in prims['Mesh']
        if 'SingleCabinet' in k
        and 'left' not in k
        and 'right' not in k
    ]
    for cabinet_mesh_path in cabinet_mesh_paths:
        # /World/SingleCabinetFactory_2727438__spawn_asset_9636407_/SingleCabinetFactory_2727438__spawn_asset_9636407_
        # /World/SingleCabinetFactory_2727438__spawn_asset_9636407__right/SingleCabinetFactory_2727438__spawn_asset_9636407__right
        cabinet_xform_path = os.path.dirname(cabinet_mesh_path)
        cabinet_xform = XFormPrim(cabinet_xform_path)
        cabinet_pos, cabinet_rot = cabinet_xform.get_world_pose()
        cabinet_global_T = np.eye(4)
        cabinet_global_T[:3, :3] = Rotation.from_quat(cabinet_rot).as_matrix()
        cabinet_global_T[:3, 3] = cabinet_pos
        print(cabinet_xform_path, cabinet_pos, cabinet_rot)

        right_door_mesh_path = cabinet_mesh_path.replace('_/', '__right/') + '_right'
        left_door_mesh_path  = cabinet_mesh_path.replace('_/', '__left/') + '_left'
        right_door_xform_path = os.path.dirname(right_door_mesh_path)
        left_door_xform_path = os.path.dirname(left_door_mesh_path)

        door_xform_mesh_paths = (
            [(right_door_mesh_path, right_door_xform_path), (left_door_mesh_path, left_door_xform_path)]
            if left_door_mesh_path in prims['Mesh']
            else [(right_door_mesh_path, right_door_xform_path)]
        )

        for door_mesh_path, door_xform_path in door_xform_mesh_paths:
            print(door_mesh_path)
            assert door_mesh_path in prims['Mesh'], f"Door mesh {door_mesh_path} not found in prims"
            door_mesh = prims_utils.get_prim_at_path(door_mesh_path)
            
            # transform
            door_xform = XFormPrim(door_xform_path)
            door_pos, door_rot = door_xform.get_world_pose()
            door_local_T = np.eye(4)
            door_local_T[:3, :3] = Rotation.from_quat(door_rot).as_matrix()
            door_local_T[:3, 3] = door_pos
            # door_global_T = cabinet_global_T @ door_local_T
            door_global_T = door_local_T @ cabinet_global_T
            door_new_pos = door_global_T[:3, 3]
            door_new_rot = Rotation.from_matrix(door_global_T[:3, :3]).as_quat()
            door_xform.set_world_pose(door_new_pos, door_new_rot)

            # collision and rigid body
            UsdPhysics.CollisionAPI.Apply(door_mesh)
            UsdPhysics.RigidBodyAPI.Apply(door_mesh)

            # create joint
            joint = UsdPhysics.RevoluteJoint.Define(stage, door_mesh_path + "/revoluteJoint")
            joint.CreateAxisAttr("Z")
            joint.CreateLowerLimitAttr(-90.0)
            joint.CreateUpperLimitAttr(90.0)
            joint.CreateBody0Rel().SetTargets([cabinet_mesh_path])
            joint.CreateBody1Rel().SetTargets([door_mesh_path])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(f) for f in door_pos]))
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(*[float(f) for f in door_rot]))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

    # save stage to output path
    stage_utils.save_stage(output_path)