from collections import defaultdict
import os
import time
import isaacsim
import omni
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp(launch_config={"renderer": "RayTracedLighting", "headless": False})
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.prims as prims_utils
from pxr import Usd, UsdLux, UsdGeom, Sdf, Gf, Tf, UsdPhysics
from omni.isaac.core.prims import XFormPrim

if __name__ == '__main__':
    input_path = 'outputs/single_cabinet_articulated/export_SingleCabinetArticulatedFactory_000.blend/export_SingleCabinetArticulatedFactory_000.usdc'
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

    # fetch prims
    light_mesh_path = prims['DomeLight'][0]
    left_door_mesh_path, right_door_mesh_path, cabinet_mesh_path = prims['Mesh']
    left_door_xform_path, right_door_xform_path = prims['Xform'][1], prims['Xform'][2]
    left_door_mesh = prims_utils.get_prim_at_path(left_door_mesh_path)
    right_door_mesh = prims_utils.get_prim_at_path(right_door_mesh_path)
    cabinet_mesh = prims_utils.get_prim_at_path(cabinet_mesh_path)
    left_door_xform = XFormPrim(left_door_xform_path)
    right_door_xform = XFormPrim(right_door_xform_path)

    # door poses
    left_door_pos, left_door_rot = left_door_xform.get_world_pose()
    right_door_pos, right_door_rot = right_door_xform.get_world_pose()

    # collision and rigid body for doors
    UsdPhysics.CollisionAPI.Apply(left_door_mesh)
    UsdPhysics.RigidBodyAPI.Apply(left_door_mesh)
    UsdPhysics.CollisionAPI.Apply(right_door_mesh)
    UsdPhysics.RigidBodyAPI.Apply(right_door_mesh)

    # left door revolute joint
    revoluteJoint = UsdPhysics.RevoluteJoint.Define(stage, left_door_mesh_path + "/revoluteJoint")

    revoluteJoint.CreateAxisAttr("Z")
    revoluteJoint.CreateLowerLimitAttr(-90.0)
    revoluteJoint.CreateUpperLimitAttr(90)

    revoluteJoint.CreateBody0Rel().SetTargets([cabinet_mesh_path])
    revoluteJoint.CreateBody1Rel().SetTargets([left_door_mesh_path])

    revoluteJoint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(f) for f in left_door_pos]))
    revoluteJoint.CreateLocalRot0Attr().Set(Gf.Quatf(*[float(f) for f in left_door_rot]))

    revoluteJoint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    revoluteJoint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

    # right door revolute joint
    revoluteJoint = UsdPhysics.RevoluteJoint.Define(stage, right_door_mesh_path + "/revoluteJoint")

    revoluteJoint.CreateAxisAttr("Z")
    revoluteJoint.CreateLowerLimitAttr(-90.0)
    revoluteJoint.CreateUpperLimitAttr(90)

    revoluteJoint.CreateBody0Rel().SetTargets([cabinet_mesh_path])
    revoluteJoint.CreateBody1Rel().SetTargets([right_door_mesh_path])

    revoluteJoint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(f) for f in right_door_pos]))
    revoluteJoint.CreateLocalRot0Attr().Set(Gf.Quatf(*[float(f) for f in right_door_rot]))

    revoluteJoint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    revoluteJoint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))


    # save stage to output path
    stage_utils.save_stage(output_path)