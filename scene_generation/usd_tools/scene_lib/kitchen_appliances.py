# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Hollow kitchen appliances and sparse, independently movable contents."""
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade
from scene_lib.bear_loop_furnishings import native

def appliance(house, name: str, position: tuple, width: float, height: float, depth: float) -> None:
    """Build an anchored cavity and passive side-hinged door, dimensions [m]."""
    x, y, z = position
    root = house.body(name, position, 45, anchored=True)
    for label, center, size in (('base', (0, 0, 0.015), (width, depth, 0.03)), ('roof', (0, 0, height - 0.015), (width, depth, 0.03)), ('left', (-width / 2 + 0.015, 0, height / 2), (0.03, depth, height)), ('right', (width / 2 - 0.015, 0, height / 2), (0.03, depth, height)), ('back', (0, depth / 2 - 0.015, height / 2), (width, 0.03, height))):
        house.box(root + '/' + label, center, size, 'metal')
    door = house.body(name + '_door', (x, y - depth / 2 - 0.045, z + height / 2), 3)
    house.box(door + '/glass', (0, 0, 0), (width - 0.02, 0.018, height - 0.04), 'glass')
    for label, center, size in (('left', (-width / 2 + 0.035, 0, 0), (0.07, 0.035, height - 0.04)), ('right', (width / 2 - 0.045, 0, 0), (0.09, 0.035, height - 0.04)), ('upper', (0, 0, height / 2 - 0.055), (width, 0.035, 0.07)), ('lower', (0, 0, -height / 2 + 0.055), (width, 0.035, 0.07))):
        house.box(door + '/' + label, center, size, 'metal')
    house.rod(door + '/handle', (width / 2 - 0.08, -0.065, -0.09), (width / 2 - 0.08, -0.065, 0.09), 0.012, 'metal')
    for i, h in enumerate((-0.09, 0.09)):
        house.rod(door + f'/mount_{i}', (width / 2 - 0.08, 0, h), (width / 2 - 0.08, -0.065, h), 0.012, 'metal')
    pivot = (x - width / 2, y - depth / 2 - 0.045, z + height / 2)
    house.joint(name + '_hinge', door, pivot, (-width / 2, 0, 0))
    joint = UsdPhysics.Joint(house.stage.GetPrimAtPath('/World/joints/' + name + '_hinge'))
    flip = Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), 180).GetQuat())
    joint.CreateLocalRot0Attr(flip)
    joint.CreateLocalRot1Attr(flip)
    house.inventory.append(dict(name=name, path=root, category='appliance', evidence='modeled household appliance with hollow cavity'))
    plate = house.body(name + '_dish', (x, y, z + .032), .25)
    for label, center, size in (
        ('base', (0,0,.006), (.23,.23,.012)),
        ('left',(-.109,0,.018),(.012,.23,.024)),
        ('right',(.109,0,.018),(.012,.23,.024)),
        ('back',(0,.109,.018),(.206,.012,.024)),
        ('front',(0,-.109,.018),(.206,.012,.024)),
    ):
        house.box(plate+'/'+label,center,size,'porcelain')
    house.stage.GetPrimAtPath(plate).SetCustomDataByKey('support',root)
    house.stage.GetPrimAtPath(plate).SetCustomDataByKey('independently_pickable',True)
    house.stored_items.append(plate)
    food = house.body(name + '_bread_roll', (x, y, z + 0.032 + 0.012 + 0.016 + 0.002), 0.09)
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=2)
    shape = UsdGeom.Mesh.Define(house.stage, food + '/shape')
    points = mesh.vertices * [0.065, 0.042, 0.027]
    points[:, 2] = points[:, 2].clip(min=-0.016)
    shape.CreatePointsAttr(points.tolist())
    shape.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    shape.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1).tolist())
    UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr('convexHull')
    UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(house.materials['bread_crust'])
    house.stage.GetPrimAtPath(food).SetCustomDataByKey('support', plate)
    house.stage.GetPrimAtPath(food).SetCustomDataByKey('independently_pickable', True)
    house.stored_items.append(food)
    house.inventory.append(dict(name=name + '_bread_roll', path=food, evidence='modeled bread roll, rigid food approximation', support=plate))

def kitchen_additions(house) -> None:
    """Add appliances and at most three items per kitchen storage compartment."""
    material = UsdShade.Material.Define(house.stage, '/World/materials/bread_crust')
    shader = UsdShade.Shader.Define(house.stage, '/World/materials/bread_crust/surface')
    from pxr import Sdf
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set((0.46, 0.2, 0.055))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.82)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    house.materials['bread_crust'] = material
    removed = {"kitchen_base_2", "kitchen_wall_2"}
    paths = {f"/World/objects/{name}" for name in removed}
    for prim in list(house.stage.GetPrimAtPath('/World/objects').GetChildren()):
        if any(prim.GetName() == name or prim.GetName().startswith(name + '_') for name in removed):
            house.stage.RemovePrim(prim.GetPath())
    for prim in list(house.stage.GetPrimAtPath('/World/joints').GetChildren()):
        if any(prim.GetName().startswith(name + '_') for name in removed):
            house.stage.RemovePrim(prim.GetPath())
    house.stored_items = [p for p in house.stored_items if house.stage.GetPrimAtPath(p)]
    house.inventory = [r for r in house.inventory if 'path' not in r or house.stage.GetPrimAtPath(r['path'])]
    house.storage_compartments = [s for s in house.storage_compartments if s['name'] not in removed]
    appliance(house, 'kitchen_oven', (12.8, 9.1, 0), 0.9, 0.9, 0.56)
    appliance(house, 'kitchen_microwave', (12.8, 9.1, 0.94), 0.9, 0.44, 0.56)
    house.box('/World/objects/kitchen_oven/microwave_support', (0, 0, .92), (.9, .56, .04), 'cream')
