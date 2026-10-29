# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sliding-part information from an authored drawer joint, as the house builder makes it."""

import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

from phygs.scene.parts import sliding_part


def drawer_stage(position, yaw):
    """A drawer body at position [m] turned by yaw [deg], on a world slide; a vertical bar handle."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    body = UsdGeom.Xform.Define(stage, "/World/drawer")
    body.AddTranslateOp().Set(Gf.Vec3d(*position))
    body.AddRotateZOp().Set(yaw)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    handle = UsdGeom.Cylinder.Define(stage, "/World/drawer/handle")
    handle.CreateRadiusAttr(0.009)
    handle.CreateHeightAttr(0.14)
    handle.AddTranslateOp().Set(Gf.Vec3d(0.2, -0.35, 0.13))
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/joints/slide")
    joint.CreateBody1Rel().SetTargets(["/World/drawer"])
    joint.CreateAxisAttr("Y")
    joint.CreateLocalPos0Attr(Gf.Vec3f(*position))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    joint.CreateLocalRot0Attr(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw + 180).GetQuat()))
    joint.CreateLocalRot1Attr(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), 180).GetQuat()))
    joint.CreateLowerLimitAttr(0.0)
    joint.CreateUpperLimitAttr(0.38)
    return stage


def test_sliding_part_follows_the_body_along_the_authored_slide():
    position, yaw, opening = np.array([2.0, 1.0, 0.5]), 90.0, 0.12
    stage = drawer_stage(position, yaw)
    rotation = Rotation.from_euler("z", yaw, degrees=True)
    outward = rotation.apply([0.0, -1.0, 0.0])  # The drawer front faces the body's -y.
    pose = np.eye(4)
    pose[:3, :3] = rotation.as_matrix()
    pose[:3, 3] = position + outward * opening  # The simulated body, pulled out.
    part = sliding_part(stage, "/World/joints/slide", "/World/drawer/handle", {"/World/drawer": pose}.__getitem__)
    assert np.allclose(part.direction, outward)
    assert part.opening == pytest.approx(opening)
    assert part.limits == (0.0, pytest.approx(0.38))
    assert np.allclose(part.handle_center, pose[:3, 3] + rotation.apply([0.2, -0.35, 0.13]))
    assert np.allclose(part.handle_axis, [0.0, 0.0, 1.0])
    assert part.handle_length == pytest.approx(0.14) and part.handle_radius == pytest.approx(0.009)


def test_sliding_part_rejects_missing_prims():
    stage = drawer_stage((0.0, 0.0, 0.0), 0.0)
    with pytest.raises(ValueError, match="prismatic joint"):
        sliding_part(stage, "/World/joints/missing", "/World/drawer/handle", lambda path: np.eye(4))


def test_a_copied_storage_unit_moves_its_bodies_and_world_slide_anchors_together(tmp_path):
    """A two-tier unit as the house builder authors it, copied to another place and turned."""
    from phygs.scene import storage_units as house_storage

    source = Usd.Stage.CreateNew(str(tmp_path / "house.usda"))
    origin = np.array([20.0, 4.0, 3.2])  # On an upper floor.

    def body(path, position, mass):
        xform = UsdGeom.Xform.Define(source, path)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        xform.AddRotateZOp().Set(0.0)
        UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateMassAttr(mass)
        xform.GetPrim().CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(16)

    body("/World/objects/unit", origin, 35.0)
    side = UsdGeom.Cube.Define(source, "/World/objects/unit/left")
    side.CreateSizeAttr(1.0)  # As the builder authors its panels.
    side.AddTranslateOp().Set(Gf.Vec3d(-0.25, 0.0, 0.3))
    side.AddScaleOp().Set(Gf.Vec3f(0.025, 0.56, 0.6))
    UsdPhysics.CollisionAPI.Apply(side.GetPrim())
    mount = UsdPhysics.FixedJoint.Define(source, "/World/joints/unit_mount")
    mount.CreateBody1Rel().SetTargets(["/World/objects/unit"])
    mount.CreateLocalPos0Attr(Gf.Vec3f(*origin))
    for name, height in (("unit", 0.055), ("unit_tier_2", 0.365)):
        body(f"/World/objects/{name}_drawer", origin + [0.0, 0.0, height], 3.0)
        handle = UsdGeom.Cylinder.Define(source, f"/World/objects/{name}_drawer/handle")
        handle.CreateRadiusAttr(0.009)
        handle.CreateHeightAttr(0.3)
        handle.CreateAxisAttr("X")
        handle.AddTranslateOp().Set(Gf.Vec3d(0.0, -0.3575, 0.1))
        slide = UsdPhysics.PrismaticJoint.Define(source, f"/World/joints/{name}_slide")
        slide.CreateBody1Rel().SetTargets([f"/World/objects/{name}_drawer"])
        slide.CreateAxisAttr("Y")
        slide.CreateLocalPos0Attr(Gf.Vec3f(*(origin + [0.0, 0.0, height])))
        slide.CreateLocalRot0Attr(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), 180).GetQuat()))
        slide.CreateLocalRot1Attr(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), 180).GetQuat()))
        slide.CreateLowerLimitAttr(0.0)
        slide.CreateUpperLimitAttr(0.38)
    source.Save()
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    unit = house_storage.copy_unit(stage, str(tmp_path / "house.usda"), "unit", 1.5, 0.0, -90.0)
    assert unit.drawers == ["/World/objects/unit_drawer", "/World/objects/unit_tier_2_drawer"]
    cache = UsdGeom.XformCache()

    def authored(path):  # The simulation starts with the drawers at their authored poses.
        return np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T

    # Turned by -90 deg, the builder's front (-y) faces -x; the floor is at z 0.
    for slide, handle, height in zip(unit.slides, unit.handles, (0.055, 0.365)):
        part = sliding_part(stage, slide, handle, authored)
        assert np.allclose(part.direction, [-1.0, 0.0, 0.0], atol=1e-6)
        assert part.opening == pytest.approx(0.0, abs=1e-6)
        assert np.allclose(part.handle_center, [1.5 - 0.3575, 0.0, height + 0.1], atol=1e-6)
        assert np.allclose(np.abs(part.handle_axis), [0.0, 1.0, 0.0], atol=1e-6)
    # The unit keeps the house's physics as built, solver iterations included.
    assert stage.GetPrimAtPath(unit.drawers[0]).GetAttribute("physxRigidBody:solverPositionIterationCount").Get() == 16
    mount = UsdPhysics.Joint(stage.GetPrimAtPath("/World/joints/unit_mount"))
    assert np.allclose(mount.GetLocalPos0Attr().Get(), [1.5, 0.0, 0.0], atol=1e-6)
    ((center, size),) = house_storage.collider_boxes(stage, [unit.carcass]).values()
    assert np.allclose(center, [1.5, 0.25, 0.3], atol=1e-6) and np.allclose(size, [0.56, 0.025, 0.6], atol=1e-6)


def test_house_storage_finds_a_unit_in_place_and_maps_colliders_per_object():
    """In the house scene itself: the unit's paths, and one map box per object or structure collider."""
    from phygs.scene import storage_units as house_storage

    stage = Usd.Stage.CreateInMemory()

    def box(path, center, size):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*center))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    for path in ("/World/objects/unit", "/World/objects/unit_drawer", "/World/objects/bottle"):
        UsdGeom.Xform.Define(stage, path)
    UsdPhysics.PrismaticJoint.Define(stage, "/World/joints/unit_slide")
    box("/World/objects/unit/left", (1.0, 0.0, 0.3), (0.02, 0.5, 0.6))  # One object: two panels.
    box("/World/objects/unit/right", (1.5, 0.0, 0.3), (0.02, 0.5, 0.6))
    box("/World/objects/bottle/shape", (1.2, 0.0, 0.5), (0.06, 0.06, 0.18))  # Excluded.
    box("/World/structure/wall", (0.0, 2.0, 1.5), (4.0, 0.1, 3.0))
    box("/World/structure/landing", (0.0, -1.0, 1.5), (2.0, 1.0, 0.2))  # Above the height band.
    box("/World/structure/far_wall", (9.0, 0.0, 1.5), (0.1, 4.0, 3.0))  # Outside the bounds.
    box("/World/envs/env_0/Robot/body", (0.0, 0.0, 0.5), (1.0, 0.5, 0.3))  # The robot is not a map box.

    unit = house_storage.unit_in_place(stage, "unit")
    assert unit.drawers == ["/World/objects/unit_drawer"] and unit.slides == ["/World/joints/unit_slide"]
    assert unit.handles == ["/World/objects/unit_drawer/handle"]
    boxes = house_storage.map_boxes(stage, (-2.0, -2.0, 3.0, 3.0), (0.05, 1.2), exclude=("bottle",))
    found = sorted((tuple(np.round(center, 3)), tuple(np.round(size, 3))) for center, size in boxes)
    assert found == [((0.0, 2.0, 1.5), (4.0, 0.1, 3.0)), ((1.25, 0.0, 0.3), (0.52, 0.5, 0.6))]
    with pytest.raises(ValueError, match="No storage unit"):
        house_storage.unit_in_place(stage, "missing")
