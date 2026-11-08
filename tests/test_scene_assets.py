# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from phygs.scene.assets import validate_spawn_disk


def test_spawn_prior_rejects_collider_and_accepts_clear_disk():
    stage = Usd.Stage.CreateInMemory()
    cube = UsdGeom.Cube.Define(stage, "/Obstacle")
    cube.CreateSizeAttr(1)
    collision = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    validate_spawn_disk(stage, (2, 0), 0.7, (0.2, 1.2))
    with pytest.raises(ValueError, match="/Obstacle"):
        validate_spawn_disk(stage, (1, 0), 0.7, (0.2, 1.2))
    collision.CreateCollisionEnabledAttr(False)
    validate_spawn_disk(stage, (1, 0), 0.7, (0.2, 1.2))


def test_floor_below_height_band_does_not_block_spawn():
    stage = Usd.Stage.CreateInMemory()
    cube = UsdGeom.Cube.Define(stage, "/Floor")
    cube.CreateSizeAttr(0.2)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    validate_spawn_disk(stage, (0, 0), 0.7, (0.2, 1.2))


def test_scene_bodies_register_as_one_collection_named_by_path():
    from types import SimpleNamespace

    from phygs.scene.assets import SCENE_BODIES, register_scene_bodies

    stage = Usd.Stage.CreateInMemory()
    for path in ("/World/objects/drawer", "/World/objects/bowl"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, path).GetPrim())
    UsdGeom.Xform.Define(stage, "/World/structure/wall")  # Static: not registered.
    config = SimpleNamespace()
    assert register_scene_bodies(stage, config) == ("/World/objects/drawer", "/World/objects/bowl")
    collection = getattr(config, SCENE_BODIES)
    assert list(collection.rigid_objects) == ["/World/objects/drawer", "/World/objects/bowl"]
    assert collection.rigid_objects["/World/objects/bowl"].prim_path == "/World/objects/bowl"
    empty = SimpleNamespace()
    assert register_scene_bodies(Usd.Stage.CreateInMemory(), empty) == () and not hasattr(empty, SCENE_BODIES)


def test_scene_bodies_read_and_write_one_body_by_path():
    import torch
    from fakes import FakeCollection

    from phygs.scene.assets import SCENE_BODIES, SceneBodies

    collection = FakeCollection(["/World/objects/drawer", "/World/objects/bowl"])
    bodies = SceneBodies({SCENE_BODIES: collection})
    assert "/World/objects/bowl" in bodies and "/World/objects/cup" not in bodies
    assert bodies.pose("/World/objects/bowl")[0] == 1.0
    bodies.write_pose("/World/objects/bowl", [5.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
    bodies.write_velocity("/World/objects/bowl", [0.0, 0.0, -1.0, 0.0, 0.0, 0.0])
    assert collection.writes == [("pose", [1]), ("velocity", [1])]  # The bowl only.
    assert bodies.pose("/World/objects/bowl")[0] == 5.0 and bodies.pose("/World/objects/drawer")[0] == 0.0
    assert bodies.velocity("/World/objects/bowl")[2] == -1.0
    with pytest.raises(KeyError, match="Not a registered scene body"):
        bodies.pose("/World/objects/cup")
    with pytest.raises(ValueError, match="finite"):
        bodies.hold_force("/World/objects/drawer", [float("nan"), 0.0, 0.0])
    bodies.hold_force("/World/objects/drawer", [0.0, -30.0, 0.0])
    assert torch.equal(bodies._forces[0], torch.tensor([0.0, -30.0, 0.0]))
    bodies.hold_force("/World/objects/drawer", None)
    assert bodies._forces == {}


def test_world_restores_the_scene_bodies_and_drops_held_forces():
    from unittest.mock import Mock

    import torch
    from fakes import FakeCollection, FakeRobot, FakeScene, isaaclab_world

    from phygs.scene.assets import SCENE_BODIES

    collection = FakeCollection(["/World/objects/drawer", "/World/objects/bowl"])
    scene = FakeScene(FakeRobot(("leg",)), collections={SCENE_BODIES: collection})
    fabric_writes = Mock()
    world = isaaclab_world(scene, fabric_writes=fabric_writes)
    world.scene_bodies.write_pose("/World/objects/bowl", [9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0])
    world.scene_bodies.hold_force("/World/objects/drawer", [0.0, -30.0, 0.0])
    world.restore()
    assert torch.equal(collection.data.body_link_pose_w.torch[0, :, 0], torch.tensor([0.0, 1.0]))
    assert world.scene_bodies._forces == {}
    assert scene.restored == [{}]
    fabric_writes.set.assert_called_once_with(True)  # Settling and the first capture render.
