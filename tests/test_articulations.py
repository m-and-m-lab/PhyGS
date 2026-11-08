# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Articulated furniture: a cabinet and its drawer and door in one PhysX articulation, as the
SimReady kitchen of bear_loop_house 1.2.0 builds its storage. Stage reads and authoring only, and
:class:`SceneBodies` against fakes (no Isaac Sim)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

from phygs.scene import reader as scene
from phygs.scene.articulations import articulated_joints, articulations, author_joint_position, joint_position
from phygs.world.spec import EpisodeSpec

CABINET, DRAWER, DOOR, BOWL = (f"/World/objects/{name}" for name in ("cabinet", "drawer", "door", "bowl"))
SLIDE, HINGE = "/World/joints/slide", "/World/joints/hinge"
GRID_DRAWER = EpisodeSpec.load(
    Path(__file__).resolve().parents[1] / "tests" / "robot_tests" / "configs" / "episodes" / "dev_grid_drawer.yaml"
)
SLIDE_AT, HINGE_AT, DOOR_EDGE = (0.0, -0.3, 0.4), (0.25, -0.3, 0.4), (-0.2, 0.0, 0.0)


def cabinet_stage() -> Usd.Stage:
    """A cabinet (turned 90 degrees) mounted to the world as an articulation root, a drawer on a
    slide along its y axis, a door on a hinge about z at the door's edge; a bowl and a room door
    hinged to the world, neither articulated."""
    stage = Usd.Stage.CreateInMemory()
    for path in (CABINET, DRAWER, DOOR, BOWL, "/World/objects/room_door"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, path).GetPrim())
    cabinet = UsdGeom.Xformable(stage.GetPrimAtPath(CABINET))
    cabinet.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 0.0))
    cabinet.AddRotateZOp().Set(90.0)
    mount = UsdPhysics.FixedJoint.Define(stage, "/World/joints/cabinet_mount")
    mount.CreateBody1Rel().SetTargets([CABINET])
    UsdPhysics.ArticulationRootAPI.Apply(mount.GetPrim())
    slide = UsdPhysics.PrismaticJoint.Define(stage, SLIDE)
    slide.CreateAxisAttr("Y")
    hinge = UsdPhysics.RevoluteJoint.Define(stage, HINGE)
    hinge.CreateAxisAttr("Z")
    for joint, child, at, edge in ((slide, DRAWER, SLIDE_AT, (0, 0, 0)), (hinge, DOOR, HINGE_AT, DOOR_EDGE)):
        joint.CreateBody0Rel().SetTargets([CABINET])
        joint.CreateBody1Rel().SetTargets([child])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*at))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*edge))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    room = UsdPhysics.RevoluteJoint.Define(stage, "/World/joints/room_door_hinge")
    room.CreateBody1Rel().SetTargets(["/World/objects/room_door"])
    return stage


def matrix(translation=(0.0, 0.0, 0.0), yaw=0.0) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    result[:3, 3] = translation
    return result


def furniture_poses(opening: float, swing: float) -> dict[str, np.ndarray]:
    """World poses (4x4) of the cabinet, the drawer ``opening`` [m] out and the door ``swing`` [rad] open."""
    cabinet = matrix((1.0, 2.0, 0.0), np.pi / 2)
    drawer = cabinet @ matrix(SLIDE_AT) @ matrix((0.0, opening, 0.0))
    door = cabinet @ matrix(HINGE_AT) @ matrix(yaw=swing) @ np.linalg.inv(matrix(DOOR_EDGE))
    return {CABINET: cabinet, DRAWER: drawer, DOOR: door}


def test_articulated_furniture_is_found_with_its_links_and_moving_joints():
    stage = cabinet_stage()
    (found,) = articulations(stage)
    assert (found.root, found.base, found.fixed) == ("/World/joints/cabinet_mount", CABINET, True)
    assert found.links == (CABINET, DRAWER, DOOR)
    assert [(joint.path, joint.kind, joint.axis) for joint in found.joints] == [
        (SLIDE, "prismatic", 1),
        (HINGE, "revolute", 2),
    ]
    assert articulated_joints(stage) == {SLIDE, HINGE}  # The room door hinged to the world is not one.
    assert articulations(stage, {BOWL}) == []  # Only articulations based on the given bodies.


def test_a_joint_position_is_found_from_its_two_bodies_poses():
    (found,) = articulations(cabinet_stage())
    slide, hinge = found.joints
    poses = furniture_poses(opening=0.2, swing=0.5)
    assert joint_position(slide, poses[CABINET], poses[DRAWER]) == pytest.approx(0.2)
    assert joint_position(hinge, poses[CABINET], poses[DOOR]) == pytest.approx(0.5)
    closed = furniture_poses(opening=0.0, swing=0.0)
    assert joint_position(slide, closed[CABINET], closed[DRAWER]) == pytest.approx(0.0, abs=1e-6)
    assert joint_position(hinge, closed[CABINET], closed[DOOR]) == pytest.approx(0.0, abs=1e-6)


def test_an_articulated_joint_starts_where_its_joint_state_says():
    stage = cabinet_stage()
    slide, hinge = articulations(stage)[0].joints
    author_joint_position(stage, slide, 0.25)
    author_joint_position(stage, hinge, np.pi / 2)
    prim = stage.GetPrimAtPath(SLIDE)  # PhysxSchema is not loaded here: read the authored list.
    assert "PhysicsJointStateAPI:linear" in prim.GetMetadata("apiSchemas").GetAddedOrExplicitItems()
    assert prim.GetAttribute("state:linear:physics:position").Get() == pytest.approx(0.25)
    assert stage.GetPrimAtPath(HINGE).GetAttribute("state:angular:physics:position").Get() == pytest.approx(90.0)


def bodies(stage: Usd.Stage) -> dict[str, str]:
    return {scene.name_of(path): path for path in scene.rigid_bodies(stage)}


def grid() -> Usd.Stage:
    """The grid scene, edited in its own session layer: the file's layer is shared by every open."""
    stage = Usd.Stage.Open(str(GRID_DRAWER.scene.usd))
    stage.SetEditTarget(stage.GetSessionLayer())
    return stage


def articulate_grid(stage: Usd.Stage) -> None:
    """Tie the grid nightstand's drawers to its carcass in one articulation (as the house's
    SimReady kitchen does): each slide's world frame becomes a frame on the carcass."""
    carcass = "/World/objects/corner_nightstand"
    world_from_carcass = scene.world_pose(stage, carcass)
    rotation = Rotation.from_matrix(world_from_carcass[:3, :3])
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.PrismaticJoint):
            joint = UsdPhysics.PrismaticJoint(prim)
            position = (np.linalg.inv(world_from_carcass) @ [*joint.GetLocalPos0Attr().Get(), 1.0])[:3]
            local = joint.GetLocalRot0Attr().Get()
            turned = (rotation.inv() * Rotation.from_quat([*local.GetImaginary(), local.GetReal()])).as_quat()
            joint.GetBody0Rel().SetTargets([carcass])
            joint.GetLocalPos0Attr().Set(Gf.Vec3f(*map(float, position)))
            joint.GetLocalRot0Attr().Set(Gf.Quatf(float(turned[3]), *map(float, turned[:3])))
        elif prim.IsA(UsdPhysics.FixedJoint) and UsdPhysics.Joint(prim).GetBody1Rel().GetTargets() == [carcass]:
            UsdPhysics.ArticulationRootAPI.Apply(prim)


def test_an_articulated_drawer_opens_by_its_joint_state_and_its_contents_ride_along():
    """The opening is authored as the slide's joint state (PhysX ignores a link's pose), the drawer's
    pose moves too (readers of the authored stage see it open), and an object settled with one
    transform op rides along."""
    stage = grid()
    articulate_grid(stage)
    found, named = scene.drawers(stage), bodies(stage)
    top = found["corner_nightstand_tier_2_drawer"]
    assert top.slide in articulated_joints(stage)
    scene.apply_initial_state(stage, named, GRID_DRAWER.objects)  # The target in the top drawer.
    target = stage.GetPrimAtPath(named["target"])
    settled = UsdGeom.Xformable(target)
    pose = settled.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    settled.ClearXformOpOrder()
    settled.AddTransformOp().Set(pose)  # As the scene's settle tool authors it.
    before = scene.body_box(stage, named["target"])[0]
    scene.apply_openings(stage, named, found, {"corner_nightstand_tier_2_drawer": {"opening": 0.25}})
    assert stage.GetPrimAtPath(top.slide).GetAttribute("state:linear:physics:position").Get() == pytest.approx(0.25)
    opened = scene.sliding_part(stage, top.slide, top.handle, lambda path: scene.world_pose(stage, path))
    assert opened.opening == pytest.approx(0.25, abs=1e-6)
    assert scene.body_box(stage, named["target"])[0] == pytest.approx(
        before + 0.25 * np.asarray(opened.direction), abs=1e-6
    )
    plain = grid()  # Not articulated: the drawer's pose alone opens it.
    scene.apply_openings(
        plain, bodies(plain), scene.drawers(plain), {"corner_nightstand_tier_2_drawer": {"opening": 0.25}}
    )
    assert not plain.GetPrimAtPath(top.slide).GetAttribute("state:linear:physics:position").IsValid()


class FakeArticulationView:
    """PhysX's articulation view of the cabinet: link and joint paths per row (padded), the writes."""

    count, max_dofs = 1, 3
    link_paths = [[CABINET, DRAWER, DOOR, ""]]
    dof_paths = [[SLIDE, HINGE, ""]]

    def __init__(self):
        self.writes = {}

    def set_dof_positions(self, data, indices):
        import warp as wp

        self.writes["positions"] = wp.to_torch(data).clone()

    def set_dof_velocities(self, data, indices):
        import warp as wp

        self.writes["velocities"] = wp.to_torch(data).clone()


class FakePhysics:
    def __init__(self, view):
        self.view, self.roots, self.kinematic_updates = view, None, 0

    def create_articulation_view(self, roots):
        self.roots = roots
        return self.view

    def update_articulations_kinematic(self):
        self.kinematic_updates += 1


def test_scene_bodies_write_articulated_links_through_their_joints():
    from fakes import FakeCollection

    from phygs.scene.assets import SCENE_BODIES, SceneBodies

    stage = cabinet_stage()
    names = scene.rigid_bodies(stage)
    collection = FakeCollection(names)
    collection._physics_sim_view = physics = FakePhysics(FakeArticulationView())
    scene_bodies = SceneBodies({SCENE_BODIES: collection}, stage=stage)
    assert scene_bodies.articulated.links == {CABINET, DRAWER, DOOR}
    poses = torch.zeros((len(names), 7))
    poses[:, 6] = 1.0
    for path, world in furniture_poses(opening=0.2, swing=0.5).items():
        poses[names.index(path), :3] = torch.tensor(world[:3, 3])
        poses[names.index(path), 3:] = torch.tensor(Rotation.from_matrix(world[:3, :3]).as_quat())
    poses[names.index(BOWL), 0] = 7.0
    scene_bodies.write_all(poses, torch.zeros((len(names), 6)))
    rigid = [names.index(BOWL), names.index("/World/objects/room_door")]
    assert collection.writes == [("pose", rigid), ("velocity", rigid)]  # Links are not written as bodies.
    assert scene_bodies.pose(BOWL)[0] == 7.0
    assert physics.roots == ["/World/joints/cabinet_mount"] and physics.kinematic_updates == 1
    written = physics.view.writes
    assert written["positions"][0].tolist() == pytest.approx([0.2, 0.5, 0.0], abs=1e-6)
    assert written["velocities"].abs().max() == 0.0
    with pytest.raises(ValueError, match="moves with its joints"):
        scene_bodies.write_pose(DRAWER, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    scene_bodies.write_pose(BOWL, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # An ordinary body still can be.


def test_an_articulated_drawer_reads_its_opening_against_its_cabinet():
    """``UsdScene.part`` takes the slide's two bodies as they are now: the drawer and its cabinet."""
    stage = grid()
    articulate_grid(stage)
    named = bodies(stage)
    house = scene.UsdScene("grid_storage", GRID_DRAWER.scene.usd, ground=True, objects={},
                           navigation_map=dict(resolution=0.05, clearance=0.0, height=(0.05, 1.2), margin=2.0),
                           robot_start=(-1.6, -0.2, 0.0))  # fmt: skip
    house._stage, house.bodies, house.drawers, house._interiors = stage, named, scene.drawers(stage), {}

    class Authored:  # Ground truth as the authored stage: each body's world pose [m, xyzw].
        def pose(self, path):
            world = scene.world_pose(stage, path)
            return torch.tensor([*world[:3, 3], *Rotation.from_matrix(world[:3, :3]).as_quat()])

    house._bodies = Authored()
    scene.apply_openings(stage, named, house.drawers, {"corner_nightstand_tier_2_drawer": {"opening": 0.1}})
    assert house.part("corner_nightstand_tier_2_drawer").opening == pytest.approx(0.1, abs=1e-6)
    assert house.part("corner_nightstand_drawer").opening == pytest.approx(0.0, abs=1e-6)
