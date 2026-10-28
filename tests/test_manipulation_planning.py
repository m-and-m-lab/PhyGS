# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Planner trajectory contracts and the shared 3D collision-map representation."""

from concurrent.futures import Future

import numpy as np
import pytest
from fakes import SPOT, FakeScene, advance, isaaclab_world, setup

from phygs.execution.manipulation.planners.interface import IkRequest, PlanRequest
from phygs.execution.manipulation.planning import JointTrajectory
from phygs.tools.collision_map import boxes_in_body

REQUEST = PlanRequest(start=[0.0] * 6, boxes=[], goal=[0.0] * 6)


def test_execute_planner_trajectory_and_settle():
    world, controller, runtime, client = setup()
    plan = JointTrajectory(("arm",), np.array([0.0, 0.5, 1.0]), np.array([[0.0], [0.2], [0.5]]), runtime.episode_id)
    command = client.execute_trajectory(plan)
    advance(runtime, client, 0.5)
    assert world.state["arm"] == pytest.approx(0.2)
    assert client.feedback(command).status == "active"
    advance(runtime, client, 0.8)
    assert client.feedback(command).status == "succeeded"
    assert world.state["arm"] == pytest.approx(0.5)


def test_joint_moves_and_trajectories_end_when_the_arm_stops_under_load():
    # A held load keeps the arm 0.05 rad short (over joint_tolerance, within tracking_tolerance).
    for mode in ("trajectory", "joints"):
        world, controller, runtime, client = setup(sag=0.05)
        if mode == "trajectory":
            plan = JointTrajectory(("arm",), np.array([0.0, 1.0]), np.array([[0.0], [0.5]]), runtime.episode_id)
            command = client.execute_trajectory(plan)
        else:
            command = client.move_joints({"arm": 0.5}, duration=1)
        advance(runtime, client, 3.5)
        assert world.state["arm"] == pytest.approx(0.45)
        assert client.feedback(command).status == "succeeded", (mode, client.feedback(command))


def test_trajectory_start_refuses_only_a_jump():
    world, controller, runtime, client = setup()
    near = JointTrajectory(("arm",), np.array([0.0, 1.0]), np.array([[0.1], [0.5]]), runtime.episode_id)
    command = client.execute_trajectory(near)  # 0.1 rad away: within the tracking limit.
    advance(runtime, client, 2.0)
    assert client.feedback(command).status == "succeeded"
    jump = JointTrajectory(("arm",), np.array([0.0, 1.0]), np.array([[0.2], [0.5]]), runtime.episode_id)
    with pytest.raises(ValueError, match="start"):
        client.execute_trajectory(jump)  # 0.3 rad from the arm at 0.5.


def test_stale_plan_and_start_mismatch_rejected_before_acquisition():
    world, controller, runtime, client = setup()
    for episode, positions, message in [(0, [[0.0], [0.5]], "episode"), (runtime.episode_id, [[0.2], [0.5]], "start")]:
        plan = JointTrajectory(("arm",), np.array([0.0, 1.0]), np.array(positions), episode)
        with pytest.raises(ValueError, match=message):
            client.execute_trajectory(plan)
        assert runtime.controller_state("manipulation") == "idle"


def test_plan_cancel_does_not_keep_following_samples():
    world, controller, runtime, client = setup()
    plan = JointTrajectory(("arm",), np.array([0.0, 1.0]), np.array([[0.0], [0.5]]), runtime.episode_id)
    command = client.execute_trajectory(plan)
    advance(runtime, client, 0.3)
    stopped = world.state["arm"]
    client.cancel(command)
    advance(runtime, client, 0.5)
    assert client.feedback(command).status == "cancelled"
    assert world.state["arm"] == stopped


def test_trajectory_times_must_increase_and_a_too_fast_plan_is_slowed_uniformly():
    with pytest.raises(ValueError, match="increase"):
        JointTrajectory(("arm",), np.array([0.0, 0.0]), np.array([[0.0], [0.5]]), 1)
    plan = JointTrajectory(("arm",), np.array([0.0, 0.1]), np.array([[0.0], [0.5]]), 1)  # 5 rad/s.
    slowed, scale = plan.slowed(1.5)
    assert scale == pytest.approx(5.0 / 1.5) and slowed.peak_speed() == pytest.approx(1.5)
    np.testing.assert_array_equal(slowed.positions, plan.positions)  # The same path, only slower.
    assert plan.slowed(10.0)[1] == 1.0


def test_known_boxes_move_into_the_body_frame():
    boxes = [dict(size=[0.2, 0.1, 0.1], pose=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])]
    world_from_body = np.eye(4)
    world_from_body[0, 3] = 10
    world_from_body[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # Body yawed +90 degrees.
    (box,) = boxes_in_body(boxes, world_from_body)
    assert box["size"] == [0.2, 0.1, 0.1]
    assert box["pose"][:3] == pytest.approx([0.0, 9.9, 0.0])
    yaw_minus_90 = [0.0, 0.0, -np.sqrt(0.5), np.sqrt(0.5)]
    assert abs(np.dot(box["pose"][3:], yaw_minus_90)) == pytest.approx(1.0)  # q and -q are the same rotation.


def test_velocity_feedforward_stays_inside_ownership_and_clears_when_holding():
    from unittest.mock import Mock

    from phygs.runtime.episode import JointTargets

    with pytest.raises(ValueError, match="owned"):
        JointTargets({"arm": 0.0}, velocities={"leg": 1.0}).validate(("arm",))
    robot = Mock(device="cpu", joint_names=["arm"], actuators={})
    world = isaaclab_world(FakeScene(robot))
    world.apply(JointTargets({"arm": 0.1}, velocities={"arm": 0.5}))
    assert robot.set_joint_velocity_target_index.call_args.kwargs["target"].tolist() == [[0.5]]
    world.apply({"arm": 0.1})
    assert robot.set_joint_velocity_target_index.call_args.kwargs["target"].tolist() == [[0.0]]


def test_move_to_pose_plans_for_the_hand_and_passes_the_held_object():
    from phygs.skills.manipulation.api import HeldObject

    world, controller, runtime, client = setup(tools=SPOT.gripper.tools)
    requests = []
    client._pose_planner = lambda pose, held=None: requests.append((pose, held)) or Future()
    held = HeldObject((0.05, 0.05, 0.1), np.eye(4), name="bottle")
    client.move_to_pose((1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0), held=held)
    ((pose, passed),) = requests
    offset = SPOT.tool("hand")[:3, 3]
    np.testing.assert_allclose(pose[:3], (1.0 - offset[0], 0.0, 0.5 - offset[2]))  # The wrist behind the hand.
    assert passed is held


def test_pose_planner_failure_does_not_use_free_space_servo():
    world, controller, runtime, client = setup()

    def fail(pose, held=None):
        raise RuntimeError("No collision-free path")

    client._pose_planner = fail
    with pytest.raises(RuntimeError, match="collision-free"):
        client.move_to_pose((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    assert runtime.controller_state("manipulation") == "idle"


def test_async_pose_holds_then_executes_without_replacing_command():
    from concurrent.futures import Future

    world, controller, runtime, client = setup()
    pending = Future()
    client._pose_planner = lambda pose, held=None: pending
    command = client.move_to_pose((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    advance(runtime, client, 0.3)
    assert world.state["arm"] == 0.0
    assert client.feedback(command).status == "active"
    pending.set_result(JointTrajectory(("arm",), [0.0, 1.0], [[0.0], [0.5]], runtime.episode_id))
    advance(runtime, client, 1.3)
    assert client.feedback(command).status == "succeeded"


def test_cancel_pending_plan_cannot_restart_motion():
    from concurrent.futures import Future

    world, controller, runtime, client = setup()
    pending = Future()
    pending.set_running_or_notify_cancel()
    client._pose_planner = lambda pose, held=None: pending
    command = client.move_to_pose((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    client.cancel(command)
    pending.set_result(JointTrajectory(("arm",), [0.0, 1.0], [[0.0], [0.5]], runtime.episode_id))
    advance(runtime, client, 0.5)
    assert client.feedback(command).status == "cancelled"
    assert world.state["arm"] == 0.0


def test_async_stale_start_fails_without_motion():
    from concurrent.futures import Future

    world, controller, runtime, client = setup()
    pending = Future()
    client._pose_planner = lambda pose, held=None: pending
    command = client.move_to_pose((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    pending.set_result(JointTrajectory(("arm",), [0.0, 1.0], [[0.5], [1.0]], runtime.episode_id))
    advance(runtime, client, 0.4)
    assert client.feedback(command).status == "failed"
    assert "start" in client.feedback(command).reason
    assert world.state["arm"] == 0.0


def test_geometry_includes_hidden_guide_colliders_and_primitive_interiors():
    from pxr import Usd, UsdGeom, UsdPhysics

    from phygs.execution.manipulation.planners.model import audit_collision_geometry, collider_spheres

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Robot")
    body = UsdGeom.Xform.Define(stage, "/Robot/body")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    box = UsdGeom.Cube.Define(stage, "/Robot/body/collider")
    box.GetSizeAttr().Set(0.1)
    box.GetPurposeAttr().Set("guide")
    box.GetVisibilityAttr().Set("invisible")
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    spheres = collider_spheres(stage, str(root.GetPath()), SPOT.planning, cell_size=0.04)
    points = np.array(
        [
            [x, y, z]
            for x in np.linspace(-0.05, 0.05, 7)
            for y in np.linspace(-0.05, 0.05, 7)
            for z in np.linspace(-0.05, 0.05, 7)
        ]
    )
    distance = np.full(len(points), np.inf)
    for sphere in spheres["body"]:
        distance = np.minimum(distance, np.linalg.norm(points - sphere["center"], axis=1) - sphere["radius"])
    assert distance.max() <= 1e-8
    assert audit_collision_geometry(stage, "/Robot", {"spheres": spheres}, SPOT.planning)["qualified"]


def test_convex_hull_spheres_exclude_empty_aabb_corners_without_losing_coverage():
    from pxr import Usd, UsdGeom, UsdPhysics

    from phygs.execution.manipulation.planners.model import collider_spheres

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/robot")
    link = UsdGeom.Xform.Define(stage, "/robot/link")
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/robot/link/collider")
    mesh.CreatePointsAttr([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    spheres = collider_spheres(stage, "/robot", SPOT.planning, cell_size=0.1)["link"]
    assert len(spheres) < 500  # Full bounding box has 1000 cells.
    centers = np.array([s["center"] for s in spheres])
    radii = np.array([s["radius"] for s in spheres])
    # Grid samples cover the interior and all four faces of the tetrahedron.
    points = np.array([[x, y, z] for x in range(11) for y in range(11) for z in range(11) if x + y + z <= 10]) / 10
    distance = np.linalg.norm(points[:, None] - centers, axis=-1)
    assert np.all(np.any(distance <= radii + 1e-12, axis=1))
    assert not np.any(np.linalg.norm(centers - [0.9, 0.9, 0.9], axis=1) <= radii)


def test_batched_sphere_transforms_match_individual_transforms(tmp_path):
    from phygs.execution.manipulation.planners.model import link_transforms, transformed_spheres

    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        '<robot name="test"><joint name="arm" type="revolute"><parent link="body"/>'
        '<child link="link"/><axis xyz="0 0 1"/><origin xyz="0.2 0.1 0"/></joint></robot>'
    )
    model = {
        "urdf": str(urdf),
        "spheres": {
            "body": [{"center": [0, 0, 0], "radius": 0.1}],
            "link": [{"center": [0.3, 0, 0], "radius": 0.02}, {"center": [0, 0.2, 0], "radius": 0.04}],
        },
    }
    pose = np.eye(4)
    pose[:3, 3] = [2, 1, 0.5]
    joints = {"arm": 0.7}
    frames = link_transforms(str(urdf), joints)
    expected = []
    for name, spheres in model["spheres"].items():
        transform = pose @ frames[name]
        for sphere in spheres:
            expected.append([*(transform[:3, :3] @ sphere["center"] + transform[:3, 3]), sphere["radius"]])
    np.testing.assert_allclose(transformed_spheres(model, joints, pose), expected, rtol=1e-14, atol=1e-14)


def test_kinematics_cache_reloads_file_and_keeps_current_joint_state(tmp_path, monkeypatch):
    from phygs.execution.manipulation.planners import model

    path = tmp_path / "robot.urdf"
    template = (
        '<robot name="test"><joint name="arm" type="revolute"><parent link="body"/>'
        '<child link="link"/><axis xyz="0 0 1"/><origin xyz="{} 0 0"/></joint></robot>'
    )
    path.write_text(template.format("1"))
    parse = model.ET.parse
    calls = []

    def counted_parse(filename):
        calls.append(filename)
        return parse(filename)

    monkeypatch.setattr(model.ET, "parse", counted_parse)
    first = model.link_transforms(str(path), {"arm": 0})["link"]
    second = model.link_transforms(str(path), {"arm": np.pi / 2})["link"]
    assert len(calls) == 1
    np.testing.assert_allclose(second[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-14)
    first[:] = 0  # Returned matrices must not mutate cached geometry.
    assert model.link_transforms(str(path), {"arm": 0})["link"][0, 3] == 1
    replacement = tmp_path / "replacement.urdf"
    replacement.write_text(template.format("2.0"))
    replacement.replace(path)
    assert model.link_transforms(str(path), {"arm": 0})["link"][0, 3] == 2
    assert len(calls) == 2
    path.unlink()
    with pytest.raises(FileNotFoundError):
        model.link_transforms(str(path), {"arm": 0})


def test_plan_request_round_trips_through_json_and_needs_one_goal():
    import json

    from phygs.execution.manipulation.planners.interface import (
        DepthImage,
        HeldShape,
        IkRequest,
        PlanRequest,
        request_from_json,
    )

    held = HeldShape(box={"size": [0.06, 0.06, 0.14], "pose": [0.2, 0, 0, 0, 0, 0, 1]}, spheres=[[0.2, 0, 0, 0.01]])
    pose = [0.5, 0, 0.3, 0, 0, 0, 1]
    image = DepthImage(path="/tmp/hand.npy", intrinsics=[[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]], pose=pose)
    request = PlanRequest(start=[0.0] * 6, boxes=[], goal_pose=pose, gripper=-0.45, held=held, images=[image])
    assert request_from_json(json.loads(json.dumps(request.to_json()))) == request
    ik = IkRequest(pose=pose, seed=[0.0] * 6)
    assert request_from_json(json.loads(json.dumps(ik.to_json()))) == ik
    with pytest.raises(ValueError, match="Unknown planner request kind"):
        request_from_json({"kind": "dance"})
    with pytest.raises(ValueError, match="exactly one"):
        PlanRequest(start=[0.0] * 6, boxes=[])
    with pytest.raises(ValueError, match="exactly one"):
        PlanRequest(start=[0.0] * 6, boxes=[], goal=[0.0] * 6, goal_pose=[0.5, 0, 0.3, 0, 0, 0, 1])


def test_planner_process_stays_busy_until_an_abandoned_request_finishes(tmp_path):
    import os
    import sys
    import textwrap
    import time

    from phygs.execution.manipulation.planning import PlannerProcess

    worker = tmp_path / "worker.py"
    worker.write_text(
        textwrap.dedent(
            """
            import argparse, json, os, sys, time
            parser = argparse.ArgumentParser()
            parser.add_argument("--config")
            parser.add_argument("--response_fd", type=int)
            out = os.fdopen(parser.parse_args().response_fd, "w")
            out.write(json.dumps({"ready": True}) + "\\n")
            out.flush()
            for line in sys.stdin:
                time.sleep(0.3)
                out.write(json.dumps({"success": True}) + "\\n")
                out.flush()
            """
        )
    )
    process = PlannerProcess([sys.executable, str(worker)], {}, tmp_path, dict(os.environ))
    try:
        process.submit(REQUEST)
        assert process.busy
        with pytest.raises(RuntimeError, match="already pending"):
            process.submit(REQUEST)
        with pytest.raises(RuntimeError, match="IK must wait"):
            process.ik(IkRequest(pose=[0.5, 0, 0.3, 0, 0, 0, 1], seed=[0.0] * 6))
        deadline = time.monotonic() + 5.0
        while process.busy and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not process.busy
        assert process.submit(REQUEST).result(timeout=5.0)["success"]
        process.submit(REQUEST).result(timeout=5.0)
        assert process.ik(IkRequest(pose=[0.5, 0, 0.3, 0, 0, 0, 1], seed=[0.0] * 6))["success"]
    finally:
        process.close()


def test_planner_process_close_kills_a_worker_that_ignores_shutdown(tmp_path):
    import os
    import sys
    import textwrap

    from phygs.execution.manipulation.planning import PlannerProcess

    worker = tmp_path / "stubborn.py"
    worker.write_text(
        textwrap.dedent(
            """
            import argparse, json, os, signal, time
            parser = argparse.ArgumentParser()
            parser.add_argument("--config")
            parser.add_argument("--response_fd", type=int)
            out = os.fdopen(parser.parse_args().response_fd, "w")
            out.write(json.dumps({"ready": True}) + "\\n")
            out.flush()
            signal.signal(signal.SIGTERM, signal.SIG_IGN)  # Like a planner busy inside native code.
            time.sleep(60)
            """
        )
    )
    process = PlannerProcess([sys.executable, str(worker)], {}, tmp_path, dict(os.environ))
    process.close(wait_seconds=0.5)
    assert process.process.returncode is not None


def test_planning_model_uses_urdf_meshes_plus_usd_boxes_and_skips_static_pairs(tmp_path):
    import xml.etree.ElementTree as ET

    from phygs.execution.manipulation.planners.model import prepare_model

    # Every joint outside the arm, as measured (the model locks them).
    locked = dict.fromkeys(set(SPOT.joint_limits) - set(SPOT.arm.joints), 0.0)
    sphere = [{"center": [0.0, 0.0, 0.0], "radius": 0.05}]
    camera = {"center": [0.3, 0.0, 0.1], "size": [0.05, 0.1, 0.03]}
    model = prepare_model(
        SPOT,
        tmp_path,
        geometry={"body": sphere},
        locked_joints=locked,
        usd_boxes={"body": [camera]},
    )
    body = next(link for link in ET.parse(model["urdf"]).getroot().findall("link") if link.get("name") == "body")
    assert body.find("collision[@name='usd_only']/geometry/box").get("size") == "0.05 0.1 0.03"
    assert body.find("collision/geometry/mesh").get("filename").startswith("file:///")  # Absolute URL.
    ignored = model["self_collision_ignore"]
    # Body and leg are not adjacent (a mesh-less hip link sits between them) but both stay still.
    assert "front_left_upper_leg" in ignored["body"]
    # Links the arm moves are still checked against the static body unless adjacent.
    assert "arm_link_el0" not in ignored["body"]
    # MoveIt's arm links collide as the referee's spheres; its SRDF skips pairs over those links.
    referee = [{"center": [0.1, 0.0, 0.0], "radius": 0.03}]
    model = prepare_model(
        SPOT,
        tmp_path,
        geometry={"body": sphere, "arm_link_sh1": referee, "arm_link_el0": referee},
        locked_joints=locked,
        usd_boxes={},
    )
    links = {link.get("name"): link for link in ET.parse(model["moveit_urdf"]).getroot().findall("link")}
    assert [c.find("geometry/sphere").get("radius") for c in links["arm_link_el0"].findall("collision")] == ["0.03"]
    assert links["arm_link_sh1"].find("collision/origin").get("xyz") == "0.1 0.0 0.0"
    assert links["body"].find("collision/geometry/mesh") is not None  # Static links keep their meshes.
    pairs = {
        frozenset((d.get("link1"), d.get("link2")))
        for d in ET.parse(model["srdf"]).getroot().iter("disable_collisions")
    }
    assert frozenset(("arm_link_sh1", "arm_link_el0")) in pairs and frozenset(("body", "arm_link_el0")) not in pairs
    model = prepare_model(
        SPOT,
        tmp_path,
        geometry={"body": sphere, **{f"arm_link_{n}": referee for n in ("el0", "el1", "wr0", "wr1")}},
        locked_joints=locked,
        usd_boxes={},
    )
    pairs = {
        frozenset((d.get("link1"), d.get("link2")))
        for d in ET.parse(model["srdf"]).getroot().iter("disable_collisions")
    }
    # Two joints apart (el1 - wr0 - wr1): their spheres overlap at wr0; three apart stays checked.
    assert (
        frozenset(("arm_link_el1", "arm_link_wr1")) in pairs
        and frozenset(("arm_link_el0", "arm_link_wr1")) not in pairs
    )
    assert "arm_link_sh1" not in model["self_collision_ignore"]  # cuRobo's model is unchanged.
    with pytest.raises(ValueError, match="unknown links"):
        prepare_model(SPOT, tmp_path, geometry={"body": sphere}, locked_joints=locked, usd_boxes={"nope": []})


def test_usd_only_boxes_cover_colliders_the_urdf_lacks():
    """The jaw's grip spheres exist only in the USD; the planner hit the clutter without them."""
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    from phygs.execution.manipulation.planners.model import usd_only_boxes

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/robot")
    jaw = UsdGeom.Xform.Define(stage, "/robot/jaw").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(jaw)
    imported = UsdGeom.Cube.Define(stage, "/robot/jaw/collisions/finger")  # From the URDF.
    grip = UsdGeom.Sphere.Define(stage, "/robot/jaw/grip_sphere")  # USD only.
    grip.CreateRadiusAttr(0.02)
    grip.AddTranslateOp().Set((0.15, 0.0, 0.0))
    for shape in (imported, grip):
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr(True)
    ((box,),) = usd_only_boxes(stage, "/robot", SPOT.planning).values()
    assert box["center"] == pytest.approx([0.15, 0.0, 0.0]) and box["size"] == pytest.approx([0.04] * 3)


def test_arm_ik_solves_reachable_poses_and_stays_inside_single_precision_limits(tmp_path):
    from scipy.spatial.transform import Rotation

    from phygs.execution.manipulation.kinematics import ArmKinematics

    links = "".join(f'<link name="{n}"/>' for n in ("body", "a", "b"))
    urdf = tmp_path / "arm.urdf"
    urdf.write_text(
        f"""<robot name="arm">{links}
        <joint name="j0" type="revolute"><parent link="body"/><child link="a"/><origin xyz="0 0 0"/>
          <axis xyz="0 0 1"/><limit lower="-2.8797932657906435" upper="2.8797932657906435"/></joint>
        <joint name="j1" type="revolute"><parent link="a"/><child link="b"/><origin xyz="0.5 0 0"/>
          <axis xyz="0 0 1"/><limit lower="-2.0" upper="2.0"/></joint></robot>"""
    )
    arm = ArmKinematics(urdf, ("j0", "j1"), tip="b")
    target = arm.forward([0.4, -0.3])
    pose = [*target[:3, 3], *Rotation.from_matrix(target[:3, :3]).as_quat()]
    np.testing.assert_allclose(arm.inverse(pose, [0.0, 0.0]), [0.4, -0.3], atol=1e-4)
    # A roll just past the URDF limit, within the IK tolerances, pins the joint to its bound. The
    # solution keeps clear of the bound: the simulator holds the limit in single precision (the
    # wrist roll's -2.8797929286956787 rad rejected a solution 4e-10 rad inside -2.87979326579 on
    # 2026-10-13).
    wrist = ArmKinematics(urdf, ("j0",), tip="a")
    edge = wrist.forward([-2.8797932657906435 - 0.0015])
    for seed in (-2.0, 0.0):
        joints = wrist.inverse([*edge[:3, 3], *Rotation.from_matrix(edge[:3, :3]).as_quat()], [seed])
        assert joints is not None and joints[0] >= -2.8797932657906435 + 5e-5
    assert arm.inverse([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0]) is None  # Beyond the 0.5 m arm.


def test_arm_ik_returns_the_solution_nearest_the_seed(tmp_path):
    """A planar three-link arm reaches a tip pose elbow up or elbow down; the seed's side is kept."""
    from scipy.spatial.transform import Rotation

    from phygs.execution.manipulation.kinematics import ArmKinematics

    links = "".join(f'<link name="{n}"/>' for n in ("body", "a", "b", "c", "tip"))
    joint = '<joint name="{}" type="revolute"><parent link="{}"/><child link="{}"/><origin xyz="{} 0 0"/>'
    joint += '<axis xyz="0 0 1"/><limit lower="-3.0" upper="3.0"/></joint>'
    urdf = tmp_path / "arm.urdf"
    urdf.write_text(
        f"""<robot name="arm">{links}{joint.format("j0", "body", "a", 0.0)}{joint.format("j1", "a", "b", 0.5)}
        {joint.format("j2", "b", "c", 0.5)}<joint name="tip" type="fixed"><parent link="c"/><child link="tip"/>
        <origin xyz="0.2 0 0"/></joint></robot>"""
    )
    arm = ArmKinematics(urdf, ("j0", "j1", "j2"), tip="tip")
    target = arm.forward([0.6, -1.2, 0.3])
    pose = [*target[:3, 3], *Rotation.from_matrix(target[:3, :3]).as_quat()]
    for seed in ([0.0, -2.5, 2.0], [0.0, 2.5, -2.0]):
        joints = arm.inverse(pose, seed)
        np.testing.assert_allclose(arm.forward(joints)[:3, 3], target[:3, 3], atol=1e-3)
        assert np.sign(joints[1]) == np.sign(seed[1])  # The elbow stays on the seed's side.
    # From this seed the search finds the elbow-down solution first; the elbow-up one is nearer.
    np.testing.assert_allclose(arm.inverse(pose, [0.0, 2.9, 0.0]), [-0.6, 1.2, -0.9], atol=1e-3)
