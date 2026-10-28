# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Skill lifecycle on the public client and the stow and look_at backends, without simulation."""

from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fakes import SPOT, TRANSIT, advance, setup
from scipy.spatial.transform import Rotation

from phygs.execution.manipulation.skills import backends
from phygs.execution.manipulation.skills.types import (
    HeldObject,
    ObjectTarget,
    PixelTarget,
    PointTarget,
    SkillConfig,
    SkillError,
    SkillFailure,
)
from phygs.observations.types import Stamp

CONFIG = replace(SkillConfig.from_robot(SPOT, "collision_avoidance_planner"), move_seconds=1.0)
JOINTS = CONFIG.joints


POINT = PointTarget((1.0, 0.0, 0.5))


class StubSkills:
    """Two low-level commands, optionally failing in between."""

    def __init__(self, client, failure=None):
        self.client, self.failure, self.attempts, self.issued = client, failure, 0, []
        self.step = ""

    def look_at(self, target, distance_m=0.5, elevation_deg=45.0, held=None):
        self.attempts = 1
        self.step = "look_at: moving"
        self.issued.append("move")
        yield self.client.move_joints({"arm": 0.5}, duration=1.0)
        if self.failure is not None:
            raise SkillError(self.failure, "Blocked")
        self.issued.append("close")
        yield self.client.close_gripper()


def test_skill_runs_commands_in_order_and_blocks_agent_commands():
    world, controller, runtime, client = setup(object_at=-0.4)
    with pytest.raises(RuntimeError, match="No manipulation skill"):
        client.look_at(POINT)
    client.bind_skills(StubSkills(client))
    with pytest.raises(RuntimeError, match="already bound"):
        client.bind_skills(StubSkills(client))
    opened = client.open_gripper()
    advance(runtime, client, 2.3)
    assert client.feedback(opened).status == "succeeded"
    skill = client.look_at(POINT)
    assert (client.feedback(skill).status, client.feedback(skill).step) == (
        "active",
        "look_at: moving",
    )
    with pytest.raises(RuntimeError, match="skill"):
        client.move_joints({"arm": 0.1})
    advance(runtime, client, 5.0)
    result = client.feedback(skill)
    assert (result.status, result.skill, result.failure, result.attempts) == (
        "succeeded",
        "look_at",
        None,
        1,
    )
    assert result.holding and client.holding and result.step == "look_at: moving"  # The last step.
    assert world.state["arm"] == pytest.approx(0.5)
    # A finished skill reports its end state like a motion does.
    assert result.progress == 1.0 and not result.stalled
    assert result.joint_positions["arm"] == pytest.approx(0.5)
    assert result.gripper_aperture == pytest.approx(world.state["claw"])


def test_skill_failure_reports_actionable_reason():
    world, controller, runtime, client = setup()
    client.bind_skills(StubSkills(client, failure=SkillFailure.PATH_UNOBSERVED))
    skill = client.look_at(POINT)
    advance(runtime, client, 2.5)
    result = client.feedback(skill)
    assert (result.status, result.failure, result.reason) == (
        "failed",
        SkillFailure.PATH_UNOBSERVED,
        "Blocked",
    )
    assert not result.holding
    client.move_joints({"arm": 0.0}, duration=1.0)  # The agent may command again.


def test_skill_cancel_and_reset_stop_issuing_commands():
    world, controller, runtime, client = setup()
    stub = StubSkills(client)
    client.bind_skills(stub)
    skill = client.look_at(POINT)
    advance(runtime, client, 0.3)
    client.cancel(skill)
    assert client.feedback(skill).status == "stopping"
    advance(runtime, client, 1.0)
    assert (client.feedback(skill).status, client.feedback(skill).failure) == (
        "cancelled",
        SkillFailure.CANCELLED,
    )
    assert stub.issued == ["move"]  # The gripper command was never issued.
    pending = client.look_at(POINT)
    client._end_episode()
    runtime.reset()
    assert client.feedback(pending).status == "cancelled"


class FakeArm:
    """Records commands; ``grasps`` decides per close whether an object ends up held."""

    episode_id = 1

    def __init__(self, grasps=(True,)):
        self.calls, self.holding, self._grasps, self.failures = (
            [],
            False,
            list(grasps),
            {},
        )

    def _call(self, *call):
        self.calls.append(call)
        return len(self.calls)

    def execute_trajectory(self, trajectory):
        """A pending plan that failed makes this command fail, as in the client."""
        if isinstance(trajectory, Future):
            if trajectory.exception() is not None:
                command = self._call("execute", "nothing")
                self.failures[command] = str(trajectory.exception())
                return command
            trajectory = trajectory.result()
        return self._call("execute", trajectory)

    def move_joints(self, target, duration):
        return self._call("move", tuple(target.values()))

    def open_gripper(self):
        self.holding = False
        return self._call("open")

    def close_gripper(self):
        self.holding = self._grasps.pop(0)
        return self._call("close")

    def minimum_move_duration(self, target):
        return 0.1


class FakeObservations:
    def __init__(self, camera=None, arm=None):
        self.camera = camera
        self.arm = [0.0] * len(JOINTS) if arm is None else list(arm)  # Measured arm joints [rad].

    def observe(self):
        return SimpleNamespace(
            localization=SimpleNamespace(valid=True, pose=SimpleNamespace(matrix=torch.eye(4))),
            joints=SimpleNamespace(names=(*JOINTS, "arm_f1x"), positions=torch.tensor([*self.arm, 0.0])),
            # The hand camera's optical frame coincides with the wrist (at the origin) unless given.
            cameras={"hand": self.camera or SimpleNamespace(extrinsics=SimpleNamespace(matrix=torch.eye(4)))},
            transform=lambda target, source, stamp: torch.eye(4),
        )


@pytest.fixture
def kinematics(monkeypatch):
    """Every pose solvable unless ``unreachable``; the wrist is at the world origin."""
    REACH.unreachable = False
    links = {"arm_link_wr1": np.eye(4), "arm_link_sh0": np.eye(4)}
    monkeypatch.setattr(backends, "link_transforms", lambda urdf, joints: links)
    return REACH


MODEL = {"urdf": "robot.urdf", "spheres": {"arm_link_wr1": []}, "locked_joints": {}}
REACH = SimpleNamespace(unreachable=False)


def fake_ik(pose, seed):
    """Stand-in for the planning library's IK: any pose is reachable unless REACH.unreachable."""
    return None if REACH.unreachable else [pose[2]] * 6


class FakeMotionPlanner:
    """Records goals; a pending plan named after the call, or refused with ``reasons`` in order."""

    def __init__(self, *reasons):
        self.requests, self.reasons = [], list(reasons)

    def __call__(self, goal, held=None):
        self.requests.append(goal)
        plan = Future()
        reason = self.reasons.pop(0) if self.reasons else None
        if reason is None:
            plan.set_result(f"plan {len(self.requests)}")
        else:
            plan.set_exception(AssertionError(f"Pose planner failed: {reason}"))
        return plan


def skills(
    arm,
    *,
    backend="collision_avoidance_planner",
    camera=None,
    ik=fake_ik,
    joint_planner=None,
    pose_planner=None,
    arm_joints=None,
):
    boxes = {"target": (np.array([0.8, 0.0, 0.46]), np.array([0.04, 0.04, 0.12]))}
    return backends.create_skills(
        arm,
        FakeObservations(camera, arm_joints),
        MODEL,
        replace(CONFIG, backend=backend),
        ik=ik,
        gripper_open={"arm_f1x": -1.4},
        object_boxes=boxes.__getitem__,
        joint_planner=FakeMotionPlanner() if joint_planner is None else joint_planner,
        pose_planner=FakeMotionPlanner() if pose_planner is None else pose_planner,
    )


def drive(sequence, failed_reason=None, arm=None):
    """Run a skill sequence; commands succeed unless failed_reason is set or the arm failed them."""
    try:
        command = next(sequence)
        while True:
            reason = failed_reason or (arm.failures.get(command) if arm is not None else None)
            command = sequence.send(SimpleNamespace(status="failed" if reason else "succeeded", reason=reason or ""))
    except StopIteration:
        return None
    except SkillError as error:
        return error


def test_pixel_targets_use_the_latest_depth_capture(kinematics):
    stamp = Stamp(1, 10, 0.32)
    depth = torch.full((4, 4), 2.0)
    valid = torch.ones((4, 4), dtype=torch.bool)
    valid[0, 0] = False
    camera = SimpleNamespace(
        stamp=stamp,
        depth=depth,
        depth_valid=valid,
        frame="hand",
        intrinsics=torch.tensor([[2.0, 0, 2], [0, 2.0, 2], [0, 0, 1]]),
    )
    backend = skills(FakeArm(), camera=camera)
    assert np.allclose(
        backend._resolve(PixelTarget("hand", stamp, 3, 2), np.eye(4))[0],
        [1.0, 0.0, 2.0],
    )
    stale = drive(backend.look_at(PixelTarget("hand", Stamp(1, 9, 0.288), 3, 2)))
    invalid = drive(backend.look_at(PixelTarget("hand", stamp, 0, 0)))
    assert stale.failure == invalid.failure == SkillFailure.TARGET_INVALID
    unprivileged = backends.DirectSkills(FakeArm(), FakeObservations(), MODEL, CONFIG, ik=fake_ik, gripper_open={})
    with pytest.raises(PermissionError, match="privileged"):
        drive(unprivileged.look_at(ObjectTarget("target")))


def test_held_object_spheres_cover_the_box_and_pose_round_trips():
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.2, 0.5]).as_matrix()
    pose[:3, 3] = [0.1, -0.05, 0.2]
    held = HeldObject((0.04, 0.06, 0.12), pose)
    world = np.eye(4)
    world[:3, 3] = [1.0, 2.0, 0.5]
    spheres = held.spheres(world, 0.04)
    samples = np.random.default_rng(1).uniform(-0.5, 0.5, (2000, 3)) * held.size
    points = samples @ (world @ pose)[:3, :3].T + (world @ pose)[:3, 3]
    covered = np.linalg.norm(points[:, None] - spheres[None, :, :3], axis=-1) <= spheres[None, :, 3] + 1e-9
    assert covered.any(axis=1).all()
    box = held.box()
    assert box["size"] == [0.04, 0.06, 0.12] and np.allclose(box["pose"][:3], pose[:3, 3])
    assert np.allclose(Rotation.from_quat(box["pose"][3:]).as_matrix(), pose[:3, :3])


def position_ik(pose, seed):
    """Joints that encode the wrist position, so a trajectory's samples show the wrist path."""
    return None if REACH.unreachable else [*pose[:3], 0.0, 0.0, 0.0]


def test_client_starts_stow_and_look_at_as_skills_with_the_held_object():
    class Recorder:
        attempts = 1

        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            def start(*args):  # A skill sequence that ends without commands.
                self.calls.append((name, *args))
                yield from ()

            return start

    world, controller, runtime, client = setup(object_at=-0.4)
    recorder = Recorder()
    client.bind_skills(recorder)
    held = HeldObject((0.04, 0.04, 0.12), np.eye(4), "target")
    for command in (client.stow(held=held), client.look_at(POINT), client.unstow()):
        assert client.feedback(command).status == "succeeded"
    assert [call[0] for call in recorder.calls] == ["stow", "look_at", "unstow"]
    assert recorder.calls[0][1] is held and recorder.calls[1][-1] is None  # Held as the agent says.
    assert client.episode_id == runtime.episode_id


def test_stow_folds_the_arm_and_closes_an_empty_gripper(kinematics):
    direct = FakeArm(grasps=(False,))
    assert drive(skills(direct, backend="direct").stow()) is None
    assert direct.calls == [("close",), ("move", CONFIG.stow_posture)]
    arm, joint_planner = FakeArm(grasps=(False,)), FakeMotionPlanner()
    assert drive(skills(arm, joint_planner=joint_planner).stow()) is None
    # Planned around obstacles to the pre-stow posture (sphere models call the stow a
    # self-collision), then straight into the stow.
    assert arm.calls == [
        ("close",),
        ("execute", "plan 1"),
        ("move", CONFIG.stow_posture),
    ]
    assert joint_planner.requests == [list(CONFIG.prestow_posture)]
    stowed, joint_planner = FakeArm(grasps=(False,)), FakeMotionPlanner()
    assert drive(skills(stowed, joint_planner=joint_planner, arm_joints=CONFIG.stow_posture).stow()) is None
    assert stowed.calls == [("close",), ("move", CONFIG.stow_posture)] and joint_planner.requests == []
    holding, joint_planner = FakeArm(), FakeMotionPlanner()
    holding.holding = True  # Carried in the stow: the gripper stays closed on it.
    assert drive(skills(holding, joint_planner=joint_planner).stow()) is None
    assert holding.calls == [("execute", "plan 1"), ("move", CONFIG.stow_posture)] and holding.holding


def test_stow_posture_is_the_posture_the_transit_footprint_was_measured_in():
    """Navigation's transit posture is the robot's stow (one source); the footprint was measured in it."""
    import json

    profile = json.loads(SPOT.navigation.footprint_profile.read_text())
    assert dict(TRANSIT) == dict(zip(SPOT.arm.joints, SPOT.arm.stow)) == dict(profile["arm_posture"])


def test_look_at_moves_the_camera_to_the_requested_view_of_the_target(kinematics):
    direct = FakeArm()
    assert drive(skills(direct, backend="direct", ik=position_ik).look_at(PointTarget((1.0, 0.0, 0.0)))) is None
    # Arm base and camera at the origin: by default 0.5 m from the target, 45 deg above it.
    side = 0.5 * np.sqrt(0.5)
    ((name, joints),) = direct.calls
    assert name == "move" and joints == pytest.approx((1.0 - side, 0.0, side, 0.0, 0.0, 0.0))
    arm, pose_planner = FakeArm(), FakeMotionPlanner()
    target = PointTarget((0.6, 0.3, -0.4))
    backend = skills(arm, ik=position_ik, pose_planner=pose_planner)
    assert drive(backend.look_at(target, 0.7, 60.0), arm=arm) is None
    assert backend.attempts == 1 and arm.calls == [("execute", "plan 1")]
    (pose,) = pose_planner.requests
    rotation = Rotation.from_quat(pose[3:])
    offset = np.array(target.position) - np.asarray(pose[:3])
    assert np.linalg.norm(offset) == pytest.approx(0.7)
    assert np.allclose(rotation.apply([0.0, 0.0, 1.0]), offset / np.linalg.norm(offset))  # Axis on the target.
    assert np.degrees(np.arcsin(-offset[2] / np.linalg.norm(offset))) == pytest.approx(60.0)
    assert rotation.apply([0.0, 1.0, 0.0])[2] < 1e-9  # Image "down" never points up: upright.
    with pytest.raises(ValueError):
        drive(backend.look_at(target, 0.5, 95.0))
    kinematics.unreachable = True
    unreachable = FakeArm()
    error = drive(skills(unreachable, ik=position_ik).look_at(PointTarget((1.0, 0.0, 0.0))))
    assert error.failure == SkillFailure.OUT_OF_REACH and unreachable.calls == []


def test_look_at_reports_a_refused_or_rejected_view_without_trying_another(kinematics):
    """Choosing another view is the agent's decision."""
    arm, refusing = FakeArm(), FakeMotionPlanner("no_plan")
    error = drive(
        skills(arm, ik=position_ik, pose_planner=refusing).look_at(PointTarget((1.0, 0.0, 0.0))),
        arm=arm,
    )
    assert error.failure == SkillFailure.PLANNING_FAILED and len(refusing.requests) == 1

    class GuardedArm(FakeArm):
        def execute_trajectory(self, trajectory):
            command = super().execute_trajectory(trajectory)
            self.failures[command] = "Trajectory intersects the known obstacle"
            return command

    arm, planner = GuardedArm(), FakeMotionPlanner()
    error = drive(
        skills(arm, ik=position_ik, pose_planner=planner).look_at(PointTarget((1.0, 0.0, 0.0))),
        arm=arm,
    )
    assert error.failure == SkillFailure.OBSTACLE_CHANGED and len(planner.requests) == 1


def test_planned_skills_leave_the_stow_through_the_prestow_posture(kinematics):
    """Planners reject the stowed posture as a start, so planned skills first unfold straight."""
    stowed = CONFIG.stow_posture
    arm = FakeArm()
    assert drive(skills(arm, ik=position_ik, arm_joints=stowed).look_at(PointTarget((1.0, 0.0, 0.0)))) is None
    assert arm.calls == [("move", CONFIG.prestow_posture), ("execute", "plan 1")]
    direct = FakeArm()  # Straight moves need no exit from the stow.
    view = skills(direct, backend="direct", ik=position_ik, arm_joints=stowed).look_at(PointTarget((1.0, 0.0, 0.0)))
    assert drive(view) is None
    assert len(direct.calls) == 1 and direct.calls[0][0] == "move" and direct.calls[0][1] != CONFIG.prestow_posture


def test_unstow_unfolds_straight_to_the_prestow_only_from_the_stow(kinematics):
    """Spot's "ready" after "stow": both skill backends unfold straight; elsewhere nothing moves."""
    for backend in ("direct", "collision_avoidance_planner"):
        arm = FakeArm()
        assert drive(skills(arm, backend=backend, arm_joints=CONFIG.stow_posture).unstow()) is None
        assert arm.calls == [("move", CONFIG.prestow_posture)]
        unfolded = FakeArm()
        assert drive(skills(unfolded, backend=backend, arm_joints=CONFIG.prestow_posture).unstow()) is None
        assert unfolded.calls == []


def test_skills_announce_their_steps_in_feedback_and_the_log(kinematics, caplog):
    caplog.set_level("INFO", logger="phygs.skills.manipulation")

    class StepArm(FakeArm):
        """Records the announced step when each command is issued."""

        def _call(self, *call):
            return super()._call(*call, backend.step)

    direct = StepArm()
    backend = skills(direct, backend="direct", ik=position_ik)
    assert drive(backend.look_at(PointTarget((1.0, 0.0, 0.0)), 0.5, 45.0)) is None
    assert [call[-1] for call in direct.calls] == ["look_at: moving straight to the view"]
    assert backend.step == "look_at: at the view; the arm has settled"
    assert [r.getMessage() for r in caplog.records] == [
        "look_at: point (world) (1.00, 0.00, 0.00) m; view 0.50 m, 45 deg up",
        "look_at: moving straight to the view",
        "look_at: at the view; the arm has settled",
    ]
