# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint execution, resource ownership and measured completion without simulation."""

from dataclasses import replace

import numpy as np
import pytest
from fakes import FakeScene, advance, isaaclab_world, setup

from phygs.execution.manipulation.controller import ManipulationController, MotionConfig
from phygs.execution.manipulation.planning import JointTrajectory
from phygs.runtime.episode import EpisodeRuntime, RunMode


def test_joint_success_requires_motion_and_settling():
    world, controller, runtime, client = setup()
    command = client.move_joints({"arm": 0.5}, duration=1)
    advance(runtime, client, 0.5)
    assert world.state["arm"] == pytest.approx(0.25)
    assert client.feedback(command).status == "active"
    advance(runtime, client, 0.8)
    assert client.feedback(command).status == "succeeded"
    assert world.state == {"arm": pytest.approx(0.5), "claw": 0}
    assert runtime.controller_state("manipulation") == "idle"


def test_a_joint_move_lasts_whole_physics_steps_never_shorter_than_asked():
    world, controller, runtime, client = setup()
    asked = 1.0 + 0.4 * runtime.physics_dt  # Between two physics steps.
    command = client.move_joints({"arm": 0.5}, duration=asked)
    assert controller.duration == pytest.approx(1.0 + runtime.physics_dt)
    advance(runtime, client, 2.0)
    assert client.feedback(command).status == "succeeded"
    with pytest.raises(ValueError, match="positive"):
        client.move_joints({"arm": 0.0}, duration=0.0)


def test_released_arm_keeps_its_commanded_target_under_load():
    world, controller, runtime, client = setup(sag=0.02)
    command = client.move_joints({"arm": 0.5}, duration=1)
    advance(runtime, client, 1.5)
    assert client.feedback(command).status == "succeeded"
    assert runtime.controller_state("manipulation") == "idle"
    advance(runtime, client, 1.0)  # Released: the hold is the commanded 0.5, the arm stays 0.02 below.
    assert world.target["arm"] == pytest.approx(0.5) and world.state["arm"] == pytest.approx(0.48)
    command = client.close_gripper()  # The next command starts from the held target.
    advance(runtime, client, 3.0)
    assert client.feedback(command).status == "succeeded"
    assert world.state["arm"] == pytest.approx(0.48)


def test_planning_wait_holds_the_commanded_arm():
    from concurrent.futures import Future

    world, controller, runtime, client = setup(sag=0.02)
    advance(runtime, client, 0.0)
    command = client.move_joints({"arm": 0.5}, duration=1)
    advance(runtime, client, 1.5)
    assert client.feedback(command).status == "succeeded"
    pending = Future()
    command = client.execute_trajectory(pending)  # The plan is computed from the measured arm.
    advance(runtime, client, 2.0)
    assert world.state["arm"] == pytest.approx(0.48)  # Not 0.46: the wait does not re-target 0.48.
    pending.set_result(JointTrajectory(("arm",), np.array([0.0, 0.5]), np.array([[0.48], [0.3]]), runtime.episode_id))
    advance(runtime, client, 2.0)
    assert client.feedback(command).status == "succeeded"


def test_failed_motion_holds_where_the_arm_stopped():
    world, controller, runtime, client = setup(sag=0.02)
    command = client.move_joints({"arm": 0.5}, duration=1)
    advance(runtime, client, 1.5)
    world.stuck = True
    command = client.move_joints({"arm": 1.0}, duration=2)
    advance(runtime, client, 7.0)
    assert client.feedback(command).status == "failed"
    assert controller.arm_hold == {} and world.target["arm"] == pytest.approx(world.state["arm"])


def test_blocked_joint_move_fails_while_moving_instead_of_reporting_success():
    world, controller, runtime, client = setup(stuck=True)
    command = client.move_joints({"arm": 0.5}, duration=1)
    advance(runtime, client, 3.3)
    result = client.feedback(command)
    assert result.status == "failed"
    assert "tracking" in result.reason  # 0.15 rad behind its moving reference, well before the timeout.
    assert world.state["arm"] == 0


def test_an_arm_that_does_not_stop_fails_the_command_not_the_episode():
    world, controller, runtime, client = setup(stuck=True)
    command = client.move_joints({"arm": 0.5}, duration=1)
    while controller.mode != "stopping":  # Blocked: tracking stops the arm.
        advance(runtime, client, runtime.physics_dt)
    step = world.step

    def drifting():  # The arm keeps creeping after the stop, 0.2 rad/s.
        step()
        world.state["arm"] += 0.2 * world.physics_dt

    world.step = drifting
    advance(runtime, client, 3.0)
    feedback = client.feedback(command)
    assert feedback.status == "failed" and "did not physically stop" in feedback.reason
    assert runtime.status.value == "running" and runtime.controller_state("manipulation") == "idle"


# The claw closes at its upper limit (0 rad), as Spot's arm_f1x does.
CLAW_AT_LIMIT = {"arm": (-2, 2), "claw": (-1, 0)}


def trajectory(runtime, positions):
    """A one-joint arm trajectory through ``positions`` [rad], 0.5 s apart."""
    times = np.arange(len(positions)) * 0.5
    return JointTrajectory(("arm",), times, np.asarray(positions, dtype=float)[:, None], runtime.episode_id)


@pytest.mark.parametrize("past", [1.8e-7, 1e-3])
def test_a_joint_measured_past_its_limit_is_held_at_the_limit(past):
    """The simulator reports a closed claw a hair past its limit (arm_f1x = 1.8e-7 rad, 2026-10-18);
    an arm command holding the uncommanded claw must not end the episode."""
    from concurrent.futures import Future

    world, controller, runtime, client = setup(limits=CLAW_AT_LIMIT)
    world.state["claw"] = past
    step = world.step

    def claw_past_limit():  # The arm follows its targets; the claw rests past its limit.
        step()
        world.state["claw"] = past

    world.step = claw_past_limit
    pending = Future()
    pending.set_result(trajectory(runtime, [0.0, 0.3]))
    command = client.execute_trajectory(pending)
    advance(runtime, client, 0.2)
    assert controller.reference["claw"] == 0.0  # Held at the limit, not past it.
    advance(runtime, client, 2.0)
    assert client.feedback(command).status == "succeeded" and runtime.status.value == "running"
    move = client.move_joints({"arm": 0.1}, duration=1)
    advance(runtime, client, 0.1)
    assert controller.reference["claw"] == 0.0
    client.cancel(move)  # The stop holds the measured claw too.
    advance(runtime, client, 0.1)
    assert controller.target["claw"] == 0.0 and runtime.status.value == "running"


def test_a_servo_target_outside_the_joint_limits_fails_the_command_not_the_episode():
    class Servo:  # IK that walks the arm toward a pose past its limit.
        def start(self, pose):
            pass

        def targets(self, fraction):
            return {"arm": 2.5 * fraction}

        def error(self):
            return (0.1, 0.0)

    world, controller, runtime, client = setup(limits=CLAW_AT_LIMIT)
    controller.pose_servo = Servo()
    command = client.move_to_pose((0, 0, 0, 0, 0, 0, 1), duration=20.0)
    advance(runtime, client, 20.0)
    feedback = client.feedback(command)
    assert feedback.status == "failed" and "outside" in feedback.reason
    assert runtime.status.value == "running" and runtime.controller_state("manipulation") == "idle"
    assert world.target["arm"] <= 2


def test_gripper_closed_follows_the_last_gripper_command_and_a_held_object():
    world, controller, runtime, client = setup(object_at=-0.4)  # Closed is 0, open -1.
    assert not client.gripper_closed  # Never commanded.
    client.open_gripper()
    advance(runtime, client, 2.3)
    assert not client.gripper_closed
    client.close_gripper()
    advance(runtime, client, 2.5)
    assert client.holding and client.gripper_closed  # Squeezing 0.05 rad past the object at -0.4.
    client.open_gripper()
    advance(runtime, client, 2.3)
    assert not client.gripper_closed


def test_a_wide_object_keeps_the_gripper_closed_and_reports_its_aperture():
    world, controller, runtime, client = setup(object_at=-0.9)  # Stops nearer the open angle (-1).
    client.open_gripper()
    advance(runtime, client, 2.3)
    command = client.close_gripper()
    advance(runtime, client, 2.5)
    assert client.holding and client.gripper_closed
    assert client.feedback(command).gripper_aperture == pytest.approx(world.state["claw"])


def test_finished_motions_report_where_they_ended():
    world, controller, runtime, client = setup()
    done = client.move_joints({"arm": 0.5}, duration=1)
    advance(runtime, client, 1.5)
    feedback = client.feedback(done)
    assert feedback.status == "succeeded" and feedback.reached_seconds == pytest.approx(1.0)
    assert feedback.joint_targets == {"arm": pytest.approx(0.5), "claw": 0.0}
    assert feedback.joint_positions == {"arm": pytest.approx(0.5), "claw": 0.0}
    assert not feedback.stalled and feedback.tool_pose is None  # No kinematics bound.

    cancelled = client.move_joints({"arm": 1.5}, duration=2)
    advance(runtime, client, 1.0)  # Halfway through the reference: 0.5 + 1.0 * 0.5 (smooth step).
    reference = controller.reference["arm"]
    client.cancel(cancelled)
    advance(runtime, client, 2.0)
    feedback = client.feedback(cancelled)
    assert feedback.status == "cancelled" and feedback.reached_seconds == pytest.approx(1.0)
    assert feedback.joint_targets["arm"] == pytest.approx(reference) == pytest.approx(1.0)
    assert feedback.joint_positions["arm"] == pytest.approx(world.state["arm"]) and not feedback.stalled


def test_a_blocked_arm_reports_a_stall():
    world, controller, runtime, client = setup(stuck=True)
    command = client.move_joints({"arm": 0.5}, duration=1)
    advance(runtime, client, 3.3)
    feedback = client.feedback(command)
    assert feedback.status == "failed" and feedback.stalled
    assert feedback.joint_positions["arm"] == 0.0 and 0.0 < feedback.joint_targets["arm"] < 0.5
    assert 0.0 < feedback.reached_seconds < 1.0


def test_cancel_retains_ownership_until_stopped():
    world, controller, runtime, client = setup()
    command = client.move_joints({"arm": 0.5}, duration=1)
    advance(runtime, client, 0.5)
    stopped_at = world.state["arm"]
    client.cancel(command)
    assert client.feedback(command).status == "stopping"
    with pytest.raises(RuntimeError, match="current"):
        client.open_gripper()
    advance(runtime, client, 0.3)
    assert client.feedback(command).status == "cancelled"
    assert world.state["arm"] == stopped_at


def test_reset_and_emergency_stop_invalidate_pending_commands():
    world, controller, runtime, client = setup()
    command = client.open_gripper()
    runtime.emergency_stop("test")
    assert client.feedback(command).status == "interrupted"
    with pytest.raises(RuntimeError, match="reset"):
        runtime.step()
    client._end_episode()
    runtime.reset()
    pending = client.open_gripper()
    client._end_episode()
    runtime.reset()
    assert client.feedback(pending).status == "cancelled"
    assert client.feedback(command).status == "interrupted"
    assert world.state == {"arm": 0, "claw": 0}


@pytest.mark.parametrize(
    "positions,duration",
    [
        ({}, 1),
        ({"bad": 0}, 1),
        ({"arm": 3}, 1),
        ({"arm": float("nan")}, 1),
        ({"arm": 1}, 0.1),
        ({"arm": 0.2}, 0.5),
        ({"arm": 0}, -1),
    ],
)
def test_invalid_command_does_not_acquire_resources(positions, duration):
    world, controller, runtime, client = setup()
    with pytest.raises(ValueError):
        client.move_joints(positions, duration=duration)
    assert runtime.controller_state("manipulation") == "idle"


def test_permissions_and_human_source_are_enforced():
    for options in ({"enabled": False}, {"source": RunMode.HUMAN}):
        world, controller, runtime, client = setup(**options)
        with pytest.raises(PermissionError):
            client.open_gripper()


def test_overlapping_controller_cannot_acquire_arm():
    world, controller, runtime, client = setup()
    other = ManipulationController(
        world.joint_names, world, controller.limits, MotionConfig(), source=RunMode.AUTONOMOUS
    )
    other.declaration = replace(other.declaration, name="whole_body")
    runtime = EpisodeRuntime(world, (controller, other), mode=RunMode.AUTONOMOUS)
    runtime.reset()
    runtime.acquire("whole_body", duration_ticks=100)
    with pytest.raises(RuntimeError, match="owned"):
        runtime.acquire("manipulation", duration_ticks=100)
    assert runtime.controller_state("manipulation") == "idle"


def test_gripper_commands_do_not_move_arm():
    world, controller, runtime, client = setup()
    opened = client.open_gripper()
    advance(runtime, client, 2.3)
    assert client.feedback(opened).status == "succeeded"
    assert world.state == {"arm": 0, "claw": -1}
    closed = client.close_gripper()
    advance(runtime, client, 2.3)
    assert client.feedback(closed).status == "succeeded"
    assert world.state == {"arm": 0, "claw": 0}


def test_cartesian_requires_explicit_servo_and_normalized_pose():
    world, controller, runtime, client = setup()
    with pytest.raises(ValueError, match="normalized"):
        client.move_to_pose((0, 0, 0, 0, 0, 0, 0))
    with pytest.raises(RuntimeError, match="servo"):
        client.move_to_pose((0, 0, 0, 0, 0, 0, 1))


def test_arm_gravity_feedforward_refreshes_each_physics_tick():
    from types import SimpleNamespace
    from unittest.mock import Mock

    import torch

    gravity = torch.arange(9, dtype=torch.float32).reshape(1, 9)
    robot = Mock(
        joint_names=["a", "b", "c"],
        num_base_dofs=6,  # Floating base coordinates precede joint coordinates.
        actuators={},
        data=SimpleNamespace(gravity_compensation_forces=SimpleNamespace(torch=gravity)),
    )
    scene = FakeScene(robot)  # No scene-body collection.
    world = isaaclab_world(scene, gravity_compensation_joints=("a", "c"))
    world.step()
    call = robot.set_joint_effort_target_index.call_args.kwargs
    assert call["joint_ids"] == [0, 2]
    assert call["target"].tolist() == [[6, 8]]
    gravity.add_(10)
    world.step()
    assert robot.set_joint_effort_target_index.call_args.kwargs["target"].tolist() == [[16, 18]]
    assert scene.writes == 2


def test_cartesian_reference_jump_fails_and_stops():
    class Servo:
        def start(self, pose):
            pass

        def targets(self, fraction):
            return {"arm": 1.0}

        def error(self):
            return (0.1, 0.0)

    world, controller, runtime, client = setup()
    controller.pose_servo = Servo()
    command = client.move_to_pose((0, 0, 0, 0, 0, 0, 1))
    advance(runtime, client, 0.3)
    result = client.feedback(command)
    assert result.status == "failed"
    assert "velocity limit" in result.reason
    assert world.state["arm"] == 0.0


@pytest.mark.parametrize("contact_velocity", [0.0, 0.8])
def test_gripper_closes_on_object_and_keeps_squeezing_through_arm_commands(contact_velocity):
    # Stillness comes from positions: a blocked claw may report 0.8 rad/s without moving.
    world, controller, runtime, client = setup(object_at=-0.4, contact_velocity=contact_velocity)
    opened = client.open_gripper()
    advance(runtime, client, 2.3)
    assert client.feedback(opened).status == "succeeded"
    closed = client.close_gripper()
    advance(runtime, client, 2.5)
    feedback = client.feedback(closed)
    assert (feedback.status, feedback.reason) == ("succeeded", "Gripper stopped on an object")
    assert world.state["claw"] == pytest.approx(-0.4)
    assert world.target["claw"] == pytest.approx(-0.35)  # Released, still squeezing 0.05 rad past contact.
    moved = client.move_joints({"arm": 0.5}, duration=1.0)
    advance(runtime, client, 0.5)
    assert world.target["claw"] == pytest.approx(-0.35)
    advance(runtime, client, 1.0)
    assert client.feedback(moved).status == "succeeded"
    assert world.state["arm"] == pytest.approx(0.5) and world.target["claw"] == pytest.approx(-0.35)
    client._end_episode()  # As RobotControl.reset does before restoring the episode.
    runtime.reset()
    assert controller.hold_targets() == {}


def test_gripper_chattering_on_a_fixed_handle_counts_as_stopped():
    """A jaw squeezing a handle fixed to the world vibrates in place; it must still stop and hold."""
    world, controller, runtime, client = setup(object_at=-0.4, chatter=0.001)  # 0.2 rad/s tick to tick.
    client.open_gripper()
    advance(runtime, client, 2.3)
    closed = client.close_gripper()
    advance(runtime, client, 2.5)
    feedback = client.feedback(closed)
    assert (feedback.status, feedback.reason) == ("succeeded", "Gripper stopped on an object")
    assert client.holding


def test_a_jaw_flipping_between_two_contact_positions_counts_as_stopped():
    """A two-tick contact limit cycle (5.6 mrad at 250 Hz, 2026-10-18) must not read as motion, whatever
    the parity of the motion window: here the window spans five ticks, so its ends fall on opposite phases."""
    world, controller, runtime, client = setup(object_at=-0.4, chatter=0.005)
    controller.config = replace(controller.config, motion_window_seconds=0.045)
    client.open_gripper()
    advance(runtime, client, 2.3)
    closed = client.close_gripper()
    advance(runtime, client, 2.5)
    feedback = client.feedback(closed)
    assert (feedback.status, feedback.reason) == ("succeeded", "Gripper stopped on an object")


def test_planned_pose_command_keeps_squeezing_a_held_object():
    from concurrent.futures import Future

    world, controller, runtime, client = setup(object_at=-0.4)
    client._pose_planner = lambda pose, held=None: Future()  # Planning stays pending; the arm holds.
    client.open_gripper()
    advance(runtime, client, 2.3)
    client.close_gripper()
    advance(runtime, client, 2.5)
    assert client.holding
    client.move_to_pose((0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0))
    advance(runtime, client, 1.0)
    assert world.target["claw"] == pytest.approx(-0.35)  # Still 0.05 rad past contact.
    assert client.holding


def test_a_yielding_object_does_not_ratchet_the_squeeze_past_the_first_contact():
    """A creeping claw pauses short of its squeeze target again and again; the squeeze is set once."""
    world, controller, runtime, client = setup(object_at=-0.4)
    world.creep = 0.0005  # 0.05 rad/s: slower than "stopped", so every pause looks like a new contact.
    client.open_gripper()
    advance(runtime, client, 2.3)
    closed = client.close_gripper()
    advance(runtime, client, 2.1)  # The closing duration (2 s) has passed: the first contact is detected.
    first = controller.gripper_setpoint
    assert first is not None and first < -0.25
    advance(runtime, client, 2.0)
    assert client.feedback(closed).reason == "Gripper stopped on an object"
    assert controller.gripper_setpoint == first  # Set once, 0.05 rad past the first contact.


def test_the_squeeze_stays_within_the_gripper_limits():
    """Contact just short of the end stop: the squeeze target is clamped, not a target the joint cannot reach."""
    world, controller, runtime, client = setup(object_at=-0.04)
    controller.limits["claw"] = (-2.0, 0.0)  # Closed is the end stop.
    client.open_gripper()
    advance(runtime, client, 2.3)
    closed = client.close_gripper()
    advance(runtime, client, 2.5)
    assert client.feedback(closed).status == "succeeded"
    assert controller.gripper_setpoint == pytest.approx(0.0) and client.holding


def test_squeeze_must_exceed_joint_tolerance_to_detect_holding():
    with pytest.raises(ValueError, match="grip_squeeze"):
        MotionConfig(grip_squeeze=0.01)


def test_plan_asks_the_planner_and_leaves_the_arm_where_it_is():
    """A plan-only call (the UI's previews): the planner gets the goal, nothing moves or is commanded."""
    from concurrent.futures import Future

    world, controller, runtime, client = setup()  # The fake world's one arm joint, "arm".
    asked, trajectory = [], JointTrajectory(("arm",), np.array([0.0, 1.0]), np.array([[0.0], [0.5]]), 1)
    client._joint_planner = lambda goal, held=None: asked.append((goal, held)) or trajectory
    plan = client.plan(goal={"arm": 0.5})
    assert isinstance(plan, Future) and plan.result() is trajectory and asked == [([0.5], None)]
    pending = Future()
    client._pose_planner = lambda pose, held=None: asked.append((pose, held)) or pending
    assert client.plan(pose_xyzw=(0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0), tool="wrist") is pending
    assert asked[-1][0] == pytest.approx((0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0))  # The wrist is the tool.
    before = dict(world.target)
    advance(runtime, client, 0.5)
    assert world.target == before and runtime.controller_state("manipulation") == "idle"
    with pytest.raises(ValueError, match="not both"):
        client.plan(goal={"arm": 0.5}, pose_xyzw=(0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="every arm joint"):
        client.plan(goal={})
    client._joint_planner = None
    with pytest.raises(RuntimeError, match="No collision planner"):
        client.plan(goal={"arm": 0.5})


def test_plan_puts_the_hand_where_move_to_pose_would():
    """plan(pose_xyzw, tool) and move_to_pose(pose_xyzw, tool) give the planner the same wrist pose."""
    from concurrent.futures import Future

    from fakes import SPOT

    world, controller, runtime, client = setup(tools=SPOT.gripper.tools)  # Spot's hand, ahead of the wrist.
    asked = []
    client._pose_planner = lambda pose, held=None: asked.append(pose) or Future()
    target = (0.6, 0.1, 0.4, 0.0, 0.7071068, 0.0, 0.7071068)
    client.plan(pose_xyzw=target, tool="hand")
    client.move_to_pose(target, tool="hand")
    assert asked[0] == pytest.approx(asked[1]) and asked[0] != pytest.approx(target)


def test_cancelling_an_arm_move_while_holding_keeps_the_grip():
    """The World's stop at the end cancels the running arm command; the held object must stay
    in the hand: the gripper setpoint survives the stop and the release."""
    world, controller, runtime, client = setup(object_at=-0.4)
    client.open_gripper()
    advance(runtime, client, 2.3)
    client.close_gripper()
    advance(runtime, client, 2.5)
    assert client.holding
    command = client.move_joints({"arm": 0.8}, duration=2.0)
    advance(runtime, client, 0.3)
    assert client.active_command == command
    client.cancel(command)
    advance(runtime, client, 2.5)
    assert client.active_command is None and runtime.controller_state("manipulation") == "idle"
    assert client.feedback(command).status == "cancelled"
    assert client.holding and client.gripper_closed  # The grip survived the stop.
    assert world.target["claw"] == pytest.approx(controller.gripper_setpoint)
