# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The arm planner host: what each plan request carries, and when it is sent."""

from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fakes import SPOT
from scipy.spatial.transform import Rotation

from phygs.execution.manipulation import arm_planner

SETTINGS = dict(
    backend="curobo",
    curobo_dependencies="",
    planning_time_seconds=30.0,
    obstacle_padding=0.01,
    held_sphere_limit=256,
    planner_resolution=0.02,
    pixel_stride=4,
    self_filter_padding=0.02,
)
JOINTS = SPOT.arm.joints
# Spot with a 32x24 px hand camera that plans, and a work volume 2 m ahead [m].
ROBOT = replace(
    SPOT,
    cameras={"hand": replace(SPOT.cameras["hand"], width=32, height=24)},
    camera_roles={"planning": ("hand",), "look": ("hand",)},
    planning=replace(SPOT.planning, workspace_lower=(0.0, -1.0, -1.0), workspace_upper=(2.0, 1.0, 1.0)),
)
K = np.array([[16.0, 0.0, 16.0], [0.0, 16.0, 12.0], [0.0, 0.0, 1.0]])
NAMES = (*JOINTS, "arm_f1x", "front_left_knee")


class Robot:
    """Observation packets of a robot at x [m] with a hand camera looking along body +x."""

    def __init__(self):
        self.x, self.knee, self.stamp, self.gripper = 0.0, 0.0, 0.0, -0.05
        self.depth = np.full((24, 32), 1.0)

    def observe(self):
        pose = torch.eye(4, dtype=torch.float64)
        pose[0, 3] = self.x
        # Optical frame: x right = body -y, y down = body -z, z forward = body +x.
        body_from_camera = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0, 0, 0, 1.0]]
        )
        camera = SimpleNamespace(
            stamp=SimpleNamespace(time_seconds=self.stamp),
            depth=torch.tensor(self.depth, dtype=torch.float32),
            depth_valid=torch.ones(self.depth.shape, dtype=torch.bool),
            intrinsics=torch.tensor(K),
            extrinsics=SimpleNamespace(matrix=body_from_camera),
        )
        positions = torch.tensor([0.0, -3.1, 3.1, 0.0, 0.0, 0.0, self.gripper, self.knee])
        return SimpleNamespace(
            cameras={"hand": camera},
            localization=SimpleNamespace(valid=True, pose=SimpleNamespace(matrix=pose)),
            joints=SimpleNamespace(names=NAMES, positions=positions),
        )


class Worker:
    startup_seconds, busy = 9.5, False
    started: list = []

    def __init__(self, command, model, folder, env):
        self.command, self.model, self.requests, self.closed = command, dict(model), [], False
        self.result = dict(success=True, joint_names=list(JOINTS), times=[0.0, 1.0], positions=[[0.0] * 6, [0.1] * 6])
        Worker.started.append(self)

    def submit(self, request):
        self.requests.append(request)
        future = Future()
        if isinstance(self.result, BaseException):  # The process broke: no response.
            future.set_exception(self.result)
        else:
            future.set_result(self.result)
        return future

    def close(self):
        self.closed = True


@pytest.fixture
def host(tmp_path, monkeypatch):
    Worker.started = []
    monkeypatch.setattr(arm_planner, "PlannerProcess", Worker)
    monkeypatch.setattr(
        arm_planner,
        "prepare_model",
        lambda *_, locked_joints, **__: dict(
            locked_joints={k: v for k, v in locked_joints.items() if k not in JOINTS}, gripper_ratios={"arm_f1x": 1.0}
        ),
    )
    robot, clock = Robot(), SimpleNamespace(t=0.0)

    def make(**sources):
        planner = arm_planner.ArmPlanner(
            ROBOT, project=tmp_path, output=tmp_path, settings=SETTINGS, geometry={}, usd_boxes={},
            observe=robot.observe, episode_id=lambda: 3, time_seconds=lambda: clock.t, **sources,
        )  # fmt: skip
        planner.start()
        return planner

    return make, robot, clock


def test_one_worker_serves_the_episode_and_each_request_carries_the_measured_legs(host):
    make, robot, _ = host
    planner = make(boxes=lambda exclude: {})
    worker = Worker.started[0]
    assert worker.model["locked_joints"] == {"arm_f1x": pytest.approx(-0.05), "front_left_knee": 0.0}
    assert worker.model["depth_map"]["center"] == [1.0, 0.0, 0.0]
    robot.x, robot.knee = 2.0, -1.5  # The base walked 2 m and the legs stand differently.
    trajectory = planner.request(world_pose=(3.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0))
    planner.poll()
    request = worker.requests[0]
    assert len(Worker.started) == 1 and request.locked_joints == {"front_left_knee": -1.5}
    assert request.goal_pose[:3] == pytest.approx([1.0, 0.0, 0.5])  # In the body frame.
    assert request.gripper == pytest.approx(-0.05) and request.images is None
    assert trajectory.result().episode_id == 3 and planner.startup_seconds == 9.5
    assert planner.status() == dict(
        running=True,
        busy=False,
        restarts=0,
        last_plan=dict(kind="pose", seconds=pytest.approx(0, abs=1), result="planned"),
    )
    planner.close()
    assert worker.closed


def test_known_boxes_leave_out_the_held_object(host):
    make, _, _ = host
    asked = []
    planner = make(boxes=lambda exclude: asked.append(exclude) or {"table": (np.array([1.0, 0, 0.2]), np.ones(3))})
    planner.request(
        goal=[0.0] * 6, held=SimpleNamespace(name="cup", size=(0.04, 0.04, 0.1), box=dict, spheres=_spheres)
    )
    planner.poll()
    planner.request(goal=[0.0] * 6)
    planner.poll()
    assert asked == [frozenset({"cup"}), frozenset()]
    assert Worker.started[0].requests[0].boxes[0]["size"] == [1.0, 1.0, 1.0]


def _spheres(wrist, cell):
    return np.array([[0.0, 0.0, 0.1, cell]])


def test_depth_waits_for_an_image_taken_after_the_request_and_keeps_only_the_work_volume(host):
    make, robot, clock = host
    planner = make(cameras=("hand",))
    worker = Worker.started[0]
    robot.depth[:, :4] = 3.0  # Beyond the 2 m work volume.
    clock.t = 1.0
    trajectory = planner.request(goal=[0.0] * 6)
    planner.poll()
    assert worker.requests == [] and not trajectory.done()  # The latest image is older than the request.
    robot.stamp = 1.0
    planner.poll()
    (image,) = worker.requests[0].images
    depth = np.load(image.path)
    assert (depth[:, :4] == 0).all() and depth[12, 16] == pytest.approx(1.0)  # The robot's pixels stay.
    camera = Rotation.from_quat(image.pose[3:]).as_matrix()
    np.testing.assert_allclose(camera, robot.observe().cameras["hand"].extrinsics.matrix[:3, :3], atol=1e-9)
    assert worker.requests[0].boxes == []
    worker.result = dict(success=False, reason="start_collision")
    failed = planner.request(goal=[0.0] * 6)
    planner.poll()
    with pytest.raises(RuntimeError, match="Joint planner failed: start_collision"):
        failed.result(timeout=0)


def test_the_planner_needs_an_obstacle_source(host):
    make, _, _ = host
    with pytest.raises(ValueError, match="depth cameras, known boxes"):
        make()


def test_moveit_workers_get_distinct_private_ros_domains():
    """MoveIt's self filter reads the robot's frames from /tf; two workers must never share a domain."""
    first, second = arm_planner.private_ros_domain(), arm_planner.private_ros_domain()
    assert first[0] != second[0] and min(first[0], second[0]) >= 215
    first[1].close()
    third = arm_planner.private_ros_domain()  # The freed domain is taken again.
    assert third[0] == first[0]
    second[1].close()
    third[1].close()


def test_a_plan_from_the_stow_starts_at_the_prestow_and_its_trajectory_unfolds_straight_there(host):
    """Planners reject the folded stow as a start (sphere models call it a self-collision): the
    request starts at the pre-stow posture and the trajectory first moves straight there."""
    make, robot, _ = host
    stow = dict(
        stow_posture=[0.0, -3.13, 3.13, 0.0, 0.0, 0.0],
        prestow_posture=[0.0, -2.83, 2.83, 0.0, 0.0, 0.0],
        move_seconds=2.0,
        tolerance=0.2,
    )
    planner = make(boxes=lambda exclude: {}, stow=stow)
    worker = Worker.started[0]
    worker.result = dict(
        success=True, joint_names=list(JOINTS), times=[0.0, 1.0], positions=[stow["prestow_posture"], [0.1] * 6]
    )
    trajectory = planner.request(goal=[0.1] * 6)
    planner.poll()
    assert worker.requests[0].start == stow["prestow_posture"]
    trajectory = trajectory.result()
    assert trajectory.positions[0].tolist() == pytest.approx([0.0, -3.1, 3.1, 0.0, 0.0, 0.0])  # Measured.
    end = int(np.flatnonzero(np.isclose(trajectory.times, 2.0))[0])
    assert trajectory.positions[end].tolist() == pytest.approx(stow["prestow_posture"])
    assert trajectory.times[-1] == pytest.approx(3.0) and trajectory.positions[-1].tolist() == pytest.approx([0.1] * 6)
    robot_unfolded = make(boxes=lambda exclude: {}, stow={**stow, "tolerance": 0.01})  # Not stowed now.
    robot_unfolded.request(goal=[0.1] * 6)
    robot_unfolded.poll()
    assert Worker.started[-1].requests[0].start == pytest.approx([0.0, -3.1, 3.1, 0.0, 0.0, 0.0])


def test_warm_up_plans_once_per_gripper_angle_before_the_episode(host):
    """cuRobo builds a motion planner per gripper angle on its first plan with it (~8 s): warm_up
    plans with the measured angle and the given ones before the clock runs."""
    make, _, _ = host
    planner = make(boxes=lambda exclude: {})
    assert planner.warm_up((-1.4,)) >= 0
    requests = Worker.started[0].requests
    assert [request.gripper for request in requests] == pytest.approx([-0.05, -1.4])
    assert all(request.boxes == [] and request.images is None for request in requests)


def test_a_new_gripper_angle_is_warmed_inside_the_tick_before_its_plan(host):
    """The jaws on a held object give an angle with no planner yet: poll warms it (synchronously, so
    the build costs no simulated time) and then sends the plan; an angle within 0.01 rad of a warmed
    one is not warmed again. A build that breaks the process is the World's failure."""
    make, robot, _ = host
    planner = make(boxes=lambda exclude: {})
    worker = Worker.started[0]
    robot.gripper = -0.443
    planner.request(goal=[0.1] * 6)
    planner.poll()
    warm, plan = worker.requests
    assert warm.gripper == plan.gripper == pytest.approx(-0.443)
    assert warm.boxes == [] and warm.images is None and plan.goal == [0.1] * 6
    robot.gripper = -0.441  # The same planner (to 0.01 rad).
    planner.request(goal=[0.2] * 6)
    planner.poll()
    assert len(worker.requests) == 3 and planner.failure is None
    robot.gripper = -0.9
    worker.result = TimeoutError("Planner response exceeded 180 wall seconds")
    broken = planner.request(goal=[0.3] * 6)
    planner.poll()
    with pytest.raises(TimeoutError):
        broken.result()
    assert planner.failure.startswith("TimeoutError") and len(worker.requests) == 4  # The plan was not sent.


def test_each_episode_writes_its_plan_records_to_its_own_folder(host, tmp_path):
    make, _, _ = host
    planner = make(boxes=lambda exclude: {})
    planner.request(goal=[0.1] * 6)
    planner.poll()
    second = tmp_path / "repeat_1"
    planner.set_output(second)
    planner.request(goal=[0.1] * 6)
    planner.poll()
    for folder in (tmp_path, second):
        assert len((folder / "planner" / "requests.jsonl").read_text().splitlines()) == 1
    assert (second / "planner" / "depth").is_dir() and len(Worker.started) == 1


def test_a_failed_plan_is_the_callers_but_a_broken_planner_process_is_recorded_for_the_world(host):
    """A plan the library could not find fails the request only; a process that raised (out of GPU
    memory, a crash, a timeout) is also kept in ``failure``, so the World ends the episode as an error."""
    make, _, _ = host
    planner = make(boxes=lambda exclude: {})
    worker = Worker.started[0]
    worker.result = dict(success=False, reason="no collision-free path")
    refused = planner.request(world_pose=(1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0))
    planner.poll()
    with pytest.raises(RuntimeError, match="no collision-free path"):
        refused.result()
    assert planner.failure is None
    worker.result = RuntimeError("OutOfMemoryError: CUDA out of memory")
    broken = planner.request(world_pose=(1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0))
    planner.poll()
    with pytest.raises(RuntimeError, match="out of memory"):
        broken.result()
    assert planner.failure == "RuntimeError: OutOfMemoryError: CUDA out of memory"
    planner.close()
