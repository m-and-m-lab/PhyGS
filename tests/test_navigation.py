# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation grids, the fine approach, command lifecycle, and independent physical task evaluation."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from phygs_robot_tests.tasks.point_navigation import PhysicalSample, PointNavigationEvaluator, PointNavigationTask

from phygs.execution.locomotion.base import DeadBand
from phygs.execution.navigation.map import GridMap, RectangleFootprint
from phygs.simulation.maps import BoxGeometry, KnownBoxMap
from phygs.skills.navigation.api import NavigationConfig, NavigationGoal


def test_segments_are_blocked_by_diagonal_corners_and_subcell_intersections():
    world = GridMap(np.array([[False, True], [True, False]]), (0, 0), 1, 0, "test")
    assert not world.segment_free((0.5, 0.5), (1.5, 1.5))
    assert not world.segment_free((-0.01, 0.5), (0.5, 0.5))
    assert world.segment_free((0.2, 0.5), (0.8, 0.5))


def test_box_map_inflates_footprint_and_updates_from_physics():
    pose = torch.tensor([[2.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]])
    scene = SimpleNamespace(
        rigid_objects={"box": SimpleNamespace(data=SimpleNamespace(root_pose_w=SimpleNamespace(torch=pose)))}
    )

    class Scene(dict):
        rigid_objects = scene.rigid_objects

    source = KnownBoxMap(
        Scene(scene.rigid_objects),
        (BoxGeometry((0.4, 0.4, 1.0), (2.0, 0.0, 0.5), "box"),),
        bounds=(-2.0, -3.0, 5.0, 3.0),
        resolution=0.1,
        footprint=RectangleFootprint(-0.5, 0.5, -0.5, 0.5),
        clearance=0.1,
    )
    first = source.snapshot()
    assert not first.free((1.5, 0))
    assert first.free((0, 0))
    assert source.snapshot() is first
    pose[0, 2] = 5
    second = source.snapshot()
    assert second.version == first.version + 1
    assert second.free((2, 0))
    assert not first.free((2, 0))
    with pytest.raises(ValueError):
        first.occupied[0, 0] = False


def advance(runtime, motion, client, count=1):
    for _ in range(count):
        client.update()
        x, y, yaw = motion.command
        runtime.velocity = torch.tensor([x, y, 0.0])
        c, s = math.cos(runtime.yaw), math.sin(runtime.yaw)
        runtime.x += (c * x - s * y) * runtime.physics_dt
        runtime.y += (s * x + c * y) * runtime.physics_dt
        runtime.yaw += yaw * runtime.physics_dt
        runtime.tick += 1


def test_fine_approach_turns_first_then_moves_along_the_larger_body_axis():
    from phygs.execution.navigation.fine_approach import FineApproach

    band = DeadBand((0.04, 0.04, 0.05), (0.05, 0.05, 0.10), (0.07, 0.05, 0.10))
    # The test's own measured moves at the lowest speed (not the calibration file's): 0.7 s and 1 s.
    moves = {direction: (0.02, 0.06) for direction in ("forward", "backward", "left")}
    coverage = FineApproach.Coverage((0.7, 1.0), {**moves, "right": (0.015, 0.057)})
    fine = FineApproach(NavigationConfig(), 0.004, band, coverage)
    goal = NavigationGoal(1.0, 0.0, heading=math.pi / 2)
    turn = fine.move(goal, (0.97, 0.02), 1.0)
    assert turn.command[:2] == (0.0, 0.0) and turn.command[2] == pytest.approx(0.5, abs=1e-9)
    # Facing +y: the remaining -0.06 m in y is backwards, 0.03 m in x is to the right.
    back = fine.move(goal, (0.97, 0.06), math.pi / 2)
    assert back.command[1:] == (0.0, 0.0) and back.command[0] == pytest.approx(-0.06)
    assert back.duration == pytest.approx(1.0)
    small = fine.move(NavigationGoal(1.0, 0.0, position_tolerance=0.01), (1.0, 0.02), 0.0)
    assert small.command == (0.0, -0.05, 0.0)  # Raised to the lowest walking speed.
    # Held as long as a measured move to the right needs for 2 cm: between 0.7 s (1.5 cm) and 1 s (5.7 cm).
    assert small.duration == pytest.approx(coverage.duration("right", 0.02), abs=0.004)
    assert 0.7 < small.duration < 1.0
    assert fine.move(goal, (0.98, 0.0), math.pi / 2) is None
    # A heading error below fine_turn_first (0.3 rad) waits until the position is within tolerance.
    ahead = fine.move(goal, (1.0, -0.08), math.pi / 2 - 0.1)
    assert ahead.command[1:] == (0.0, 0.0) and ahead.command[0] > 0
    last = fine.move(goal, (1.0, -0.01), math.pi / 2 - 0.1)
    assert last.command[:2] == (0.0, 0.0) and last.command[2] > 0


def sample(time, x=0.0, y=0.0, speed=0.0, force=0.0, z=0.5):
    return PhysicalSample(time, x, y, z, 0.0, speed, 0.0, 1.0, force)


def test_evaluator_requires_physical_arrival_and_stopped_hold():
    task = PointNavigationTask("scene", "initial", (0, 0, 0.5), NavigationGoal(1, 0))
    evaluator = PointNavigationEvaluator(task)
    assert evaluator.update(sample(0)) == "running"
    assert evaluator.update(sample(1, x=1, speed=0.4)) == "running"
    assert evaluator.update(sample(2, x=1)) == "running"
    assert evaluator.update(sample(2.3, x=1)) == "succeeded"
    assert evaluator.result()["travel_distance_m"] == 1


@pytest.mark.parametrize(
    "measurement, expected",
    [(sample(1, force=2), "collision"), (sample(1, z=0.1), "fallen"), (sample(46), "timed_out")],
)
def test_evaluator_failure_conditions(measurement, expected):
    task = PointNavigationTask("scene", "initial", (0, 0, 0.5), NavigationGoal(1, 0), minimum_height=0.35)
    evaluator = PointNavigationEvaluator(task)
    evaluator.update(sample(0))
    assert evaluator.update(measurement) == expected


def test_a_robot_that_cannot_fall_is_judged_without_a_height():
    # A base on planar joints (fallen_height None): its base link sits at z = 0 and is not fallen.
    evaluator = PointNavigationEvaluator(PointNavigationTask("scene", "initial", (0, 0, 0.0), NavigationGoal(1, 0)))
    assert evaluator.update(sample(0, z=0.0)) == "running"
    assert evaluator.update(sample(1, x=1, z=0.0)) == "running"
    assert evaluator.update(sample(1.3, x=1, z=0.0)) == "succeeded"


def test_task_action_permissions_reject_commands_before_backend_access():
    from phygs.robots.control import ArmClient
    from phygs.skills.locomotion.api import LocomotionClient

    controller = SimpleNamespace(declaration=SimpleNamespace(name="test"))
    motion = LocomotionClient(None, controller, enabled=False)
    with pytest.raises(PermissionError, match="Direct motion"):
        motion.command_velocity(0.1, 0, 0)
    arm = ArmClient(None, controller, None, {}, enabled=False)
    with pytest.raises(PermissionError, match="Arm commands"):
        arm.command_joint_positions({"arm_sh0": 0.1})


def test_the_navigation_course_refuses_its_goals_for_the_stated_reason():
    """The navigation rung's course (tests/robot_tests/scenes/nav_course): the invalid goal lies inside the wall and
    the unreachable goal inside the closed pen, clear of its walls (a free cell no route reaches)."""
    from pathlib import Path

    import yaml
    from pxr import Usd

    from phygs.episodes.scenes import scene_file
    from phygs.simulation.scene import collider_boxes

    project = Path(__file__).resolve().parents[1]
    cases = yaml.safe_load((project / "tests/robot_tests/configs/navigation.yaml").read_text())["cases"]
    goals = {case["name"]: case["goal"] for case in cases["nav_course"]}
    stage = Usd.Stage.Open(str(scene_file("nav_course")))
    boxes = {}
    for _, body, center, size in collider_boxes(stage):
        offset = np.array(stage.GetPrimAtPath(body).GetAttribute("xformOp:translate").Get())
        boxes[body.rsplit("/", 1)[-1]] = (center + offset, size)

    def inside(point, box, margin=0.0):
        center, size = box
        return all(abs(p - c) < s / 2 - margin for p, c, s in zip(point, center[:2], size[:2]))

    assert inside(goals["invalid"][:2], boxes["wall"])
    pen = [box for name, box in boxes.items() if name.startswith("pen_")]
    low = np.min([center[:2] - size[:2] / 2 for center, size in pen], axis=0)
    high = np.max([center[:2] + size[:2] / 2 for center, size in pen], axis=0)
    goal = np.asarray(goals["unreachable"][:2])
    assert np.all(goal - low > 0.6) and np.all(high - goal > 0.6)  # Enclosed, half a robot from every wall.
    assert not any(inside(goal, box) for box in boxes.values())


SPOT = RectangleFootprint(-0.6, 0.6, -0.26, 0.26, padding=0.05)


def corridor_grid(width_cells: int, *, room: bool = True) -> GridMap:
    """An L: a corridor east along y = 1.55 m, then north along x = 6.55 m, ``width_cells`` x 0.1 m
    wide, with a 2.1 m square room at the bend (or none)."""
    obstacles = np.ones((60, 90), dtype=bool)
    low = 15 - width_cells // 2
    obstacles[low : low + width_cells, 5:72] = False
    obstacles[low:55, 65 - width_cells // 2 : 65 - width_cells // 2 + width_cells] = False
    if room:
        obstacles[5:26, 55:76] = False
    return GridMap.with_footprint(obstacles, SPOT, (0.0, 0.0), 0.1, 0, "test")


def test_footprint_grid_fits_a_corridor_lengthwise_but_not_across():
    grid = corridor_grid(11)  # 1.1 m: narrower than the footprint's 1.31 m length.
    assert grid.pose_free((3.0, 1.55), 0.0) and not grid.pose_free((3.0, 1.55), math.pi / 2)
    assert grid.free((3.0, 1.55))  # Some heading fits.
    assert not grid.turn_free((3.0, 1.55), 0.0, math.pi / 2)
    assert grid.turn_free((6.55, 1.55), 0.0, math.pi / 2)  # In the room.


def test_base_standing_within_the_padding_may_leave():
    obstacles = np.zeros((40, 60), dtype=bool)
    obstacles[25:, :] = True  # A wall from y = 2.5 m.
    grid = GridMap.with_footprint(obstacles, SPOT, (0.0, 0.0), 0.1, 0, "test")
    start = (2.0, 2.5 - 0.26 - 0.08)  # The side 8 cm from the wall: inside the 5 cm padding plus a cell.
    assert grid.pose_free(start, 0.0, padded=False) and not grid.pose_free(start, 0.0)
    # Leaving: sideways away from the wall, without the padding near the start.
    assert grid.segment_free(start, (2.0, 1.5), 0.0, bare_start=0.1 * math.sqrt(2))
    assert not grid.segment_free(start, (2.0, 1.5), 0.0)


def test_fine_moves_are_checked_at_the_base_heading():
    from phygs.execution.navigation.fine_approach import FineApproach, FineMove

    grid = corridor_grid(11)
    sideways = FineMove((0.0, 0.08, 0.0), 0.5, "sideways")
    forward = FineMove((0.08, 0.0, 0.0), 0.5, "forward")
    turn = FineMove((0.0, 0.0, 0.5), 3.0, "turn")
    assert FineApproach.move_free(grid, forward, (3.0, 1.55), 0.0)
    assert FineApproach.move_free(grid, sideways, (3.0, 1.55), 0.0)  # 4 cm across a 1.1 m corridor.
    assert not FineApproach.move_free(grid, turn, (3.0, 1.55), 0.0)


def test_move_coverage_interpolates_measured_short_moves():
    from phygs.execution.navigation.fine_approach import FineApproach

    coverage = FineApproach.Coverage(
        (0.3, 0.5, 1.0), {"forward": (0.03, 0.05, 0.1), "backward": (0.02, 0.03, 0.07), "left": (0.04, 0.06, 0.1),
                          "right": (0.005, 0.004, 0.06)}
    )  # fmt: skip
    assert coverage.duration("forward", 0.04) == pytest.approx(0.4)
    assert coverage.duration("right", 0.005) == pytest.approx(0.3)  # The 0.5 s move covered no more.
    assert coverage.duration("right", 0.03) == pytest.approx(0.3 + 0.7 * 0.025 / 0.055)
    assert coverage.duration("left", 0.2) is None  # Beyond the longest measured move.
    with pytest.raises(ValueError, match="one distance per duration"):
        FineApproach.Coverage((0.3,), {d: (0.1, 0.2) for d in ("forward", "backward", "left", "right")})
