# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grounded object-search evidence without Isaac Sim or a detector model."""

from dataclasses import replace

import pytest
import torch
from phygs_robot_tests.tasks.object_search import (
    ObjectSearchEvaluator,
    ObjectSearchTask,
    TargetBounds,
    TargetDeclaration,
)
from phygs_robot_tests.tasks.object_search_api import ObjectSearchClient

from phygs.observations.types import (
    CameraObservation,
    ObservationPacket,
    Stamp,
    Transform,
)


def packet(*, episode: int = 1, tick: int = 0, depth_m: float = 2.0) -> ObservationPacket:
    stamp = Stamp(episode, tick, tick * 0.032)
    identity = torch.eye(4)
    optical = Transform("body", "inspection_optical", stamp, identity, "calibration")
    world = Transform("world", "body", stamp, identity, "ground_truth_localization")
    camera = CameraObservation(
        stamp,
        "inspection_optical",
        torch.full((5, 5, 3), 150, dtype=torch.uint8),
        torch.full((5, 5), depth_m),
        torch.ones((5, 5), dtype=torch.bool),
        torch.tensor([[10.0, 0.0, 2.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]]),
        optical,
        True,
    )
    return ObservationPacket(stamp, None, None, None, {"inspection": camera}, (optical, world))


def task() -> ObjectSearchTask:
    return ObjectSearchTask("boxes", "seed-7", "green cuboid", "inspection")


def bounds() -> TargetBounds:
    return TargetBounds((-0.3, -0.3, 1.9), (0.3, 0.3, 2.1))


def test_agent_declaration_is_grounded_in_exact_captured_depth():
    evaluator = ObjectSearchEvaluator(task(), episode_id=1)
    first = packet(depth_m=1.0)
    assert evaluator.record(first, bounds())
    assert not evaluator.record(first, bounds())
    second = packet(tick=1)
    assert evaluator.record(second, bounds())
    client = ObjectSearchClient(evaluator.submit, camera="inspection")
    assert client.declare_target(second.stamp, 2, 2) == "succeeded"
    assert evaluator.result()["declaration"]["stamp"]["tick"] == 1
    assert evaluator.result()["frames_recorded"] == 2
    with pytest.raises(RuntimeError, match="terminal"):
        client.declare_target(first.stamp, 2, 2)


def test_occluded_target_and_wrong_pixel_are_false_positives():
    occluded = ObjectSearchEvaluator(task(), episode_id=1)
    near = packet(depth_m=1.0)
    occluded.record(near, bounds())
    assert occluded.submit(TargetDeclaration("inspection", near.stamp, 2, 2)) == "false_positive"

    wrong = ObjectSearchEvaluator(task(), episode_id=1)
    visible = packet()
    wrong.record(visible, bounds())
    assert wrong.submit(TargetDeclaration("inspection", visible.stamp, 0, 0)) == "false_positive"


def test_stale_cross_episode_and_invalid_evidence_fail_explicitly():
    evaluator = ObjectSearchEvaluator(task(), episode_id=1, history_frames=1)
    first = packet()
    evaluator.record(first, bounds())
    evaluator.record(packet(tick=1), bounds())
    with pytest.raises(ValueError, match="expired"):
        evaluator.submit(TargetDeclaration("inspection", first.stamp, 2, 2))
    with pytest.raises(ValueError, match="episode"):
        evaluator.submit(TargetDeclaration("inspection", Stamp(2, 1, 0.032), 2, 2))
    with pytest.raises(ValueError, match="outside"):
        evaluator.submit(TargetDeclaration("inspection", Stamp(1, 1, 0.032), 8, 2))
    with pytest.raises(ValueError, match="another episode"):
        evaluator.record(packet(episode=2), bounds())


def test_invalid_camera_and_time_budget():
    evaluator = ObjectSearchEvaluator(task(), episode_id=1)
    image = packet()
    without_depth = replace(image.cameras["inspection"], depth=None)
    with pytest.raises(ValueError, match="RGB-D"):
        evaluator.record(replace(image, cameras={"inspection": without_depth}), bounds())
    evaluator.record(image, bounds())
    assert evaluator.update(45.0) == "timed_out"
    with pytest.raises(RuntimeError, match="terminal"):
        evaluator.submit(TargetDeclaration("inspection", image.stamp, 2, 2))


def test_task_validation_rejects_invalid_permissions_and_geometry():
    with pytest.raises(ValueError, match="action"):
        ObjectSearchTask("boxes", "seed", "green cuboid", "inspection", allowed_actions=("arm",))
    with pytest.raises(ValueError, match="positive size"):
        TargetBounds((0, 0, 2), (1, 1, 2))
