# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The retrieve evaluator scores a settled end state from ground truth: outcome and conduct apart."""

from __future__ import annotations

import numpy as np
import pytest
from phygs_robot_tests.tasks import load_task
from phygs_robot_tests.tasks.retrieve import THRESHOLDS, RetrieveTask, load_thresholds

from phygs.world.task import EpisodeEnd, NoTask, Task


class FakeTruth:
    """Ground truth of one end state: the target's height, whether it is held, the arm, the base."""

    def __init__(self, *, held=True, height=0.9, stow_error=0.01, up_cos=1.0, openings=None):
        self._held, self._height, self._stow, self._up = held, height, stow_error, up_cos
        self._openings = {} if openings is None else openings
        self.asked = None

    def object_pose(self, name):
        pose = np.eye(4)
        pose[2, 3] = self._height
        return pose

    def held(self, name, *, lift_m, reach_m):
        self.asked = (name, lift_m, reach_m)
        return self._held

    def stow_error_rad(self):
        return self._stow

    def base_pose(self):
        pose = np.eye(4)
        pose[2, 2] = self._up
        return pose

    def part_openings(self):
        return dict(self._openings)


def evaluate(truth, ended_by="finish", contacts=(), at_start=None):
    task = RetrieveTask("the box", "box")
    task.reset(FakeTruth(openings=at_start or {}))  # The settled world before the agent starts.
    task.update(truth)  # Nothing to sample for this evaluator.
    return task.result(truth, EpisodeEnd(ended_by, 30.0, tuple(contacts)))


def test_the_thresholds_file_is_complete_and_a_missing_key_is_an_error(tmp_path):
    thresholds = load_thresholds()
    assert set(thresholds) >= {"held_lift_m", "held_reach_m", "stow_tolerance_rad", "upright_min_cos"}
    assert THRESHOLDS.exists()
    partial = tmp_path / "retrieve.yaml"
    partial.write_text("held_lift_m: 0.05\n")
    with pytest.raises(ValueError, match="lacks"):
        load_thresholds(partial)


def test_success_needs_held_stowed_upright_and_in_time():
    truth = FakeTruth()
    result = evaluate(truth)
    assert result["outcome"] == "success" and result["clean"] is True
    assert truth.asked == ("box", 0.05, 0.35)  # The evaluator's thresholds reach the ground truth.
    assert result["measured"]["target_height_m"] == 0.9
    for truth, ended_by, missing in (
        (FakeTruth(held=False), "finish", "held"),
        (FakeTruth(stow_error=0.2), "finish", "stowed"),
        (FakeTruth(up_cos=0.5), "finish", "upright"),
        (FakeTruth(), "budget", "in_time"),
    ):
        result = evaluate(truth, ended_by)
        assert result["outcome"] == "failure" and result["checks"][missing] is False
        assert missing in result["reason"]


def test_conduct_is_reported_beside_the_outcome_never_in_it():
    contacts = [
        dict(link="arm_link_fngr", body="table", peak_n=12.0),
        dict(link="arm_link_wr1", body="shelf", peak_n=0.2),
    ]
    result = evaluate(FakeTruth(), contacts=contacts)
    assert result["outcome"] == "success" and result["clean"] is False
    assert result["conduct"]["contacts"] == 1 and result["conduct"]["peak_contact_n"] == 12.0  # 0.2 N is under the bar.


def test_a_drawer_the_agent_left_open_counts_but_one_open_at_the_start_does_not():
    truth = FakeTruth(openings={"top": 0.25, "bottom": 0.30, "side": 0.0})
    result = evaluate(truth, at_start={"bottom": 0.30})
    assert result["conduct"]["drawers_left_open"] == {"top": 0.25} and result["clean"] is False


def test_a_goal_of_none_is_not_evaluated_and_retrieve_needs_a_target():
    none = load_task({"goal": "none"})
    none.reset(FakeTruth())
    assert (
        isinstance(none, NoTask)
        and none.result(FakeTruth(), EpisodeEnd("finish", 1.0, ()))["outcome"] == "not_evaluated"
    )
    with pytest.raises(ValueError, match="target"):
        RetrieveTask("the box", "")


def test_the_task_object_is_what_the_environment_needs_and_names_its_rules():
    task = load_task(dict(goal="retrieve", description="the box", target="box"))
    assert isinstance(task, Task) and isinstance(NoTask(), Task)  # The seam: episodes/task.py.
    assert task.brief() == "Retrieve the box: take it into the gripper and stow the arm." and task.target == "box"
    other = RetrieveTask("x", "y", dict(task.thresholds, contact_n=9.0))
    assert task.provenance()["evaluator_thresholds"] != other.provenance()["evaluator_thresholds"]
