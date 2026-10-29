# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The check before the agent starts (``phygs.world.start``) and the report's status, on a fake
ground truth: no simulator."""

from __future__ import annotations

import math

import numpy as np
import pytest
import yaml

from phygs.world.build import SETTINGS
from phygs.world.report import STATUSES, status
from phygs.world.start import LIMITS, check

LIMITS_NOW = yaml.safe_load(SETTINGS.read_text())["start_check"]


class FakeTruth:
    """A settled world: the base at ``height`` [m], tilted by ``tilt_deg``; bodies (speed, shift); drawers."""

    def __init__(self, height=0.5, tilt_deg=0.0, motion=None, openings=None) -> None:
        tilt = math.radians(tilt_deg)
        self.pose = np.eye(4)
        self.pose[:3, :3] = [[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)], [0, math.sin(tilt), math.cos(tilt)]]
        self.pose[2, 3] = height
        self.motion = {"table": (0.0, 0.0), "box": (0.001, 0.004)} if motion is None else motion
        self.openings = {"drawer_0": 0.0} if openings is None else openings

    def base_pose(self):
        return self.pose

    def body_motion(self):
        return self.motion

    def part_openings(self):
        return self.openings


def test_the_settings_hold_every_limit():
    assert set(LIMITS) == set(LIMITS_NOW)


def test_a_settled_world_passes_and_reports_what_it_measured():
    result = check(FakeTruth(), fallen_height=0.25, limits=LIMITS_NOW)
    assert result["passed"] and result["problems"] == []
    measured = result["measured"]
    assert measured["fastest_body"] == "box" and measured["farthest_shift_m"] == pytest.approx(0.004)
    assert measured["bodies"] == 2 and measured["widest_drawer"] == "drawer_0"


@pytest.mark.parametrize(
    ("truth", "words"),
    [
        (FakeTruth(tilt_deg=30.0), "tilted"),
        (FakeTruth(height=0.1), "down"),
        (FakeTruth(motion={"box": (0.4, 0.0)}), "still move"),
        (FakeTruth(motion={"box": (0.0, 0.2)}), "left their places"),
        (FakeTruth(openings={"drawer_0": 0.2}), "drawers are open"),
    ],
)
def test_each_unsettled_world_is_a_setup_problem(truth, words):
    result = check(truth, fallen_height=0.25, limits=LIMITS_NOW)
    assert not result["passed"] and len(result["problems"]) == 1 and words in result["problems"][0]


def test_a_robot_that_cannot_fall_and_drawers_allowed_open_are_not_checked():
    limits = dict(LIMITS_NOW, drawer_open_max_m=None)
    assert check(FakeTruth(height=0.0, openings={"drawer_0": 0.3}), fallen_height=None, limits=limits)["passed"]


def test_a_problem_names_five_bodies_and_counts_the_rest():
    motion = {f"box_{i}": (1.0, 0.0) for i in range(8)}
    (problem,) = check(FakeTruth(motion=motion), fallen_height=None, limits=LIMITS_NOW)["problems"]
    assert problem.startswith("8 bodies still move") and problem.endswith("and 3 more")


def test_missing_limits_are_an_error():
    with pytest.raises(ValueError, match="start_check lacks"):
        check(FakeTruth(), fallen_height=None, limits={})


@pytest.mark.parametrize(
    ("ended_by", "expected"),
    [
        ("finish", "completed"),
        ("agent_returned", "completed"),
        ("agent_error", "completed"),  # The agent's crash is the agent's failure.
        ("budget", "completed"),
        ("runtime_interrupted", "completed"),  # A fall: an emergency stop the agent caused.
        ("setup_error", "setup_error"),
        ("runtime_failed", "error"),
        ("environment_error", "error"),
    ],
)
def test_the_status_says_whether_an_episode_can_be_scored(ended_by, expected):
    assert status(ended_by) == expected and expected in STATUSES


def test_a_broken_component_is_an_error_even_when_the_agent_stopped_first():
    """The arm planner process ran out of GPU memory: the skill failed, the agent gave up, and the
    runtime had failed; the agent is not scored for it."""
    assert status("agent_failed", runtime_failed=True) == "error"
    assert status("agent_failed") == "completed"
