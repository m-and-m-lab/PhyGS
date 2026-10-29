# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Skill step narration collected for printing and video overlays."""

import logging

from phygs.recording.narration import Narration


def test_narration_keeps_every_step_with_sim_time_and_the_latest_per_kind(capsys):
    clock = iter((1.0, 2.5, 4.0))
    narration = Narration(lambda: next(clock)).attach()
    try:
        logging.getLogger("phygs.skills.navigation").info("navigate_to: following a route")
        logging.getLogger("phygs.skills.manipulation").info("pick: closing the gripper")
        logging.getLogger("phygs.skills.navigation").info("navigate_to: controlled stop (stopping)")
    finally:
        narration.detach()
    assert narration.log == [
        (1.0, "navigation", "navigate_to: following a route"),
        (2.5, "manipulation", "pick: closing the gripper"),
        (4.0, "navigation", "navigate_to: controlled stop (stopping)"),
    ]
    assert narration.lines() == [
        "manipulation: pick: closing the gripper",
        "navigation: navigate_to: controlled stop (stopping)",
    ]
    assert capsys.readouterr().out.splitlines()[0] == "[   1.000 s] navigate_to: following a route"
