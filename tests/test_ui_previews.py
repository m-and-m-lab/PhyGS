# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plan previews that run as shown."""

from dataclasses import replace

import numpy as np
import pytest
from fakes import SPOT
from ui_fakes import START, Manipulation, Ui, fake_control, run_requests

from phygs.execution.manipulation.planning import JointTrajectory
from phygs.ui import schema
from phygs.ui.bridge import Bridge

ARM_JOINTS, ARM_LINKS = SPOT.arm.joints, SPOT.arm_joint_links


def preview_session():
    control = fake_control()
    control.manipulation = Manipulation()
    plans = control.manipulation.plans  # manipulation.plan calls: the agent API's plan-only call.

    def forward(joints):  # Each link 0.1 m further ahead, all of them moved by the shoulder angle.
        links = {}
        for index, name in enumerate(ARM_LINKS):
            links[name] = np.eye(4)
            links[name][0, 3] = 0.1 * (index + 1) + joints["arm_sh0"]
        return links

    bridge = Bridge({"views": ["front"]})
    ahead = np.eye(4)
    ahead[0, 3] = 0.2  # A grasp point 0.2 m along the wrist x.
    robot = replace(SPOT, gripper=replace(SPOT.gripper, tools={**SPOT.gripper.tools, "grasp": ahead}))
    session = Ui(control, bridge, robot=robot, cameras=(), forward=forward)
    bridge.request(START)
    session.update()
    return control, bridge, session, plans


def test_a_preview_plans_without_moving_and_runs_the_plan_it_showed():
    control, bridge, session, plans = preview_session()
    pose = {name: 0.0 for name in ARM_JOINTS}
    bridge.request({"type": "preview", "skill": "move_joints", "inputs": {"positions": pose}})
    session.update()
    assert plans == [dict(goal=pose, pose_xyzw=None, tool="hand", held=None)]
    assert control.manipulation.calls == []  # Nothing moved.
    previews = [event for _, event in bridge.events_after(0) if event["type"] == "preview"]
    assert previews[-1]["status"] == "planning"
    trajectory = JointTrajectory(ARM_JOINTS, np.array([0.0, 1.0, 2.0]), np.zeros((3, 6)) + [[0.0], [0.5], [1.0]], 3)
    session._previews["p1"]["plan"].set_result(trajectory)
    session.update()
    ready = [event for _, event in bridge.events_after(0) if event["type"] == "preview"][-1]
    assert ready["status"] == "ready" and ready["duration"] == 2.0 and len(ready["path_body"]) == 3
    assert len(ready["arm_body"]) == len(ARM_LINKS) + 1 and len(ready["path_world"][0]) == 2
    bridge.request({"type": "run", "skill": "move_joints", "inputs": {"preview": "p1"}})
    bridge.request({"type": "run", "skill": "move_joints", "inputs": {"preview": "p9"}})
    session.update()
    assert control.manipulation.calls == [("execute_trajectory", trajectory)]  # The very plan shown.
    commands = [event for _, event in bridge.events_after(0) if event["type"] == "command"]
    assert commands[0]["skill"] == "move_joints" and commands[0]["inputs"]["preview"] == "p1"
    assert "No ready plan 'p9'" in commands[-1]["feedback"]["reason"]


def test_a_failed_preview_shows_the_planners_reason():
    control, bridge, session, plans = preview_session()
    target = {"point": [1.0, 0.0, 0.5], "normal": [0, 0, 1], "view": "hand"}
    bridge.request(
        {
            "type": "preview",
            "skill": "move_to_pose",
            "inputs": {"target": target, "orientation": "down", "standoff_m": 0.1},
        }
    )
    session.update()
    assert plans[0]["pose_xyzw"][:3] == pytest.approx((1.0, 0.0, 0.8)) and plans[0]["tool"] == "wrist"
    session._previews["p1"]["plan"].set_exception(RuntimeError("Pose planner failed: no_plan_within_budget"))
    session.update()
    failed = [event for _, event in bridge.events_after(0) if event["type"] == "preview"][-1]
    assert (
        failed["status"] == "failed" and failed["reason"] == "RuntimeError: Pose planner failed: no_plan_within_budget"
    )


def test_preview_needs_a_planner_and_the_schema_says_which_skills_preview():
    control = fake_control()
    events = run_requests(control, {"type": "preview", "skill": "move_joints", "inputs": {"positions": {}}})
    assert events[-1]["feedback"]["reason"] == "RuntimeError: Plan preview needs the robot's kinematic model."
    listed = {
        skill["name"]: skill for skill in schema.skills({"max_speed": 1, "max_yaw_speed": 1}, {}, True, preview=True)
    }
    assert listed["move_joints"]["preview"] and listed["move_to_pose"]["preview"]
    assert not listed["move_along"]["preview"]
    assert "declare_target" not in listed
    # The skills tier has no base velocity client: no Drive or Stop tiles.
    skills_tier = schema.skills({"max_speed": 1, "max_yaw_speed": 1}, {}, True, actions=("navigation", "manipulation"))
    assert {"drive", "stop"}.isdisjoint(skill["name"] for skill in skills_tier)
