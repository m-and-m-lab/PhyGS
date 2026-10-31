# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A run's trace from its logs, and the overview strip drawn from it (no Isaac Sim)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from phygs.recording import trace
from phygs.recording.overview import ROBOT, Overview
from phygs.recording.state_log import StateLog
from phygs.scene.plan import Floor, Plan

COMMANDS = [  # In the timeline's order: a command a skill issues opens before the skill's own span.
    dict(lane="arm", name="move_joints", start=0.0, end=0.5, status="succeeded"),
    dict(lane="arm", name="stow", start=0.0, end=1.0, status="succeeded"),
    dict(lane="base", name="navigate_to", start=1.0, end=5.0, status="succeeded"),
    dict(lane="arm", name="close_gripper", start=6.0, end=6.5, status="succeeded"),
    dict(lane="arm", name="execute_trajectory", start=6.5, end=None, status="active"),
    dict(lane="arm", name="move_to_pose", start=6.5, end=None, status="active"),
]
REPORT = dict(
    sim_seconds=8.0,
    grasp_contacts=[
        dict(link="jaw", body="cup", first_s=6.2, last_s=7.0, peak_n=12.0, ticks=5, step="close_gripper"),
        dict(link="finger", body="cup", first_s=6.1, last_s=8.0, peak_n=20.0, ticks=5, step="close_gripper"),
    ],
    contacts=[dict(link="arm_link_wr1", body="table", first_s=3.0, last_s=3.2, peak_n=40.0, ticks=2, step="stow")],
)


def mark(t, name, event, depth, **extra):
    return dict(t=t, call="session.step", name=name, event=event, depth=depth, **extra)


AUDIT = [
    mark(0.0, "stow", "begin", 0),
    dict(t=0.0, call="manipulation.stow", result=1),
    mark(1.0, "stow", "end", 0),
    dict(t=1.0, call="hints.target", result={"TargetHint": {"name": "cup", "center": [4.0, 1.0, 0.8]}}),
    mark(1.0, "go_to(table)", "begin", 0),
    mark(5.0, "go_to(table)", "end", 0, error="AgentFailed: blocked"),
    mark(6.0, "pick(cup)", "begin", 0),
    mark(6.0, "look_at(target)", "begin", 1),
    mark(6.0, "look_at(target)", "end", 1),
]


def write_run(folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(json.dumps(REPORT))
    (folder / "audit.jsonl").write_text("".join(json.dumps(line) + "\n" for line in AUDIT))
    log = StateLog(folder / "state.npz", joint_names=("j",), body_names=("cup",), meta=dict(period=0.5))
    for k in range(17):
        t = 0.5 * k
        x = min(max(t - 1.0, 0.0), 4.0)  # Walks 1 m/s from t = 1 s to 5 s.
        root = np.array([x, 1.0, 0.5, 0.0, 0.0, 0.0, 1.0])
        log.sample(t, root, [0.0], [[4.0, 1.0, 0.8, 0, 0, 0, 1]], holding=6.5 <= t < 7.5)
        if t == 1.0:
            log.route(t, (4.0, 1.0, 0.0), ((0.0, 1.0), (2.0, 1.0), (4.0, 1.0)))
        if t == 5.0:
            log.route(t, None, ())
    log.write(COMMANDS)
    return folder


def test_the_trace_has_the_agents_steps_the_commands_and_the_contacts(tmp_path):
    built = trace.build(write_run(tmp_path / "run"))
    lanes = {lane["name"]: lane["items"] for lane in built["lanes"]}
    assert [lane["name"] for lane in built["lanes"]] == list(trace.LANES)
    assert [(item["label"], item["s"], item["e"], item["status"]) for item in lanes["Agent"]] == [
        ("stow", 0.0, 1.0, "succeeded"),
        ("go_to(table)", 1.0, 5.0, "failed"),
        ("pick(cup)", 6.0, 8.0, "open"),  # The run ended inside it.
    ]
    assert lanes["Agent"][1]["note"] == "AgentFailed: blocked"
    assert [(s["label"], s["depth"]) for s in built["steps"]][-1] == ("look_at(target)", 1)
    assert [item["label"] for item in lanes["Base"]] == ["navigate_to"]
    assert [item["label"] for item in lanes["Arm"]] == ["stow", "move_to_pose"]  # Not the commands inside them.
    assert lanes["Arm"][1]["e"] == 8.0  # Still running at the end.
    assert [item["label"] for item in lanes["Gripper"]] == ["close_gripper"]
    assert lanes["Grip"] == [dict(label="grip cup", s=6.1, e=8.0, status="grip", body="cup", peak=20.0)]
    assert lanes["Bumps"][0]["peak"] == 40.0 and lanes["Bumps"][0]["body"] == "table"
    assert built["holding"] == [dict(s=6.5, e=7.5, label="cup", status="succeeded")]
    path = built["path"]
    assert path["target"] == [4.0, 1.0] and path["z"] == 0.5 and built["duration"] == 8.0
    assert [route["t"] for route in path["routes"]] == [1.0, 5.0] and path["routes"][1]["path"] == []
    assert path["poses"][30][1:3] == pytest.approx((2.0, 1.0))  # 3 s at 10 per second.


def test_a_run_without_steps_or_audit_still_has_its_commands(tmp_path):
    run = write_run(tmp_path / "run")
    (run / "audit.jsonl").unlink()
    built = trace.build(run)
    assert built["lanes"][0]["items"] == [] and built["path"]["target"] is None
    assert len(built["lanes"][1]["items"]) == 1


def plan_of(tmp_path):
    """A 400x200 plan of x in [0, 8] and y in [-1, 3] at 50 px/m."""
    folder = tmp_path / "plan"
    folder.mkdir()
    Image.new("RGB", (400, 200), (200, 200, 200)).save(folder / "ground.png")
    plan = Plan(folder, (Floor("ground", "ground.png", 0.0, 2.4, 400, 200, (4.0, 1.0), 50.0),))
    plan.save()
    return Plan.load(folder)


@pytest.mark.parametrize("with_plan", [True, False])
def test_the_overview_follows_the_robot_where_the_trace_puts_it(tmp_path, with_plan):
    built = trace.build(write_run(tmp_path / "run"))
    overview = Overview(plan_of(tmp_path) if with_plan else None, width=960, height=260)
    center = ((overview.box[0] + overview.box[2]) // 2, (overview.box[1] + overview.box[3]) // 2)
    for t in (0.0, 3.0, 8.0):
        image = overview.draw(built, t)
        assert image.shape == (260, 960, 3)
        x = 0.0 if t == 0.0 else min(t - 1.0, 4.0)
        assert overview._center == pytest.approx((x, 1.0))  # The window follows the robot.
        patch = image[center[1] - 2 : center[1] + 3, center[0] - 2 : center[0] + 3].reshape(-1, 3)
        assert any(tuple(pixel) == ROBOT for pixel in patch)  # The robot's arrow, in the middle.
    assert not np.array_equal(overview.draw(built, 2.0), overview.draw(built, 6.0))  # The playhead moved.
    u0, v0 = overview._to_window(0.0, 1.0)
    u1, v1 = overview._to_window(1.0, 2.0)
    assert u1 - u0 == pytest.approx(overview.scale) and v0 - v1 == pytest.approx(overview.scale)  # +x right, +y up.
    assert overview.draw(built, 3.0, duration=60.0).shape == (260, 960, 3)  # A recording's growing axis.


def test_a_recordings_trace_is_composed_from_what_the_run_logged_so_far():
    commands = [dict(lane="base", name="navigate_to", start=1.0, end=None, status="active")]
    live = trace.compose(
        end=2.0,
        title="dev | retrieve: the cup",
        audit=AUDIT[:5],  # The walk has begun, not ended.
        commands=commands,
        contacts=dict(contacts=REPORT["contacts"][:0]),
        narration=[(1.0, "navigation", "navigate_to: following")],
        poses=[(0.0, 0.0, 1.0, 0.0), (1.5, 0.5, 1.0, 0.0)],
    )
    lanes = {lane["name"]: lane["items"] for lane in live["lanes"]}
    assert [(item["label"], item["e"], item["status"]) for item in lanes["Agent"]] == [
        ("stow", 1.0, "succeeded"),
        ("go_to(table)", 2.0, "open"),
    ]
    assert lanes["Base"][0]["e"] == 2.0 and live["holding"] == [] and live["path"]["target"] == [4.0, 1.0]
    assert live["path"]["poses"][-1] == [1.5, 0.5, 1.0, 0.0] and live["title"].startswith("dev")
    image = Overview(None, width=960, height=260).draw(live, 2.0, duration=60.0)
    assert image.shape == (260, 960, 3)
