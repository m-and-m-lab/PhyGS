# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The command timeline (spans of client commands) and the videos' one look, without simulation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from phygs.recording.panels import layout
from phygs.recording.timeline import CommandTimeline
from phygs.recording.trace import compose
from phygs.recording.video import CHASE, CHASE_KEEP, Chase, Video, live_axis, strip_height


class Client:
    """Commands that end when ``finish`` says so; feedback reads the status."""

    def __init__(self):
        self.next, self.statuses = 1, {}

    def _issue(self, *args, **kwargs):
        command, self.next = self.next, self.next + 1
        self.statuses[command] = "active"
        return command

    look_at = stow = move_to_pose = navigate_to = command_velocity = _issue

    def feedback(self, command):
        return SimpleNamespace(status=self.statuses[command])


def setup():
    clock = SimpleNamespace(now=0.0)
    arm, base = Client(), Client()
    timeline = CommandTimeline(lambda: clock.now)
    timeline.watch(SimpleNamespace(manipulation=arm, navigation=base, locomotion=base))
    return clock, arm, base, timeline


def test_commands_become_spans_closed_with_their_final_status():
    clock, arm, base, timeline = setup()
    look = arm.look_at("target")
    walk = base.navigate_to(1.0, 0.0, 0.0)
    clock.now = 2.0
    arm.statuses[look] = "succeeded"
    timeline.update()
    clock.now = 3.0
    base.statuses[walk] = "blocked"
    timeline.update()
    assert [(s.lane, s.name, s.start, s.end, s.status) for s in timeline.spans] == [
        ("arm", "look_at", 0.0, 2.0, "succeeded"),
        ("base", "navigate_to", 0.0, 3.0, "blocked"),
    ]


def test_the_active_command_of_a_lane_is_the_latest_still_running():
    clock, arm, base, timeline = setup()
    assert timeline.active("arm") is None
    look = arm.look_at("target")
    move = arm.move_to_pose((0, 0, 0, 0, 0, 0, 1))
    assert timeline.active("arm") == "move_to_pose" and timeline.active("base") is None
    arm.statuses[move] = "succeeded"
    timeline.update()
    assert timeline.active("arm") == "look_at"
    arm.statuses[look] = "failed"
    timeline.update()
    assert timeline.active("arm") is None


def test_keep_alives_are_skipped_and_repeated_commands_extend_one_span():
    clock, arm, base, timeline = setup()
    base.command_velocity(0.0, 0.0, 0.0, duration=0.5)  # Standing still: not an action.
    assert timeline.spans == []
    first = base.command_velocity(0.3, 0.0, 0.0, duration=0.5)
    clock.now = 0.5
    base.statuses[first] = "done"
    timeline.update()
    clock.now = 0.6  # Issued again within the merge window: the same span goes on.
    second = base.command_velocity(0.3, 0.0, 0.0, duration=0.5)
    clock.now = 1.1
    base.statuses[second] = "done"
    timeline.update()
    assert [(s.name, s.start, s.end, s.status) for s in timeline.spans] == [("command_velocity", 0.0, 1.1, "succeeded")]


def test_a_video_is_the_outside_view_with_the_panels_right_and_the_strip_under_both(tmp_path):
    import imageio.v2 as imageio

    outside = np.full((450, 800, 3), 127, dtype=np.uint8)
    spans = [dict(lane="arm", name="look_at", start=1.0, end=5.0, status="succeeded")]
    video = Video(tmp_path / "v.mp4", fps=10.0)
    for k in range(3):
        trace = compose(end=0.5 * k, commands=spans, narration=[(0.0, "manipulation", "look_at: planning")])
        video.frame(outside, sources=None, trace=trace, sim_time=0.5 * k, duration=live_axis(0.5 * k))
    video.close()
    frames = imageio.mimread(tmp_path / "v.mp4")
    grid = layout(outside, {})  # The panels' grid, "no data" without sources.
    assert len(frames) == 3 and frames[0].shape == (450 + strip_height(450), grid.shape[1], 3)
    assert grid.shape[1] > 800 and (np.abs(frames[0][:440, :790].astype(int) - 127) < 12).all()  # Image untouched.
    plain = Video(tmp_path / "plain.mp4", fps=10.0, panels=False, overview=False)
    plain.frame(outside, sources=None, trace=trace, sim_time=0.0)
    plain.close()
    assert imageio.mimread(tmp_path / "plain.mp4")[0].shape == (450, 800, 3)


def test_the_chase_camera_stays_behind_the_robot_and_turns_with_it_smoothly():
    chase = Chase(0.032)
    first, target, near = chase.aim([2.0, 1.0, 0.5, 0.0, 0.0, 0.0, 1.0])  # Heading +x.
    assert target == pytest.approx([2.35, 1.0, 0.45]) and first == pytest.approx(target + np.array(CHASE))
    assert near == pytest.approx(np.linalg.norm(CHASE) - CHASE_KEEP)  # Walls between are cut away.
    turned = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]  # Heading +y.
    eye, _, _ = chase.aim([2.0, 1.0, 0.5, *turned])
    assert 0.0 < np.linalg.norm(eye - first) < 0.3  # It starts to swing round, without a jump.
    for _ in range(200):
        eye, target, _ = chase.aim([2.0, 1.0, 0.5, *turned])
    assert eye == pytest.approx(target + np.array([-CHASE[1], CHASE[0], CHASE[2]]), abs=1e-3)
    chase.reset()
    assert chase.aim([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])[0] == pytest.approx(np.array([0.35, 0.0, 0.45]) + CHASE)
    fixed = Chase(0.032, eye=(5.0, 5.0, 2.0))
    assert fixed.aim([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])[0].tolist() == [5.0, 5.0, 2.0]
    assert live_axis(10.0) == 60.0 and live_axis(61.0) == 90.0


def test_a_new_episode_starts_an_empty_timeline():
    clock, arm, base, timeline = setup()
    arm.look_at("target")
    timeline.clear()
    clock.now = 0.0
    arm.stow()
    assert [(s.name, s.start) for s in timeline.spans] == [("stow", 0.0)]
