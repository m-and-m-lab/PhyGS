# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The command timeline (spans of client commands) and the video overlay, without simulation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from phygs.recording.timeline import CommandTimeline, Span
from phygs.recording.video import COMMAND_COLORS, FAILED, compose


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


def test_text_and_timeline_are_strips_around_the_unchanged_image_with_a_constant_size():
    frame = np.full((450, 800, 3), 127, dtype=np.uint8)
    spans = [Span("arm", "look_at", 1.0, 5.0, "succeeded"), Span("base", "navigate_to", 0.0, None, "active")]
    one = compose(frame, title="check", stats="RTF 0.8", sim_time=10.0, lines=["manipulation: x"], spans=spans[:1])
    spans.append(Span("arm", "move_to_pose", 6.0, 9.0, "failed"))
    lines = ["manipulation: x", "navigation: y", "locomotion: z"]
    many = compose(frame, title="check", stats="RTF 0.8", sim_time=10.0, lines=lines, spans=spans)
    assert one.shape == many.shape and one.shape[1] == 800 and one.shape[0] > 450  # Same size every frame.
    top = next(row for row in range(many.shape[0]) if (many[row] == 127).all())
    assert (many[top : top + 450] == 127).all()  # Nothing drawn over the image.
    strip = many[top + 450 :]
    for color in (COMMAND_COLORS["look_at"], COMMAND_COLORS["navigate_to"], FAILED):
        assert (np.abs(strip.astype(int) - np.array(color)).sum(axis=-1) < 30).any(), color


def test_a_new_episode_starts_an_empty_timeline():
    clock, arm, base, timeline = setup()
    arm.look_at("target")
    timeline.clear()
    clock.now = 0.0
    arm.stow()
    assert [(s.name, s.start) for s in timeline.spans] == [("stow", 0.0)]
