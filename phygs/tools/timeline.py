# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Commands issued to the robot's clients, as spans on a base and an arm lane (for videos).

:class:`CommandTimeline` wraps the command methods of the client instances it watches: each call
opens a span with the command's name and the simulation time, and :meth:`CommandTimeline.update`
closes it with the command's final status. Repeated calls of the same command back to back extend
one span, so repeated velocity keep-alive commands do not flood the lane.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ACTIVE = frozenset({"active", "stopping"})
# Commands per lane and client attribute of RobotControl.
COMMANDS = {
    "base": {"navigation": ("navigate_to",), "locomotion": ("command_velocity", "stop")},
    "arm": {
        "manipulation": (
            "move_joints",
            "move_to_pose",
            "execute_trajectory",
            "open_gripper",
            "close_gripper",
            "stow",
            "look_at",
        ),
        "arm": ("command_joint_positions",),
    },
}


@dataclass
class Span:
    """One command on a lane: its name, start and end [s sim] (None while it runs) and status."""

    lane: str
    name: str
    start: float
    end: float | None = None
    status: str = "active"


class CommandTimeline:
    """Spans of the commands issued to the watched clients."""

    MERGE_S = 0.2  # The same command issued again within this [s] of the last one extends its span.

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self.spans: list[Span] = []
        self._open: list[tuple[Span, Any, int]] = []  # (span, client, command id)

    def watch(self, control: Any) -> None:
        """Wrap the command methods of a RobotControl's clients (those it has)."""
        for lane, clients in COMMANDS.items():
            for attribute, names in clients.items():
                client = getattr(control, attribute, None)
                if client is None:
                    continue
                for name in names:
                    if callable(getattr(client, name, None)):
                        setattr(client, name, self._wrapped(lane, client, name, getattr(client, name)))

    def _wrapped(self, lane: str, client: Any, name: str, method: Callable) -> Callable:
        @functools.wraps(method)
        def call(*args, **kwargs):
            command = method(*args, **kwargs)
            if name == "command_velocity" and not any(args[:3]):
                return command  # Standing still: a keep-alive, not an action.
            self._opened(lane, name, client, command)
            return command

        return call

    def _opened(self, lane: str, name: str, client: Any, command: int) -> None:
        now = self._clock()
        last = next((span for span in reversed(self.spans) if span.lane == lane), None)
        if last is not None and last.name == name and (last.end is None or now - last.end <= self.MERGE_S):
            last.end, last.status = None, "active"
            self._open = [entry for entry in self._open if entry[0] is not last]
        else:
            last = Span(lane, name, now)
            self.spans.append(last)
        self._open.append((last, client, command))

    def active(self, lane: str) -> str | None:
        """The name of the latest running command on ``lane``, or None."""
        return next((span.name for span, _, _ in reversed(self._open) if span.lane == lane), None)

    def clear(self) -> None:
        """Forget every span (a new episode: simulated time starts over)."""
        self.spans, self._open = [], []

    def update(self) -> None:
        """Close the spans whose command finished, with its status (``done`` reads ``succeeded``)."""
        still = []
        for span, client, command in self._open:
            status = client.feedback(command).status
            if status in ACTIVE:
                still.append((span, client, command))
            else:
                span.end, span.status = self._clock(), "succeeded" if status == "done" else status
        self._open = still
