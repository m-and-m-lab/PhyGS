# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared fakes for the robot UI unit tests: a fake robot control, skill clients and camera frames,
and :class:`Ui`, a UiSession with a fake control as its agent session and its episode view."""

import enum
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from fakes import SPOT

from phygs.episodes.viewer import EpisodeView
from phygs.runtime.episode import EpisodeStatus
from phygs.tools.front_view import CameraFrame
from phygs.ui.bridge import Bridge
from phygs.ui.session import UiSession

ARM_JOINTS = SPOT.arm.joints

SECRET = "privileged-target-pose-1234"  # Privileged text that must never reach the browser.


START = {"type": "episode", "action": "start"}


class Failure(enum.Enum):
    OUT_OF_REACH = "out_of_reach"


@dataclass(frozen=True)
class Feedback:
    """Command feedback as the skill clients return it."""

    command_id: int
    status: str
    reason: str = ""
    failure: Failure | None = None


class Runtime:
    """The episode runtime the session reads: episode 3 at 12.5 s, 250 Hz physics."""

    def __init__(self):
        self.status, self.episode_id, self.time_seconds = EpisodeStatus.RUNNING, 3, 12.5
        self.physics_dt = 0.004
        self.stops = []

    def emergency_stop(self, reason):
        self.stops.append(reason)
        self.status, self.reason = EpisodeStatus.INTERRUPTED, reason

    def complete(self, reason):
        self.status, self.reason = EpisodeStatus.COMPLETED, reason

    def result(self):
        return {"reason": getattr(self, "reason", "")}


class Client:
    """Feedback that advances one status per poll."""

    def __init__(self, statuses):
        self.statuses = list(statuses)

    def feedback(self, command_id):
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        failure = Failure.OUT_OF_REACH if status == "failed" else None
        return Feedback(
            command_id, status, "This view of the target is out of reach; move the base." if failure else "", failure
        )


def camera(tick, value):
    """A valid 4x6 RGB-D camera sample filled with ``value``, captured at ``tick``."""
    stamp = SimpleNamespace(tick=tick, time_seconds=tick * 0.004, episode_id=3)
    return SimpleNamespace(
        stamp=stamp,
        valid=True,
        frame=f"cam{value}",
        rgb=torch.full((4, 6, 3), value, dtype=torch.uint8),
        depth=torch.full((4, 6), 1.5),
        depth_valid=torch.ones(4, 6, dtype=torch.bool),
        intrinsics=torch.eye(3),
    )


def fake_control():
    """A robot control with cameras, a map, narration and privileged fields the page must not see."""
    names = (*ARM_JOINTS, "arm_f1x", "front_left_knee")
    stow = [0.0, -3.13, 3.13, 0.0, 0.0, 0.0]
    packet = SimpleNamespace(
        stamp=SimpleNamespace(tick=16, time_seconds=0.064, episode_id=3),
        cameras={"hand": camera(16, 10), "frontleft": camera(16, 20), "third_person": camera(16, 99)},
        localization=SimpleNamespace(
            valid=True, pose=SimpleNamespace(matrix=torch.eye(4)), linear_velocity=torch.zeros(3)
        ),
        joints=SimpleNamespace(
            valid=True,
            names=names,
            positions=torch.tensor([*stow, -0.4, -1.5]),
            velocities=torch.tensor([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ),
        transform=lambda target, source, stamp: torch.eye(4),
    )
    snapshot = SimpleNamespace(
        cells=np.zeros((3, 4), dtype=np.int8), origin=(1.0, 2.0), resolution=0.05, frames_integrated=5
    )
    return SimpleNamespace(
        observe=lambda: packet,
        product=lambda name: snapshot,
        runtime=Runtime(),
        narration=SimpleNamespace(
            log=[(1.0, "locomotion", "command_velocity: v_x 0.40 m/s"), (1.2, "check", "episode check: " + SECRET)]
        ),
        locomotion=Client(["active", "done"]),
        manipulation=SimpleNamespace(holding=False, feedback=Client(["active", "failed"]).feedback),
        scene=SimpleNamespace(target_pose=SECRET),  # Privileged: must never reach the browser.
        evaluator=SimpleNamespace(answer=SECRET),
    )


class Navigation:
    """A navigation client recording its goals and cancellations."""

    def __init__(self):
        self.goals, self.cancelled = [], []

    def navigate_to(self, x, y, heading=None, *, position_tolerance, heading_tolerance):
        if x > 100:
            raise ValueError("Goal is outside the map")
        self.goals.append((x, y, heading, position_tolerance, heading_tolerance))
        return len(self.goals)

    def feedback(self, command_id):
        return Feedback(command_id, "cancelled" if command_id in self.cancelled else "active")

    def cancel(self, command_id):
        self.cancelled.append(command_id)


class AgentSession:
    """The agent session over a fake control: its clients, its observation and time; ``finished``
    keeps the reasons the agent finished the episode with."""

    def __init__(self, control):
        self._control, self.finished, self.time_calls = control, [], 0
        self.ended: str | None = None  # Set: every call raises EpisodeEnded, as after the budget.

    def __getattr__(self, name):
        if name in ("navigation", "manipulation", "locomotion"):
            self._check()
            return getattr(self._control, name)
        raise AttributeError(name)

    def _check(self):
        from phygs.episodes.session import EpisodeEnded

        if self.ended is not None:
            raise EpisodeEnded(self.ended)

    def observe(self):
        self._check()
        return self._control.observe()

    def time(self):
        self._check()
        self.time_calls += 1
        return self._control.runtime.time_seconds

    def finish(self, reason=""):
        self.finished.append(reason)


class Ui:
    """A UiSession whose agent session and episode view come from a fake control. :meth:`update`
    is one agent pass, then one viewer update; other members are the UiSession's."""

    def __init__(self, control, bridge, *, costmap=None, **options):
        options.setdefault("feedback_period", 0.0)  # Every update reads the feedback.
        self.session = UiSession(bridge, **{"robot": SPOT, **options})
        self.control, self.costmap, self.robot = control, costmap, AgentSession(control)

    def view(self):
        control = self.control
        manipulation = getattr(control, "manipulation", None)
        return EpisodeView(
            episode_id=control.runtime.episode_id,
            time_seconds=control.runtime.time_seconds,
            status=control.runtime.status.value,
            observe=control.observe,
            costmap=self.costmap or (lambda: None),
            narration=lambda start: list(control.narration.log[start:]),
            holding=bool(getattr(manipulation, "holding", False)),
        )

    def update(self):
        self.session.agent_step(self.robot)
        self.session.update(self.view())

    def track(self, client, skill, command_id, inputs=None):
        return self.session.track(client, skill, command_id, inputs, self.robot)

    def __getattr__(self, name):
        return getattr(self.session, name)


def run_requests(control, *requests, limits=None):
    """Start an episode, send ``requests`` from the page, and return the command events."""
    bridge = Bridge({"views": ["front"]})
    ui = Ui(control, bridge, cameras=(), drive_limits=limits)
    for request in (START, *requests):
        assert bridge.request(request)
    ui.update()
    return [event for _, event in bridge.events_after(0) if event["type"] == "command"]


def frame(name, value, tick=40):
    """A 48x64 camera frame filled with ``value``, 2 m deep."""
    return CameraFrame(
        name=name,
        time_seconds=0.08,
        tick=tick,
        rgb=np.full((48, 64, 3), value, dtype=np.uint8),
        depth=np.full((48, 64), 2.0, dtype=np.float32),
        valid=np.ones((48, 64), dtype=bool),
        intrinsics=np.array([[32.0, 0, 32], [0, 32.0, 24], [0, 0, 1]]),
        body_from_camera=np.eye(4),
    )


class Manipulation:
    """A manipulation client recording its calls."""

    def __init__(self):
        self.calls, self.holding, self.plans = [], False, []
        self.tool_queries = []  # tool_pose() reads.

    def plan(self, *, goal=None, pose_xyzw=None, tool="hand", held=None):
        from concurrent.futures import Future

        self.plans.append(dict(goal=goal, pose_xyzw=pose_xyzw, tool=tool, held=held))
        return Future()

    def minimum_move_duration(self, positions):
        return 1.2501  # Not a whole number of 0.004 s physics steps.

    def move_joints(self, positions, *, duration):
        self.calls.append(("move_joints", positions, duration))
        return len(self.calls)

    def execute_trajectory(self, trajectory):
        self.calls.append(("execute_trajectory", trajectory))
        return len(self.calls)

    def stow(self, *, held=None):
        self.calls.append(("stow",))
        return len(self.calls)

    def look_at(self, target, *, distance_m, elevation_deg, held=None):
        self.calls.append(("look_at", target, distance_m, elevation_deg))
        return len(self.calls)

    def move_to_pose(self, pose, *, tool="hand", held=None):  # The defaults are the API's.
        self.calls.append(("move_to_pose", pose, tool))
        return len(self.calls)

    def tool_pose(self, tool="hand"):
        self.tool_queries.append(tool)
        return (0.5, 0.0, 0.6, 0.0, 0.0, 0.0, 1.0)

    def open_gripper(self, *, duration):
        self.calls.append(("open_gripper", duration))
        return len(self.calls)

    def feedback(self, command_id):
        return Feedback(command_id, "active")
