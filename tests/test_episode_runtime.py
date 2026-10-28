# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime authority, atomic ownership, expiry, and failure checks."""

import math
from functools import partial

import pytest
from fakes import FakeWorld

from phygs.runtime.episode import ControllerDeclaration, EpisodeRuntime, EpisodeStatus, RunMode

JOINTS = ("leg", "arm", "gripper")


class PositionController:
    def __init__(self, name, joints, source=RunMode.AUTONOMOUS):
        self.declaration = ControllerDeclaration(name, joints, source)
        self.calls = 0
        self.target = 0.25

    def reset(self, positions):
        self.initial = positions
        self.calls = 0

    def step(self, dt):
        self.calls += 1
        return dict.fromkeys(self.declaration.joints, self.target)


def build(mode=RunMode.AUTONOMOUS):
    world = FakeWorld(JOINTS)
    controllers = (
        PositionController("legs", ("leg",)),
        PositionController("arm", ("arm",)),
        PositionController("whole", world.joint_names),
        PositionController("human", ("leg",), RunMode.HUMAN),
    )
    return world, controllers, EpisodeRuntime(world, controllers, mode=mode)


def test_ownership_is_atomic_and_handoff_uses_current_state():
    world, controllers, runtime = build()
    runtime.reset()
    runtime.acquire("legs", duration_ticks=10)
    with pytest.raises(RuntimeError, match="already owned"):
        runtime.acquire("whole", duration_ticks=10)
    runtime.acquire("arm", duration_ticks=10)
    runtime.step()
    assert world.state == {"leg": 0.25, "arm": 0.25, "gripper": 0.0}
    runtime.release("legs")
    runtime.release("arm")
    runtime.acquire("whole", duration_ticks=10)
    assert controllers[2].initial == world.state


def test_expiration_holds_and_does_not_tick_expired_controller():
    world, controllers, runtime = build()
    runtime.reset()
    runtime.acquire("legs", duration_ticks=2)
    for _ in range(4):
        runtime.step()
    assert controllers[0].calls == 2
    assert world.state["leg"] == 0.25
    assert runtime.time_seconds == pytest.approx(0.016)  # Four 250 Hz steps.
    assert any(event.kind == "expired" and event.tick == 2 for event in runtime.events)


@pytest.mark.parametrize("mode", list(RunMode))
def test_human_authority_is_assigned_at_setup(mode):
    _, _, runtime = build(mode)
    runtime.reset()
    if mode == RunMode.AUTONOMOUS:
        with pytest.raises(PermissionError):
            runtime.acquire("human", duration_ticks=1)
    else:
        runtime.acquire("human", duration_ticks=1)
        runtime.step()


def test_emergency_stop_is_terminal_and_reset_creates_new_episode():
    world, controllers, runtime = build()
    runtime.reset(settling_ticks=3)
    runtime.acquire("legs", duration_ticks=10)
    runtime.step()
    runtime.emergency_stop("operator stop")
    for action in (runtime.step, lambda: runtime.acquire("arm", duration_ticks=1)):
        with pytest.raises(RuntimeError, match="interrupted"):
            action()
    assert world.steps == 4
    assert runtime.result()["status"] == "interrupted"
    runtime.reset()
    assert runtime.episode_id == 2
    assert runtime.tick == 0
    assert world.state == dict.fromkeys(world.joint_names, 0.0)
    assert all(controller.calls == 0 for controller in controllers)
    runtime.acquire("whole", duration_ticks=1)


@pytest.mark.parametrize("target", [math.nan, math.inf])
def test_invalid_output_fails_before_physics(target):
    world, controllers, runtime = build()
    runtime.reset()
    controllers[0].target = target
    runtime.acquire("legs", duration_ticks=1)
    with pytest.raises(ValueError, match="nonfinite"):
        runtime.step()
    assert world.steps == 0
    assert runtime.status == EpisodeStatus.FAILED
    with pytest.raises(RuntimeError, match="failed"):
        runtime.step()


def test_controller_cannot_write_another_owners_joint():
    world, controllers, runtime = build()
    runtime.reset()
    controllers[0].step = lambda dt: {"arm": 1.0}
    runtime.acquire("legs", duration_ticks=1)
    with pytest.raises(ValueError, match="owned joints"):
        runtime.step()
    assert world.steps == 0
    assert world.state["arm"] == 0.0


def test_invalid_declarations_fail_before_episode():
    world = FakeWorld(JOINTS)
    with pytest.raises(ValueError, match="Unknown joints"):
        EpisodeRuntime(world, (PositionController("bad", ("missing",)),), mode=RunMode.AUTONOMOUS)
    with pytest.raises(ValueError, match="unique"):
        ControllerDeclaration("bad", ("leg", "leg"))
    controller = PositionController("same", ("leg",))
    with pytest.raises(ValueError, match="Duplicate"):
        EpisodeRuntime(world, (controller, controller), mode=RunMode.AUTONOMOUS)


@pytest.mark.parametrize("ticks", [-1, 1.5, True])
def test_invalid_timing_is_rejected(ticks):
    _, _, runtime = build()
    with pytest.raises(ValueError):
        runtime.reset(settling_ticks=ticks)
    runtime.reset()
    with pytest.raises(ValueError):
        runtime.acquire("legs", duration_ticks=ticks)


def test_backend_failure_is_terminal():
    world, _, runtime = build()
    runtime.reset()

    def fail():
        raise RuntimeError("physics failed")

    world.step = fail
    with pytest.raises(RuntimeError, match="physics failed"):
        runtime.step()
    assert runtime.status == EpisodeStatus.FAILED
    assert runtime.tick == 0


def test_complete_prevents_further_execution():
    _, _, runtime = build()
    with pytest.raises(RuntimeError, match="ready"):
        runtime.step()
    runtime.reset()
    runtime.complete("diagnostic complete")
    with pytest.raises(RuntimeError, match="completed"):
        runtime.step()


@pytest.mark.parametrize("phase", ["reset", "acquire", "release"])
def test_lifecycle_failure_terminates_episode(phase):
    world, controllers, runtime = build()
    runtime.reset()

    def fail(*args):
        raise RuntimeError("lifecycle failure")

    if phase == "reset":
        world.restore = fail
        operation = runtime.reset
    elif phase == "acquire":
        controllers[0].reset = fail
        operation = partial(runtime.acquire, "legs", duration_ticks=1)
    else:
        runtime.acquire("legs", duration_ticks=1)
        world.apply = fail
        operation = partial(runtime.release, "legs")
    with pytest.raises(RuntimeError, match="lifecycle failure"):
        operation()
    assert runtime.status == EpisodeStatus.FAILED
    with pytest.raises(RuntimeError, match="failed"):
        runtime.step()


class TimedStop(PositionController):
    def __init__(self, ticks):
        super().__init__("timed", ("leg",))
        self.declaration = ControllerDeclaration("timed", ("leg",), stop_ticks=5)
        self.ticks = ticks

    def stop(self):
        self.stopped = True

    def stop_duration_ticks(self):
        return self.ticks


def test_a_timed_stop_replaces_the_declared_interval():
    runtime = EpisodeRuntime(FakeWorld(JOINTS), (TimedStop(12),), mode=RunMode.AUTONOMOUS)
    runtime.reset()
    runtime.acquire("timed", duration_ticks=100)
    runtime.request_stop("timed")
    for _ in range(11):
        runtime.step()
    assert runtime.controller_state("timed") == "stopping"
    runtime.step()
    runtime.step()
    assert runtime.controller_state("timed") == "idle"


def test_an_invalid_timed_stop_fails_the_episode():
    runtime = EpisodeRuntime(FakeWorld(JOINTS), (TimedStop(0),), mode=RunMode.AUTONOMOUS)
    runtime.reset()
    runtime.acquire("timed", duration_ticks=100)
    with pytest.raises(ValueError, match="invalid stop interval"):
        runtime.request_stop("timed")
    assert runtime.status == EpisodeStatus.FAILED
