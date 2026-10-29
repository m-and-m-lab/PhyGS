# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The agent session: calls on the main thread, simulated time only while waiting, the audit
log, privileged hints by grant, and the episode ending under a waiting agent (no Isaac Sim)."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass

import numpy as np
import pytest
from fakes import SPOT

from phygs.session.hints import PrivilegedHints
from phygs.session.session import (
    AgentSession,
    Audit,
    Command,
    EpisodeEnded,
    MainThread,
    ObjectTarget,
    PublicTask,
)

DT = 0.01


@dataclass(frozen=True)
class Feedback:
    status: str
    reason: str = ""


class World:
    """A fake simulation: a clock and the thread that steps it."""

    def __init__(self):
        self.time, self.thread = 0.0, None

    def step(self):
        assert threading.current_thread() is self.thread, "only the main thread steps"
        self.time += DT


class FakeNavigation:
    """navigate_to takes 0.5 s of simulated time; it records which thread ran each call."""

    def __init__(self, world):
        self.world, self.ends, self.threads = world, {}, []
        self.navigation = object()  # A backend object an agent must not reach.
        self.active = False

    def navigate_to(self, x, y, heading=None):
        self.threads.append(threading.current_thread())
        command = len(self.ends) + 1
        self.ends[command] = self.world.time + 0.5
        return command

    def feedback(self, command_id):
        return Feedback("active" if self.world.time < self.ends[command_id] - 1e-9 else "succeeded")

    def update(self):
        raise AssertionError("agents must not tick clients")


class FakeManipulation:
    def __init__(self):
        self.targets = []

    def look_at(self, target, **options):
        self.targets.append(target)
        raise RuntimeError("no plan")


class FakeScene:
    def object_box(self, name):
        return (1.0, 2.0, 0.5), (0.04, 0.04, 0.12)

    def placement(self, name):
        return "drawer", None

    def part(self, name):
        return f"part {name}"


def fake_boxes():
    return {"/World/objects/table/box": (np.array([0.85, 0.0, 0.2]), np.array([0.5, 1.0, 0.4]))}


def make_session(tmp_path, grants=("object_targets", "scene_hints")):
    world, main = World(), MainThread()
    navigation, manipulation = FakeNavigation(world), FakeManipulation()
    audit = Audit(tmp_path / "audit.jsonl", lambda: world.time)
    finished = []
    session = AgentSession(
        task=PublicTask("retrieve", "the small green box"),
        robot=SPOT.info(),
        grants=grants,
        clients=dict(navigation=navigation, manipulation=manipulation, locomotion=None, arm=None),
        hints=PrivilegedHints(FakeScene(), "target", grants, boxes=fake_boxes),
        observe=lambda: dict(time=world.time),
        time_seconds=lambda: world.time,
        finish=finished.append,
        camera_period=0.1,
        main=main,
        audit=audit,
        privileged=lambda client, member, args, kwargs: [
            "object_targets" for value in (*args, *kwargs.values()) if isinstance(value, ObjectTarget)
        ],
    )
    return world, main, session, navigation, manipulation, audit, finished


def run(world, main, agent, *, clock="charged", end_after=None):
    """The World's loop: serve requests; step only while the agent waits (charged clock)."""
    world.thread = threading.current_thread()
    outcome = {}

    def target():
        try:
            outcome["result"] = agent()
        except BaseException as error:
            outcome["error"] = error

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    while thread.is_alive():
        main.serve()
        if end_after is not None and world.time >= end_after:
            main.end("budget")
            break
        if clock == "realtime" or main.waiting:
            world.step()
            main.wake()
        else:
            main.serve(timeout=0.005)
    thread.join(timeout=5)
    assert not thread.is_alive()
    return outcome


def test_calls_run_on_the_main_thread_and_time_passes_only_in_wait(tmp_path):
    world, main, session, navigation, _, _, _ = make_session(tmp_path)
    seen = {}

    def agent():
        seen["before"] = session.time()
        command = session.navigation.navigate_to(1.0, 2.0)
        seen["after_call"] = session.time()  # Charged clock: no time passed while the agent did not wait.
        seen["command"] = command
        seen["feedback"] = session.wait(command)
        seen["after_wait"] = session.time()

    assert "error" not in run(world, main, agent)
    assert navigation.threads == [world.thread]
    assert isinstance(seen["command"], Command) and seen["command"].client == "navigation" and seen["command"] == 1
    assert seen["before"] == seen["after_call"] == 0.0
    assert seen["feedback"] == Feedback("succeeded")
    assert seen["after_wait"] == pytest.approx(0.5)


def test_sleep_and_stream_ask_for_simulated_time(tmp_path):
    world, main, session, *_ = make_session(tmp_path)
    seen = {}

    def agent():
        session.sleep(0.25)
        seen["slept"] = session.time()
        seen["packets"] = [packet["time"] for packet in session.stream(seconds=0.3)]

    assert "error" not in run(world, main, agent)
    assert seen["slept"] == pytest.approx(0.25)
    assert seen["packets"] == pytest.approx([0.35, 0.45, 0.55])


def test_agents_see_only_public_client_members(tmp_path):
    world, main, session, *_ = make_session(tmp_path)
    errors = {}

    def agent():
        assert session.navigation.active is False  # Plain data is readable.
        assert session.locomotion is None and session.arm is None  # Not in the tier.
        for member in (
            "update",
            "navigation",
            "_backend",
            "backend",
            "controller",
        ):  # The last two: trusted diagnostics.
            try:
                getattr(session.navigation, member)
            except AttributeError as error:
                errors[member] = str(error)
        try:
            session.navigation.active = True
        except AttributeError as error:
            errors["set"] = str(error)

    assert "error" not in run(world, main, agent)
    assert set(errors) == {"update", "navigation", "_backend", "backend", "controller", "set"}
    assert all("not available to agents" in errors[member] for member in ("backend", "controller"))


def test_errors_reach_the_agent_and_every_call_is_audited(tmp_path):
    world, main, session, _, manipulation, audit, _ = make_session(tmp_path)
    seen = {}

    def agent():
        try:
            session.manipulation.look_at(ObjectTarget("target"))
        except RuntimeError as error:
            seen["error"] = str(error)
        session.wait(session.navigation.navigate_to(0.0, 1.0))
        seen["hint"] = session.hints.target()
        seen["part"] = session.hints.part("drawer")

    assert "error" not in run(world, main, agent)
    audit.close()
    assert seen["error"] == "no plan" and manipulation.targets == [ObjectTarget("target")]
    assert seen["hint"].inside == "drawer" and seen["part"] == "part drawer"
    lines = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert [line["call"] for line in lines] == [
        "manipulation.look_at",
        "navigation.navigate_to",
        "navigation.wait",
        "hints.target",
        "hints.part",
    ]
    assert lines[0]["error"] == "RuntimeError: no plan" and lines[0]["privileged"] == ["object_targets"]
    assert lines[2]["feedback"] == {"Feedback": {"status": "succeeded", "reason": ""}}
    assert lines[3]["privileged"] == ["object_targets"] and lines[4]["privileged"] == ["scene_hints"]
    summary = audit.summary()
    assert summary["privileged_reads"] == {"object_targets": 2, "scene_hints": 1}
    assert summary["levels_used"] == ["skills"] and summary["by_client"]["navigation"] == 2
    assert len(audit.commands) == 1  # The waited command, with its final feedback.


def test_hints_refuse_what_the_spec_does_not_grant(tmp_path):
    world, main, session, *_ = make_session(tmp_path, grants=("scene_hints",))
    seen = {}

    def agent():
        for call in (session.hints.target, session.hints.boxes):
            try:
                call()
            except PermissionError as error:
                seen.setdefault("refused", []).append(str(error))
        seen["part"] = session.hints.part("drawer")  # Granted.
        try:
            session.hints.stance  # Stances are the robot tests' own data, not a hint.
        except AttributeError as error:
            seen["no stances"] = str(error)

    assert "error" not in run(world, main, agent)
    assert "object_targets grant" in seen["refused"][0] and "known_map grant" in seen["refused"][1]
    assert seen["part"] == "part drawer" and "no stances" in seen


def test_the_episode_ending_releases_a_waiting_agent(tmp_path):
    world, main, session, *_ = make_session(tmp_path)

    def agent():
        session.sleep(10.0)

    outcome = run(world, main, agent, end_after=0.2)
    assert isinstance(outcome["error"], EpisodeEnded) and str(outcome["error"]) == "budget"
    with pytest.raises(EpisodeEnded):
        session.observe()


def test_finish_and_the_realtime_clock(tmp_path):
    world, main, session, _, _, _, finished = make_session(tmp_path)
    seen = {}

    def agent():
        start = session.time()
        while session.time() - start < 0.05:  # Realtime: time passes while the agent works.
            pass
        seen["moved"] = session.time() > start
        session.finish("done")

    assert "error" not in run(world, main, agent, clock="realtime")
    assert seen["moved"] and finished == ["done"]


def test_wait_takes_only_session_command_ids(tmp_path):
    world, main, session, *_ = make_session(tmp_path)
    outcome = run(world, main, lambda: session.wait(1))
    assert isinstance(outcome["error"], TypeError)


def test_feedback_that_carries_the_session_command_id_is_audited():
    """A running skill's feedback holds the id the agent passed, a session Command; the audit copies
    feedback to JSON and could not copy a Command (the depth example agent crashed, 2026-10-25)."""
    import copy

    from phygs.session.session import Command, jsonable

    @dataclass(frozen=True)
    class Feedback:
        command_id: int
        status: str

    command = Command(7, "manipulation")
    assert copy.deepcopy(command) == 7 and copy.deepcopy(command).client == "manipulation"
    assert jsonable(Feedback(command, "active")) == {"Feedback": {"command_id": 7, "status": "active"}}
