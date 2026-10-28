# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The robot UI's episode as the Environment's agent: Start, End, E-stop and an episode ended from outside."""

from ui_fakes import START, Navigation, Ui, fake_control

from phygs.ui.bridge import Bridge

TASK = {"goal": "retrieve", "description": "the small green box"}


def episode(budget=420.0):
    control = fake_control()  # Its clock reads 12.5 s.
    control.navigation = Navigation()
    bridge = Bridge({"views": ["front", "hand"]})
    return control, bridge, Ui(control, bridge, cameras=(), task=TASK, budget_seconds=budget)


def events(bridge, kind):
    return [event for _, event in bridge.events_after(0) if event["type"] == kind]


def test_nothing_reaches_the_robot_before_start_and_end_finishes_the_episode():
    control, bridge, ui = episode()
    bridge.request({"type": "run", "skill": "navigate_to", "inputs": {"x": 1.0, "y": 0.0}})
    ui.update()
    # No session call yet: the Environment holds its clock until the agent's first call.
    assert ui.phase == "ready" and control.navigation.goals == [] and ui.robot.time_calls == 0
    refused = events(bridge, "command")[-1]["feedback"]["reason"]
    assert refused == "RuntimeError: The episode has not started; press Start."
    state = bridge.state()[1]
    assert state["phase"] == "ready" and state["task"] == {**TASK, "budget": 420.0, "elapsed": 0.0}
    bridge.request(START)
    bridge.request({"type": "run", "skill": "navigate_to", "inputs": {"x": 1.0, "y": 0.0}})
    ui.update()
    assert ui.phase == "running" and len(control.navigation.goals) == 1 and ui.robot.time_calls == 1
    bridge.request({"type": "episode", "action": "end"})
    ui.update()
    assert ui.phase == "ended" and ui.robot.finished == ["Ended from the UI"]
    ended = events(bridge, "episode")[-1]
    assert ended["outcome"] == "ended" and ended["report"]["commands"]["navigate_to"] == {"failed": 1}
    assert ended["report"]["failures"]["navigate_to"] == {"rejected": 1}  # The one sent before Start.
    bridge.request({"type": "run", "skill": "stop", "inputs": {}})
    ui.update()
    assert events(bridge, "command")[-1]["feedback"]["reason"] == "RuntimeError: The episode has ended."
    # The Environment's report follows: its evaluation on the result card.
    shown = ui.show_report({"sim_seconds": 30.0, "evaluation": {"outcome": "success"}, "ended_by": "finish"})
    assert shown["evaluation"] == {"outcome": "success"} and shown["sim_seconds"] == 30.0
    assert events(bridge, "episode")[-1]["report"] == shown


def test_an_episode_ended_by_the_environment_ends_the_page():
    control, bridge, ui = episode(budget=10.0)
    bridge.request(START)
    ui.update()
    ui.robot.ended = "budget"  # The Environment spent the budget: the next call raises EpisodeEnded.
    bridge.request({"type": "run", "skill": "navigate_to", "inputs": {"x": 1.0, "y": 0.0}})
    ui.update()
    assert ui.phase == "ended" and events(bridge, "episode")[-1]["outcome"] == "budget"
    assert not ui.agent_step(ui.robot)  # The agent returns.


def test_estop_stops_the_base_and_finishes_the_episode_as_interrupted():
    control, bridge, ui = episode()
    stops = []
    control.locomotion.stop = lambda: stops.append("stop") or 7
    bridge.request(START)
    bridge.request({"type": "estop"})
    ui.update()
    assert ui.phase == "ended" and events(bridge, "episode")[-1]["outcome"] == "interrupted"
    assert stops == ["stop"] and ui.robot.finished == ["E-stop from the UI"]
    assert ui.summary()["reason"] == "E-stop from the UI"
    # A failing step is shown, and the episode still ends.
    control, bridge, ui = episode()
    bridge.request(START)
    bridge.request({"type": "estop"})
    ui.update()  # The fake base has no stop(): the error is shown.
    assert ui.robot.finished == ["E-stop from the UI"]
    assert "E-stop could not stop the base" in events(bridge, "error")[-1]["message"]
