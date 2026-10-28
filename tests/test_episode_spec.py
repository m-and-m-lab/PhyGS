# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode specs (self-contained: scene, starts, task, grants), the robot tests' stances and agent
references (no Isaac Sim)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from phygs.episodes.spec import ACTION_SETS, ACTIONS, EpisodeSpec, load_agent
from phygs_robot_tests.tasks import load_task

PROJECT = Path(__file__).resolve().parents[1]
EPISODES = PROJECT / "phygs_robot_tests" / "configs" / "episodes"
STANCES = PROJECT / "phygs_robot_tests" / "configs" / "stances"
SPECS = sorted(EPISODES.glob("*.yaml")) + sorted((PROJECT / "phygs" / "configs" / "ui").glob("*_drawer.yaml"))


def spec_data(**changes) -> dict:
    data = dict(
        schema=3,
        id="test",
        scene=dict(id="grid_storage", ground=True),
        robot=dict(model="spot", start=[-1.6, -0.2, 0.0]),
        task=dict(goal="retrieve", description="the small green box", target="target"),
        actions="skills",
        observations=dict(cameras=["hand"]),
        grants=["object_targets"],
        budget=dict(sim_seconds=60),
        seed=3,
    )
    data.update(changes)
    return {key: value for key, value in data.items() if value is not None}


@pytest.mark.parametrize("path", SPECS, ids=lambda path: path.stem)
def test_every_episode_spec_loads_with_its_scene_file(path):
    spec = EpisodeSpec.load(path)
    assert spec.id.removeprefix("ui_") == path.stem
    assert spec.actions and set(spec.actions) <= set(ACTIONS)
    assert spec.scene.usd.is_absolute() and spec.scene.usd.is_file(), spec.scene.usd  # Found by the scene's id.
    load_task(spec.task)  # Every task block builds its task object.


def test_a_spec_runs_with_another_robot_from_its_own_start_and_its_cameras_by_role():
    from phygs.robots.description import load_robot

    data = spec_data(
        robot=dict(model="spot", start=[0.0, 0.0, 0.0], starts={"ridgeback_franka": [-0.15, 0.0, 0.0]}),
        observations=dict(cameras=["hand"], camera_roles=["planning"]),
    )
    spec = EpisodeSpec.from_dict(data)
    assert spec.with_robot("spot").robot.start == (0.0, 0.0, 0.0)  # No entry of its own: the start.
    other = spec.with_robot("ridgeback_franka")
    assert other.robot.model == "ridgeback_franka" and other.robot.start == (-0.15, 0.0, 0.0)
    assert spec.observations.for_robot(load_robot("spot")) == ("hand", "frontleft", "frontright")
    assert spec.observations.for_robot(load_robot("ridgeback_franka")) == ("hand", "front")
    with pytest.raises(ValueError, match="start"):
        EpisodeSpec.from_dict(spec_data(robot=dict(model="spot", start=[0, 0, 0], starts={"other": [0, 0]})))


def test_a_spec_maps_a_named_action_set_to_actions_and_keeps_its_public_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("PHYGS_SCENES", str(tmp_path))
    (tmp_path / "grid_storage").mkdir()
    (tmp_path / "grid_storage" / "scene.usda").write_text("#usda 1.0\n")
    spec = EpisodeSpec.from_dict(spec_data())
    assert spec.scene.usd == tmp_path / "grid_storage" / "scene.usda" and spec.robot.start == (-1.6, -0.2, 0.0)
    assert spec.scene.id == spec.scene.name == "grid_storage"
    assert spec.robot.model == "spot" and spec.summary()["robot"] == "spot"
    mine = EpisodeSpec.from_dict(
        spec_data(robot=dict(model="robots/mine/robot.yaml", start=[0, 0, 0])), base=Path("/specs")
    )
    assert mine.robot.model == "/specs/robots/mine/robot.yaml"  # A robot.yaml path: resolved against the spec.
    assert spec.objects == {} and spec.record_view is None
    assert spec.actions == ("navigation", "manipulation") and spec.observations.cameras == ("hand",)
    assert spec.with_seed(9).seed == 9
    summary = spec.summary()
    assert summary["scene_id"] == "grid_storage" and "goal" not in summary  # The task block is the task's.
    assert spec.task == dict(goal="retrieve", description="the small green box", target="target")
    custom = EpisodeSpec.from_dict(spec_data(actions=["motion", "arm"]))
    assert custom.actions == ("motion", "arm") and ACTION_SETS["skills"] == spec.actions
    house = EpisodeSpec.from_dict(spec_data(scene=dict(id="bear_loop_house", name="house", ground=False)))
    assert house.scene.name == "house"


def test_a_missing_scene_names_the_roots_searched(tmp_path, monkeypatch):
    monkeypatch.setenv("PHYGS_SCENES", str(tmp_path))
    spec = EpisodeSpec.from_dict(spec_data(scene=dict(id="moon", ground=True)))  # Validates without the file ...
    with pytest.raises(FileNotFoundError, match=f"moon.*{tmp_path}"):
        spec.scene.usd  # ... and fails on first use.


@pytest.mark.parametrize(
    "changes, message",
    [
        (dict(schema=0), "schema"),
        (dict(colour="red"), "Unknown spec fields"),
        (dict(tier="skills"), "replaced by actions"),
        (dict(actions="whole_body"), "not available yet"),
        (dict(actions="everything"), "Unknown action set"),
        (dict(actions=["fly"]), "subset"),
        (dict(grants=["god_mode"]), "Grants"),
        (dict(grants=["known_map", "known_map"]), "Grants"),
        (dict(scene=dict(name="moon", usd="moon.usda", ground=True)), "named by id"),
        (dict(scene=dict(id="moon")), "scene is"),
        (dict(scene=dict(id="../moon", ground=True)), "no '..'"),
        (dict(scene=dict(id="scenes/moon", ground=True)), "no '..'"),
        (dict(robot=dict(model="../robots/mine/robot.yaml", start=[0, 0, 0])), "climbs out"),
        (dict(robot=dict(model="spot", start=[0, 0])), "a model and a start"),
        (dict(robot=dict(model="", start=[0, 0, 0])), "a model and a start"),
        (dict(robot=dict(start=[0, 0, 0], height=0.5)), "model, start"),  # Schema 1's spawn height.
        (dict(objects=dict(target=dict(position=[0, 0]))), "position"),
        (dict(record_view=dict(eye=[0, 0, 1])), "record view"),
        (dict(initial_state="target_on_moon"), "Unknown spec fields"),
        (dict(budget=dict(sim_seconds=0)), "budget"),
        (dict(budget=dict(sim_seconds=60, tool_calls=10)), "one field"),
        (dict(task="retrieve"), "task is a mapping"),
        (dict(seed=1.5), "seed"),
    ],
)
def test_invalid_specs_are_refused_before_anything_starts(changes, message):
    with pytest.raises(ValueError, match=message):
        EpisodeSpec.from_dict(spec_data(**changes))


def test_load_names_the_file_in_errors(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("schema: 0\nid: x\n")
    with pytest.raises(ValueError, match="broken.yaml"):
        EpisodeSpec.load(path)


def test_the_routines_have_stances_for_every_episode_scene():
    """The robot tests' scripted plans stand where configs/stances/<scene>.yaml says (their own data,
    not served to agents); every retrieve episode's scene has a file, every file a scene, and every
    stance is (x, y, heading). The navigation courses need none: their plans walk to given goals."""
    specs = [EpisodeSpec.load(path) for path in sorted(EPISODES.glob("*.yaml"))]
    files = {path.stem for path in STANCES.glob("*.yaml")}
    assert {spec.scene.name for spec in specs if spec.task["goal"] == "retrieve"} <= files
    assert files <= {spec.scene.name for spec in specs}
    for path in STANCES.glob("*.yaml"):
        for place, uses in yaml.safe_load(path.read_text())["stances"].items():
            assert "open" not in uses, place  # The open stance is derived from the drawer (routines).
            for stance in uses.values():  # (x, y, heading) [m, m, rad].
                assert len(stance) == 3 and all(math.isfinite(value) for value in stance), (place, stance)


@pytest.mark.parametrize(
    "block, message",
    [
        (dict(goal="retrieve", description="a box"), "all given"),
        (dict(goal="retrieve", description="a box", target=""), "all given"),
        (dict(goal="deliver", description="a box", target="target"), "Unknown goal"),
        (dict(goal="none", target="target"), "no other fields"),
        (dict(description="a box"), "needs a goal"),
    ],
)
def test_task_blocks_are_validated_by_the_task_package_not_the_spec(block, message):
    EpisodeSpec.from_dict(spec_data(task=block))  # The framework's spec takes any mapping ...
    with pytest.raises(ValueError, match=message):
        load_task(block)  # ... the task package refuses a bad one.


def test_agents_resolve_by_name_or_by_file(tmp_path):
    name, run = load_agent("template")
    assert name == "template" and run.__module__ == "phygs.agents.template"
    script = tmp_path / "mine.py"
    script.write_text("def go(session):\n    return 'went'\n")
    assert load_agent(f"{script}:go")[1](None) == "went"
    with pytest.raises(ValueError, match="No agent"):
        load_agent("nobody")
    with pytest.raises(ValueError, match="no function"):
        load_agent(f"{script}:run")
