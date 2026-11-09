# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The robot-test ladder file (tests/robot_tests/ladder.json) and how run_suite.py selects from it."""

from pathlib import Path

import pytest
import yaml
from phygs_robot_tests import run_suite

ROOT = Path(run_suite.__file__).parent
ENTRIES = run_suite.ladder()

# The suite's selections and test arguments before ladder.json (run_suite.py's LADDER and GROUPS, 2026-11-09).
SELECTIONS = [  # (groups, --from, --upto, --only, the suite names run, in order)
    (
        ["ladder"],
        1,
        15,
        None,
        [
            "scene_reset",
            "sensors",
            "locomotion",
            "footprint",
            "arm",
            "navigation_course",
            "navigation_house",
            "obstacle_avoidance",
            "arm_planning_known",
            "arm_planning_known_moveit",
            "arm_planning_depth",
            "arm_planning_depth_moveit",
            "pick_place_known",
            "pick_place_known_moveit",
            "pick_place_depth",
            "pick_place_depth_moveit",
            "drawer_grid",
            "drawer_house",
            "look_table",
            "look_grid",
            "look_house",
            "drawer_place",
            "search_house",
            "retrieve_grid",
            "retrieve_house",
            "retrieve_repeat",
            "entry_point",
        ],
    ),
    (["base"], 1, 15, None, ["scene_reset", "sensors", "locomotion", "footprint"]),
    (["nav2"], 1, 15, None, ["navigation_course", "navigation_house", "obstacle_avoidance", "search_house"]),
    (
        ["manipulation"],
        1,
        15,
        None,
        [
            "arm",
            "arm_planning_known",
            "arm_planning_known_moveit",
            "arm_planning_depth",
            "arm_planning_depth_moveit",
            "pick_place_known",
            "pick_place_known_moveit",
            "pick_place_depth",
            "pick_place_depth_moveit",
        ],
    ),
    (
        ["episodes"],
        1,
        15,
        None,
        [
            "drawer_grid",
            "drawer_house",
            "look_table",
            "look_grid",
            "look_house",
            "drawer_place",
            "retrieve_grid",
            "retrieve_house",
            "retrieve_repeat",
            "entry_point",
        ],
    ),
    (["ladder"], 1, 1, None, ["scene_reset"]),
    (
        ["ladder"],
        6,
        10,
        None,
        [
            "navigation_course",
            "navigation_house",
            "obstacle_avoidance",
            "arm_planning_known",
            "arm_planning_known_moveit",
            "arm_planning_depth",
            "arm_planning_depth_moveit",
            "pick_place_known",
            "pick_place_known_moveit",
            "pick_place_depth",
            "pick_place_depth_moveit",
        ],
    ),
    (
        ["base", "nav2"],
        1,
        6,
        None,
        ["scene_reset", "sensors", "locomotion", "footprint", "navigation_course", "navigation_house"],
    ),
    (["manipulation"], 1, 15, ["pick_place_known"], ["pick_place_known"]),
    (["episodes"], 14, 15, ["retrieve_house", "entry_point"], ["retrieve_house", "entry_point"]),
]
ARGUMENTS = {
    "scene_reset": [],
    "sensors": [],
    "locomotion": [],
    "footprint": [],
    "arm": [],
    "navigation_course": ["--spec", "configs/episodes/nav_course.yaml"],
    "navigation_house": ["--spec", "configs/episodes/nav_house.yaml"],
    "obstacle_avoidance": ["--spec", "configs/episodes/nav_dynamic.yaml"],
    "arm_planning_known": ["--spec", "configs/episodes/table_reach.yaml"],
    "arm_planning_known_moveit": ["--spec", "configs/episodes/table_reach.yaml", "--planner", "moveit"],
    "arm_planning_depth": ["--spec", "configs/episodes/table_reach_depth.yaml"],
    "arm_planning_depth_moveit": ["--spec", "configs/episodes/table_reach_depth.yaml", "--planner", "moveit"],
    "pick_place_known": ["--spec", "configs/episodes/table_reach.yaml"],
    "pick_place_known_moveit": ["--spec", "configs/episodes/table_reach.yaml", "--planner", "moveit"],
    "pick_place_depth": ["--spec", "configs/episodes/table_reach_depth.yaml"],
    "pick_place_depth_moveit": ["--spec", "configs/episodes/table_reach_depth.yaml", "--planner", "moveit"],
    "drawer_grid": ["--spec", "configs/episodes/dev_grid_drawer.yaml"],
    "drawer_house": ["--spec", "configs/episodes/dev_house_drawer.yaml"],
    "look_table": ["--spec", "configs/episodes/table_reach_depth.yaml"],
    "look_grid": ["--spec", "configs/episodes/dev_grid_drawer.yaml"],
    "look_house": ["--spec", "configs/episodes/dev_house_drawer.yaml"],
    "drawer_place": [],
    "search_house": [],
    "retrieve_grid": ["--spec", "configs/episodes/dev_grid_drawer.yaml"],
    "retrieve_house": ["--spec", "configs/episodes/dev_house_drawer.yaml"],
    "retrieve_repeat": ["--spec", "configs/episodes/dev_grid_drawer.yaml", "--repeats", "3"],
    "entry_point": [],
}


def test_the_ladder_has_its_27_entries_over_rungs_1_to_15_in_build_order():
    rungs = [entry["rung"] for entry in ENTRIES]
    assert len(ENTRIES) == 27 and sorted(set(rungs)) == list(range(1, 16)) and rungs == sorted(rungs)
    assert len({entry["id"] for entry in ENTRIES}) == 27 and len({entry["name"] for entry in ENTRIES}) == 27
    assert all(entry["id"] == f"R{entry['rung']}.{entry['name']}" for entry in ENTRIES)


@pytest.mark.parametrize(("groups", "first", "upto", "only", "names"), SELECTIONS)
def test_the_suite_selects_what_it_selected_before_the_ladder_file(groups, first, upto, only, names):
    assert [entry["name"] for entry in run_suite.select(ENTRIES, groups, first, upto, only)] == names
    if upto == 15:  # The default --upto is the top rung.
        assert run_suite.select(ENTRIES, groups, first, None, only) == run_suite.select(
            ENTRIES, groups, first, 15, only
        )


def test_each_test_gets_the_arguments_it_got_before_the_ladder_file():
    given = {
        entry["name"]: [
            str(Path(a).relative_to(ROOT)) if isinstance(a, Path) else a for a in run_suite.arguments(entry)
        ]
        for entry in ENTRIES
    }
    assert given == ARGUMENTS


def test_every_path_and_dependency_in_the_ladder_exists():
    ids = {entry["id"] for entry in ENTRIES}
    for entry in ENTRIES:
        assert (ROOT / entry["test"]).is_file(), entry["id"]
        assert entry["spec"] is None or (run_suite.SPECS / f"{entry['spec']}.yaml").is_file(), entry["id"]
        criteria = ROOT / entry["criteria"]["file"]
        assert criteria.is_file(), entry["id"]
        if "keys" in entry["criteria"]:  # Keys of one thresholds file shared by several tests.
            assert set(entry["criteria"]["keys"]) <= set(yaml.safe_load(criteria.read_text())), entry["id"]
        assert set(entry["depends_on"]) <= ids - {entry["id"]}, entry["id"]
        assert set(entry["requires"]) <= {"ros2", "curobo", "explore"}, entry["id"]
        assert entry["level"] == ("component" if entry["test"].startswith("components/") else "agent"), entry["id"]


def test_evidence_is_a_run_folder_and_a_commit_only_with_a_pass():
    for entry in ENTRIES:
        assert entry["last_run"] is None or entry["last_run"].startswith("output_check/"), entry["id"]
        assert entry["passed_at_commit"] is None or (entry["passes"] and len(entry["passed_at_commit"]) == 40), entry[
            "id"
        ]


def test_the_readme_table_names_every_entry_at_its_rung():
    rows = {}
    for line in (ROOT / "README.md").read_text().splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) > 2 and cells[1].isdigit():
            rows.setdefault(int(cells[1]), "")
            rows[int(cells[1])] += cells[2]
    for entry in ENTRIES:
        assert f"`{entry['name']}`" in rows[entry["rung"]] or (
            entry["name"].endswith("_moveit") and f"`{entry['name'][: -len('_moveit')]}`" in rows[entry["rung"]]
        ), entry["id"]
