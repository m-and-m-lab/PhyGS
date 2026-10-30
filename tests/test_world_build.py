# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The World's build steps that need no simulator: the settings with a benchmark's overrides, the
combinations refused up front, the cameras captured, and who may read which observations."""

from __future__ import annotations

import pytest
import yaml

from phygs.recording.panels import PanelFeed
from phygs.robots.description import load_robot
from phygs.world import build
from phygs.world.build import SETTINGS
from phygs.world.spec import EpisodeSpec


def spec(**changes) -> EpisodeSpec:
    data = dict(
        schema=3,
        id="test",
        scene=dict(id="grid_storage", ground=True),
        robot=dict(model="spot", start=[-1.6, -0.2, 0.0]),
        task=dict(goal="none"),
        actions="skills",
        observations=dict(cameras=["hand"], joint_state=True),
        grants=["gt_localization", "known_map"],
        budget=dict(sim_seconds=60),
        seed=0,
    )
    data.update(changes)
    return EpisodeSpec.from_dict(data)


@pytest.fixture(scope="module")
def spot():
    return load_robot("spot")


def test_overrides_replace_a_key_and_merge_one_level_into_a_mapping(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(dict(end_settle_seconds=0.5, planner=dict(backend="curobo", cell=0.04))))
    settings = build.load_settings(path, {"end_settle_seconds": 2.0, "planner": {"backend": "moveit"}, "new": 1})
    assert settings == dict(end_settle_seconds=2.0, planner=dict(backend="moveit", cell=0.04), new=1)
    assert build.load_settings(path) == yaml.safe_load(path.read_text())  # The file is never changed.


def test_the_framework_settings_hold_every_key_a_benchmark_is_expected_to_set():
    settings = build.load_settings(SETTINGS)
    for key in ("stop_all_timeout_seconds", "end_settle_seconds", "contact_record_n", "agent_wall_seconds"):
        assert key in settings, key
    assert "backend" in settings["skills"] and "backend" in settings["planner"]


def test_unbuildable_combinations_are_refused_before_the_simulator_starts(spot):
    build.check_supported(spec(), spot)
    with pytest.raises(ValueError, match="Unknown cameras"):
        build.check_supported(spec(observations=dict(cameras=["periscope"])), spot)
    with pytest.raises(NotImplementedError, match="gt_localization"):
        build.check_supported(spec(grants=["known_map"]), spot)


def test_the_captured_cameras_are_the_agents_then_what_skills_video_and_viewer_need(spot):
    known = build.captured_cameras(spec(), spot, ("hand",))
    assert known[0] == "hand" and spot.look_camera in known and len(set(known)) == len(known)
    depth = build.captured_cameras(spec(grants=["gt_localization"]), spot, ("hand",))
    assert set(spot.cameras_for("planning")) | set(spot.cameras_for("navigation")) <= set(depth)
    recorded = build.captured_cameras(spec(), spot, ("hand",), record=True)
    assert set(PanelFeed.cameras(spot)) <= set(recorded)
    with pytest.raises(ValueError, match="viewer asks for unknown cameras"):
        build.captured_cameras(spec(), spot, ("hand",), viewer_cameras=("periscope",))


def test_the_agent_reads_only_what_the_spec_grants_and_the_others_read_what_they_need(spot):
    config, consumers = build.observation_pipeline(
        spec(grants=["object_targets"], actions=["manipulation"]),
        spot,
        ("hand",),
        ("hand", "frontleft"),
        dt=0.004,
        period=0.032,
        record=True,
        viewer_cameras=("frontleft",),
    )
    assert config.agent.localization == "none" and [c.name for c in config.agent.cameras] == ["hand"]
    assert [c.name for c in config.capture.cameras] == ["hand", "frontleft", "third_person"]
    assert [c.name for c in consumers] == ["skills", "viewer", "recorder"]
    assert config.privileged_sources == ()
    config, consumers = build.observation_pipeline(spec(), spot, ("hand",), ("hand",), dt=0.004, period=0.032)
    assert config.agent.localization == "ground_truth" and [c.name for c in consumers] == ["skills"]
    assert config.privileged_sources == ("privileged_known_box_geometry",)


def test_the_third_person_view_is_the_specs_or_behind_the_start(spot):
    eye, target = build.record_view(spec(record_view=dict(eye=[1, 2, 3], target=[0, 0, 0])))
    assert eye.tolist() == [1, 2, 3] and target.tolist() == [0, 0, 0]
    eye, target = build.record_view(spec())
    assert eye[0] < -1.6 < target[0] and eye[2] > target[2]  # Behind and above, looking ahead.
    camera = build.third_person_camera(dict(width=640, height=360), (eye, target))
    assert camera["width"] == 640 and len(camera["offset"]["rot"]) == 4
