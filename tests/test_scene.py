# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene loading reads drawers, colliders and where objects start from the USD itself (stage
reads only, no Isaac Sim): on the grid storage scene file and on the house."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pxr import Usd

from phygs.episodes.spec import EpisodeSpec
from phygs.simulation import scene

EPISODES = Path(__file__).resolve().parents[1] / "tests" / "robot_tests" / "configs" / "episodes"
GRID_DRAWER = EpisodeSpec.load(EPISODES / "dev_grid_drawer.yaml")
GRID = GRID_DRAWER.scene.usd
HOUSE = EpisodeSpec.load(EPISODES / "dev_house_drawer.yaml").scene.usd
MAP = dict(resolution=0.05, clearance=0.0, height=(0.05, 1.2), margin=2.0)


def bodies(stage):
    return {scene.name_of(path): path for path in scene.rigid_bodies(stage)}


def test_the_grid_file_has_its_objects_and_drawers_by_name():
    stage = Usd.Stage.Open(str(GRID))
    assert set(bodies(stage)) == {
        "table", "target", "clutter", "corner_nightstand", "corner_nightstand_drawer", "corner_nightstand_tier_2_drawer"
    }  # fmt: skip
    found = scene.drawers(stage)
    assert set(found) == {"corner_nightstand_drawer", "corner_nightstand_tier_2_drawer"}
    assert found["corner_nightstand_tier_2_drawer"].handle.endswith("/handle")
    owners = {body for _, body, _, _ in scene.collider_boxes(stage)}
    assert "/World/objects/table" in owners and None not in owners  # Every grid collider belongs to a body.


def test_where_an_object_starts_is_found_and_an_initial_state_moves_it():
    stage = Usd.Stage.Open(str(GRID))
    found, named = scene.drawers(stage), bodies(stage)
    assert scene.placement(stage, named["target"], found) == (None, "table")
    scene.apply_initial_state(stage, named, GRID_DRAWER.objects)  # The episode moves it into the drawer.
    assert scene.placement(stage, named["target"], found) == ("corner_nightstand_tier_2_drawer", None)
    center, size = scene.body_box(stage, named["target"])
    assert size == pytest.approx([0.04, 0.04, 0.12])
    drawer = scene.world_pose(stage, named["corner_nightstand_tier_2_drawer"])
    assert (np.linalg.inv(drawer) @ [*center, 1.0])[:3] == pytest.approx([0.0, -0.134, 0.0725], abs=1e-6)


def test_a_drawer_interior_has_a_ceiling():
    """An object in the top drawer is not inside the bottom drawer below it."""
    stage = Usd.Stage.Open(str(GRID))
    found = scene.drawers(stage)
    lower, upper = found["corner_nightstand_drawer"], found["corner_nightstand_tier_2_drawer"]
    bottom, top = scene.drawer_interior(stage, lower), scene.drawer_interior(stage, upper)
    assert all(low < high for low, high in zip(bottom[0::2], bottom[1::2]))
    middle = [(low + high) / 2 for low, high in zip(top[0::2], top[1::2])]  # Middle of the top drawer, its frame.
    world = scene.world_pose(stage, upper.body) @ [*middle, 1.0]
    below = (np.linalg.inv(scene.world_pose(stage, lower.body)) @ world)[:3]
    assert scene.inside_interior(top, middle)
    assert bottom[0] < below[0] < bottom[1] and bottom[2] < below[1] < bottom[3]  # Right above the bottom drawer...
    assert not scene.inside_interior(bottom, below)  # ...but not inside it.


@pytest.mark.skipif(not HOUSE.exists(), reason="the house scene file is not present")
def test_the_house_target_and_what_rides_with_it_start_in_the_island_drawer():
    stage = Usd.Stage.Open(str(HOUSE))
    found, named = scene.drawers(stage), bodies(stage)
    assert len(named) == len(scene.rigid_bodies(stage))  # Body names are unique.
    assert "kitchen_island_drawer" in found
    for item in ("kitchen_island_compartment_0_item_1", "kitchen_island_compartment_0_item_0"):
        assert scene.placement(stage, named[item], found) == ("kitchen_island_drawer", None)


def test_the_map_covers_the_scene_colliders_and_the_robot_start():
    """Bounds from the USD: the colliders in the height band and the start, grown by the margin, on whole cells."""
    stage = Usd.Stage.Open(str(GRID))
    bounds = scene.map_bounds(stage, MAP["height"], MAP["resolution"], MAP["margin"], (-1.6, -0.2))
    assert bounds == pytest.approx((-3.6, -2.5, 3.1, 5.2))  # Furniture x -0.28..1.1, y -0.5..3.19; start x -1.6.
    assert all(abs(value / 0.05 - round(value / 0.05)) < 1e-6 for value in bounds)
    far = scene.map_bounds(stage, MAP["height"], MAP["resolution"], MAP["margin"], (6.0, 0.0))
    assert far[2] == pytest.approx(8.0)  # A start outside the furniture widens the map.


def test_a_scene_without_its_own_physics_scene_is_layered_onto_the_stage():
    stage = Usd.Stage.CreateInMemory()
    options = dict(ground=True, objects={}, navigation_map=MAP, robot_start=(-1.6, -0.2, 0.0))
    grid = scene.UsdScene("grid_storage", GRID, **options)
    grid._load(stage)
    assert stage.GetRootLayer().subLayerPaths == [grid.usd] and stage.GetPrimAtPath("/World/objects/target").IsValid()
    grid._load(stage)  # Already there: not added twice.
    assert len(stage.GetRootLayer().subLayerPaths) == 1
    if HOUSE.exists():
        house = scene.UsdScene("house", HOUSE, **{**options, "ground": False})
        with pytest.raises(RuntimeError, match="open it as the stage"):
            house._load(Usd.Stage.CreateInMemory())
