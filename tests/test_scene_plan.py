# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene floor plans: the pixel mapping, the release's plan before the cache, refusals (no Isaac Sim)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from phygs.scene import plan as plans
from phygs.scene.plan import Floor, Plan

SHA = "ab" * 32


def floor(name="ground", elevation=0.0):
    return Floor(name, f"{name}.png", elevation, elevation + 2.4, 1800, 1240, (11.2, 5.6), 64.42)


def save(folder, *floors):
    folder.mkdir(parents=True, exist_ok=True)
    for entry in floors:
        Image.new("RGB", (8, 8)).save(folder / entry.image)
    return Plan(folder, floors).save()


def test_pixels_and_world_points_map_both_ways():
    ground = floor()
    assert tuple(map(float, ground.pixel(11.2, 5.6))) == (900.0, 620.0)  # The centre.
    column, row = ground.pixel(12.2, 6.6)
    assert (float(column), float(row)) == pytest.approx((900 + 64.42, 620 - 64.42))  # +x right, +y up.
    xs, ys = np.array([0.0, 3.0, 20.0]), np.array([-1.0, 4.0, 9.0])
    back = ground.world(*ground.pixel(xs, ys))
    assert np.allclose(back[0], xs) and np.allclose(back[1], ys)


def test_a_plan_reads_back_and_picks_the_floor_under_a_point(tmp_path):
    save(tmp_path / "p", floor("upper", 3.0), floor())
    plan = Plan.load(tmp_path / "p")
    assert [f.name for f in plan.floors] == ["ground", "upper"]  # Lowest first.
    assert plan.floor_at(0.5).name == "ground" and plan.floor_at(3.6).name == "upper"
    assert plan.floor_at(-2.0).name == "ground"
    assert plan.image_path(plan.floors[1]) == tmp_path / "p" / "upper.png"
    assert plan.floors[0] == floor()


def test_the_release_plan_comes_before_the_cache(tmp_path, monkeypatch):
    usd = tmp_path / "release" / "scene.usdc"
    usd.parent.mkdir()
    usd.touch()
    monkeypatch.setenv("PHYGS_PLANS", str(tmp_path / "cache"))
    assert plans.find(usd, SHA) is None
    save(plans.cached_folder(SHA), floor())
    assert plans.find(usd, SHA).folder == tmp_path / "cache" / SHA
    save(usd.parent / "plan", floor("ground"), floor("upper", 3.0))
    assert len(plans.find(usd, SHA).floors) == 2  # The release's.
    with pytest.raises(ValueError):
        plans.cached_folder("../escape")


def test_bad_plans_are_refused(tmp_path):
    save(tmp_path / "p", floor())
    data = json.loads((tmp_path / "p" / "plan.json").read_text())
    for change, error in (
        (dict(schema=2), ValueError),
        (dict(floors=[]), ValueError),
        (dict(floors=[dict(data["floors"][0], image="../x.png")]), ValueError),
        (dict(floors=[dict(data["floors"][0], image="missing.png")]), FileNotFoundError),
        (dict(floors=[dict(data["floors"][0], ppm=0)]), ValueError),
    ):
        (tmp_path / "p" / "plan.json").write_text(json.dumps(dict(data, **change)))
        with pytest.raises(error):
            Plan.load(tmp_path / "p")


def test_fit_frames_the_bounds_with_their_margin():
    center, ppm = plans.fit((0.0, 0.0, 20.0, 10.0), 1800, 1000, 1.0)
    assert center == (10.0, 5.0) and ppm == pytest.approx(min(1800 / 22.0, 1000 / 12.0))
