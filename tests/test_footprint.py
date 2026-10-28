# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Geometry calibration and independent comparison measurements."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from phygs.execution.navigation.footprint import (
    FootprintSweep,
    convex_hull,
    load_footprint,
    path_error,
    transit_footprint,
)
from phygs.execution.navigation.map import RectangleFootprint
from phygs.execution.navigation.metrics import NavigationMeasurements
from phygs.simulation.maps import BoxGeometry, KnownBoxMap


def test_swept_hull_includes_motion_and_enclosing_radius():
    sweep = FootprintSweep()
    rectangle = np.array([[-1, -0.5], [1, -0.5], [1, 0.5], [-1, 0.5]])
    sweep.update(rectangle)
    sweep.update(rectangle + (0, 0.5))
    result = sweep.result()
    assert result["length"] == 2
    assert result["width"] == 1.5
    assert result["radius"] == pytest.approx(2**0.5)
    assert result["samples"] == 2
    with pytest.raises(ValueError, match="collinear"):
        convex_hull(np.array([[0, 0], [1, 1], [2, 2]]))


STANDING = dict(
    bounds=[-0.6, 0.6, -0.3, 0.3], padding=0.05, polygon=[[-0.6, -0.3], [0.6, -0.3], [0.6, 0.3], [-0.6, 0.3]]
)


def test_profile_rejects_changed_asset_and_understated_radius(tmp_path):
    asset, policy, profile = (tmp_path / name for name in ("robot", "policy", "profile"))
    asset.write_bytes(b"robot")
    policy.write_bytes(b"policy")
    data = dict(
        status="passed",
        schema=2,
        standing=copy.deepcopy(STANDING),
        asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
        policy_config_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
        envelope=dict(polygon=[[-1, -1], [1, -1], [0, 1]], radius=2),
    )
    profile.write_text(json.dumps(data))
    files = dict(policy_sha256=policy, policy_config_sha256=policy)  # The base's files, by profile key.
    assert load_footprint(profile, asset, files)["schema"] == 2
    assert transit_footprint(load_footprint(profile, asset, files)) == RectangleFootprint(-0.6, 0.6, -0.3, 0.3, 0.05)
    data["standing"]["bounds"][1] = 0.5  # Smaller than the recorded outline.
    profile.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Standing bounds"):
        load_footprint(profile, asset, files)
    data["standing"] = copy.deepcopy(STANDING)
    data["envelope"]["radius"] = 1
    profile.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="enclose"):
        load_footprint(profile, asset, files)
    asset.write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset_sha256"):
        load_footprint(profile, asset, files)
    asset.write_bytes(b"robot")
    with pytest.raises(ValueError, match="base_config_sha256"):  # A base file the profile has no hash for.
        load_footprint(profile, asset, dict(base_config_sha256=policy))


def test_integer_polyline_and_duplicate_vertices():
    assert path_error((1, 1), ((0, 0), (0, 0), (2, 0))) == 1
    assert path_error((3, 0), ((0, 0), (2, 0))) == 1


def test_external_map_has_no_robot_inflation_and_clearance_is_physical():
    maps = KnownBoxMap(
        SimpleNamespace(rigid_objects={}),
        (BoxGeometry((1, 1, 1), (2, 0, 0.5)),),
        bounds=(-3, -3, 5, 3),
        resolution=0.1,
        footprint=RectangleFootprint(-0.5, 0.5, -0.5, 0.5),
        clearance=0.2,
    )
    assert maps.raw_snapshot().free((1.0, 0.0))
    assert not maps.snapshot().free((1.0, 0.0))
    assert maps.physical_clearance(0, 0, 0.0) == pytest.approx(1)
    assert maps.physical_clearance(2, 0, 0.0) == pytest.approx(-1)


def test_measurements_are_based_on_physical_motion():
    measurements = NavigationMeasurements()
    measurements.update(SimpleNamespace(x=0, y=1, time=0), ((0, 0), (4, 0)), 0.5)
    measurements.update(SimpleNamespace(x=3, y=1, time=1), ((0, 0), (4, 0)), 0.3, stopping=True)
    result = measurements.result()
    assert result["travel_distance_m"] == 3
    assert result["tracking_error_rms_m"] == 1
    assert result["minimum_envelope_clearance_m"] == 0.3
    assert result["stopping_distance_m"] == 3
    assert result["stopping_time_seconds"] == 1


def test_navigation_rejects_settings_outside_calibration():
    from dataclasses import replace

    from fakes import SPOT, TRANSIT

    from phygs.execution.navigation.footprint import validate_transit
    from phygs.execution.navigation.types import NavigationConfig

    profile = json.loads(SPOT.navigation.footprint_profile.read_text())
    config = NavigationConfig(**SPOT.navigation.limits, arm_posture=TRANSIT)
    validate_transit(profile, config, 0.004)
    with pytest.raises(ValueError, match="timestep"):
        validate_transit(profile, config, 0.002)
    with pytest.raises(ValueError, match="speed"):
        validate_transit(profile, replace(config, max_speed=1.1), 0.004)
    with pytest.raises(ValueError, match="posture"):
        validate_transit(profile, replace(config, arm_posture=()), 0.004)


def test_collision_geometry_identity_does_not_depend_on_where_the_robot_stands():
    """Link-frame bounds are composed through world transforms; placement must not change the hash."""
    from types import SimpleNamespace

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from phygs.simulation.footprint import CollisionFootprint

    def identity(position, yaw):
        stage = Usd.Stage.CreateInMemory()
        root = UsdGeom.Xform.Define(stage, "/World/Robot")
        root.AddTranslateOp().Set(Gf.Vec3d(*position))
        root.AddRotateZOp().Set(yaw)
        link = UsdGeom.Xform.Define(stage, "/World/Robot/link")
        link.AddTranslateOp().Set(Gf.Vec3d(0.3, -0.1, 0.2))
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        shape = UsdGeom.Cube.Define(stage, "/World/Robot/link/collision")
        shape.AddTranslateOp().Set(Gf.Vec3d(0.05, 0.02, -0.03))
        shape.AddRotateXOp().Set(12.5)
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        robot = SimpleNamespace(body_names=["link"], device="cpu")
        return CollisionFootprint(stage, robot, "/World/Robot", base=None).geometry_sha256

    reference = identity((0.0, 0.0, 0.5), 0.0)
    assert identity((-1.6, -0.2, 0.5), 0.0) == reference
    assert identity((10.2, 6.0, 0.6), 37.0) == reference


def test_box_distances_are_exact_near_boxes_and_never_too_small():
    """Each box updates only its neighbourhood (a whole-house map costs per box): the clearance is
    exact wherever it is below the reach and never below the true clearance elsewhere."""
    boxes = (BoxGeometry((1, 1, 1), (2, 0, 0.5)), BoxGeometry((0.4, 2.0, 1), (-1.2, 1.0, 0.5)))
    maps = KnownBoxMap(
        SimpleNamespace(rigid_objects={}), boxes, bounds=(-3, -3, 5, 3), resolution=0.1,
        footprint=RectangleFootprint(-0.5, 0.5, -0.5, 0.5), clearance=0.0,
    )  # fmt: skip
    yy, xx = np.indices((60, 80))
    x, y = -3 + (xx + 0.5) * 0.1, -3 + (yy + 0.5) * 0.1
    margin, reach = 0.07, 0.4
    true = np.minimum.reduce((x + 3, 5 - x, y + 3, 3 - y)) - margin
    for box in boxes:
        dx = np.maximum(np.abs(x - box.center[0]) - box.size[0] / 2, 0)
        dy = np.maximum(np.abs(y - box.center[1]) - box.size[1] / 2, 0)
        true = np.minimum(true, np.hypot(dx, dy) - margin)
    fast = maps._distance(margin, reach=reach)
    assert np.all(fast >= true - 1e-9)
    near = true < reach
    np.testing.assert_allclose(fast[near], true[near], atol=1e-9)
