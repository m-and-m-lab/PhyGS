# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sensor-owned Nav2 geometry and permission boundaries."""

import numpy as np
import pytest

from phygs.execution.navigation.backends.nav2 import Nav2BackendFactory, Nav2Config
from phygs.execution.navigation.floor import DeclaredFloorMap
from phygs.execution.navigation.map import RectangleFootprint
from phygs.observations.types import CameraPermission, ObservationAccess


def test_floor_prior_contains_no_obstacle_geometry():
    floor = DeclaredFloorMap(
        bounds=(-2.0, -2.0, 2.0, 2.0), resolution=0.1, footprint=RectangleFootprint(-0.7, 0.7, -0.7, 0.7), clearance=0.2
    )
    assert not np.any(floor.raw_snapshot().occupied)
    assert floor.snapshot().free((0.0, 0.0))
    assert not floor.snapshot().free((1.9, 0.0))
    assert floor.configuration["source"] == "declared_flat_floor"


def test_nav2_sensor_requires_depth_only_for_private_navigation_view():
    floor = DeclaredFloorMap(
        bounds=(-2.0, -2.0, 2.0, 2.0), resolution=0.1, footprint=RectangleFootprint(-0.7, 0.7, -0.7, 0.7), clearance=0.2
    )
    factory = Nav2BackendFactory(Nav2Config(sensor_cameras=("inspection",)), maps=floor)
    declaration = factory.declaration
    assert declaration.inputs.cameras == (CameraPermission("inspection", rgb=False),)
    assert "simulated_rgbd_camera" in declaration.information_sources
    assert declaration.privileged_sources == ()
    with pytest.raises(PermissionError):
        declaration.inputs.require(
            ObservationAccess(joint_state=True, localization="ground_truth"), consumer="navigation"
        )


def test_invalid_camera_and_floor_fail_early():
    with pytest.raises(ValueError, match="camera"):
        Nav2Config(sensor_cameras=("not/a/camera",))
    with pytest.raises(ValueError, match="align"):
        DeclaredFloorMap(
            bounds=(0, 0, 1.05, 1), resolution=0.1, footprint=RectangleFootprint(-0.5, 0.5, -0.5, 0.5), clearance=0.1
        )
