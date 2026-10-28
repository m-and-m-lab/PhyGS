# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The route the bridge hands Nav2's controller (no ROS needed)."""

import math

import pytest

from phygs.execution.navigation.backends.ros2 import dense_route


def test_routes_are_resampled_at_the_costmap_resolution():
    """MPPI's route critics count route points; Smac Lattice routes are ~0.15 m apart."""
    route = dense_route([(0.0, 0.0, 0.0), (0.0, 0.15, 0.0), (0.0, 0.15, math.pi / 2), (0.05, 0.15, -3.0)], 0.05)
    start = [coordinate for pose in route[:4] for coordinate in pose[:2]]
    assert start == pytest.approx([0, 0, 0, 0.05, 0, 0.1, 0, 0.15])
    assert list(route[4]) == pytest.approx([0.0, 0.15, math.pi / 2])  # A turn in place keeps one pose.
    assert route[-1][2] == pytest.approx(-3.0)  # Headings turn the shorter way: +pi/2 to -3.0 via pi.
    assert all(math.dist(a[:2], b[:2]) <= 0.05 + 1e-9 for a, b in zip(route, route[1:]))
    with pytest.raises(ValueError, match="spacing"):
        dense_route(route, 0.0)
