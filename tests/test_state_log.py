# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The live video's states (recording.state_log): samples kept as they were, Nav2's routes when they
change, the base path resampled (no Isaac Sim)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phygs.recording.state_log import StateLog


def identity(x=0.0, y=0.0, z=0.0, yaw=0.0):
    return np.array([x, y, z, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)], dtype=np.float32)


def test_a_sample_keeps_its_values_when_the_simulation_overwrites_its_buffer():
    log, root = StateLog(), identity()
    for k in range(3):
        root[0] = k  # The same buffer, updated in place as the simulation's is.
        log.sample(0.1 * k, root, holding=k == 2)
    states = log.states()
    assert len(log) == 3 and states.root[:, 0].tolist() == [0.0, 1.0, 2.0]
    assert states.holding.tolist() == [False, False, True]


def test_routes_are_logged_when_they_change():
    log = StateLog()
    log.sample(0.0, identity(), holding=False)
    path = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
    log.route(0.0, None, ())
    log.route(0.5, (1.0, 1.0, None), path)
    log.route(0.6, (1.0, 1.0, None), path)  # The same route object: not again.
    log.route(0.7, (1.0, 1.0, None), [list(point) for point in path])  # Equal values: not again.
    log.route(1.0, (1.0, 1.0, None), path[1:])  # Replanned.
    log.route(2.0, None, ())  # Arrived: no route.
    states = log.states()
    assert [route.time for route in states.routes] == [0.0, 0.5, 1.0, 2.0]
    assert states.routes[0].goal is None and len(states.routes[0].points) == 0
    assert states.routes[1].goal == (1.0, 1.0, None) and states.routes[1].points.tolist() == [list(p) for p in path]
    assert len(states.routes[2].points) == 2 and states.routes[3].goal is None
    assert states.route_at(0.9).time == 0.5 and states.route_at(-1.0) is None


def test_the_base_path_is_resampled_with_its_heading():
    log = StateLog()
    for k in range(11):
        log.sample(0.1 * k, identity(0.1 * k, yaw=0.1 * k), holding=False)
    path = log.states().base_path(rate=5.0)
    assert [p[0] for p in path] == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert path[-1][1:] == pytest.approx((1.0, 0.0, 1.0), abs=1e-3)
