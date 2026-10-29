# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The state log: samples written once and read back, only moved bodies logged, routes on change (no Isaac Sim)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from phygs.recording.state_log import POSITION_STEP, ROTATION_STEP, SCHEMA, StateLog, States

JOINTS = ("hip", "knee", "arm")
BODIES = ("/World/table", "/World/drawer", "/World/cup")


def identity(x=0.0, y=0.0, z=0.0, yaw=0.0):
    return np.array([x, y, z, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)], dtype=np.float32)


def scene(drawer_x=0.0, cup_yaw=0.0):
    return np.stack([identity(1.0), identity(drawer_x, 2.0), identity(3.0, yaw=cup_yaw)])


def make_log(tmp_path):
    return StateLog(tmp_path / "state.npz", joint_names=JOINTS, body_names=BODIES, meta=dict(period=0.032))


def test_only_bodies_that_moved_are_logged_and_every_sample_reads_back_whole(tmp_path):
    log = make_log(tmp_path)
    log.sample(0.0, identity(), np.zeros(3), scene(), holding=False)
    log.sample(0.032, identity(0.1), np.ones(3), scene(drawer_x=0.5 * POSITION_STEP), holding=False)  # Too little.
    log.sample(0.064, identity(0.2), np.ones(3), scene(drawer_x=0.2), holding=True)
    log.sample(0.096, identity(0.3), np.ones(3), scene(drawer_x=0.2, cup_yaw=2 * ROTATION_STEP), holding=True)
    log.write([dict(lane="base", name="navigate_to", start=0.0, end=None, status="active")])
    with np.load(tmp_path / "state.npz") as arrays:
        assert arrays["body_sample"].tolist() == [0, 0, 0, 2, 3]  # All at first, then the drawer, then the cup.
        assert arrays["body_index"].tolist() == [0, 1, 2, 1, 2]
    states = States.load(tmp_path)  # The folder or the file.
    assert len(states) == 4 and states.holding.tolist() == [False, False, True, True]
    assert states.joint_names == JOINTS and states.body_names == BODIES
    assert states.meta["schema"] == SCHEMA and states.meta["period"] == 0.032
    assert states.commands == [dict(lane="base", name="navigate_to", start=0.0, end=None, status="active")]
    sequence = [poses.copy() for poses in states.bodies()]
    for sample, poses in enumerate(sequence):
        assert np.array_equal(poses, states.bodies_at(sample))
    assert sequence[1][1, 0] == 0.0 and sequence[2][1, 0] == pytest.approx(0.2)  # The drawer: logged when it moved.
    assert sequence[3][2] == pytest.approx(scene(drawer_x=0.2, cup_yaw=2 * ROTATION_STEP)[2])
    assert states.index(0.07) == 2 and states.index(-1.0) == 0 and states.index(9.0) == 3


def test_a_sample_keeps_its_values_when_the_simulation_overwrites_its_buffers(tmp_path):
    log = make_log(tmp_path)
    root, joints, bodies = identity(), np.zeros(3, dtype=np.float32), scene()
    for k in range(3):
        root[0], joints[:] = k, k  # The same buffers, updated in place as the simulation's are.
        bodies[2, 0] = 3.0 + k
        log.sample(0.1 * k, root, joints, bodies, holding=False)
    log.write()
    states = States.load(tmp_path)
    assert states.root[:, 0].tolist() == [0.0, 1.0, 2.0] and states.joints[:, 1].tolist() == [0.0, 1.0, 2.0]
    assert [poses[2, 0] for poses in states.bodies()] == [3.0, 4.0, 5.0]


def test_routes_are_logged_when_they_change(tmp_path):
    log = make_log(tmp_path)
    log.sample(0.0, identity(), np.zeros(3), scene(), holding=False)
    path = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
    log.route(0.0, None, ())
    log.route(0.5, (1.0, 1.0, None), path)
    log.route(0.6, (1.0, 1.0, None), path)  # The same route object: not again.
    log.route(0.7, (1.0, 1.0, None), [list(point) for point in path])  # Equal values: not again.
    log.route(1.0, (1.0, 1.0, None), path[1:])  # Replanned.
    log.route(2.0, None, ())  # Arrived: no route.
    log.write()
    states = States.load(tmp_path / "state.npz")
    assert [route.time for route in states.routes] == [0.0, 0.5, 1.0, 2.0]
    assert states.routes[0].goal is None and len(states.routes[0].points) == 0
    assert states.routes[1].goal == (1.0, 1.0, None) and states.routes[1].points.tolist() == [list(p) for p in path]
    assert len(states.routes[2].points) == 2 and states.routes[3].goal is None
    assert states.route_at(0.9).time == 0.5 and states.route_at(-1.0) is None


def test_the_base_path_is_resampled_with_its_heading(tmp_path):
    log = make_log(tmp_path)
    for k in range(11):
        log.sample(0.1 * k, identity(0.1 * k, yaw=0.1 * k), np.zeros(3), scene(), holding=False)
    log.write()
    path = States.load(tmp_path).base_path(rate=5.0)
    assert [p[0] for p in path] == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert path[-1][1:] == pytest.approx((1.0, 0.0, 1.0), abs=1e-3)


def test_a_log_of_another_schema_is_refused(tmp_path):
    log = make_log(tmp_path)
    log.sample(0.0, identity(), np.zeros(3), scene(), holding=False)
    log.write()
    with np.load(tmp_path / "state.npz") as arrays:
        data = {key: arrays[key] for key in arrays.files}
    data["meta"] = np.asarray(json.dumps(dict(json.loads(str(data["meta"])), schema=SCHEMA + 1)))
    with pytest.raises(ValueError, match="schema"):
        States(data)
    with pytest.raises(IndexError):
        States.load(tmp_path).bodies_at(1)
