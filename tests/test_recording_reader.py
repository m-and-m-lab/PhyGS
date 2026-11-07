# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The recording read back (phygs.recording.reader) holds what the state log holds: the same samples,
bodies, routes, costmaps, arm plans, command spans and audit lines (no Isaac Sim)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

rr = pytest.importorskip("rerun")

from phygs.recording.log import EpisodeLog  # noqa: E402
from phygs.recording.panels import PlannerView  # noqa: E402
from phygs.recording.reader import Recording  # noqa: E402
from phygs.recording.state_log import StateLog, States  # noqa: E402

SPOT_URDF = Path(__file__).resolve().parents[1] / "phygs" / "robots" / "spot" / "assets" / "spot.urdf"
JOINTS = ("arm_sh0", "front_left_knee", "not in.the urdf")  # The last: plotted, not drawn; escaped in paths.
BODIES = ("/World/table", "/World/drawer", "/World/cup")
META = dict(period=0.032, physics_dt=0.004, robot="spot", scene=dict(id="grid", sha256="abc"), start=[0.0, 0.0, 0.0])
SPANS = [dict(lane="base", name="navigate_to", start=0.1, end=None, status="active")]


def pose(x=0.0, y=0.0, z=0.0, yaw=0.0):
    return np.array([x, y, z, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def plan(label: str, shift: float) -> PlannerView:
    return PlannerView(
        label=label,
        voxels=np.array([[0.5, 0.0, 0.2], [0.52, 0.0, 0.2]]) + shift,
        voxel_size=0.02,
        boxes=(((1.0, 0.0, 0.4), (0.6, 0.4, 0.8)),),
        arm_start=np.array([[0.3, 0.0, 0.5], [0.6, 0.0, 0.5]]),
        arm_path=(np.array([[0.3, 0, 0.5], [0.7, 0, 0.4]]), np.array([[0.3, 0, 0.5], [0.8, 0, 0.3], [0.9, 0, 0.3]])),
        arm_end=np.array([[0.3, 0.0, 0.5], [0.9, 0.0, 0.3]]),
        end_spheres=np.array([[0.9, 0.0, 0.3, 0.05]]),
        tool_path=np.array([[0.7, 0.0, 0.4], [0.9, 0.0, 0.3]]),
        floor=None if shift else -0.5,
    )


def record_both(tmp_path: Path) -> tuple[States, Recording]:
    """One episode's states written by the state log and by the recording, read back by each."""
    state = StateLog(tmp_path / "state.npz", joint_names=JOINTS, body_names=BODIES, meta=META)
    log = EpisodeLog(tmp_path / "episode.rrd", run_id="run")
    log.properties(replay=META)
    log.robot_model(SPOT_URDF, JOINTS)
    log.scene([(None, np.zeros(3), np.ones(3))], BODIES)
    route = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    first_plan, second_plan = plan("plan 1: ok", 0.0), plan("plan 2: ok", 0.1)
    for sample, tick in enumerate([0, 0, *range(8, 400, 8)]):  # Two samples in the first tick, as a run's start.
        time = tick * 0.004
        bodies = np.stack([pose(1.0), pose(min(time, 0.5), 2.0), pose(3.0, yaw=0.2 if time > 1.0 else 0.0)])
        root, joints = pose(0.01 * sample, z=0.5, yaw=0.1 * sample), np.array([0.1, -1.2, 0.01 * sample])
        holding, command = time > 0.8, None if time < 0.2 else (0.3, 0.0, 0.1)
        state.sample(time, root, joints, bodies, holding=holding, command=command)
        log.at(tick, time)
        log.robot_state(root, joints, holding=holding, command=command)
        log.body_poses(bodies)
        goal, points = ((1.0, 1.0, None), route) if 0.2 <= time < 1.2 else (None, ())
        state.route(time, goal, points)
        log.route(goal, points)
        if sample in (2, 30):
            costs = np.full((4, 5), 3 * sample, dtype=np.uint8)
            costs[0, 1] = 254
            state.costmap(time, costs, (-1.0, -2.0), 0.05)
            log.costmap(costs, (-1.0, -2.0), 0.05)
        view = first_plan if sample >= 10 else None
        view = second_plan if sample >= 40 else view
        if view is not None:
            state.plan(time, view)
            log.plan(view)
    log.agent_call(dict(t=0.4, call="navigation.navigate_to", args={"x": 1.0}, result="accepted"))
    log.agent_call(dict(t=0.5, call="session.step", name="open the drawer", event="start"))
    log.commands(SPANS)
    state.write(SPANS)
    assert log.close()["status"] == "complete"
    return States.load(tmp_path), Recording(tmp_path)


def test_the_recording_reads_back_what_the_state_log_holds(tmp_path):
    expected, recording = record_both(tmp_path)
    states = recording.states()
    assert len(states) == len(expected) == 51
    assert np.allclose(states.time, expected.time, atol=1e-9)
    for name in ("root", "joints", "holding"):
        assert np.array_equal(getattr(states, name), getattr(expected, name)), name
    assert np.array_equal(states.command, expected.command, equal_nan=True)
    assert states.joint_names == expected.joint_names and states.body_names == expected.body_names
    for sample in range(len(states)):
        assert np.array_equal(states.bodies_at(sample), expected.bodies_at(sample)), sample
    assert len(states.routes) == len(expected.routes) == 3
    for got, want in zip(states.routes, expected.routes):
        assert got.time == pytest.approx(want.time) and got.goal == want.goal
        assert np.array_equal(got.points, want.points)
    for time in (*expected.costmap_times, 0.0, 9.0):
        got, want = states.costmap_at(time), expected.costmap_at(time)
        assert (got is None) == (want is None)
        if want is not None:
            assert np.array_equal(got[0], want[0]) and got[1:] == want[1:]
    assert np.allclose(states.plan_times, expected.plan_times)
    for time in expected.plan_times:
        got, want = states.plan_at(time), expected.plan_at(time)
        fields = ("label", "voxel_size", "floor", "boxes")
        assert [getattr(got, name) for name in fields] == [getattr(want, name) for name in fields]
        for name in ("voxels", "arm_start", "arm_end", "end_spheres", "tool_path"):
            assert np.array_equal(getattr(got, name), getattr(want, name)), name
        assert all(np.array_equal(a, b) for a, b in zip(got.arm_path, want.arm_path, strict=True))
    assert states.commands == expected.commands == SPANS
    assert states.meta == expected.meta


def test_the_audit_lines_come_back_in_order(tmp_path):
    _, recording = record_both(tmp_path)
    assert [line["call"] for line in recording.audit] == ["navigation.navigate_to", "session.step"]
    assert recording.audit[0]["args"] == {"x": 1.0} and recording.properties["run_id"] == "run"


TINY_URDF = """<robot name="tiny">
  <link name="base"/><link name="arm"/>
  <joint name="j" type="revolute"><parent link="base"/><child link="arm"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
</robot>
"""


def test_a_run_that_died_reads_back_to_its_last_whole_chunk(tmp_path):
    (tmp_path / "tiny.urdf").write_text(TINY_URDF)
    run = tmp_path / "run"
    run.mkdir()
    log = EpisodeLog(run / "episode.rrd", run_id="died")
    log.properties(replay=META)
    log.robot_model(tmp_path / "tiny.urdf", ("j",))
    log.scene([], BODIES[:1])
    for k in range(300):
        log.at(8 * k, 0.032 * k)
        log.robot_state(pose(0.01 * k, z=0.5), [0.001 * k], holding=k > 150)
        log.body_poses([pose(1.0 + (0.01 if k > 50 else 0.0))])
        if k == 200:
            log.phase("agent")  # What was logged so far is made safe on disk.
            safe = (run / "episode.rrd").stat().st_size
    log.close()
    data = (run / "episode.rrd").read_bytes()
    died = tmp_path / "died"
    died.mkdir()
    (died / "episode.rrd").write_bytes(data[:safe])  # Killed right after the phase.
    states = Recording(died).states()
    assert len(states) == 201 and states.joints[:, 0] == pytest.approx(0.001 * np.arange(201))
    assert states.bodies_at(200)[0, 0] == pytest.approx(1.01) and states.holding.sum() == 50
    (died / "episode.rrd").write_bytes(data[: (safe + len(data)) // 2])  # Killed in the middle of a message.
    recording = Recording(died)
    assert recording.repaired and not Recording(run).repaired
    assert recording.properties["run_id"] == "died" and recording.properties["replay"] == META
    assert 201 <= len(recording.states()) <= 300


def test_a_recording_of_another_schema_is_refused(tmp_path):
    log = EpisodeLog(tmp_path / "episode.rrd", run_id="old")
    log.properties(phygs_log_schema=0)
    log.close()
    with pytest.raises(ValueError, match="schema 0"):
        Recording(tmp_path)
