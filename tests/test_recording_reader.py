# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The recording read back (phygs.recording.reader): the samples, the scene's bodies, Nav2's routes
and costmaps, the arm plans and their requests, the command spans and the audit lines, as they were
logged (no Isaac Sim)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

rr = pytest.importorskip("rerun")

from phygs.observations.types import CameraObservation, ObservationPacket, Stamp, Transform  # noqa: E402
from phygs.recording.log import POSITION_STEP, EpisodeLog  # noqa: E402
from phygs.recording.panels import PlannerView  # noqa: E402
from phygs.recording.reader import Recording  # noqa: E402

SPOT_URDF = Path(__file__).resolve().parents[1] / "phygs" / "robots" / "spot" / "assets" / "spot.urdf"
JOINTS = ("arm_sh0", "front_left_knee", "not in.the urdf")  # The last: plotted, not drawn; escaped in paths.
BODIES = ("/World/table", "/World/drawer", "/World/cup")
META = dict(period=0.032, physics_dt=0.004, robot="spot", scene=dict(id="grid", sha256="abc"), start=[0.0, 0.0, 0.0])
SPANS = [dict(lane="base", name="navigate_to", start=0.1, end=None, status="active")]
TICKS = [0, 0, *range(8, 400, 8)]  # Two samples in the first tick, as a run's start; then every 32 ms.
ROUTE = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))


def pose(x=0.0, y=0.0, z=0.0, yaw=0.0):
    return np.array([x, y, z, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)], dtype=np.float32)


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


FIRST_PLAN, SECOND_PLAN = plan("plan 1: ok", 0.0), plan("plan 2: ok", 0.1)


def sample(index: int, time: float) -> tuple:
    """The state at sample ``index``: root, joints, bodies, holding, command."""
    bodies = np.stack([pose(1.0), pose(min(time, 0.5), 2.0), pose(3.0, yaw=0.2 if time > 1.0 else 0.0)])
    root, joints = pose(0.01 * index, z=0.5, yaw=0.1 * index), np.array([0.1, -1.2, 0.25 * time], np.float32)
    return root, joints, bodies, time > 0.8, None if time < 0.2 else (0.3, 0.0, 0.1)


def costs(index: int) -> np.ndarray:
    found = np.full((4, 5), 3 * index, dtype=np.uint8)
    found[0, 1] = 254
    return found


def record(tmp_path: Path) -> Recording:
    """One episode's recording, read back."""
    log = EpisodeLog(tmp_path / "episode.rrd", run_id="run")
    log.properties(replay=META)
    log.robot_model(SPOT_URDF, JOINTS)
    log.scene([(None, np.zeros(3), np.ones(3))], BODIES)
    for index, tick in enumerate(TICKS):
        time = tick * 0.004
        root, joints, bodies, holding, command = sample(index, time)
        log.at(tick, time)
        log.joint_state(torch.as_tensor(joints), torch.as_tensor(joints) + 0.01, torch.zeros(3))
        log.robot_state(root, joints, holding=holding, command=command)
        log.body_poses(bodies)
        if 12 <= index <= 15:  # A finger touches the cup harder at each read, then lets go.
            log.contacts([("arm_link_fngr", "/World/cup", 10.0 + index)] if index < 15 else [], "pick")
        route = (1.0, 1.0, None) if 0.2 <= time < 1.2 else None
        log.route(route, ROUTE if route else ())
        log.route(route, [list(point) for point in ROUTE] if route else [])  # Equal values: not again.
        if index in (2, 30):
            log.costmap(costs(index), (-1.0, -2.0), 0.05)
        if index == 31:
            log.costmap(costs(30), (-1.0, -2.0), 0.05)  # The same costmap: not again.
        view = SECOND_PLAN if index >= 40 else FIRST_PLAN if index >= 10 else None
        if view is not None:
            log.plan(view)  # Logged when it is another object.
    log.agent_call(dict(t=0.4, call="navigation.navigate_to", args={"x": 1.0}, result="accepted"))
    log.agent_call(dict(t=0.5, call="session.step", name="open the drawer", event="begin"))
    for index, span in enumerate(SPANS):
        log.command_span(index, span)
    assert log.close()["status"] == "complete"
    return Recording(tmp_path)


def test_the_recording_reads_back_the_samples_the_scene_and_the_panels_as_logged(tmp_path):
    states = record(tmp_path).states()
    times = np.array(TICKS) * 0.004
    assert len(states) == len(TICKS) and np.allclose(states.time, times, atol=1e-9)
    for index, time in enumerate(times):
        root, joints, bodies, holding, command = sample(index, time)
        assert np.array_equal(states.root[index], root) and np.array_equal(states.joints[index], joints)
        assert states.holding[index] == holding
        want = np.full(3, np.nan) if command is None else command
        assert np.array_equal(states.command[index], np.asarray(want, np.float32), equal_nan=True)
        # Each body's last logged pose: within the logging step of where it is.
        assert np.abs(states.bodies_at(index)[:, :3] - bodies[:, :3]).max() <= POSITION_STEP
    assert states.joint_names == JOINTS and states.body_names == BODIES
    changes = [0.0, times[times >= 0.2][0], times[times >= 1.2][0]]  # No route, the route, arrived.
    assert [route.time for route in states.routes] == pytest.approx(changes)
    assert states.routes[0].goal is None and states.routes[1].goal == (1.0, 1.0, None)
    assert states.routes[1].points.tolist() == [list(point) for point in ROUTE] and len(states.routes[2].points) == 0
    assert np.allclose(states.costmap_times, times[[2, 30]]) and states.costmap_at(times[2] - 0.01) is None
    found, origin, cell = states.costmap_at(times[31])
    assert np.array_equal(found, costs(30)) and origin == (-1.0, -2.0) and cell == 0.05
    assert np.allclose(states.plan_times, times[[10, 40]]) and states.plan_at(times[10] - 0.01) is None
    for time, want in ((times[10], FIRST_PLAN), (times[45], SECOND_PLAN)):
        got = states.plan_at(time)
        assert (got.label, got.voxel_size, got.floor) == (want.label, want.voxel_size, want.floor)
        assert np.allclose(np.ravel(got.boxes), np.ravel(want.boxes))
        for name in ("voxels", "arm_start", "arm_end", "end_spheres", "tool_path"):
            assert np.allclose(getattr(got, name), getattr(want, name)), name
        assert all(np.allclose(a, b) for a, b in zip(got.arm_path, want.arm_path, strict=True))
    assert states.commands == SPANS and states.meta == dict(META, commands=SPANS)


def test_the_contact_forces_read_back_over_time(tmp_path):
    forces = record(tmp_path).contact_forces()
    times, values = forces[("arm_link_fngr", "/World/cup")]
    assert values.tolist() == [22.0, 23.0, 24.0, 0.0] and np.allclose(times, np.array(TICKS[12:16]) * 0.004)


def camera_packet(tick: int, shade: int) -> ObservationPacket:
    """A packet with a 4x6 hand camera image captured at ``tick``."""
    stamp = Stamp(0, tick, tick * 0.004)
    depth = torch.full((4, 6), 1.5)
    depth[0, 0] = 0.0  # No depth there.
    camera = CameraObservation(
        stamp,
        "hand_optical",
        torch.full((4, 6, 3), shade, dtype=torch.uint8),
        depth,
        depth > 0,
        torch.tensor([[5.0, 0, 3], [0, 5.0, 2], [0, 0, 1]]),
        Transform("body", "hand_optical", stamp, torch.eye(4) * 2, "ideal_robot_kinematics"),
        True,
    )
    return ObservationPacket(stamp, None, None, None, {"hand": camera}, ())


@pytest.mark.parametrize("level", ["full", "all_frames"])
def test_what_the_agent_received_and_every_capture_read_back(tmp_path, level):
    log = EpisodeLog(tmp_path / "episode.rrd", run_id="frames", level=level)
    log.properties(replay=META, settings=dict(recording=dict(level=level)))
    for tick, shade in ((8, 40), (16, 90)):
        log.at(tick, tick * 0.004)
        log.camera_frames(camera_packet(tick, shade))  # Level all_frames only.
    log.at(20, 0.08)
    log.observed(camera_packet(16, 90))
    log.at(22, 0.088)
    log.observed(camera_packet(16, 90))  # The same capture: its images once.
    log.close()
    recording = Recording(tmp_path)
    assert [summary["cameras"] for _, summary in recording.observations()] == [{"hand": 16}, {"hand": 16}]
    (received,) = recording.images("agent_view", "hand")  # At all_frames: found among every capture.
    assert received.tick == 16 and received.rgb[0, 0].tolist() == [90, 90, 90] and received.depth[0, 0] == 0.0
    frames = list(recording.images("sensors", "hand"))
    assert [frame.tick for frame in frames] == ([8, 16] if level == "all_frames" else [])
    if frames:
        assert np.array_equal(frames[1].body_from_camera, np.eye(4) * 2) and frames[1].intrinsics[0, 0] == 5.0
    assert recording.entities("sensors") == [] and recording.entities("agent_view", "images") == []  # Not loaded.


def test_the_audit_lines_come_back_in_order(tmp_path):
    recording = record(tmp_path)
    assert [line["call"] for line in recording.audit] == ["navigation.navigate_to", "session.step"]
    assert recording.audit[0]["args"] == {"x": 1.0} and recording.properties["run_id"] == "run"


def test_an_arm_plan_request_reads_back_with_its_depth_images_lossless(tmp_path):
    log = EpisodeLog(tmp_path / "episode.rrd", run_id="plans")
    rng = np.random.default_rng(0)
    depth = {"hand": rng.uniform(0.2, 3.0, (24, 32)).astype(np.float32), "front": np.zeros((24, 32), np.float32)}
    images = [dict(camera=name, intrinsics=[[30.0, 0, 16], [0, 30.0, 12], [0, 0, 1]], pose=[0.0] * 7) for name in depth]
    for k in range(2):
        log.at(100 * k, 0.4 * k)
        request = dict(kind="plan", start=[0.1 * k] * 6, goal=[0.0] * 6, images=images if k == 0 else None)
        log.plan_request(request, dict(success=k == 0, plan_seconds=0.05), depth if k == 0 else {})
    log.close()
    first, second = Recording(tmp_path).plan_requests()
    assert first.request["images"] == images and first.request["start"] == [0.0] * 6
    assert all(np.array_equal(first.depth[name], depth[name]) for name in depth)  # Every float as it was.
    assert first.outcome == dict(success=True, plan_seconds=0.05) and first.time == 0.0
    assert second.request["images"] is None and second.depth == {} and second.time == pytest.approx(0.4)


TINY_URDF = """<robot name="tiny">
  <link name="base"/><link name="arm"/>
  <joint name="j" type="revolute"><parent link="base"/><child link="arm"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
</robot>
"""


def test_an_agent_s_records_read_back_with_their_images_exactly_as_given(tmp_path):
    import io

    from PIL import Image

    rgb = np.random.default_rng(0).integers(0, 255, (24, 32, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    png = buffer.getvalue()
    log = EpisodeLog(tmp_path / "episode.rrd", run_id="run")
    log.at(10, 0.04)
    log.agent_log("model/turn", {"turn": 1, "reply": "stow"}, {"frontleft": png, "gray": rgb[..., 0]})
    log.agent_log("model/point", {"turn": 1, "target": "the cup"}, {"rgba": np.dstack([rgb, rgb[..., :1]])})
    log.at(50, 0.2)
    log.agent_log("model/turn", {"turn": 2, "reply": "done"}, {"frontleft": png})
    log.agent_log("model", {"deeper topics stay apart": True}, {})
    assert log.close()["status"] == "complete"

    recording = Recording(tmp_path)
    turns = recording.agent_log("model/turn")
    assert [(r.record["turn"], r.time, r.topic) for r in turns] == [(1, 0.04, "model/turn"), (2, 0.2, "model/turn")]
    assert [r.topic for r in recording.agent_log()] == ["model/turn", "model/point", "model/turn", "model"]
    assert turns[0].wall_time <= turns[1].wall_time and turns[0].seq < turns[1].seq
    images = recording.agent_images("model/turn")
    assert set(images) == {turns[0].seq, turns[1].seq} and images[turns[0].seq]["frontleft"] == png  # The bytes.
    assert np.array_equal(images[turns[0].seq]["gray"], rgb[..., 0]) and set(images[turns[1].seq]) == {"frontleft"}
    (point,) = recording.agent_images("model/point").values()
    assert np.array_equal(point["rgba"], np.dstack([rgb, rgb[..., :1]]))
    assert recording.agent_images("model") == {}  # Its own images only, none of model/turn's.


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
        log.joint_state(torch.tensor([0.001 * k]), torch.zeros(1), torch.zeros(1))
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
