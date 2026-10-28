# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The robot UI's runner-side session (what reaches the browser) and its server protocol."""

import asyncio
import collections
import inspect
import json
from dataclasses import replace

import numpy as np
import pytest
from ui_fakes import (
    ARM_JOINTS,
    SECRET,
    START,
    Feedback,
    Manipulation,
    Navigation,
    Ui,
    fake_control,
    frame,
    run_requests,
)

from phygs.skills.navigation.api import NavigationClient
from phygs.tools.front_view import CameraFrame
from phygs.ui import schema
from phygs.ui.bridge import Bridge, MapImage
from phygs.ui.codecs import parse_binary_message
from phygs.ui.server import UiServer

FRONT = ("frontleft", "frontright")  # Spot's front cameras, stitched into the Front view.
STATE_KEYS = {
    *("type", "sim_time", "wall_time", "rtf", "episode_id", "status"),
    *("pose", "speed", "arm", "holding", "phase", "task", "telemetry"),
}


def test_session_publishes_only_the_agent_view_the_listed_cameras_and_skill_steps():
    control = fake_control()
    bridge = Bridge({"views": ["front", "hand", "frontleft"]})
    session = Ui(
        control,
        bridge,
        cameras=("hand", "frontleft"),
        costmap=lambda: (np.array([[0, 253, 254, 255]], dtype=np.uint8), (1.0, 2.0), 0.05),
    )
    session.update()
    assert set(bridge.frames()) == {"hand", "frontleft"}  # Not third_person.
    assert bridge.frames()["hand"].rgb[0, 0, 0] == 10
    version, state = bridge.state()
    assert set(state) == STATE_KEYS
    assert state["arm"]["stowed"] is True and state["holding"] is False and state["status"] == "running"
    events = [event for _, event in bridge.events_after(0)]
    assert [event["type"] for event in events] == ["step"]  # The "check" line is a test line, not a skill step.
    image = bridge.map()
    assert isinstance(image, MapImage) and image.origin == (1.0, 2.0) and image.version == 1
    assert image.cells.tolist() == [[0, 0, 100, -1]]  # Free, too close (free), obstacle, unknown.
    assert image.blocked.tolist() == [[False, True, True, False]]  # Where the robot's centre may not be.
    text = json.dumps([state, events, image.origin, image.resolution])
    assert SECRET not in text


def test_session_reports_command_changes_with_raw_feedback():
    control = fake_control()
    bridge = Bridge({"views": ["front"]})
    session = Ui(control, bridge, cameras=())
    session.track("locomotion", "command_velocity", 1, {"velocity": (0.4, 0.0, 0.0)})
    session.track("manipulation", "look_at", 2)
    session.update()
    session.update()
    commands = [event for _, event in bridge.events_after(0) if event["type"] == "command"]
    assert [(c["skill"], c["status"], c["state"]) for c in commands] == [
        ("command_velocity", "active", "executing"),
        ("look_at", "active", "executing"),
        ("command_velocity", "done", "succeeded"),
        ("look_at", "failed", "failed"),
    ]
    failed = commands[-1]
    assert failed["channel"] == "arm"
    assert failed["feedback"]["failure"] == "out_of_reach"
    assert failed["feedback"]["reason"] == "This view of the target is out of reach; move the base."
    assert commands[0]["inputs"] == {"velocity": [0.4, 0.0, 0.0]}
    json.dumps(commands)  # Plain JSON.


class Locomotion:
    def __init__(self):
        self.commands, self.navigating = [], False

    def command_velocity(self, v_x, v_y, v_rot, *, duration):
        if self.navigating:
            raise RuntimeError("Cancel navigation and wait for its controlled stop before direct motion.")
        self.commands.append((v_x, v_y, v_rot, duration))
        return len(self.commands)

    def stop(self):
        return self.command_velocity(0.0, 0.0, 0.0, duration=0.5)

    def feedback(self, command_id):
        latest = len(self.commands)
        return Feedback(command_id, "active" if command_id == latest else "overridden")


def test_navigate_to_runs_with_the_page_inputs_and_the_public_api_defaults():
    control = fake_control()
    control.navigation = Navigation()
    events = run_requests(
        control, {"type": "run", "skill": "navigate_to", "inputs": {"x": 1.5, "y": -0.5, "heading": None}}
    )
    signature = inspect.signature(NavigationClient.navigate_to).parameters  # The public API's defaults.
    defaults = (signature["position_tolerance"].default, signature["heading_tolerance"].default)
    assert control.navigation.goals == [(1.5, -0.5, None, *defaults)]
    listed = {item["name"]: item["default"] for item in schema.NAVIGATE_TO["inputs"] if "default" in item}
    assert (listed["position_tolerance"], listed["heading_tolerance"]) == defaults  # The page offers the same.
    assert events[0]["skill"] == "navigate_to" and events[0]["channel"] == "base" and events[0]["state"] == "executing"
    assert events[0]["inputs"]["x"] == 1.5


def test_drive_renewals_stay_one_command_and_are_clamped_to_the_limits():
    control = fake_control()
    control.locomotion = Locomotion()
    drive = {"type": "drive", "v_x": 5.0, "v_y": 0.0, "v_rot": -9.0}
    events = run_requests(control, drive, drive, drive, limits={"max_speed": 0.8, "max_yaw_speed": 1.0})
    assert control.locomotion.commands == [(0.8, 0.0, -1.0, 0.5)] * 3
    assert [event["skill"] for event in events] == ["drive"]  # One timeline entry for a held key.


def test_cancel_and_rejections_show_the_raw_api_result():
    control = fake_control()
    control.navigation, control.locomotion = Navigation(), Locomotion()
    control.locomotion.navigating = True
    bridge = Bridge({"views": ["front"]})
    session = Ui(control, bridge, cameras=())
    session.track("navigation", "navigate_to", 1)
    for request in (
        START,
        {"type": "drive", "v_x": 0.3},
        {"type": "run", "skill": "navigate_to", "inputs": {"x": 500.0, "y": 0.0}},
        {"type": "run", "skill": "dance", "inputs": {}},
        {"type": "cancel", "id": "c1"},
    ):
        bridge.request(request)
    session.update()
    commands = [event for _, event in bridge.events_after(0) if event["type"] == "command"]
    rejected = [event for event in commands if event["status"] == "rejected"]
    assert [event["feedback"]["reason"] for event in rejected] == [
        "RuntimeError: Cancel navigation and wait for its controlled stop before direct motion.",
        "ValueError: Goal is outside the map",
        "ValueError: The UI cannot run 'dance'",
    ]
    assert all(event["state"] == "failed" for event in rejected)
    assert control.navigation.cancelled == [1]
    assert commands[-1]["skill"] == "navigate_to" and commands[-1]["state"] == "cancelled"


def test_estop_request_ends_the_episode_once():
    control = fake_control()
    bridge = Bridge({"views": ["front"]})
    session = Ui(control, bridge, cameras=())
    assert bridge.request({"type": "estop"}) and bridge.request({"type": "estop"})
    session.update()
    assert session.robot.finished == ["E-stop from the UI"]  # The episode ends through the session.
    episodes = [event for _, event in bridge.events_after(0) if event["type"] == "episode"]
    assert len(episodes) == 1 and episodes[-1]["status"] == "interrupted"
    assert bridge.state()[1]["phase"] == "ended"


def test_server_sends_hello_state_frames_and_map_to_one_client_with_the_token():
    pytest.importorskip("aiohttp")
    import aiohttp

    bridge = Bridge({"episode_id": 3, "task": "watch", "views": ["front", "hand"], "max_depth": 6.0})
    bridge.publish_state({"type": "state", "sim_time": 1.0, "status": "running"})
    bridge.publish_frame(frame("hand", 50))
    bridge.publish_map(MapImage(np.zeros((3, 4), dtype=np.int8), (0.0, 0.0), 0.05, 1))
    bridge.push_event({"type": "step", "sim_time": 0.5, "kind": "locomotion", "text": "stop"})
    server = UiServer(bridge, port=0, token="secret-token", thumbnail_period=0.1)
    server.start()

    async def session():
        base = f"http://127.0.0.1:{server.port}"
        received = []
        async with aiohttp.ClientSession() as http:
            async with http.get(base + "/") as response:
                assert response.status == 200 and "app.js" in await response.text()
            async with http.get(base + "/static/secrets.py") as response:
                assert response.status == 404
            async with http.ws_connect(base + "/ws?token=wrong") as socket:
                message = await socket.receive()
                assert message.type == aiohttp.WSMsgType.CLOSE and message.data == 4003
            async with http.ws_connect(base + "/ws?token=secret-token") as socket:
                async with http.ws_connect(base + "/ws?token=secret-token") as second:
                    message = await second.receive()
                    assert message.type == aiohttp.WSMsgType.CLOSE and message.data == 4001
                await socket.send_str(json.dumps({"type": "view", "main": "hand", "mode": "depth"}))
                end = asyncio.get_running_loop().time() + 1.5
                while asyncio.get_running_loop().time() < end:
                    try:
                        message = await socket.receive(timeout=0.3)
                    except TimeoutError:
                        continue
                    if message.type == aiohttp.WSMsgType.TEXT:
                        received.append(json.loads(message.data))
                    elif message.type == aiohttp.WSMsgType.BINARY:
                        received.append(parse_binary_message(message.data)[0])
                await socket.send_str(json.dumps({"type": "estop"}))
                await asyncio.sleep(0.2)
        return received

    try:
        received = asyncio.run(session())
    finally:
        server.stop()
    types = [message["type"] for message in received]
    assert types[0] == "hello" and received[0]["views"] == ["front", "hand"]
    assert "state" in types and "step" in types and "map" in types
    frames = [message for message in received if message["type"] == "frame"]
    assert {(f["view"], f["mode"]) for f in frames} >= {("hand", "depth")}  # The main view, as asked.
    assert all(f["view"] != "front" for f in frames)  # No front pair captured: no Front view.
    assert bridge.take_requests() == [{"type": "estop"}]


def test_server_reports_a_port_in_use_at_start():
    first = UiServer(Bridge({"views": ["front"]}), port=0, token="t")
    first.start()
    try:
        second = UiServer(Bridge({"views": ["front"]}), port=first.port, token="t")
        with pytest.raises(RuntimeError, match="could not start"):
            second.start()
    finally:
        first.stop()


def test_arm_requests_call_the_manipulation_api_with_the_page_inputs():
    control = fake_control()
    control.manipulation = Manipulation()
    pose = {name: 0.1 * i for i, name in enumerate(ARM_JOINTS)}
    events = run_requests(
        control,
        {"type": "run", "skill": "move_joints", "inputs": {"positions": pose, "duration": None}},
        {
            "type": "run",
            "skill": "execute_trajectory",
            "inputs": {
                "waypoints": [
                    {"positions": pose, "seconds": 2.0},
                    {"positions": {n: 0.0 for n in ARM_JOINTS}, "seconds": 1.5},
                ]
            },
        },
        {"type": "run", "skill": "stow", "inputs": {}},
        {"type": "run", "skill": "open_gripper", "inputs": {}},
    )
    calls = control.manipulation.calls
    assert calls[0] == ("move_joints", pose, 1.2501)  # No duration: the shortest (the API rounds to steps).
    trajectory = calls[1][1]
    assert trajectory.names == ARM_JOINTS and list(trajectory.times) == [0.0, 2.0, 3.5]
    assert trajectory.positions[0] == pytest.approx([0.0, -3.13, 3.13, 0.0, 0.0, 0.0])  # From the measured pose.
    assert calls[2:] == [("stow",), ("open_gripper", 2.0)]
    assert [event["skill"] for event in events] == ["move_joints", "execute_trajectory", "stow", "open_gripper"]
    assert all(event["channel"] == "arm" for event in events)


def test_a_pointed_pixel_resolves_in_the_frame_that_was_sent():
    from phygs.ui.targets import resolve_camera_pixel

    depth = np.full((48, 64), 2.0, dtype=np.float32)
    valid = np.ones((48, 64), dtype=bool)
    valid[10, 20] = False  # No depth at the pointed pixel: the nearest valid one answers.
    world_from_camera = np.eye(4)
    world_from_camera[:3, 3] = [1.0, 2.0, 0.5]
    frame = CameraFrame(
        name="hand",
        time_seconds=0.1,
        tick=50,
        rgb=np.zeros((48, 64, 3), np.uint8),
        depth=depth,
        valid=valid,
        intrinsics=np.array([[32.0, 0, 32], [0, 32.0, 24], [0, 0, 1]]),
        body_from_camera=np.eye(4),
        world_from_camera=world_from_camera,
    )
    found = resolve_camera_pixel(frame, 20, 10)
    assert found["pixel"] != [20, 10] and abs(found["pixel"][0] - 20) + abs(found["pixel"][1] - 10) == 1
    assert found["depth"] == pytest.approx(2.0)
    x, y = found["pixel"]
    assert found["world"] == pytest.approx([1.0 + (x - 32) / 32 * 2.0, 2.0 + (y - 24) / 32 * 2.0, 2.5])
    assert resolve_camera_pixel(frame, 100, 10) is None  # Outside the image.
    # The camera moved 3 m between two sent frames: a click on the older one resolves in it.
    later = np.eye(4)
    later[:3, 3] = [4.0, 2.0, 0.5]
    server = UiServer(Bridge({"views": ["front", "hand"]}), token="t", front_cameras=FRONT)
    sent = server._sent_frames.setdefault("hand", collections.OrderedDict())
    server._remember(sent, 50, frame, 8)
    server._remember(sent, 60, replace(frame, tick=60, world_from_camera=later), 8)
    assert server.probe({"view": "hand", "tick": 50, "x": 32, "y": 24})["found"]["world"] == pytest.approx(
        [1.0, 2.0, 2.5]
    )
    assert server.probe({"view": "hand", "tick": 60, "x": 32, "y": 24})["found"]["world"] == pytest.approx(
        [4.0, 2.0, 2.5]
    )
    assert server.probe({"view": "frontleft", "tick": 1, "x": 0, "y": 0})["found"] is None


def test_target_skills_get_api_arguments_translated_from_the_click():
    control = fake_control()
    control.manipulation = Manipulation()
    target = {"point": [1.2, 0.1, 0.4], "normal": [-1.0, 0.0, 0.0], "view": "hand", "pixel": [320, 240]}
    events = run_requests(
        control,
        {"type": "run", "skill": "look_at", "inputs": {"target": target, "distance_m": 0.6, "elevation_deg": 30.0}},
        {
            "type": "run",
            "skill": "move_to_pose",
            "inputs": {"target": target, "orientation": "down", "standoff_m": 0.1},
        },
        {"type": "run", "skill": "move_along", "inputs": {"direction": "up", "distance_m": 0.05}},
    )
    calls = control.manipulation.calls
    assert calls[0][0] == "look_at" and calls[0][1].position == (1.2, 0.1, 0.4) and calls[0][2:] == (0.6, 30.0)
    # The click's wrist pose already holds the robot's grasp tool frame, so it moves the wrist: Spot's
    # grasp point is 0.183 m along the wrist x and 0.029 m along its z (the heading, top-down at yaw 0).
    assert calls[1][0] == "move_to_pose" and calls[1][2] == "wrist"
    assert calls[1][1][:3] == pytest.approx((1.2 - 0.029, 0.1, 0.4 + 0.183 + 0.1))
    assert "wrist" in control.manipulation.tool_queries  # Move along starts from where the wrist is.
    line = calls[2][1]  # A wrist line 0.05 m up at 0.1 m/s.
    assert calls[2][0] == "execute_trajectory" and line.tool == "wrist" and line.frame == "world"
    assert list(line.times) == pytest.approx([0.0, 0.5])
    assert line.poses[1] == pytest.approx([0.5, 0.0, 0.65, 0.0, 0.0, 0.0, 1.0])
    assert [event["skill"] for event in events] == ["look_at", "move_to_pose", "move_along"]
    assert events[1]["inputs"]["pose"][:3] == pytest.approx([1.171, 0.1, 0.683])


def test_a_run_with_a_clicked_pixel_uses_the_point_resolved_at_the_click():
    depth = np.full((48, 64), 2.0, dtype=np.float32)
    frame = CameraFrame(
        name="hand",
        time_seconds=0.1,
        tick=50,
        rgb=np.zeros((48, 64, 3), np.uint8),
        depth=depth,
        valid=np.ones((48, 64), bool),
        intrinsics=np.array([[32.0, 0, 32], [0, 32.0, 24], [0, 0, 1]]),
        body_from_camera=np.eye(4),
        world_from_camera=np.eye(4),
    )
    server = UiServer(Bridge({"views": ["front", "hand"]}), token="t", front_cameras=FRONT)
    server._remember(server._sent_frames.setdefault("hand", collections.OrderedDict()), 50, frame, 8)
    clicked = {"view": "hand", "tick": 50, "x": 32, "y": 24}
    assert server.probe(clicked)["found"]["world"] == pytest.approx([0.0, 0.0, 2.0])
    server._sent_frames["hand"].clear()  # The frame is gone by the time the person presses Run.
    request = {"type": "run", "skill": "look_at", "inputs": {"target": dict(clicked), "distance_m": 0.6}}
    assert server.resolve_target(request) is None
    assert request["inputs"]["target"]["point"] == pytest.approx([0.0, 0.0, 2.0])
    assert request["inputs"]["target"]["normal"] == pytest.approx([0.0, 0.0, -1.0])  # Facing the camera.
    missing = {"type": "run", "skill": "look_at", "inputs": {"target": {"view": "hand", "tick": 99, "x": 1, "y": 1}}}
    assert "No depth" in server.resolve_target(missing)


def test_a_navigation_map_error_is_reported_once_and_the_map_still_streams():
    control = fake_control()
    bridge = Bridge({"views": ["front"]})
    readings = [ValueError("Nav2 costmap geometry changed"), ValueError("Nav2 costmap geometry changed"), None]
    readings.append((np.zeros((3, 4), dtype=np.uint8), (1.0, 2.0), 0.05))

    def costmap():
        reading = readings.pop(0)
        if isinstance(reading, Exception):
            raise reading
        return reading

    session = Ui(control, bridge, cameras=(), costmap=costmap, map_period=0.0)
    for _ in range(4):
        session.update()
    errors = [event for _, event in bridge.events_after(0) if event["type"] == "error"]
    assert [event["message"] for event in errors] == ["Navigation map: Nav2 costmap geometry changed"]
    assert bridge.map().version == 1 and not bridge.map().blocked.any()
    assert session.map_updates == 1


def test_command_feedback_is_read_every_feedback_period_not_every_pass():
    """Each feedback read is an audited session call: polled every pass, a 3-minute run logged ~1,600."""
    control = fake_control()
    control.navigation = Navigation()
    reads = []
    feedback = control.navigation.feedback
    control.navigation.feedback = lambda command_id: reads.append(command_id) or feedback(command_id)
    now = [0.0]
    bridge = Bridge({"views": ["front"]})
    session = Ui(control, bridge, cameras=(), feedback_period=0.25, clock=lambda: now[0])
    bridge.request(START)
    bridge.request({"type": "run", "skill": "navigate_to", "inputs": {"x": 1.0, "y": 0.0}})
    session.update()
    assert len(reads) == 1  # Read once when it starts.
    for _ in range(4):  # 0.2 s of passes: no feedback due yet.
        now[0] += 0.05
        session.update()
    assert len(reads) == 1
    now[0] += 0.05
    session.update()
    assert len(reads) == 2
