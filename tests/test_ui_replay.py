# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recording what the page received and sent, and replaying it to a read-only page."""

import asyncio
import json

import aiohttp
import numpy as np
import pytest
from ui_fakes import frame

from phygs.ui.bridge import Bridge, MapImage
from phygs.ui.codecs import parse_binary_message
from phygs.ui.recording import Recorder, Recording
from phygs.ui.replay import ReplayServer
from phygs.ui.server import UiServer


async def talk(port, token, outgoing=(), seconds=1.5):
    """Connect, send ``outgoing``, and collect (kind, message or header, payload) for ``seconds``."""
    received = []
    async with aiohttp.ClientSession() as http:
        async with http.ws_connect(f"http://127.0.0.1:{port}/ws?token={token}") as socket:
            for message in outgoing:
                await socket.send_str(json.dumps(message))
            end = asyncio.get_running_loop().time() + seconds
            while asyncio.get_running_loop().time() < end:
                try:
                    message = await socket.receive(timeout=0.2)
                except TimeoutError:
                    continue
                if message.type == aiohttp.WSMsgType.TEXT:
                    received.append(("text", json.loads(message.data), None))
                elif message.type == aiohttp.WSMsgType.BINARY:
                    header, payload = parse_binary_message(message.data)
                    received.append(("binary", header, payload))
    return received


def test_recorder_keeps_messages_both_ways_and_image_bytes(tmp_path):
    recorder = Recorder(tmp_path)
    recorder.text("out", {"type": "state", "sim_time": 1.0})
    recorder.binary({"type": "frame", "view": "hand"}, b"jpeg-bytes")
    recorder.text("in", {"type": "view", "main": "hand"})
    recorder.binary({"type": "map"}, b"png")
    recorder.close()
    recording = Recording(tmp_path)
    assert [(item.direction, (item.message or item.header)["type"]) for item in recording.items] == [
        ("out", "state"),
        ("out", "frame"),
        ("in", "view"),
        ("out", "map"),
    ]
    assert recording.payload(recording.items[1]) == b"jpeg-bytes" and recording.payload(recording.items[3]) == b"png"
    assert recording.duration == recording.items[-1].time >= 0.0
    recording.close()


def test_a_recorded_session_replays_read_only_with_the_persons_view(tmp_path):
    bridge = Bridge({"views": ["front", "hand"], "task": "Find the green box."})
    bridge.publish_state({"type": "state", "sim_time": 1.0, "status": "running", "phase": "running"})
    bridge.publish_frame(frame("hand", 50))
    bridge.publish_map(MapImage(np.zeros((3, 4), dtype=np.int8), (0.0, 0.0), 0.05, 1))
    bridge.push_event({"type": "command", "id": "c1", "skill": "stop", "state": "succeeded", "sim_time": 0.5})
    recorder = Recorder(tmp_path / "recording")
    live = UiServer(bridge, port=0, token="live", thumbnail_period=0.1, recorder=recorder)
    live.start()
    try:
        recorded = asyncio.run(talk(live.port, "live", [{"type": "view", "main": "hand", "mode": "rgb"}]))
        recorded += asyncio.run(talk(live.port, "live", seconds=0.5))  # A reconnect sends the events again.
    finally:
        live.stop()
        recorder.close()
    commands = [m for kind, m, _ in recorded if kind == "text" and m["type"] == "command"]
    assert len(commands) == 2  # Once per connection.
    sent_frames = [(h, p) for kind, h, p in recorded if kind == "binary" and h["type"] == "frame"]

    with pytest.raises(ValueError, match="speed"):
        ReplayServer(tmp_path / "recording", port=0, token="replay", speed=3.0)
    replay = ReplayServer(tmp_path / "recording", port=0, token="replay", speed=8.0)
    replay.start()
    try:
        played = asyncio.run(talk(replay.port, "replay", [{"type": "estop"}, {"type": "run", "skill": "stop"}]))
    finally:
        replay.stop()
    texts = [m for kind, m, _ in played if kind == "text"]
    assert texts[0]["type"] == "hello" and texts[0]["task"] == "Find the green box."
    assert texts[0]["replay"]["duration"] > 0 and 8.0 in texts[0]["replay"]["speeds"]
    assert [m for m in texts if m["type"] == "hello"] == texts[:1]  # The reconnect's hello is not replayed.
    assert len([m for m in texts if m["type"] == "command"]) == 1  # Nor its repeated events.
    assert {"type": "replay_view", "main": "hand", "mode": "rgb"} in texts
    assert any(m["type"] == "replay_clock" for m in texts) and any(m["type"] == "state" for m in texts)
    played_frames = [(h, p) for kind, h, p in played if kind == "binary" and h["type"] == "frame"]
    assert played_frames[0] == sent_frames[0]  # The same header and JPEG bytes the person got.
    # Read-only: the page's estop and run requests reach no bridge a runner would take them from.
    assert replay.bridge.take_requests() == []
