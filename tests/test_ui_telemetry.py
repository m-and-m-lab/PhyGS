# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The telemetry the page's drawer shows and the launcher saves, and a clean stop with a page connected."""

import asyncio
import json
import math
import threading
import time
from types import SimpleNamespace

import pytest
import torch
from fakes import SPOT
from ui_fakes import START, Feedback, Ui, camera, fake_control

from phygs.ui.bridge import Bridge
from phygs.ui.server import UiServer


def turned(yaw):
    matrix = torch.eye(4)
    matrix[:2, :2] = torch.tensor([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    return matrix


def test_telemetry_comes_from_the_agent_view_and_is_saved_every_period(tmp_path):
    control = fake_control()
    packet = control.observe()
    packet.localization.linear_velocity = torch.tensor([0.3, -0.1, 0.0])
    control.manipulation = SimpleNamespace(holding=False, feedback=lambda command_id: Feedback(command_id, "active"))
    wrist = torch.eye(4).numpy()
    wrist[:3, 3] = [0.8, 0.0, 0.5]  # The wrist link in the body frame [m].
    bridge = Bridge({"views": ["front", "hand"]})
    health = {"physics_ms": 40.0, "ui_ms": 6.0, "planner": {"running": True, "busy": False, "restarts": 0}}
    session = Ui(
        control,
        bridge,
        cameras=("hand",),
        health=lambda: health,
        forward=lambda joints: {"arm_link_wr1": wrist},  # The robot's model: joints -> body-frame links.
        telemetry_path=tmp_path / "telemetry.jsonl",
        telemetry_period=0.5,
    )
    bridge.request(START)
    session.track("manipulation", "move_to_pose", 1)
    for step, (time_s, yaw) in enumerate(((12.5, 0.0), (12.6, 0.05), (13.1, 0.05))):
        control.runtime.time_seconds = time_s
        packet.localization.pose.matrix = turned(yaw)
        packet.cameras["hand"] = camera(8 * (step + 1), 10)  # A capture every 8 ticks of 0.004 s.
        session.update()
        if step == 1:
            telemetry = bridge.state()[1]["telemetry"]
            assert telemetry["base"]["yaw_rate"] == pytest.approx(0.5)  # 0.05 rad in 0.1 s.
    session.close()
    telemetry = bridge.state()[1]["telemetry"]
    assert telemetry["base"]["vx"] == pytest.approx(0.3) and telemetry["base"]["vy"] == pytest.approx(-0.1)
    assert telemetry["joint_speeds"]["arm_sh0"] == pytest.approx(0.1)
    hand = turned(0.05).numpy() @ wrist @ SPOT.tool("hand")  # From the view's pose and joints.
    assert telemetry["hand"] == pytest.approx(hand[:3, 3].tolist())
    assert telemetry["tracking"]["skill"] == "move_to_pose"  # The running arm command.
    assert telemetry["cameras"]["hand"]["fps"] == pytest.approx(31.25)
    assert telemetry["health"] == health and "commands" in telemetry and "failures" in telemetry
    lines = [json.loads(line) for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()]
    assert [line["sim_time"] for line in lines] == [12.5, 13.1]  # Every 0.5 s of sim time.
    assert lines[-1]["base"]["vx"] == pytest.approx(0.3) and "commands" not in lines[-1]


def test_the_server_stops_cleanly_with_a_page_still_connected():
    aiohttp = pytest.importorskip("aiohttp")
    server = UiServer(Bridge({"views": ["front"]}), port=0, token="t")
    server.start()
    connected, closed = threading.Event(), {}

    async def page():
        async with aiohttp.ClientSession() as http:
            async with http.ws_connect(f"http://127.0.0.1:{server.port}/ws?token=t") as socket:
                connected.set()
                async for _ in socket:
                    pass
                closed["code"] = socket.close_code

    thread = threading.Thread(target=lambda: asyncio.run(page()), daemon=True)
    thread.start()
    assert connected.wait(5.0)
    started = time.monotonic()
    server.stop()  # Must not raise, as the page holds its connection open.
    thread.join(5.0)
    assert time.monotonic() - started < 8.0
    assert closed.get("code") == 1001  # The page was told the server went away.
