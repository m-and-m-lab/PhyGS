# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a recorded UI session to a read-only page, without the simulation.

The page gets the recorded messages on their recorded wall timing, scaled by the replay speed,
and follows the view the person had open. ``replay`` messages from the page pause, play, change
the speed or restart; every other request is ignored.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from aiohttp import WSMsgType, web

from phygs.ui.bridge import Bridge
from phygs.ui.codecs import binary_message
from phygs.ui.recording import Recording
from phygs.ui.server import UiServer

SPEEDS = (0.5, 1.0, 2.0, 4.0, 8.0)
# Messages a reconnect sends again; replayed once.
EVENTS = ("step", "command", "episode")


class ReplayServer(UiServer):
    """Serves the page and plays ``recording`` to one client at a time from its start."""

    def __init__(self, recording: Path, *, host: str = "127.0.0.1", port: int = 8766, token: str, speed: float = 1.0):
        self.recording = Recording(recording)
        hello = next(
            (item.message for item in self.recording.items if item.message and item.message.get("type") == "hello"),
            None,
        )
        if hello is None:
            raise ValueError(f"No page connected during the recording in {recording}; nothing to replay.")
        if speed not in SPEEDS:
            raise ValueError(f"Replay speed must be one of {SPEEDS}.")
        self.speed = speed
        super().__init__(Bridge(hello), host=host, port=port, token=token)

    def stop(self) -> None:
        super().stop()
        self.recording.close()

    async def _socket(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse(heartbeat=10.0, compress=False)
        await socket.prepare(request)
        if request.query.get("token") != self.token:
            await socket.close(code=4003, message=b"Wrong access token")
            return socket
        if self._client is not None and not self._client.closed:
            await socket.close(code=4001, message=b"Another client is connected")
            return socket
        self._client = socket
        player = dict(position=0.0, speed=self.speed, paused=False, restart=False)
        duration = self.recording.duration
        await socket.send_str(json.dumps({**self.bridge.hello, "replay": dict(duration=duration, speeds=SPEEDS)}))
        playing = asyncio.ensure_future(self._play(socket, player))
        try:
            async for message in socket:
                if message.type != WSMsgType.TEXT:
                    continue
                data = json.loads(message.data)
                if data.get("type") != "replay":
                    continue  # Read-only: the robot is not there.
                action = data.get("action")
                if action == "pause":
                    player["paused"] = True
                elif action == "play":
                    player["paused"] = False
                elif action == "speed" and data.get("speed") in SPEEDS:
                    player["speed"] = data["speed"]
                elif action == "restart":
                    player.update(restart=True, paused=False)
        finally:
            playing.cancel()
            if self._client is socket:
                self._client = None
        return socket

    async def _play(self, socket: web.WebSocketResponse, player: dict) -> None:
        items, duration = self.recording.items, self.recording.duration
        index, seen, last, clock_sent = 0, set(), time.monotonic(), 0.0
        while not socket.closed:
            await asyncio.sleep(0.02)
            now = time.monotonic()
            if player["restart"]:
                player.update(restart=False, position=0.0)
                index, seen = 0, set()
                await socket.send_str(json.dumps(dict(type="replay_restart")))
                await socket.send_str(
                    json.dumps({**self.bridge.hello, "replay": dict(duration=duration, speeds=SPEEDS)})
                )
            if not player["paused"]:
                player["position"] = min(duration, player["position"] + (now - last) * player["speed"])
            last = now
            while index < len(items) and items[index].time <= player["position"]:
                item = items[index]
                index += 1
                if item.direction == "in":
                    if item.message.get("type") == "view":  # Follow the view the person had open.
                        view = dict(type="replay_view", main=item.message.get("main"), mode=item.message.get("mode"))
                        await socket.send_str(json.dumps(view))
                    continue
                if item.header is not None:
                    await socket.send_bytes(binary_message(item.header, self.recording.payload(item)))
                    continue
                kind = item.message.get("type")
                if kind == "hello":
                    continue
                if kind in EVENTS:
                    key = json.dumps(item.message, sort_keys=True)
                    if key in seen:
                        continue
                    seen.add(key)
                await socket.send_str(json.dumps(item.message))
            if player["position"] >= duration:
                player["paused"] = True
            if now - clock_sent > 0.25:
                clock_sent = now
                clock = dict(type="replay_clock", position=player["position"], duration=duration)
                clock.update(speed=player["speed"], paused=player["paused"])
                await socket.send_str(json.dumps(clock))
