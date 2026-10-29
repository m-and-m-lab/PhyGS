# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UI server thread: serves the page and one WebSocket client from the bridge's plain data.

Protocol (docs/design/ui/design.html, section 13). Text messages are JSON:
``hello``, ``state``, ``step``, ``command``, ``episode``, ``error``, ``probe`` from the server;
``view`` ({main, mode}), ``episode`` ({action: start or end}), ``run`` ({skill, inputs}), ``drive``
({v_x, v_y, v_rot}), ``preview`` ({skill, inputs}), ``cancel`` ({id}), ``probe`` ({view, tick,
x, y}) and ``estop`` from the client. Binary messages carry images
(:func:`~phygs.ui.codecs.binary_message`) with header kinds ``frame`` and ``map``.
The main view is sent at every new capture (at most ``main_rate`` per wall second); the
other views as thumbnails every ``thumbnail_period`` [s]. A ``recorder``
(:class:`~phygs.ui.recording.Recorder`) keeps every message both ways for replay.
"""

from __future__ import annotations

import asyncio
import collections
import json
import threading
import time
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

from phygs.observations.front_view import CameraFrame, FrontStitcher, FrontView, depth_colors
from phygs.ui.bridge import Bridge
from phygs.ui.codecs import binary_message, jpeg, map_png
from phygs.ui.recording import Recorder
from phygs.ui.targets import resolve_camera_pixel, resolve_front_pixel

WEB = Path(__file__).parent / "web"
STATIC = {"app.js": "text/javascript", "style.css": "text/css"}
FRONT, MAP = "front", "map"
REQUESTS = ("episode", "run", "drive", "preview", "cancel")  # Passed on to the session.


class UiServer:
    """Runs an aiohttp app in a daemon thread on host:port; one client at a time, with an access token.

    ``front_cameras`` are the robot's left and right front cameras the Front view stitches (none: no
    Front view)."""

    def __init__(
        self,
        bridge: Bridge,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        token: str,
        front_cameras: tuple[str, ...] = (),
        stitcher: FrontStitcher | None = None,
        max_depth: float = 6.0,
        main_rate: float = 15.0,
        thumbnail_period: float = 1.0,
        recorder: Recorder | None = None,
    ) -> None:
        if not token:
            raise ValueError("The UI requires an access token.")
        self.bridge, self.host, self.port, self.token = bridge, host, port, token
        if len(front_cameras) not in (0, 2):
            raise ValueError("The Front view stitches a left and a right camera.")
        self.front_cameras = tuple(front_cameras)
        self.stitcher = stitcher or FrontStitcher()
        self.max_depth, self.main_rate, self.thumbnail_period = max_depth, main_rate, thumbnail_period
        self.recorder = recorder
        self._client: web.WebSocketResponse | None = None
        self._front: FrontView | None = None
        # Frames sent recently, by view and capture tick: a pointed-at pixel is resolved in the
        # frame the person saw. Front views keep their virtual origin and world-from-body [m].
        self._sent_frames: dict[str, collections.OrderedDict[int, CameraFrame]] = {}
        self._sent_fronts: collections.OrderedDict[int, tuple[FrontView, np.ndarray, np.ndarray | None]] = (
            collections.OrderedDict()
        )
        # Pixels resolved recently, by (view, tick, x, y): a run uses what the person saw when clicking.
        self._resolved: collections.OrderedDict[tuple, dict | None] = collections.OrderedDict()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: web.AppRunner | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._serve, name="ui-server", daemon=True)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/?token={self.token}"

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(10.0):
            raise RuntimeError("The UI server did not start within 10 s.")
        if self._error is not None:
            raise RuntimeError(
                f"The UI server could not start on {self.host}:{self.port}: {self._error}"
            ) from self._error

    def stop(self) -> None:
        """Close the page's connection, then the server; never raises, so it cannot hide why a run ended."""
        if self._loop is not None and self._runner is not None and self._error is None and self._loop.is_running():
            client = self._client
            steps = [] if client is None else [("close the page", client.close(code=1001, message=b"Server stopped"))]
            steps.append(("stop the server", self._runner.cleanup()))
            for what, step in steps:
                try:
                    asyncio.run_coroutine_threadsafe(step, self._loop).result(5.0)
                except Exception as error:  # A timeout or a closed loop: report it and go on stopping.
                    print(f"UI server: could not {what} within 5 s ({type(error).__name__}).", flush=True)
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(5.0)

    def _serve(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/static/{name}", self._static)
        app.router.add_get("/ws", self._socket)
        self._runner = web.AppRunner(app, access_log=None)
        try:
            self._loop.run_until_complete(self._runner.setup())
            self._loop.run_until_complete(web.TCPSite(self._runner, self.host, self.port).start())
        except OSError as error:  # For example, the port is in use.
            self._error = error
            self._ready.set()
            return
        if self.port == 0:  # An ephemeral port, for tests.
            self.port = self._runner.addresses[0][1]
        self._ready.set()
        self._loop.run_forever()

    async def _index(self, request: web.Request) -> web.Response:
        return web.FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})

    async def _static(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        if name not in STATIC:
            raise web.HTTPNotFound()
        return web.FileResponse(WEB / name, headers={"Content-Type": STATIC[name], "Cache-Control": "no-store"})

    async def _socket(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse(heartbeat=10.0, compress=False)  # JPEG frames gain nothing from it.
        await socket.prepare(request)
        if request.query.get("token") != self.token:
            await socket.close(code=4003, message=b"Wrong access token")
            return socket
        if self._client is not None and not self._client.closed:
            await socket.close(code=4001, message=b"Another client is connected")
            return socket
        self._client = socket
        view = dict(main=FRONT, mode="rgb")
        await self._text(socket, {**self.bridge.hello, "type": "hello"})
        sender = asyncio.ensure_future(self._send(socket, view))
        try:
            async for message in socket:
                if message.type != WSMsgType.TEXT:
                    continue
                data = json.loads(message.data)
                if self.recorder is not None:
                    self.recorder.text("in", data)
                if data.get("type") == "view" and data.get("main") in (*self.bridge.hello["views"], MAP):
                    view.update(main=data["main"], mode="depth" if data.get("mode") == "depth" else "rgb")
                elif data.get("type") == "estop":
                    self.bridge.request(dict(type="estop"))
                elif data.get("type") == "probe":
                    await self._text(socket, self.probe(data))
                elif data.get("type") in REQUESTS:  # Checked by the session against the API.
                    problem = self.resolve_target(data)
                    if problem:
                        await self._text(socket, dict(type="error", message=problem))
                    elif not self.bridge.request(data):
                        await self._text(socket, dict(type="error", message="Too many requests; try again"))
        finally:
            sender.cancel()
            if self._client is socket:
                self._client = None
        return socket

    async def _send(self, socket: web.WebSocketResponse, view: dict) -> None:
        state_version, event_sequence, map_version = -1, 0, None
        sent: dict[tuple[str, str], int] = {}  # (view, mode) -> capture tick sent.
        thumbnails: dict[str, float] = {}
        last_main = 0.0
        while not socket.closed:
            await asyncio.sleep(0.05)
            now = time.monotonic()
            version, state = self.bridge.state()
            if version != state_version and state:
                state_version = version
                await self._text(socket, state)
            for event_sequence, event in self.bridge.events_after(event_sequence):
                await self._text(socket, event)
            image = self.bridge.map()
            if image is not None and image.version != map_version:
                map_version = image.version
                header = dict(
                    type="map",
                    version=image.version,
                    origin=image.origin,
                    resolution=image.resolution,
                    width=image.cells.shape[1],
                    height=image.cells.shape[0],
                )
                await self._binary(socket, header, map_png(image.cells, image.blocked))
            frames = self.bridge.frames()
            for name in self.bridge.hello["views"]:
                main = name == view["main"]
                if main and now - last_main < 1.0 / self.main_rate:
                    continue
                if not main and now - thumbnails.get(name, -np.inf) < self.thumbnail_period:
                    continue
                mode = view["mode"] if main else "rgb"
                encoded = self._encode(name, mode, frames, sent)
                if encoded is None:
                    continue
                header, payload = encoded
                header.update(main=main)
                await self._binary(socket, header, payload)
                if main:
                    last_main = now
                else:
                    thumbnails[name] = now

    async def _text(self, socket: web.WebSocketResponse, message: dict) -> None:
        await socket.send_str(json.dumps(message))
        if self.recorder is not None:
            self.recorder.text("out", message)

    async def _binary(self, socket: web.WebSocketResponse, header: dict, payload: bytes) -> None:
        await socket.send_bytes(binary_message(header, payload))
        if self.recorder is not None:
            self.recorder.binary(header, payload)

    def _encode(self, name: str, mode: str, frames: dict, sent: dict) -> tuple[dict, bytes] | None:
        """The newest image of a view not yet sent in this mode, as (header, JPEG)."""
        if name == FRONT:
            pair = [frames.get(camera) for camera in self.front_cameras]
            if len(pair) != 2 or None in pair or pair[0].tick != pair[1].tick:
                return None
            tick, time_seconds = pair[0].tick, pair[0].time_seconds
            if sent.get((name, mode)) == tick:
                return None
            if self._front is None or self._front.tick != tick:
                self._front = self.stitcher.stitch(*pair)
                origin = (pair[0].body_from_camera[:3, 3] + pair[1].body_from_camera[:3, 3]) / 2
                world_from_body = None
                if pair[0].world_from_camera is not None:
                    world_from_body = pair[0].world_from_camera @ np.linalg.inv(pair[0].body_from_camera)
                self._remember(self._sent_fronts, tick, (self._front, origin, world_from_body), 4)
            front = self._front
            if mode == "depth":
                distance = front.depth()
                rgb = front.outlined(depth_colors(distance, np.isfinite(distance), self.max_depth))
            else:
                rgb = front.outlined(front.rgb)
        else:
            frame = frames.get(name)
            if frame is None or sent.get((name, mode)) == frame.tick:
                return None
            tick, time_seconds = frame.tick, frame.time_seconds
            self._remember(self._sent_frames.setdefault(name, collections.OrderedDict()), tick, frame, 8)
            rgb = depth_colors(frame.depth, frame.valid, self.max_depth) if mode == "depth" else frame.rgb
        sent[(name, mode)] = tick
        header = dict(
            type="frame", view=name, mode=mode, tick=tick, time=time_seconds, width=rgb.shape[1], height=rgb.shape[0]
        )
        if mode == "depth":
            header["max_depth"] = self.max_depth
        if name == FRONT:  # Its pinhole, for drawing body-frame points on it: x ahead, y left, z up [m],
            # tilted down by pitch [rad].
            header.update(
                origin=self._sent_fronts[tick][1].tolist(), focal=self.stitcher.focal, pitch=self.stitcher.pitch
            )
        return header, jpeg(rgb)

    @staticmethod
    def _remember(history: collections.OrderedDict, tick: int, value, size: int) -> None:
        history[tick] = value
        while len(history) > size:
            history.popitem(last=False)

    def probe(self, request: dict) -> dict:
        """What the person points at: ``{view, tick, x, y}`` (source pixels) -> its depth and 3D point.

        Resolved in the sent frame with that tick, or the latest one sent if it is gone.
        """
        view, tick = request.get("view"), request.get("tick")
        x, y = int(request.get("x", -1)), int(request.get("y", -1))
        key = (view, tick, x, y)
        if key in self._resolved:
            return dict(type="probe", view=view, tick=tick, x=x, y=y, found=self._resolved[key])
        reply = dict(type="probe", view=view, tick=tick, x=x, y=y, found=None)
        if view == FRONT:
            if not self._sent_fronts:
                return reply
            front, origin, world_from_body = self._sent_fronts.get(tick) or next(reversed(self._sent_fronts.values()))
            reply["found"] = resolve_front_pixel(front, x, y, origin, world_from_body)
        else:
            history = self._sent_frames.get(view)
            if not history:
                return reply
            frame = history.get(tick) or next(reversed(history.values()))
            reply["found"] = resolve_camera_pixel(frame, x, y)
        if (view == FRONT and tick in self._sent_fronts) or tick in self._sent_frames.get(view, {}):
            self._remember(self._resolved, key, reply["found"], 64)  # Only answers from the exact frame.
        return reply

    def resolve_target(self, request: dict) -> str | None:
        """Replace a pointed pixel ``inputs.target = {view, tick, x, y}`` by its world point and normal.

        Resolved in the frame the person clicked (section 9 of the design). Returns why it cannot be
        resolved, or None.
        """
        target = (request.get("inputs") or {}).get("target")
        if not isinstance(target, dict) or "view" not in target:
            return None
        found = self.probe(target).get("found")
        if found is None:
            return "No depth at that pixel; click a surface the camera sees."
        if found["world"] is None:
            return "The robot's pose is unknown, so the pixel has no world position."
        request["inputs"]["target"] = dict(
            point=found["world"],
            normal=found["normal"],
            view=target["view"],
            pixel=found["pixel"],
            tick=target.get("tick"),
        )
        return None
