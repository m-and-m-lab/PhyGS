# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thread-safe exchange of plain data between the runner thread and the UI server thread.

The runner publishes the latest frame per camera, the latest robot state and map, and a
bounded list of events (skill step lines, command changes). The server reads them and
queues the person's requests for the runner. Nothing here refers to simulation objects.
"""

from __future__ import annotations

import collections
import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from phygs.tools.front_view import CameraFrame


@dataclass(frozen=True)
class MapImage:
    """Occupancy grid [y,x] (-1 unknown, 0 free, 100 occupied), its origin (lowest x, y) [m] and cell size [m];
    ``blocked`` [y,x]: where the navigator will not put the robot's centre (None: no navigator)."""

    cells: np.ndarray
    origin: tuple[float, float]
    resolution: float
    version: int
    blocked: np.ndarray | None = None


class Bridge:
    """Latest values per stream with version counters, a bounded event list, and a request queue."""

    def __init__(self, hello: dict, *, log_path: Path | None = None, max_events: int = 4000) -> None:
        self.hello = hello  # Static episode information sent when a client connects.
        self._lock = threading.Lock()
        self._frames: dict[str, CameraFrame] = {}
        self._state: dict = {}
        self._state_version = 0
        self._map: MapImage | None = None
        self._events: collections.deque[tuple[int, dict]] = collections.deque(maxlen=max_events)
        self._next_event = 1
        self._requests: queue.Queue[dict] = queue.Queue(maxsize=64)
        self._log = None if log_path is None else log_path.open("a")

    # Runner side.
    def publish_frame(self, frame: CameraFrame) -> None:
        with self._lock:
            self._frames[frame.name] = frame

    def publish_state(self, state: dict) -> None:
        with self._lock:
            self._state, self._state_version = state, self._state_version + 1

    def publish_map(self, image: MapImage) -> None:
        with self._lock:
            self._map = image

    def push_event(self, event: dict) -> None:
        with self._lock:
            self._events.append((self._next_event, event))
            self._next_event += 1
        if self._log is not None:
            self._log.write(json.dumps(event, separators=(",", ":")) + "\n")
            self._log.flush()

    def take_requests(self) -> list[dict]:
        """Requests queued since the last call, oldest first."""
        requests = []
        while True:
            try:
                requests.append(self._requests.get_nowait())
            except queue.Empty:
                return requests

    # Server side.
    def frames(self) -> dict[str, CameraFrame]:
        with self._lock:
            return dict(self._frames)

    def state(self) -> tuple[int, dict]:
        with self._lock:
            return self._state_version, self._state

    def map(self) -> MapImage | None:
        with self._lock:
            return self._map

    def events_after(self, sequence: int) -> list[tuple[int, dict]]:
        """Events newer than sequence, as (sequence, event); old events beyond the bound are gone."""
        with self._lock:
            return [(number, event) for number, event in self._events if number > sequence]

    def request(self, request: dict[str, Any]) -> bool:
        """Queue a request for the runner; False when the queue is full."""
        try:
            self._requests.put_nowait(request)
        except queue.Full:
            return False
        return True

    def close(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None
