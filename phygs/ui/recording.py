# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""What the page received and sent, recorded for replay without the simulation.

A recording is a folder with ``messages.jsonl`` (one line per message: wall time [s] since the
recording started, direction ``out`` to the page or ``in`` from it, and the JSON message, or for
an image its header and where its bytes are) and ``payloads.bin`` (the JPEG and PNG bytes).
:class:`Recording` reads it back; ``phygs/ui/replay_cli.py`` serves it to a read-only page.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path


class Recorder:
    """Appends each message with its wall time; safe to call from several threads."""

    def __init__(self, directory: Path, clock=time.monotonic) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self._clock, self._started = clock, clock()
        self._lock = threading.Lock()
        self._messages = (directory / "messages.jsonl").open("a")
        self._payloads = (directory / "payloads.bin").open("ab")
        self._offset = self._payloads.tell()

    def text(self, direction: str, message: dict) -> None:
        """A JSON message, ``out`` to the page or ``in`` from it."""
        self._write(dict(t=self._now(), dir=direction, message=message))

    def binary(self, header: dict, payload: bytes) -> None:
        """An image sent to the page: its header, and its bytes in ``payloads.bin``."""
        with self._lock:
            if self._payloads.closed:
                return
            self._payloads.write(payload)
            offset, self._offset = self._offset, self._offset + len(payload)
            line = dict(t=self._now(), dir="out", header=header, offset=offset, length=len(payload))
            self._messages.write(json.dumps(line, separators=(",", ":")) + "\n")

    def close(self) -> None:
        with self._lock:
            self._messages.close()
            self._payloads.close()

    def _now(self) -> float:
        return round(self._clock() - self._started, 4)

    def _write(self, line: dict) -> None:
        with self._lock:
            if not self._messages.closed:
                self._messages.write(json.dumps(line, separators=(",", ":")) + "\n")


@dataclass(frozen=True)
class Recorded:
    """One recorded message: wall time [s], direction, and the JSON message or an image header
    with where its bytes are in ``payloads.bin``."""

    time: float
    direction: str
    message: dict | None = None
    header: dict | None = None
    offset: int = 0
    length: int = 0


class Recording:
    """A recording read back: its messages in order; image bytes are read when asked for."""

    def __init__(self, directory: Path) -> None:
        self.items: list[Recorded] = []
        for line in (directory / "messages.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if "header" in item:
                self.items.append(
                    Recorded(
                        item["t"], item["dir"], header=item["header"], offset=item["offset"], length=item["length"]
                    )
                )
            else:
                self.items.append(Recorded(item["t"], item["dir"], message=item["message"]))
        self._payloads = (directory / "payloads.bin").open("rb")

    @property
    def duration(self) -> float:
        """Wall time [s] from the start of the recording to its last message."""
        return self.items[-1].time if self.items else 0.0

    def payload(self, item: Recorded) -> bytes:
        self._payloads.seek(item.offset)
        return self._payloads.read(item.length)

    def close(self) -> None:
        self._payloads.close()
