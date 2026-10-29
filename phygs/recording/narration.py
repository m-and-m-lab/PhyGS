# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Skill step narration: what manipulation, navigation and locomotion skills do now.

Skills log one line per step under ``phygs.skills.<kind>`` (``manipulation``,
``navigation``, ``locomotion``; checks may add ``check``). :class:`Narration` collects them
with the simulation time, prints them, and keeps the latest line per kind for video overlays.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

ROOT = "phygs.skills"


class Narration(logging.Handler):
    """Collects skill step lines with the simulation time [s]; prints each one."""

    def __init__(self, clock: Callable[[], float], *, echo: bool = True) -> None:
        super().__init__(logging.INFO)
        self.clock, self._echo = clock, echo  # Simulation time [s]; may be set once a runtime exists.
        self.latest: dict[str, str] = {}
        self.log: list[tuple[float, str, str]] = []  # (time [s], kind, text)

    def emit(self, record: logging.LogRecord) -> None:
        kind, text, time = record.name.rsplit(".", 1)[-1], record.getMessage(), float(self.clock())
        self.latest[kind] = text
        self.log.append((round(time, 3), kind, text))
        if self._echo:
            print(f"[{time:8.3f} s] {text}", flush=True)

    def attach(self) -> Narration:
        """Receive every skill's steps (INFO and above)."""
        root = logging.getLogger(ROOT)
        root.setLevel(logging.INFO)
        root.addHandler(self)
        return self

    def detach(self) -> None:
        logging.getLogger(ROOT).removeHandler(self)

    def lines(self, kinds: tuple[str, ...] = ("manipulation", "navigation", "locomotion", "check")) -> list[str]:
        """The latest line of each kind that has one, for an overlay."""
        return [f"{kind}: {self.latest[kind]}" for kind in kinds if kind in self.latest]
