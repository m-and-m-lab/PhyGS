# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""An episode's output folder: ``report.json``, the state log, the third-person video and contact records."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from phygs.recording.panels import PanelSources
from phygs.recording.state_log import StateLog
from phygs.recording.video import Video
from phygs.scene.plan import Plan


def pinched_bodies(contacts: list[tuple[str, str, float]], jaw_links: tuple[str, ...]) -> set[str]:
    """The bodies every jaw link presses on in one tick's arm contacts (link, body, force [N])."""
    touched: dict[str, set[str]] = {}
    for link, body, _ in contacts:
        touched.setdefault(body, set()).add(link)
    return {body for body, links in touched.items() if set(jaw_links) <= links}


class Recorder:
    """Writes the report and, with ``video``, ``third_person.mp4`` (recording.video: the outside view,
    the panels and the overview strip, with the scene's floor ``plan``).

    Arm contacts are recorded apart as the grasp when, from the gripper's close command to its
    opening (one hold), a jaw link touches a body both jaws pinched at some tick of that hold: the
    squeeze on a picked object or a drawer handle, including the jaw contact before the second jaw
    lands and the one-jaw load while pulling. Every other arm contact, including an open hand resting
    both jaws on a surface, is a ``contact``.

    Contacts are read every ``read_ticks`` physics ticks; a record's ``ticks`` counts each read as
    that many ticks. ``state`` is the run's state log (:mod:`phygs.recording.state_log`), sampled by
    the World and written with the report.
    """

    def __init__(
        self,
        output: Path,
        *,
        video: bool,
        fps: float,
        jaw_links: tuple[str, ...] = (),
        read_ticks: int = 1,
        state: StateLog | None = None,
        plan: Plan | None = None,
    ) -> None:
        self.output = output
        self.state = state
        self._video = Video(output / "third_person.mp4", fps=fps, plan=plan) if video else None
        self._jaw_links = tuple(jaw_links)
        self._read_ticks = read_ticks
        self.contacts: dict[tuple[str, str, bool], dict] = {}
        self._hold: dict[tuple[str, str], dict] = {}  # Jaw contacts of the current hold, not yet sorted.
        self._pinched: set[str] = set()  # Bodies both jaws pinched in the current hold.

    def frame(
        self,
        outside: np.ndarray,
        *,
        sources: PanelSources | None,
        trace: dict,
        sim_time: float,
        duration: float | None = None,
    ) -> None:
        """One video frame: the ``outside`` image with the panels (cameras, costmap, arm plan) from
        ``sources`` beside it and the overview strip of ``trace`` under both (recording.video)."""
        if self._video is not None:
            self._video.frame(outside, sources=sources, trace=trace, sim_time=sim_time, duration=duration)

    def contacts_tick(
        self, time_seconds: float, contacts: list[tuple[str, str, float]], step: str, *, gripper_closed: bool
    ) -> None:
        """One tick's arm contacts above the record threshold (link, body, force [N]); ``step`` labels
        them and ``gripper_closed`` says whether the gripper is commanded closed."""
        if not gripper_closed:
            self._end_hold()
        for link, body, force in contacts:
            if gripper_closed and link in self._jaw_links:
                record = self._hold.setdefault((link, body), _record(link, body, time_seconds, step))
                _merge(record, time_seconds, force, self._read_ticks)
            else:
                self.contact(time_seconds, link, body, force, step)
        if gripper_closed and self._jaw_links:
            self._pinched |= pinched_bodies(contacts, self._jaw_links)

    def contact(
        self, time_seconds: float, link: str, body: str, force: float, step: str, *, grasp: bool = False
    ) -> None:
        """Arm contact above the record threshold: first and last time, peak force [N], ticks; a
        ``grasp`` contact is recorded apart from the others."""
        record = self.contacts.setdefault((link, body, grasp), _record(link, body, time_seconds, step))
        _merge(record, time_seconds, force, self._read_ticks)

    def _end_hold(self) -> None:
        """Sort the hold's jaw contacts: the grasp if both jaws pinched their body in the hold."""
        self.contacts = _sorted(self.contacts, self._hold, self._pinched)
        self._hold, self._pinched = {}, set()

    def contact_report(self) -> dict[str, list[dict]]:
        """The report's ``contacts`` (unintended) and ``grasp_contacts`` (the squeeze on a grasped body)."""
        self._end_hold()
        return _report(self.contacts)

    def contacts_so_far(self) -> dict[str, list[dict]]:
        """:meth:`contact_report` now, as if the current hold ended (copies; the hold goes on)."""
        return _report(_sorted({key: dict(record) for key, record in self.contacts.items()}, self._hold, self._pinched))

    def write(self, report: dict, commands: Sequence[dict] = ()) -> Path:
        """Write ``report.json``, and the state log with the command timeline's spans ``commands``."""
        path = self.output / "report.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        if self.state is not None:
            self.state.write(commands)
        return path

    def close(self) -> None:
        if self._video is not None:
            self._video.close()
            self._video = None


def _sorted(contacts: dict, hold: dict, pinched: set[str]) -> dict:
    """``contacts`` with a hold's jaw contacts merged in: the grasp for a body both jaws pinched."""
    for (link, body), held in hold.items():
        grasp = body in pinched
        record = contacts.get((link, body, grasp))
        if record is None:
            contacts[(link, body, grasp)] = dict(held)
        else:
            record.update(last_s=held["last_s"], ticks=record["ticks"] + held["ticks"])
            record["peak_n"] = max(record["peak_n"], held["peak_n"])
    return contacts


def _report(contacts: dict) -> dict[str, list[dict]]:
    records = {False: [], True: []}
    for (_, _, grasp), record in contacts.items():
        records[grasp].append(record)
    return dict(contacts=records[False], grasp_contacts=records[True])


def _record(link: str, body: str, time_seconds: float, step: str) -> dict:
    return dict(link=link, body=body, first_s=round(time_seconds, 3), peak_n=0.0, ticks=0, step=step)


def _merge(record: dict, time_seconds: float, force: float, ticks: int) -> None:
    record.update(last_s=round(time_seconds, 3), ticks=record["ticks"] + ticks)
    record["peak_n"] = round(max(record["peak_n"], force), 1)
