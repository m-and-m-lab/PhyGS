# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Internal command ownership and episode-scoped feedback bookkeeping."""

from dataclasses import dataclass

from phygs.runtime.episode import EpisodeRuntime, EpisodeStatus


@dataclass(frozen=True)
class CommandFeedback:
    """Execution status; completion is not independent evidence of physical success."""

    command_id: int
    episode_id: int
    status: str


class CommandClient:
    """Shared ownership submission and per-command feedback."""

    def __init__(self, runtime: EpisodeRuntime, name: str) -> None:
        self._runtime = runtime
        self._name = name
        self._records: dict[int, CommandFeedback] = {}
        self._current: int | None = None
        self._next_id = 1

    @property
    def name(self) -> str:
        """The runtime controller this client commands."""
        return self._name

    def _submit(self, ticks: int) -> int:
        previous = self.feedback(self._current) if self._current is not None else None
        name = self._name
        if self._runtime.controller_state(name) == "idle":
            self._runtime.acquire(name, duration_ticks=ticks)
        else:
            self._runtime.renew(name, duration_ticks=ticks)
        if previous is not None:
            status = "overridden" if previous.status in {"active", "stopping"} else previous.status
            self._records[previous.command_id] = CommandFeedback(previous.command_id, previous.episode_id, status)
        command_id = self._next_id
        self._next_id += 1
        self._current = command_id
        self._records[command_id] = CommandFeedback(command_id, self._runtime.episode_id, "active")
        return command_id

    def _end_episode(self) -> None:
        for command_id, record in tuple(self._records.items()):
            if record.status in {"active", "stopping"}:
                feedback = self.feedback(command_id)
                status = "cancelled" if feedback.status in {"active", "stopping"} else feedback.status
                self._records[command_id] = CommandFeedback(command_id, record.episode_id, status)

    def feedback(self, command_id: int) -> CommandFeedback:
        """Return active/stopping/done or a terminal cancellation/failure status."""
        record = self._records[command_id]
        if record.status not in {"active", "stopping"}:
            return record
        if record.episode_id != self._runtime.episode_id:
            status = "cancelled"
        elif self._runtime.status == EpisodeStatus.INTERRUPTED:
            status = "interrupted"
        elif self._runtime.status == EpisodeStatus.FAILED:
            status = "failed"
        elif self._runtime.status != EpisodeStatus.RUNNING:
            status = "cancelled"
        else:
            state = self._runtime.controller_state(self._name)
            status = "done" if state == "idle" else state
        result = CommandFeedback(record.command_id, record.episode_id, status)
        if status not in {"active", "stopping"}:
            self._records[command_id] = result
        return result
