# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run one skill sequence: a generator that yields low-level command ids and receives their feedback."""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Any

from phygs.execution.manipulation.skills.types import SkillError, SkillFailure

SkillSequence = Generator[int, Any, None]


class SkillRun:
    """Advance a skill when its current low-level command ends; record the outcome.

    ``advancing`` is true only while the sequence issues its next command, so the client
    can accept that command while rejecting agent commands during the skill.
    """

    def __init__(self, sequence: SkillSequence, feedback: Callable[[int], Any]) -> None:
        self._sequence, self._feedback = sequence, feedback
        self.command: int | None = None
        self.done = False
        self.failure: SkillFailure | None = None
        self.message = ""
        self.advancing = False
        self._send(None)

    def update(self) -> None:
        """Send the finished command's feedback to the sequence; wait while it runs."""
        if self.done:
            return
        feedback = self._feedback(self.command)
        if feedback.status not in {"active", "stopping"}:
            self._send(feedback)

    def cancel(self) -> None:
        """Stop issuing commands; the caller cancels the current low-level command."""
        if not self.done:
            self._sequence.close()
            self._finish(SkillFailure.CANCELLED, "Cancelled by agent")

    def _send(self, value: Any) -> None:
        self.advancing = True
        try:
            self.command = self._sequence.send(value)
        except StopIteration:
            self._finish(None, "Skill completed")
        except SkillError as error:
            self._finish(error.failure, error.message)
        finally:
            self.advancing = False

    def _finish(self, failure: SkillFailure | None, message: str) -> None:
        self.done, self.failure, self.message = True, failure, message
