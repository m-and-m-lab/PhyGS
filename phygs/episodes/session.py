# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""What an agent gets: the session, its requests for simulated time, and the audit log.

No Isaac Sim imports. The Environment owns the simulation loop on the main thread (agents never
step physics); an agent runs ``run(session)`` in its own thread. Every client call is carried
out on the main thread between physics ticks (:class:`MainThread`). Simulated time passes only
while the agent waits (``charged`` clock) or always (``realtime``), as the Environment decides.

Agents import from this module only (with the standard library, NumPy and PyYAML); it re-exports
the target types skills accept. ``session.robot`` (:class:`RobotInfo`) says which robot the agent
drives: its arm joints, gripper, tools and cameras.
"""

from __future__ import annotations

import json
import math
import queue
import threading
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from phygs.execution.manipulation.lowering import TrajectoryRefused  # noqa: F401 (re-exported for agents)
from phygs.execution.manipulation.planning import (  # noqa: F401 (re-exported for agents)
    EndEffectorTrajectory,
    JointTrajectory,
)
from phygs.execution.manipulation.skills.types import (  # noqa: F401 (re-exported for agents)
    ArticulatedPart,
    HeldObject,
    ObjectTarget,
    PixelTarget,
    PointTarget,
    SkillFailure,
    entry_depth,
    motion_failure,
    unit_approach,
)
from phygs.robots.description import RobotInfo  # noqa: F401 (re-exported for agents)

# Command statuses of a command still running (every skill client).
ACTIVE = frozenset({"active", "stopping"})
# Session client name -> RobotControl action interface it needs, and the level it is recorded as.
CLIENTS = {
    "navigation": ("navigation", "skills"),
    "manipulation": ("manipulation", "skills"),
    "locomotion": ("motion", "motion"),
    "arm": ("arm", "joints"),
}
# Client members only the Environment uses: ticking, resets, wiring.
ENVIRONMENT_ONLY = frozenset(
    {"update", "reset", "close", "bind_skills", "bind_kinematics", "bind_navigation", "backend", "controller"}
)


class EpisodeEnded(RuntimeError):
    """The episode ended (finish, budget, emergency stop) while or before the agent called."""


class AgentFailed(RuntimeError):
    """Raised by an agent that gives up; the reason goes into the report."""


@dataclass(frozen=True)
class PublicTask:
    """What the agent is told: the goal and the target's description (not its scene name)."""

    goal: str
    description: str


@dataclass(frozen=True)
class TargetHint:
    """Privileged (object_targets): the target's scene name, its box now (center, size) [m], and
    the part it is inside (``inside``, e.g. a drawer) or the furniture it rests on (``on``)."""

    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    inside: str | None
    on: str | None


class Command(int):
    """A command id that remembers its client, so :meth:`AgentSession.wait` knows whom to ask."""

    client: str

    def __new__(cls, value: int, client: str) -> Command:
        command = int.__new__(cls, value)
        command.client = client
        return command

    def __getnewargs__(self) -> tuple[int, str]:
        # Copies (the audit copies feedback that holds the id the agent passed) keep the client.
        return int(self), self.client


class _Request:
    def __init__(self, function: Callable[[], Any]) -> None:
        self.function, self.done = function, threading.Event()
        self.result: Any = None
        self.error: BaseException | None = None


class MainThread:
    """Carries agent requests to the main thread and holds agents until a condition holds there.

    Agent side: :meth:`call` and :meth:`wait_until` block. Main side: :meth:`serve` runs queued
    requests, :meth:`wake` releases waiters whose condition holds after a tick, :meth:`end`
    releases everyone with :class:`EpisodeEnded`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: queue.Queue[_Request] = queue.Queue()
        self._waiters: list[tuple[Callable[[], bool], _Request]] = []
        self._ended: str | None = None
        self.served = 0  # Requests run so far.

    # Agent side.
    def call(self, function: Callable[..., Any], *args, **kwargs) -> Any:
        """Run ``function(*args, **kwargs)`` on the main thread and return its result or raise its error."""
        request = _Request(lambda: function(*args, **kwargs))
        with self._lock:
            if self._ended is not None:
                raise EpisodeEnded(self._ended)
            self._requests.put(request)
        return self._result(request)

    def wait_until(self, condition: Callable[[], bool]) -> None:
        """Block until ``condition()``, evaluated on the main thread after each tick, holds."""
        request = _Request(condition)
        with self._lock:
            if self._ended is not None:
                raise EpisodeEnded(self._ended)
            self._waiters.append((condition, request))
        self._result(request)

    @staticmethod
    def _result(request: _Request) -> Any:
        request.done.wait()
        if request.error is not None:
            raise request.error
        return request.result

    # Main side.
    @property
    def waiting(self) -> bool:
        """True while an agent waits for simulated time."""
        with self._lock:
            return bool(self._waiters)

    @property
    def ended(self) -> str | None:
        return self._ended

    def serve(self, timeout: float = 0.0) -> int:
        """Run the queued requests; with nothing queued, wait up to ``timeout`` [s wall] for one."""
        served = 0
        try:
            request = self._requests.get(timeout=timeout) if timeout > 0 else self._requests.get_nowait()
        except queue.Empty:
            return 0
        while True:
            try:
                request.result = request.function()
            except BaseException as error:  # Delivered to the agent's thread.
                request.error = error
            request.done.set()
            served += 1
            self.served += 1
            try:
                request = self._requests.get_nowait()
            except queue.Empty:
                return served

    def wake(self) -> None:
        """Release the waiters whose condition holds now."""
        with self._lock:
            waiters = list(self._waiters)
        for condition, request in waiters:
            try:
                if not condition():
                    continue
                request.result = True
            except BaseException as error:
                request.error = error
            with self._lock:
                self._waiters.remove((condition, request))
            request.done.set()

    def end(self, reason: str) -> None:
        """End the episode for the agent: every blocked and later call raises :class:`EpisodeEnded`."""
        with self._lock:
            self._ended = reason
            pending = [request for _, request in self._waiters]
            self._waiters.clear()
            while not self._requests.empty():
                pending.append(self._requests.get_nowait())
        for request in pending:
            request.error = EpisodeEnded(reason)
            request.done.set()


def jsonable(value: Any) -> Any:
    """A JSON-friendly copy of call arguments, results and feedback."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return {type(value).__name__: jsonable(asdict(value))}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return repr(value)


class Audit:
    """One JSON line per agent call (``audit.jsonl``) and the totals for the report."""

    def __init__(self, path: Path | None, time_seconds: Callable[[], float]) -> None:
        self._file = None if path is None else path.open("w")
        self._time = time_seconds
        self.calls: Counter[str] = Counter()
        self.privileged: Counter[str] = Counter()
        self.levels: set[str] = set()
        self.commands: list[dict] = []

    def record(self, call: str, *, level: str | None = None, privileged=(), **fields) -> None:
        """Record one call; ``fields`` (arguments, result, error, feedback) are stored as JSON."""
        self.calls[call.split(".")[0]] += 1
        self.privileged.update(privileged)
        if level is not None:
            self.levels.add(level)
        line = dict(t=round(self._time(), 4), call=call, **{k: jsonable(v) for k, v in fields.items()})
        if privileged:
            line["privileged"] = list(privileged)
        if "feedback" in fields:
            self.commands.append(line)
        if self._file is not None:
            self._file.write(json.dumps(line) + "\n")
            self._file.flush()

    def summary(self) -> dict:
        return dict(
            calls=sum(self.calls.values()),
            by_client=dict(self.calls),
            privileged_reads=dict(self.privileged),
            levels_used=sorted(self.levels),
        )

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


class _Proxy:
    """A client as an agent sees it: public members only, every call on the main thread and audited."""

    def __init__(self, name: str, target: Any, main: MainThread, audit: Audit, level: str | None, privileged) -> None:
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_main", main)
        object.__setattr__(self, "_audit", audit)
        object.__setattr__(self, "_level", level)
        object.__setattr__(self, "_privileged", privileged)

    def __getattr__(self, member: str) -> Any:
        if member.startswith("_") or member in ENVIRONMENT_ONLY:
            raise AttributeError(f"{self._name}.{member} is not available to agents")
        value = self._main.call(getattr, self._target, member)
        if not callable(value):
            if not _plain(value):
                raise AttributeError(f"{self._name}.{member} is not available to agents")
            return value
        call = f"{self._name}.{member}"

        def invoke(*args, **kwargs):
            privileged = self._privileged(member, args, kwargs)
            try:
                result = self._main.call(value, *args, **kwargs)
            except BaseException as error:
                self._audit.record(
                    call,
                    level=self._level,
                    privileged=privileged,
                    args=args,
                    kwargs=kwargs,
                    error=f"{type(error).__name__}: {error}",
                )
                raise
            if type(result) is int:  # A command id.
                result = Command(result, self._name)
            self._audit.record(call, level=self._level, privileged=privileged, args=args, kwargs=kwargs, result=result)
            return result

        return invoke

    def __setattr__(self, member: str, value: Any) -> None:
        raise AttributeError(f"{self._name} is read-only for agents")

    def __dir__(self) -> list[str]:
        return [m for m in dir(self._target) if not m.startswith("_") and m not in ENVIRONMENT_ONLY]


def _plain(value: Any) -> bool:
    """Data an agent may read from a client attribute (not another client or backend object)."""
    if isinstance(value, (bool, int, float, str, tuple, list, dict, np.ndarray)) or value is None:
        return True
    return is_dataclass(value) and not isinstance(value, type)


class AgentSession:
    """The only object an agent receives.

    ``robot`` describes the robot (:class:`RobotInfo`: arm joints, gripper, tools, cameras).
    ``navigation``, ``manipulation``, ``locomotion`` and ``arm`` are the skill clients the tier
    allows (None otherwise); ``hints`` are the privileged hints, each call refused without its
    grant. Command ids are :class:`Command` ints; :meth:`wait` returns a command's final feedback.
    """

    def __init__(
        self,
        *,
        task: PublicTask,
        robot: RobotInfo,
        grants: tuple[str, ...],
        clients: dict[str, Any],
        hints: Any,
        observe: Callable[[], Any],
        time_seconds: Callable[[], float],
        finish: Callable[[str], None],
        camera_period: float,
        main: MainThread,
        audit: Audit,
        privileged: Callable[[str, str, tuple, dict], list[str]],
    ) -> None:
        if set(clients) != set(CLIENTS):
            raise ValueError(f"Expected clients {sorted(CLIENTS)}.")
        self.task, self.robot, self.grants = task, robot, tuple(grants)
        self._main, self._audit = main, audit
        self._observe, self._time, self._finish = observe, time_seconds, finish
        self._camera_period = camera_period
        self._clients = {name: client for name, client in clients.items() if client is not None}
        for name, client in clients.items():
            proxy = None
            if client is not None:
                proxy = _Proxy(name, client, main, audit, CLIENTS[name][1], partial_privileged(privileged, name))
            setattr(self, name, proxy)
        self.hints = _Proxy("hints", hints, main, audit, None, lambda member, args, kwargs: hints.grant(member))

    def observe(self) -> Any:
        """The latest observation packet the spec allows; time does not pass."""
        return self._main.call(self._observe)

    def time(self) -> float:
        """Simulated episode time [s]."""
        return self._main.call(self._time)

    def wait(self, command: Command) -> Any:
        """Let simulated time pass until the command ends; return its final feedback."""
        if not isinstance(command, Command):
            raise TypeError("wait() takes a command id returned by a session client.")
        client = self._clients[command.client]
        self._main.wait_until(lambda: client.feedback(command).status not in ACTIVE)
        feedback = self._main.call(client.feedback, command)
        self._audit.record(f"{command.client}.wait", command=int(command), feedback=feedback)
        return feedback

    def sleep(self, seconds: float) -> None:
        """Let ``seconds`` [s] of simulated time pass."""
        if not (math.isfinite(seconds) and seconds >= 0):
            raise ValueError("sleep() takes a finite, non-negative duration [s].")
        end = self.time() + seconds
        self._main.wait_until(lambda: self._time() >= end - 1e-9)

    def stream(self, seconds: float | None = None, until: Callable[[Any], bool] | None = None) -> Iterator[Any]:
        """Yield one observation per camera period while time passes, for ``seconds`` [s] or ``until(packet)``."""
        if seconds is None and until is None:
            raise ValueError("stream() needs seconds, until, or both.")
        end = None if seconds is None else self.time() + seconds
        while True:
            self.sleep(self._camera_period)
            packet = self.observe()
            yield packet
            if (end is not None and self.time() >= end - 1e-9) or (until is not None and until(packet)):
                return

    def finish(self, reason: str = "") -> None:
        """End the episode now; returning from ``run`` does the same."""
        self._main.call(self._finish, reason)


def partial_privileged(privileged: Callable[[str, str, tuple, dict], list[str]], client: str):
    """Bind the client name of the Environment's privileged-use classifier."""
    return lambda member, args, kwargs: privileged(client, member, args, kwargs)
