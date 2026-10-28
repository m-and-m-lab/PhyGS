# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-environment lifecycle and exclusive controller scheduling."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from phygs.observations.types import ObservationPacket


class RunMode(str, Enum):
    """Authority assigned when an experiment is constructed."""

    AUTONOMOUS = "autonomous"
    HUMAN = "human"


class EpisodeStatus(str, Enum):
    """An episode must reset before leaving a terminal state."""

    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


@dataclass(frozen=True)
class ControllerDeclaration:
    """Controller identity, exact owned joints, and command origin."""

    name: str
    joints: tuple[str, ...]
    source: RunMode = RunMode.AUTONOMOUS
    stop_ticks: int = 0

    def __post_init__(self) -> None:
        if type(self.stop_ticks) is not int or self.stop_ticks < 0:
            raise ValueError("stop_ticks must be a nonnegative integer.")
        if not self.name or not self.joints or len(set(self.joints)) != len(self.joints):
            raise ValueError("A controller needs a name and unique, nonempty joints.")
        if any(not joint for joint in self.joints) or not isinstance(self.source, RunMode):
            raise ValueError("Invalid controller joint or source.")


@dataclass(frozen=True)
class JointTargets:
    """Position targets [rad] with optional gains for the same owned joints.

    Stiffness is [N·m/rad]; damping is [N·m·s/rad]. Gain changes must specify both.
    ``stiffness`` and ``damping`` are explicit-PD gains of the joint's actuator.
    ``drive_stiffness`` and ``drive_damping`` hand the joint to PhysX joint drives (implicit PD on
    the undelayed set-point) instead; explicit gains hand it back to its actuator.
    Optional desired velocities are [rad/s]; omitted joints use zero velocity.
    """

    positions: dict[str, float]
    stiffness: dict[str, float] = field(default_factory=dict)
    damping: dict[str, float] = field(default_factory=dict)
    velocities: dict[str, float] = field(default_factory=dict)
    drive_stiffness: dict[str, float] = field(default_factory=dict)
    drive_damping: dict[str, float] = field(default_factory=dict)

    def validate(self, joints: tuple[str, ...]) -> None:
        """Reject incomplete positions or gains outside declared ownership."""
        if set(self.positions) != set(joints):
            raise ValueError("Controller output does not match its owned joints.")
        if not set(self.velocities).issubset(joints):
            raise ValueError("Velocity targets must remain within owned joints.")
        for stiffness, damping in ((self.stiffness, self.damping), (self.drive_stiffness, self.drive_damping)):
            if set(stiffness) != set(damping) or not set(stiffness).issubset(joints):
                raise ValueError("Gain changes must match and remain within owned joints.")
        if set(self.stiffness) & set(self.drive_stiffness):
            raise ValueError("A joint takes either explicit or PhysX drive gains.")
        gains = (self.stiffness, self.damping, self.drive_stiffness, self.drive_damping)
        if any(not math.isfinite(v) for values in (self.positions, self.velocities, *gains) for v in values.values()):
            raise ValueError("Controller produced nonfinite targets.")
        if any(v < 0 for values in gains for v in values.values()):
            raise ValueError("Controller gains must be nonnegative.")


class Controller(Protocol):
    """Position controller; references are initialized at every acquisition."""

    declaration: ControllerDeclaration

    def reset(self, positions: dict[str, float]) -> None:
        """Clear memory and initialize from measured joint positions [rad]."""
        ...

    def step(self, dt: float) -> dict[str, float] | JointTargets:
        """Return declared joint position targets [rad] after interval [s]."""
        ...


@runtime_checkable
class HoldingController(Protocol):
    """Optional interface: owned joints that keep a commanded setpoint after release."""

    def hold_targets(self) -> dict[str, float]:
        """Return setpoints [rad] for owned joints to keep, e.g. a closed gripper."""
        ...


@runtime_checkable
class StoppableController(Protocol):
    """Additional interface required when a controller declares stop ticks."""

    def stop(self) -> None:
        """Begin controlled stopping without relinquishing resources."""
        ...


@runtime_checkable
class TimedStopController(Protocol):
    """Optional interface: a stop interval that depends on the stoppable controller's state."""

    def stop_duration_ticks(self) -> int:
        """Ticks [physics steps] the stop requested now needs, in place of the declared ``stop_ticks``."""
        ...


class World(Protocol):
    """Trusted backend owned exclusively by an episode runtime."""

    physics_dt: float
    joint_names: tuple[str, ...]

    def restore(self) -> None: ...
    def positions(self) -> dict[str, float]: ...
    def apply(self, targets: dict[str, float] | JointTargets) -> None: ...
    def step(self) -> None: ...


@dataclass(frozen=True)
class Event:
    """Lifecycle event indexed by episode and physics tick."""

    episode_id: int
    tick: int
    kind: str
    detail: str


class ObservationSource(Protocol):
    """Trusted sensor adapter scheduled exclusively by the episode runtime."""

    @property
    def configuration(self) -> dict: ...
    def reset(self, episode_id: int) -> None: ...
    def update(self, episode_id: int, tick: int) -> None: ...
    def observe(self) -> ObservationPacket: ...
    def observe_internal(self) -> ObservationPacket: ...


class EpisodeRuntime:
    """Own one world's clock and exclusive position-controller execution.

    This first runtime supports single-environment, revolute-joint position control.
    Controller registration is trusted setup; agents only select registered names.
    """

    def __init__(
        self,
        world: World,
        controllers: tuple[Controller, ...],
        *,
        mode: RunMode,
        reset_targets: JointTargets | None = None,
        observations: ObservationSource | None = None,
        configuration: dict | None = None,
    ) -> None:
        if not isinstance(mode, RunMode):
            raise ValueError("mode must be a RunMode.")
        if not math.isfinite(world.physics_dt) or world.physics_dt <= 0:
            raise ValueError("physics_dt must be positive and finite.")
        if reset_targets is not None:
            reset_targets.validate(tuple(reset_targets.positions))
            if not set(reset_targets.positions).issubset(world.joint_names):
                raise ValueError("Reset targets contain unknown joints.")
        self._configuration = copy.deepcopy(configuration)
        self._observations = observations
        self._reset_targets = reset_targets
        self._world = world
        self._mode = mode
        self._controllers: dict[str, Controller] = {}
        for controller in controllers:
            spec = controller.declaration
            if spec.stop_ticks and not isinstance(controller, StoppableController):
                raise TypeError(f"Controller {spec.name} declares stop ticks but has no stop method.")
            if spec.name in self._controllers:
                raise ValueError(f"Duplicate controller: {spec.name}")
            unknown = set(spec.joints) - set(world.joint_names)
            if unknown:
                raise ValueError(f"Unknown joints for {spec.name}: {sorted(unknown)}")
            self._controllers[spec.name] = controller
        self._active: dict[str, int] = {}
        self._stopping: set[str] = set()
        self._events: list[Event] = []
        self._status = EpisodeStatus.READY
        self._episode_id = 0
        self._tick = 0
        self._settling_ticks = 0

    @property
    def physics_dt(self) -> float:
        """Physics interval [s]."""
        return self._world.physics_dt

    @property
    def status(self) -> EpisodeStatus:
        return self._status

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def episode_id(self) -> int:
        return self._episode_id

    @property
    def time_seconds(self) -> float:
        """Elapsed episode simulation time [s], excluding reset settling."""
        return self._tick * self._world.physics_dt

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    def reset(self, *, settling_ticks: int = 0) -> None:
        """Start a new episode after restoring and executing fixed settling ticks."""
        if type(settling_ticks) is not int or settling_ticks < 0:
            raise ValueError("settling_ticks must be a nonnegative integer.")
        self._active.clear()
        self._stopping.clear()
        self._episode_id += 1
        self._tick = 0
        self._settling_ticks = settling_ticks
        self._status = EpisodeStatus.READY
        try:
            self._world.restore()
            self._world.apply(self._world.positions())
            if self._reset_targets is not None:
                self._world.apply(self._reset_targets)
            for _ in range(settling_ticks):
                self._world.step()
            positions = self._world.positions()
            for controller in self._controllers.values():
                controller.reset({joint: positions[joint] for joint in controller.declaration.joints})
            if self._observations is not None:
                self._observations.reset(self._episode_id)
        except Exception as exc:
            self._terminate(EpisodeStatus.FAILED, str(exc))
            raise
        self._status = EpisodeStatus.RUNNING
        self._record("reset", f"settling_ticks={settling_ticks}")

    def acquire(self, name: str, *, duration_ticks: int) -> None:
        """Acquire all declared joints atomically for a bounded command duration."""
        self._require_running()
        if type(duration_ticks) is not int or duration_ticks <= 0:
            raise ValueError("duration_ticks must be a positive integer.")
        controller = self._controllers[name]
        spec = controller.declaration
        if self._mode == RunMode.AUTONOMOUS and spec.source == RunMode.HUMAN:
            self._record("rejected", f"human controller: {name}")
            raise PermissionError("Autonomous episodes reject human control.")
        occupied = {joint for owner in self._active for joint in self._controllers[owner].declaration.joints}
        conflict = occupied.intersection(spec.joints)
        if conflict:
            self._record("rejected", f"ownership conflict: {name}")
            raise RuntimeError(f"Joints already owned: {sorted(conflict)}")
        try:
            positions = self._world.positions()
            controller.reset({joint: positions[joint] for joint in spec.joints})
        except Exception as exc:
            self._terminate(EpisodeStatus.FAILED, str(exc))
            raise
        self._active[name] = self._tick + duration_ticks
        self._record("acquired", name)

    def controller_state(self, name: str) -> str:
        """Return idle, active, or stopping for a registered controller."""
        if name not in self._controllers:
            raise KeyError(name)
        return "stopping" if name in self._stopping else "active" if name in self._active else "idle"

    def renew(self, name: str, *, duration_ticks: int) -> None:
        """Extend an owner's command without resetting policy memory."""
        self._require_running()
        if name not in self._active:
            raise KeyError(f"Controller is not active: {name}")
        if type(duration_ticks) is not int or duration_ticks <= 0:
            raise ValueError("duration_ticks must be a positive integer.")
        self._active[name] = self._tick + duration_ticks
        self._stopping.discard(name)

    def request_stop(self, name: str) -> None:
        """Run a declared stop sequence while retaining ownership until it finishes."""
        self._require_running()
        if name not in self._active:
            raise KeyError(f"Controller is not active: {name}")
        controller = self._controllers[name]
        if not controller.declaration.stop_ticks:
            self.release(name)
            return
        try:
            cast(StoppableController, controller).stop()
            ticks = controller.declaration.stop_ticks
            if isinstance(controller, TimedStopController):
                ticks = controller.stop_duration_ticks()
                if type(ticks) is not int or ticks <= 0:
                    raise ValueError(f"Controller {name} returned an invalid stop interval: {ticks!r} ticks.")
        except Exception as exc:
            self._terminate(EpisodeStatus.FAILED, str(exc))
            raise
        self._stopping.add(name)
        self._active[name] = self._tick + ticks
        self._record("stopping", name)

    def release(self, name: str) -> None:
        """Release ownership and hold measured joint positions [rad], except declared hold setpoints."""
        self._require_running()
        if name not in self._active:
            raise KeyError(f"Controller is not active: {name}")
        try:
            controller = self._controllers[name]
            positions = self._world.positions()
            held = {joint: positions[joint] for joint in controller.declaration.joints}
            if isinstance(controller, HoldingController):
                kept = controller.hold_targets()
                if not set(kept) <= set(held) or not all(math.isfinite(v) for v in kept.values()):
                    raise ValueError("Hold setpoints must be finite and within owned joints.")
                held.update(kept)
            self._world.apply(held)
        except Exception as exc:
            self._terminate(EpisodeStatus.FAILED, str(exc))
            raise
        del self._active[name]
        self._stopping.discard(name)
        self._record("released", name)

    def step(self) -> None:
        """Advance exactly one physics tick; terminate on controller/backend failure."""
        self._require_running()
        try:
            for name, deadline in tuple(self._active.items()):
                if self._tick >= deadline:
                    if name in self._stopping or not self._controllers[name].declaration.stop_ticks:
                        self.release(name)
                    else:
                        self.request_stop(name)
                    self._record("expired", name)
            merged = JointTargets({})
            for name in self._active:
                controller = self._controllers[name]
                output = controller.step(self._world.physics_dt)
                output = JointTargets(output) if isinstance(output, dict) else output
                output.validate(controller.declaration.joints)
                for part in ("positions", "stiffness", "damping", "velocities", "drive_stiffness", "drive_damping"):
                    getattr(merged, part).update(getattr(output, part))
            plain = not (merged.stiffness or merged.velocities or merged.drive_stiffness)
            self._world.apply(merged.positions if plain else merged)
            self._world.step()
            self._tick += 1
            if self._observations is not None:
                self._observations.update(self._episode_id, self._tick)
        except Exception as exc:
            self._terminate(EpisodeStatus.FAILED, str(exc))
            raise

    def emergency_stop(self, reason: str) -> None:
        """Interrupt this episode permanently; further physics stepping is rejected."""
        self._require_running()
        self._terminate(EpisodeStatus.INTERRUPTED, reason)

    def fail(self, reason: str) -> None:
        """End execution after an environment component reports failure."""
        self._require_running()
        self._terminate(EpisodeStatus.FAILED, reason)

    def complete(self, reason: str) -> None:
        """End this episode after an external evaluator determines completion."""
        self._require_running()
        self._terminate(EpisodeStatus.COMPLETED, reason)

    def observe(self) -> ObservationPacket:
        """Return the latest permitted sensor samples without advancing the clock."""
        if self._observations is None:
            raise RuntimeError("No observation configuration was supplied for this environment.")
        if self._status == EpisodeStatus.READY:
            raise RuntimeError("Reset the environment before requesting observations.")
        packet = self._observations.observe()
        if packet.stamp.episode_id != self._episode_id:
            raise RuntimeError("No observations are available for the current episode.")
        return packet

    def result(self) -> dict:
        """Return JSON-serializable lifecycle diagnostics, not task-success evidence."""
        return {
            "episode_id": self._episode_id,
            "mode": self._mode.value,
            "configuration": copy.deepcopy(self._configuration),
            "status": self._status.value,
            "physics_dt": self._world.physics_dt,
            "ticks": self._tick,
            "time_seconds": self.time_seconds,
            "settling_ticks": self._settling_ticks,
            "observations": self._observations.configuration if self._observations is not None else None,
            "events": [vars(event) for event in self._events if event.episode_id == self._episode_id],
        }

    def _require_running(self) -> None:
        if self._status != EpisodeStatus.RUNNING:
            raise RuntimeError(f"Episode is {self._status.value}; reset before execution.")

    def _record(self, kind: str, detail: str) -> None:
        self._events.append(Event(self._episode_id, self._tick, kind, detail))

    def _terminate(self, status: EpisodeStatus, reason: str) -> None:
        if self._status == status:
            return
        self._active.clear()
        self._stopping.clear()
        self._status = status
        self._record(status.value, reason)
