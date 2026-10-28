# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured joint execution; no grasp inference."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from phygs.runtime.episode import ControllerDeclaration, JointTargets, RunMode


class JointReader(Protocol):
    def positions(self) -> dict[str, float]: ...
    def velocities(self) -> dict[str, float]: ...


class PoseServo(Protocol):
    def start(self, pose: tuple[float, ...]) -> None: ...
    def targets(self, fraction: float) -> dict[str, float]: ...
    def error(self) -> tuple[float, float]: ...


# The simulator keeps joint limits in float32: a 0.04 m limit reads 0.0399999991 m, and a target at
# the limit (an open gripper) is not outside it.
LIMIT_SLACK = 1e-6


@dataclass(frozen=True)
class MotionConfig:
    """Joint limits [rad/s, rad/s²], tolerances [rad, rad/s, m, rad], and waits [s].

    ``gripper_tolerance`` [m or rad, the gripper joint's] is how near its target a commanded gripper
    must come (else it closed on an object), and ``gripper_stopped_velocity`` [m/s or rad/s] when the
    gripper joints count as still (a prismatic gripper on a revolute arm needs its own). ``grip_squeeze``
    [m or rad] is how far past the contact position a gripper that stopped on an object keeps pressing;
    it bounds the holding force instead of crushing.
    ``motion_window_seconds`` [s]: joint motion is the net position change over this window;
    a jaw squeezing a handle fixed to the world chatters in place, which per-tick changes
    would read as motion.
    """

    max_velocity: float = 1.5
    max_acceleration: float = 3.0
    tracking_tolerance: float = 0.15
    tool_position_tolerance: float = 0.02  # End-effector tracking of an end-effector trajectory [m].
    tool_orientation_tolerance: float = 0.05  # [rad]
    joint_tolerance: float = 0.03
    gripper_tolerance: float = 0.03
    stopped_velocity: float = 0.08
    gripper_stopped_velocity: float = 0.08
    position_tolerance: float = 0.01
    orientation_tolerance: float = 0.05
    settle_seconds: float = 0.15
    timeout_seconds: float = 2.0
    stop_timeout_seconds: float = 2.0
    grip_squeeze: float = 0.05
    motion_window_seconds: float = 0.05

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value <= 0 for value in vars(self).values()):
            raise ValueError("Motion limits, tolerances and durations must be positive and finite.")
        if self.grip_squeeze <= self.gripper_tolerance:
            # Holding is detected as the gripper staying short of its squeeze setpoint.
            raise ValueError("grip_squeeze must exceed gripper_tolerance, or a held object cannot be detected.")


class ManipulationController:
    """Own arm and gripper together; finish only after measured convergence.

    A joint move or a trajectory is checked while it runs (``tracking_tolerance`` against its
    reference) and at its end only has to stop: a held object's load holds arm joints a few
    hundredths of a radian off their targets at 250 Hz (0.039 rad on a drawer handle; a stow with
    an object timed out on the old 0.03 rad end check, 2026-10-17). A commanded gripper still has
    to reach its target or stop on an object. The Cartesian servo (``move_to_pose`` without a
    planner) keeps its joint and pose checks.

    The last commanded gripper angle [rad] stays the gripper target in later arm
    commands, when stopping, and after release, as a physical gripper keeps squeezing.
    After a motion succeeds, the arm keeps its commanded joint targets: on release and as
    the start of the next command (``arm_hold``), not its measured joints. Under a held
    object's load the arm sits a little off its targets; re-targeting the measured joints
    let it give way again each time (0.034 rad on arm_el1 while a stow was planned at
    250 Hz, 2026-10-16). A failed or cancelled motion holds where the arm stopped.
    A commanded gripper that stops moving after the motion time has closed on an object.
    Stillness is judged from measured position changes over ``motion_window_seconds``: in
    sustained contact the simulator's reported joint velocity does not describe motion.

    ``gripper`` is the gripper's primary joint; ``gripper_followers`` (joint -> ratio) are owned
    gripper joints that always follow it (a parallel gripper's second finger): their targets are
    the primary's times the ratio, and they are never commanded on their own.
    """

    def __init__(
        self,
        joints: tuple[str, ...],
        reader: JointReader,
        limits: dict[str, tuple[float, float]],
        config: MotionConfig,
        *,
        source: RunMode,
        pose_servo: PoseServo | None = None,
        gripper: str | None = None,
        gripper_followers: dict[str, float] | None = None,
    ) -> None:
        self.gripper_followers = dict(gripper_followers or {})
        if gripper is not None and gripper not in joints:
            raise ValueError("The gripper must be one of the owned joints.")
        if not set(self.gripper_followers) <= set(joints) - {gripper} or (self.gripper_followers and gripper is None):
            raise ValueError("Gripper followers must be owned joints other than the gripper's primary.")
        self._gripper_joints = frozenset({gripper, *self.gripper_followers} - {None})
        self.declaration = ControllerDeclaration("manipulation", joints, source)
        self.reader, self.limits, self.config = reader, limits, config
        self.pose_servo = pose_servo
        self.gripper, self.gripper_setpoint = gripper, None
        self.object_held = False
        self.arm_hold: dict[str, float] = {}  # Commanded arm targets [rad] after a successful motion.
        self.reset({joint: 0.0 for joint in joints})

    def reset(self, positions: dict[str, float]) -> None:
        self.initial = positions.copy()
        self.target = positions.copy()
        self.reference = positions.copy()
        self.elapsed = self.stable = 0.0
        self.duration = 1.0
        self.trajectory = None
        self.mode = "joints"
        self.outcome = ""
        self.reason = ""
        self.ready = False
        self.error_rad = 0.0
        self.pose_error = (0.0, 0.0)
        # An end-effector trajectory's tool tracking: (measured joints, elapsed [s]) -> error [m, rad].
        self.tool_error: Callable[[dict[str, float], float], tuple[float, float]] | None = None
        self.progress = 0.0  # Fraction of the reference duration reached.
        self.time_scale = 1.0  # How much the trajectory was slowed down to the joint speed limit.
        self.gripper_commanded = False
        self.stalled = False  # The arm stopped following its reference (tracking failure or timeout).
        self.final_reference: dict[str, float] = {}  # The reference [rad] when the motion ended.
        self.commanded: set[str] = set()
        self.history = deque([(0.0, positions.copy())])  # (elapsed [s], measured positions [rad])

    def hold_targets(self) -> dict[str, float]:
        """Setpoints [m or rad] kept after release: the arm's commanded targets after a successful
        motion (else none: the runtime holds the measured arm) and the gripper setpoint."""
        gripper = {} if self.gripper_setpoint is None else self._gripper_targets(self.gripper_setpoint)
        return {**self.arm_hold, **gripper}

    def _gripper_targets(self, primary: float) -> dict[str, float]:
        """Every gripper joint's target [m or rad] for the primary's ``primary``."""
        return {self.gripper: primary, **{joint: primary * r for joint, r in self.gripper_followers.items()}}

    def clear_hold(self) -> None:
        """Forget the gripper setpoint and the arm hold, e.g. when the episode ends."""
        self.gripper_setpoint, self.object_held = None, False
        self.arm_hold = {}

    @property
    def holding(self) -> bool:
        """Whether the gripper closed on an object and is still held short of its squeeze setpoint.

        A lost object lets the jaws reach the setpoint, so this becomes false.
        """
        if not self.object_held:
            return False
        error = abs(self.reader.positions()[self.gripper] - self.gripper_setpoint)
        return error > self.config.gripper_tolerance

    def validate(self, target: dict[str, float]) -> None:
        if not target or not set(target).issubset(self.declaration.joints):
            raise ValueError("Specify declared arm or gripper joints.")
        if set(target) & set(self.gripper_followers):
            raise ValueError(
                f"Joints {sorted(self.gripper_followers)} follow the gripper's {self.gripper!r}; command it."
            )
        violation = self._limit_violation(target)
        if violation is not None:
            raise ValueError(violation)

    def _limit_violation(self, target: dict[str, float]) -> str | None:
        """Why ``target`` [m or rad] is outside the joint limits, or None."""
        for joint, value in target.items():
            lower, upper = self.limits[joint]
            if not math.isfinite(value) or not lower - LIMIT_SLACK <= value <= upper + LIMIT_SLACK:
                return f"Target {joint}={value} is outside [{lower}, {upper}] (m or rad)."
        return None

    def _measured_targets(self) -> dict[str, float]:
        """The measured joints [rad] as targets, within the joint limits: the simulator reports a joint
        resting on a limit slightly past it (a closed gripper at +1.8e-7 rad over its 0 rad limit,
        2026-10-18), and holding that angle is not a command outside the limits."""
        positions = self.reader.positions()
        return {j: min(max(positions[j], self.limits[j][0]), self.limits[j][1]) for j in self.declaration.joints}

    def start(self, target: dict[str, float], duration: float) -> None:
        self.reset(self._measured_targets())
        for joint, value in self.arm_hold.items():  # Start from the held arm targets.
            self.initial[joint] = self.target[joint] = self.reference[joint] = value
        target = dict(target)
        self.commanded = set(target)
        if self.gripper in target:
            self.gripper_setpoint, self.gripper_commanded = target[self.gripper], True
            self.object_held = False
        elif self.gripper_setpoint is not None:
            # Keep squeezing from the first tick; never ramp a held gripper from its measured angle.
            target[self.gripper] = self.initial[self.gripper] = self.reference[self.gripper] = self.gripper_setpoint
        if self.gripper in target:
            target.update(self._gripper_targets(target[self.gripper]))  # The followers go along.
        self.target.update(target)
        self.duration = duration

    def _stopped(self, motion: dict[str, float]) -> bool:
        """Every joint slower than its threshold: the gripper joints' own, the arm's for the rest."""
        return all(
            speed
            <= (self.config.gripper_stopped_velocity if joint in self._gripper_joints else self.config.stopped_velocity)
            for joint, speed in motion.items()
        )

    def _motion(self, positions: dict[str, float], dt: float) -> dict[str, float]:
        """Joint speeds [m/s or rad/s] from the net position change over the motion window: the mean of its
        newer half against the mean of its older half, so a jaw flipping between two contact positions
        every tick reads as still whatever the window's parity (a 5.6 mrad limit cycle at 250 Hz)."""
        self.history.append((self.elapsed, positions.copy()))
        while len(self.history) > 2 and self.history[1][0] <= self.elapsed - self.config.motion_window_seconds:
            self.history.popleft()
        samples = list(self.history)
        half = max(len(samples) // 4 * 2, 1)  # Even, so the two positions weigh the same in each half.
        older, newer = samples[:half], samples[-half:]
        span = max(sum(s[0] for s in newer) / half - sum(s[0] for s in older) / half, dt)
        return {
            j: abs(sum(s[1][j] for s in newer) - sum(s[1][j] for s in older)) / half / span
            for j in self.declaration.joints
        }

    def stop(self, outcome: str, reason: str, *, stalled: bool = False) -> None:
        self.final_reference, self.stalled = self.reference.copy(), stalled
        self.arm_hold = {}  # Hold where the arm stopped, not a target it may not reach.
        positions = self.reader.positions()
        self.target = self._measured_targets()
        self.target.update(self.hold_targets())
        self.reference = self.target.copy()
        self.history = deque([(0.0, positions.copy())])
        self.mode, self.outcome, self.reason = "stopping", outcome, reason
        self.elapsed = self.stable = 0.0

    def step(self, dt: float) -> dict[str, float] | JointTargets:
        targets = self._step(dt)
        if self.gripper_followers:
            positions = targets.positions if isinstance(targets, JointTargets) else targets
            positions.update(self._gripper_targets(positions[self.gripper]))
        return targets

    def _step(self, dt: float) -> dict[str, float] | JointTargets:
        self.elapsed += dt
        positions, velocities = self.reader.positions(), self.reader.velocities()
        if any(not math.isfinite(values[j]) for values in (positions, velocities) for j in self.declaration.joints):
            raise RuntimeError("Nonfinite measured manipulation state.")
        motion = self._motion(positions, dt)
        if self.mode == "planning":
            return self.reference.copy()
        if self.mode != "stopping":
            t = min(self.elapsed / self.duration, 1.0)
            self.progress = t
            fraction = t**3 * (10 - 15 * t + 6 * t * t)
            reference = {j: p + fraction * (self.target[j] - p) for j, p in self.initial.items()}
            if self.mode == "trajectory":
                if (
                    max(abs(positions[j] - self.reference[j]) for j in self.trajectory.names)
                    > self.config.tracking_tolerance
                ):
                    self.stop("failed", "Trajectory tracking error exceeded tolerance", stalled=True)
                    return self.reference.copy()
                if self.tool_error is not None:
                    self.pose_error = self.tool_error(positions, self.elapsed)
                    if (
                        self.pose_error[0] > self.config.tool_position_tolerance
                        or self.pose_error[1] > self.config.tool_orientation_tolerance
                    ):
                        self.stop(
                            "failed",
                            f"End-effector tracking error exceeded tolerance: {self.pose_error[0]:.3f} m,"
                            f" {self.pose_error[1]:.3f} rad",
                            stalled=True,
                        )
                        return self.reference.copy()
                reference.update(self.trajectory.sample(self.elapsed))
                # Admission checks the trajectory's samples; interpolation must not leave the limits.
                violation = self._limit_violation(reference)
                if violation is not None:
                    self.stop("failed", violation)
                    return self.reference.copy()
            if self.mode == "pose":
                reference.update(self.pose_servo.targets(fraction))
                violation = self._limit_violation(reference)  # The servo's IK may walk past a limit.
                if violation is not None:
                    self.stop("failed", violation)
                    return self.reference.copy()
                # A servo output is not a trajectory with certified joint speed limits.
                if max(abs(reference[j] - self.reference[j]) for j in reference) > self.config.max_velocity * dt:
                    self.stop("failed", "Cartesian IK exceeded the joint reference velocity limit")
                    return self.reference.copy()
                self.pose_error = self.pose_servo.error()
            self.reference = reference
            goal = self.reference if self.mode == "pose" else self.target
            # A held (uncommanded) gripper may rest on an object; arm commands check arm joints only.
            arm = [j for j in goal if j not in self._gripper_joints]
            if self.mode == "joints" and (
                max((abs(positions[j] - reference[j]) for j in arm), default=0.0) > self.config.tracking_tolerance
            ):
                self.stop("failed", "Joint tracking error exceeded tolerance", stalled=True)
                return self.reference.copy()
            # A gripper-only command checks the gripper; the arm must only be still.
            gripper_only = self.gripper_commanded and self.commanded == {self.gripper}
            checked = [self.gripper] if gripper_only else list(goal) if self.gripper_commanded else arm
            self.error_rad = max((abs(positions[j] - goal[j]) for j in checked), default=0.0)
            if self.mode == "pose":
                converged = self.error_rad <= self.config.joint_tolerance
            else:
                # Joint moves and trajectories are tracked while running and end when the arm stops;
                # a held object's load can hold an arm joint a few hundredths of a radian off its
                # target. A commanded gripper still has to reach its target or stop on an object.
                gripper = [self.gripper] if self.gripper_commanded else []
                converged = all(abs(positions[j] - goal[j]) <= self.config.gripper_tolerance for j in gripper)
            grasped = False
            if self.gripper_commanded and not converged and self.elapsed >= self.duration:
                # The gripper stopped short of its target: it closed on an object.
                others = [j for j in checked if j not in self._gripper_joints] if self.mode == "pose" else []
                arm_error = max((abs(positions[j] - goal[j]) for j in others), default=0.0)
                grasped = converged = arm_error <= self.config.joint_tolerance and (
                    motion[self.gripper] <= self.config.gripper_stopped_velocity
                )
            if self.mode == "pose":
                converged &= (
                    self.pose_error[0] <= self.config.position_tolerance
                    and self.pose_error[1] <= self.config.orientation_tolerance
                )
            stopped = self._stopped(motion)
            self.stable = self.stable + dt if self.elapsed >= self.duration and converged and stopped else 0.0
            if grasped and not self.object_held:
                # Hold a bounded squeeze past the first contact angle, not the full closing target. Once:
                # a soft or slipping object lets the jaw creep, and every pause would push it further.
                direction = math.copysign(1.0, goal[self.gripper] - positions[self.gripper])
                lower, upper = self.limits[self.gripper]
                squeeze = positions[self.gripper] + direction * self.config.grip_squeeze
                self.gripper_setpoint = min(max(squeeze, lower), upper)
                self.target.update(self._gripper_targets(self.gripper_setpoint))
                reference.update(self._gripper_targets(self.gripper_setpoint))
                self.object_held = True
            if self.stable >= self.config.settle_seconds:
                self.outcome, self.ready = "succeeded", True
                self.final_reference = self.reference.copy()
                self.arm_hold = {j: self.reference[j] for j in self.declaration.joints if j not in self._gripper_joints}
                self.reason = "Gripper stopped on an object" if grasped else "Measured target reached and settled"
            elif self.elapsed >= self.duration + self.config.timeout_seconds:
                self.stop("failed", "Motion did not reach its target before timeout", stalled=True)
        else:
            stopped = self._stopped(motion)
            self.stable = self.stable + dt if stopped else 0.0
            self.ready = self.stable >= self.config.settle_seconds
            if not self.ready and self.elapsed >= self.config.stop_timeout_seconds:
                # A failure the agent sees (execution_stalled); the episode decides whether it is fatal.
                joint = max(motion, key=motion.get)
                self.outcome, self.ready = "failed", True
                self.reason = (
                    f"Manipulation did not physically stop before its deadline: {joint} moves"
                    f" {motion[joint]:.3f} m/s or rad/s after stopping ({self.reason})"
                )
        if self.mode == "trajectory":
            return JointTargets(self.reference.copy(), velocities=self.trajectory.velocity(self.elapsed))
        return self.reference.copy()
