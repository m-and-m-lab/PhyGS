# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional Nav2 execution adapter; ROS imports live only in its transport module."""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, replace
from typing import Protocol

import numpy as np
import torch

from phygs.execution.locomotion.base import DeadBand
from phygs.execution.navigation.fine_approach import FineApproach, FineMove
from phygs.execution.navigation.interfaces import NavigationContext
from phygs.execution.navigation.map import GridMap, RawMapProvider
from phygs.execution.navigation.types import (
    STEPS,
    NavigationConfig,
    NavigationFeedback,
    NavigationGoal,
    angle_error,
    describe_goal,
    describe_path,
)
from phygs.observations.types import (
    CameraObservation,
    CameraPermission,
    ObservationAccess,
    ObservationReader,
)
from phygs.runtime.episode import EpisodeStatus
from phygs.runtime.pipeline import ServiceDeclaration
from phygs.runtime.timing import duration_ticks


@dataclass(frozen=True)
class Nav2Config:
    """ROS namespace, timeouts [s], and soft inflation decay distance [m]."""

    namespace: str = "spot_navigation"
    service_timeout: float = 10.0
    # Nav2 runs beside the simulation, which never waits for its velocity commands: each control
    # period applies the newest one for what is left of command_timeout [s] after its stamp. A goal
    # with no fresh command for command_grace [s] fails (Nav2 stalled).
    command_timeout: float = 0.2
    command_grace: float = 2.0
    inflation_cost_distance: float = 0.3
    # Depth cameras whose points Nav2's obstacle layer maps (a marking and a clearing source each;
    # their ranges limit the points), sent every sensor_period [s]; none: the map comes only from the
    # map provider.
    sensor_cameras: tuple[str, ...] = ()
    sensor_period: float = 0.032
    # Every this many pixel rows and columns go to Nav2 (stride² fewer points).
    sensor_pixel_stride: int = 1
    # One point per voxel of this size [m] (their centroid) goes to Nav2, as a voxel grid filter
    # before a costmap does; 0 sends every point. Nav2 raytraces each point it receives.
    sensor_voxel_size: float = 0.0
    # Points in this fraction of each image's bottom rows only clear (``clearing/<camera>``); the rest
    # also mark (``points/<camera>``). A voxel seen at the bottom edge of the view from a slightly lower
    # camera (the body bobs) is crossed by no ray from the next pose and would stay marked.
    sensor_mark_margin: float = 0.0
    # Inside the fine approach radius Nav2 keeps driving until the heading is within this [rad] of
    # the goal's, for at most handover_wait [s], before the fine moves take over: Nav2's controller
    # turns while it walks, where a fine move turns standing (about 2.5 s per move).
    handover_heading_tolerance: float = 0.1
    handover_wait: float = 2.0
    # A goal Nav2's costmap rejects as occupied (its footprint over a low open drawer the robot
    # reaches into) is finished with fine moves from within this distance [m], on the navigation map.
    occupied_goal_reach: float = 0.35
    sensor_replan_period: float = 0.1
    max_sensor_replans: int = 2
    # Nav2's costmaps activate only once they see the bridge's transform and clock; the bridge keeps
    # sending them while it waits this long [s] for the planner and controller to become active.
    bringup_timeout: float = 60.0
    # A lifecycle node still unconfigured or inactive this long [s] into the wait, with no state
    # change, is driven through its transition by the bridge: Nav2's lifecycle manager can lose a
    # transition response and stall its bringup.
    stalled_transition_seconds: float = 15.0
    # Frontier exploration (``explore``): explore_lite (m-explore-ros2) on Nav2's global costmap picks
    # frontiers every 1 / explore_planner_frequency [s of wall time] (only frontiers at least
    # explore_min_frontier_size [m] long; a goal not approached for explore_progress_timeout [s] is
    # given up) and sends each as a NavigateToPose goal, which the bridge runs as a navigation leg
    # reaching within explore_arrival_tolerance [m] at any heading. explore_lite's output goes to
    # explore_log (empty: discarded).
    explore_planner_frequency: float = 1.0
    explore_progress_timeout: float = 30.0
    explore_min_frontier_size: float = 0.5
    explore_arrival_tolerance: float = 0.3
    explore_log: str = ""

    def __post_init__(self):
        if not self.namespace.isidentifier():
            raise ValueError("Nav2 namespace must be a single identifier.")
        for value in (self.service_timeout, self.command_timeout, self.command_grace):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Nav2 timeouts must be positive and finite.")
        if not math.isfinite(self.inflation_cost_distance) or not 0 < self.inflation_cost_distance <= 1.0:
            raise ValueError("Nav2 inflation cost distance must be in (0, 1] m.")
        if not math.isfinite(self.handover_heading_tolerance) or self.handover_heading_tolerance < 0:
            raise ValueError("Nav2 hand-over heading tolerance must be finite and nonnegative.")
        if not math.isfinite(self.occupied_goal_reach) or self.occupied_goal_reach < 0:
            raise ValueError("Nav2 occupied goal reach must be finite and nonnegative.")
        if not math.isfinite(self.handover_wait) or self.handover_wait < 0:
            raise ValueError("Nav2 hand-over wait must be finite and nonnegative.")
        if isinstance(self.sensor_pixel_stride, bool) or not isinstance(self.sensor_pixel_stride, int):
            raise ValueError("Nav2 sensor pixel stride must be an integer.")
        if self.sensor_pixel_stride < 1:
            raise ValueError("Nav2 sensor pixel stride must be positive.")
        for value in (
            self.explore_planner_frequency,
            self.explore_progress_timeout,
            self.explore_min_frontier_size,
            self.explore_arrival_tolerance,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Nav2 exploration settings must be positive and finite.")
        if not math.isfinite(self.sensor_mark_margin) or not 0 <= self.sensor_mark_margin < 0.5:
            raise ValueError("Nav2 sensor mark margin must be in [0, 0.5).")
        if not math.isfinite(self.sensor_voxel_size) or self.sensor_voxel_size < 0:
            raise ValueError("Nav2 sensor voxel size must be finite and nonnegative.")
        if not math.isfinite(self.sensor_period) or self.sensor_period <= 0:
            raise ValueError("Nav2 sensor period must be positive and finite.")
        if not math.isfinite(self.sensor_replan_period) or self.sensor_replan_period <= 0:
            raise ValueError("Nav2 sensor replan period must be positive and finite.")
        if not math.isfinite(self.bringup_timeout) or self.bringup_timeout <= 0:
            raise ValueError("Nav2 bringup timeout must be positive and finite.")
        if not math.isfinite(self.stalled_transition_seconds) or self.stalled_transition_seconds <= 0:
            raise ValueError("Nav2 stalled transition seconds must be positive and finite.")
        if (
            isinstance(self.max_sensor_replans, bool)
            or not isinstance(self.max_sensor_replans, int)
            or self.max_sensor_replans < 1
        ):
            raise ValueError("Nav2 maximum sensor replans must be a positive integer.")
        object.__setattr__(self, "sensor_cameras", tuple(self.sensor_cameras))
        if not all(isinstance(name, str) and name.isidentifier() for name in self.sensor_cameras):
            raise ValueError("Nav2 sensor cameras must be configured camera names.")
        if len(set(self.sensor_cameras)) != len(self.sensor_cameras):
            raise ValueError("Nav2 sensor cameras must not repeat.")


# Statuses that end a whole exploration, not just one of its legs: the agent or the runtime took over.
EXPLORATION_ENDINGS = frozenset({"overridden", "interrupted", "cancelled", "timed_out"})


@dataclass(frozen=True)
class Nav2State:
    """Permitted world pose [m, rad], body velocity [m/s, rad/s], and simulation time [s]."""

    episode_id: int
    time: float
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    yaw_rate: float


@dataclass(frozen=True)
class Nav2Update:
    """Transport status and body velocity [m/s, m/s, rad/s], age measured in simulation [s]."""

    status: str
    command: tuple[float, float, float] | None = None
    age: float = 0.0
    reason: str = ""
    # A failed route because the controller found no collision-free command near the robot (an
    # obstacle in the way): navigation replans instead of failing.
    blocked: bool = False


class Nav2Transport(Protocol):
    """ROS operations; cancellation must finish before reset or a replacement goal."""

    def publish(self, state: Nav2State, raw_map: GridMap) -> None: ...
    def publish_camera(self, name: str, camera: CameraObservation, map_from_camera: torch.Tensor) -> None: ...
    def costmap(self) -> tuple[np.ndarray, tuple[float, float], float] | None: ...
    def start(self, goal: NavigationGoal) -> tuple[tuple[float, float], ...]: ...
    def poll(self) -> Nav2Update: ...
    def path_blocked(self, path: tuple[tuple[float, float], ...], position: tuple[float, float]) -> bool: ...
    def cancel(self) -> None: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...
    def explore_start(self) -> None: ...
    def explore_poll(self) -> ExploreEvent: ...
    def explore_finish(self, succeeded: bool, reason: str) -> None: ...
    def explore_stop(self) -> None: ...


@dataclass(frozen=True)
class ExploreEvent:
    """What the frontier explorer did since the last poll: a new frontier goal (x, y) [m] to go to
    (its predecessor, if any, is superseded), whether it found no frontier left, and whether its
    process still runs."""

    goal: tuple[float, float] | None = None
    complete: bool = False
    running: bool = True


class Nav2PlanningError(RuntimeError):
    """A Nav2 planner returned a terminal planning failure."""

    def __init__(self, reason: str, *, unreachable: bool = False, goal_occupied: bool = False):
        super().__init__(reason)
        self.unreachable, self.goal_occupied = unreachable, goal_occupied


@dataclass(frozen=True)
class Nav2BackendFactory:
    """Construct the optional backend; importing the core package never imports ROS."""

    config: Nav2Config = Nav2Config()
    maps: RawMapProvider | None = None
    localization_mode: str = "ground_truth"
    dependencies: tuple[str, ...] = ()

    @property
    def configuration(self) -> dict:
        return {
            "backend": "nav2",
            "transport": asdict(self.config),
            "map": self.maps.configuration if self.maps is not None else None,
            "localization": self.localization_mode,
        }

    @property
    def declaration(self) -> ServiceDeclaration:
        if not isinstance(self.maps, RawMapProvider):
            raise ValueError("This Nav2 configuration requires a raw map provider; configure it on the factory.")
        source = self.maps.configuration["source"]
        return ServiceDeclaration(
            "navigation",
            ObservationAccess(
                joint_state=True,
                localization=self.localization_mode,
                cameras=tuple(CameraPermission(name, rgb=False) for name in self.config.sensor_cameras),
            ),
            information_sources=(source,) + (("simulated_rgbd_camera",) if self.config.sensor_cameras else ()),
            privileged_sources=(source,) if source.startswith("privileged_") else (),
            dependencies=self.dependencies,
            outputs=("body_velocity_commands", "navigation_feedback"),
        )

    def __call__(self, context: NavigationContext):
        from phygs.execution.navigation.backends.ros2 import RclpyNav2Transport

        maps = self.maps
        if not isinstance(maps, RawMapProvider):
            raise ValueError("Nav2 requires a RawMapProvider exposing uninflated raw_snapshot().")
        transport = RclpyNav2Transport(self.config, maps.configuration)
        base = context.locomotion.controller
        moves = base.lowest_speed_moves
        return Nav2NavigationBackend(
            context.runtime,
            context.locomotion,
            maps,
            context.config,
            self.config,
            transport,
            observations=context.observations,
            dead_band=base.dead_band,
            coverage=None if moves is None else FineApproach.Coverage.from_moves(moves),
        )


class Nav2NavigationBackend:
    """Delegate planning and following to Nav2, applying only expiring runtime commands.

    This map-fed configuration uses declared localization, a known or observed
    grid, and the folded-arm transit posture. The backend contract itself does
    not require a framework map or framework localization.
    """

    def __init__(
        self,
        runtime,
        locomotion,
        maps,
        config: NavigationConfig,
        nav2_config: Nav2Config,
        transport: Nav2Transport,
        *,
        observations: ObservationReader | None = None,
        dead_band: DeadBand,
        coverage: FineApproach.Coverage | None = None,
    ):
        """``dead_band`` and ``coverage``: the base's command dead band and lowest-speed moves, for the
        fine approach (:class:`~phygs.execution.navigation.fine_approach.FineApproach`)."""
        self._observations = observations if observations is not None else runtime
        self._runtime, self._motion, self._maps = runtime, locomotion, maps
        self._config, self._nav2, self._transport = config, nav2_config, transport
        self._period = duration_ticks(config.period, runtime.physics_dt)
        self._sensor_period = (
            duration_ticks(nav2_config.sensor_period, runtime.physics_dt) if nav2_config.sensor_cameras else None
        )
        duration_ticks(nav2_config.command_timeout, runtime.physics_dt)
        if nav2_config.sensor_cameras:
            duration_ticks(nav2_config.sensor_replan_period, runtime.physics_dt)
        self._records: dict[int, NavigationFeedback] = {}
        self._current: int | None = None
        self._goal: NavigationGoal | None = None
        self._motion_id: int | None = None
        self._finish = ""
        self._deadline: int | None = None
        self._map_version = -1
        self._previous_state: Nav2State | None = None
        self._last_tick = -1
        self._last_sensor_tick = -1
        self._last_path_check_tick = -1
        self._external_active = False
        self._fresh_command_time = 0.0  # Simulation time [s] of the newest fresh Nav2 command.
        self._saturation_warned = False
        self._fine = FineApproach(config, runtime.physics_dt, dead_band, coverage)
        self._fine_move: FineMove | None = None
        self._fine_moves = 0
        self._fine_started = False
        self._within_since: float | None = None  # Simulation time [s] the base entered the fine radius.
        self._corrections = 0  # Routes followed in again from a settled pose outside the tolerances.
        # A frontier exploration (explore) runs as one command made of legs: each of explore_lite's
        # frontier goals is followed like a navigate_to goal (self._goal; None between legs).
        self._exploring = False
        self._ending = False  # The exploration itself ends with the next terminal status.
        self._legs = 0

    @property
    def configuration(self) -> dict:
        return dict(
            backend="nav2",
            transport=asdict(self._nav2),
            limits=asdict(self._config),
            information=self._maps.configuration["source"],
            sensor_cameras=list(self._nav2.sensor_cameras),
            goal_frame=self._config.reference_frame,
            optional_heading=True,
        )

    def _arm_folded(self, positions: dict[str, float]) -> bool:
        """Whether the arm joints [rad] are within ``arm_tolerance`` of the transit posture."""
        return all(
            abs(positions[name] - value) <= self._config.arm_tolerance for name, value in self._config.arm_posture
        )

    def costmap(self) -> tuple[np.ndarray, tuple[float, float], float] | None:
        """Nav2's global costmap for display (trusted runners such as the robot UI): cost per cell
        (253 the robot's centre may not be there, 254 an obstacle, 255 unknown), origin and cell [m];
        None before Nav2 serves it."""
        return self._transport.costmap()

    def route(self) -> tuple[NavigationGoal, tuple[tuple[float, float], ...]] | None:
        """The running command's goal and route [m] (the dense route followed; empty before Nav2 plans)
        for display; None when no navigation command runs."""
        if not self.active or self._goal is None:
            return None
        return self._goal, self._records[self._current].path

    @property
    def active(self) -> bool:
        return self._current is not None and self.feedback(self._current).status in {"active", "stopping"}

    @property
    def active_command(self) -> int | None:
        return self._current if self.active else None

    def feedback(self, command_id: int) -> NavigationFeedback:
        record = self._records[command_id]
        if record.status in {"active", "stopping"}:
            if record.episode_id != self._runtime.episode_id:
                record = replace(record, status="cancelled", reason="Episode reset")
            elif self._runtime.status != EpisodeStatus.RUNNING:
                record = replace(record, status=self._runtime.status.value, reason="Episode terminated")
            self._records[command_id] = record
        return record

    def _set(self, status: str, reason: str) -> None:
        name = "explore" if self._exploring else "navigate_to"
        if self._exploring and status not in {"active", "stopping"}:
            if self._goal is not None and not self._ending and status not in EXPLORATION_ENDINGS:
                # A leg ended: explore_lite hears how, and skips a frontier given up.
                self._transport.explore_finish(status == "succeeded", f"{status}: {reason}")
                self._goal, self._motion_id, self._fine_move = None, None, None
                status, reason = "active", f"Exploring; frontier {self._legs} {status}: {reason}"
            else:
                self._transport.explore_stop()
                self._exploring = self._ending = False
                self._goal = None
        record = self._records[self._current]
        if (record.status, record.reason) != (status, reason):
            STEPS.info(f"{name}: {reason} ({status})")
        self._records[self._current] = replace(record, status=status, reason=reason)

    def _state(self) -> Nav2State:
        packet = self._observations.observe()
        pose, joints = packet.localization, packet.joints
        if pose is None or not pose.valid or joints is None or not joints.valid:
            raise ValueError("Nav2 requires permitted localization and joint observations.")
        if pose.pose.target != self._config.reference_frame:
            raise ValueError("Localization does not match the navigation goal frame.")
        if self._runtime.time_seconds - pose.stamp.time_seconds > self._config.period:
            raise ValueError("Nav2 localization is stale.")
        positions = dict(zip(joints.names, joints.positions.tolist(), strict=True))
        if not self._arm_folded(positions):
            raise ValueError("Arm left the measured folded transit posture.")
        matrix = pose.pose.matrix
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
        # Camera and control updates can consume the same localization sample.
        yaw_rate = self._previous_state.yaw_rate if self._previous_state is not None else 0.0
        if self._previous_state is not None and pose.stamp.time_seconds > self._previous_state.time:
            yaw_rate = angle_error(yaw, self._previous_state.yaw) / (
                pose.stamp.time_seconds - self._previous_state.time
            )
        state = Nav2State(
            self._runtime.episode_id,
            pose.stamp.time_seconds,
            float(matrix[0, 3]),
            float(matrix[1, 3]),
            yaw,
            *map(float, pose.linear_velocity[:2]),
            yaw_rate,
        )
        self._previous_state = state
        return state

    def _map_raw(self) -> GridMap:
        raw = self._maps.raw_snapshot()
        if raw.frame != self._config.reference_frame:
            raise ValueError("Map does not match the navigation goal frame.")
        if raw.episode_id is not None and raw.episode_id != self._runtime.episode_id:
            raise RuntimeError("Navigation map belongs to another episode.")
        return raw

    def _publish(self, state: Nav2State) -> None:
        """Publish only inputs declared for this navigation service."""
        self._transport.publish(state, self._map_raw())
        if self._nav2.sensor_cameras:
            packet = self._observations.observe()
            for name in self._nav2.sensor_cameras:
                camera = packet.cameras[name]
                self._transport.publish_camera(
                    name, camera, packet.transform(self._config.reference_frame, camera.frame, camera.stamp)
                )

    def _plan(self, state: Nav2State, *, replan: bool) -> None:
        if replan:
            record = self._records[self._current]
            self._records[self._current] = replace(record, replans=record.replans + 1)
            STEPS.info(f"navigate_to: replanning with Nav2 ({record.replans + 1})")
        self._transport.cancel()
        self._external_active = False
        raw = self._map_raw()
        self._publish(state)
        try:
            path = self._transport.start(self._goal)
        except Nav2PlanningError as exc:
            distance = math.hypot(state.x - self._goal.x, state.y - self._goal.y)
            reach = self._goal.position_tolerance < self._config.fine_approach_radius and (
                distance <= self._nav2.occupied_goal_reach
            )
            if exc.goal_occupied and reach:
                # The goal's footprint touches something the costmap sees (a low open drawer the robot
                # reaches into), but it is within the fine approach's reach: fine moves finish it,
                # checked against the navigation map, not the costmap.
                STEPS.info(f"navigate_to: goal occupied in Nav2's costmap, {distance:.2f} m away; fine approach")
                self._fine_started = True
                self._next_fine_move(state)
                return
            status = "blocked" if replan and exc.unreachable else "unreachable" if exc.unreachable else "failed"
            if self._motion_id is not None:
                self._stop(status, str(exc))
            else:
                self._set(status, str(exc))
            return
        if len(path) < 2 or not all(math.isfinite(value) for point in path for value in point):
            raise ValueError("Nav2 returned an invalid path.")
        self._external_active = True
        self._fresh_command_time = self._runtime.time_seconds  # The grace for the first command starts.
        self._map_version = raw.version
        record = self._records[self._current]
        self._records[self._current] = replace(record, path=path)
        STEPS.info(f"navigate_to: Nav2 route of {describe_path(path)}; following")

    def navigate_to(
        self,
        x: float,
        y: float,
        heading: float | None = None,
        *,
        position_tolerance: float = 0.04,
        heading_tolerance: float = 0.04,
        timeout: float | None = None,
    ) -> int:
        if self._runtime.status != EpisodeStatus.RUNNING or self.active:
            raise RuntimeError("Nav2 requires a running episode with no active navigation goal.")
        goal = NavigationGoal(x, y, heading, position_tolerance, heading_tolerance)
        budget = self._config.timeout if timeout is None else timeout  # [s]; None: no budget.
        ticks = None if budget is None else duration_ticks(budget, self._runtime.physics_dt)
        state = self._state()
        grid = self._maps.snapshot()
        self._map_raw()
        self._current = len(self._records) + 1
        self._records[self._current] = NavigationFeedback(
            self._current, self._runtime.episode_id, "active", "Nav2 planning"
        )
        self._goal, self._deadline = goal, None if ticks is None else self._runtime.tick + ticks
        self._motion_id, self._last_tick = None, -1
        self._fine_move, self._fine_moves, self._fine_started = None, 0, False
        self._within_since = None
        self._corrections = 0
        STEPS.info(f"navigate_to: {describe_goal(goal)}; planning (Nav2)")
        self._saturation_warned = False
        if not grid.free((x, y)):
            row, col = grid.cell((x, y))
            unseen = (
                grid.unknown is not None
                and 0 <= row < grid.unknown.shape[0]
                and 0 <= col < grid.unknown.shape[1]
                and grid.unknown[row, col]
            )
            self._set(
                "invalid_goal",
                "Goal is in unobserved space" if unseen else "Goal violates the common robot envelope or map bounds",
            )
            return self._current
        try:
            self._plan(state, replan=False)
        except Exception:
            self._transport.cancel()
            self._set("failed", "Nav2 setup or planning failed")
            raise
        return self._current

    def explore(self, *, timeout: float | None = None) -> int:
        """Explore until no frontier is left: explore_lite picks frontiers on Nav2's global costmap and
        each becomes a navigation leg (any heading, within ``explore_arrival_tolerance``). Needs a map
        prior with unexplored space. ``timeout`` [s]: the whole exploration's budget (None: the
        configured navigation default)."""
        if self._runtime.status != EpisodeStatus.RUNNING or self.active:
            raise RuntimeError("Nav2 requires a running episode with no active navigation goal.")
        raw = self._map_raw()
        if raw.unknown is None or not raw.unknown.any():
            raise ValueError("Exploration needs unexplored space on the navigation map (an unexplored floor prior).")
        budget = self._config.timeout if timeout is None else timeout  # [s]; None: no budget.
        if budget is not None and not (math.isfinite(budget) and budget > 0):
            raise ValueError("An exploration timeout must be positive and finite, or None.")
        ticks = None if budget is None else duration_ticks(budget, self._runtime.physics_dt)
        self._state()
        self._current = len(self._records) + 1
        self._records[self._current] = NavigationFeedback(
            self._current, self._runtime.episode_id, "active", "Exploring: waiting for a frontier"
        )
        self._goal, self._deadline = None, None if ticks is None else self._runtime.tick + ticks
        self._motion_id, self._last_tick, self._fine_move = None, -1, None
        self._legs, self._ending, self._saturation_warned = 0, False, False
        STEPS.info("explore: explore_lite on Nav2's costmap; waiting for a frontier")
        try:
            self._transport.explore_start()
        except Exception:
            self._set("failed", "explore_lite did not start")
            raise
        self._exploring = True
        return self._current

    def _explore_step(self, state: Nav2State) -> bool:
        """Advance an exploration: the budget, explore_lite's next frontier (it supersedes a leg being
        followed), its end. True when no leg is to be followed this tick."""
        if self.feedback(self._current).status == "stopping":
            return False  # A controlled stop finishes first (a leg's arrival, a cancel).
        if self._deadline is not None and self._runtime.tick >= self._deadline:
            self._ending = True
            if self._goal is not None:
                self._stop("timed_out", "Exploration budget exhausted")
            else:
                self._set("timed_out", "Exploration budget exhausted")
            return True
        event = self._transport.explore_poll()
        if event.goal is not None:
            self._legs += 1
            x, y = event.goal
            self._goal = NavigationGoal(x, y, None, self._nav2.explore_arrival_tolerance, math.pi)
            self._fine_move, self._fine_moves, self._fine_started = None, 0, False
            self._within_since, self._corrections = None, 0
            self._set("active", f"Exploring: frontier {self._legs} at ({x:.2f}, {y:.2f}) m")
            if not self._maps.snapshot().free((x, y)):
                self._set("invalid_goal", "Frontier goal violates the common robot envelope or map bounds")
                return True
            self._plan(state, replan=False)
            return self._goal is None or self.feedback(self._current).status != "active"
        if self._goal is not None:
            return False
        if event.complete:
            self._ending = True
            self._set("succeeded", f"No frontier left after {self._legs} frontier goals")
        elif not event.running:
            self._ending = True
            self._set("failed", "explore_lite exited")
        return True

    def _stop(self, status: str, reason: str) -> None:
        self._transport.cancel()
        self._external_active = False
        self._motion_id = self._motion._stop()
        self._finish = status
        self._fine_move = None
        self._set("stopping", reason)

    def cancel(self, command_id: int) -> None:
        if command_id != self._current or not self.active:
            raise ValueError("Only the active navigation goal can be cancelled.")
        self._ending = self._exploring
        self._stop("cancelled", "Cancelled by agent")

    def reset(self) -> None:
        self._transport.reset()
        self._external_active = False
        if self.active:
            self._ending = self._exploring
            self._set("cancelled", "Episode reset")
        self._exploring = self._ending = False
        self._motion_id, self._previous_state, self._last_tick = None, None, -1
        self._fine_move = None
        self._last_sensor_tick = -1
        self._last_path_check_tick = -1

    def costmap_cost(self, x: float, y: float) -> int:
        """Nav2's global costmap cost at a world point [m] (trusted diagnostics)."""
        return self._transport.costmap_cost(x, y)

    def close(self) -> None:
        self._transport.close()
        self._external_active = False

    def update(self) -> None:
        if self._runtime.status != EpisodeStatus.RUNNING:
            if self._external_active:
                self._transport.cancel()
                self._external_active = False
            if self._exploring:
                self._transport.explore_stop()
                self._exploring = self._ending = False
            return
        if (
            self._nav2.sensor_cameras
            and self._runtime.tick % self._sensor_period == 0
            and self._last_sensor_tick != self._runtime.tick
        ):
            self._last_sensor_tick = self._runtime.tick
            # Nav2 keeps mapping between goals, but only with the arm folded: the front cameras see an
            # arm that is out (a look at a low drawer handle), and Nav2 would keep it as an obstacle.
            joints = self._observations.observe().joints
            if joints is not None and joints.valid:
                if self._arm_folded(dict(zip(joints.names, joints.positions.tolist(), strict=True))):
                    self._publish(self._state())
        if self._last_tick == self._runtime.tick or self._runtime.tick % self._period:
            return
        self._last_tick = self._runtime.tick
        if not self.active:
            return
        try:
            self._advance()
        except Exception:
            self._transport.cancel()
            self._external_active = False
            self._ending = self._exploring
            self._set("failed", "Nav2 transport or command validation failed")
            raise

    def _replan_observed_blockage(self, state: Nav2State) -> None:
        """Stop stale motion before requesting a bounded sensor-triggered replan."""
        if self._records[self._current].replans >= self._nav2.max_sensor_replans:
            self._stop("blocked", "Observed route remained blocked after replanning")
            return
        self._motion_id = self._motion._stop()
        self._plan(state, replan=True)

    def _next_fine_move(self, state: Nav2State) -> None:
        """Start the next single-axis move from the settled pose, or end the command."""
        if self._deadline is not None and self._runtime.tick >= self._deadline:
            self._set("timed_out", "Navigation budget exhausted")
            return
        position = (state.x, state.y)
        move = self._fine.move(self._goal, position, state.yaw)
        if not FineApproach.move_free(self._maps.snapshot(), move, position, state.yaw):
            self._set("blocked", f"Fine approach move ({move.description}) crosses occupied map cells")
            return
        self._fine_moves += 1
        self._fine_move = move
        self._set("active", f"Fine approach move {self._fine_moves}: {move.description}")
        self._motion_id = self._motion._command_velocity(*move.command, duration=move.duration)

    def _advance(self) -> None:
        state = self._state()
        self._publish(state)
        if self._exploring and self._explore_step(state):
            return
        self._follow(state)

    def _follow(self, state: Nav2State) -> None:
        """Follow the goal (or exploration leg): motion outcomes, fine moves, replans, Nav2's commands."""
        if self._motion_id is not None:
            motion = self._motion.feedback(self._motion_id)
            if motion.status == "overridden":
                self._transport.cancel()
                self._external_active = False
                self._set("overridden", "Direct motion superseded Nav2")
                return
            if motion.status in {"failed", "interrupted", "cancelled"}:
                self._set(motion.status, "Locomotion terminated")
                return
            if self._fine_move is not None or self.feedback(self._current).status == "stopping":
                # A fine move ends with the runtime's controlled stop, like a navigation stop.
                if motion.status != "done":
                    return
                fine_move, self._fine_move = self._fine_move, None
                goal = self._goal
                distance = math.hypot(state.x - goal.x, state.y - goal.y)
                arrived = distance <= goal.position_tolerance and (
                    goal.heading is None or abs(angle_error(goal.heading, state.yaw)) <= goal.heading_tolerance
                )
                if fine_move is None and (self._finish != "succeeded" or arrived and not self._fine_moves):
                    self._set(self._finish, self.feedback(self._current).reason)
                    return
                if arrived:
                    self._set("succeeded", "Settled pose satisfies goal tolerances")
                    return
                if self._fine.applies(goal, distance, self._fine_started):
                    self._next_fine_move(state)
                    return
                self._fine_started = False  # Beyond the exit radius: follow a Nav2 route in again.
                if self._corrections >= self._config.max_goal_corrections:
                    self._set(
                        "failed",
                        f"Settled pose stays outside the goal tolerances after {self._corrections} corrections"
                        f" ({distance:.3f} m"
                        + ("" if goal.heading is None else f", {abs(angle_error(goal.heading, state.yaw)):.3f} rad")
                        + ")",
                    )
                    return
                self._corrections += 1
                self._set("active", f"Correcting settled goal pose ({self._corrections})")
                self._motion_id = None
                self._plan(state, replan=True)
                if self.feedback(self._current).status != "active":
                    return
        if self._deadline is not None and self._runtime.tick >= self._deadline:
            self._stop("timed_out", "Navigation budget exhausted")
            return
        if self._map_raw().version != self._map_version:
            self._plan(state, replan=True)
            if self.feedback(self._current).status != "active":
                return
        elif (
            self._nav2.sensor_cameras
            and self._last_path_check_tick != self._runtime.tick
            and (
                self._last_path_check_tick < 0
                or self._runtime.tick - self._last_path_check_tick
                >= duration_ticks(self._nav2.sensor_replan_period, self._runtime.physics_dt)
            )
        ):
            self._last_path_check_tick = self._runtime.tick
            path = self._records[self._current].path
            # Fine moves to an occupied goal close by have no Nav2 route to check (they keep to the map).
            if len(path) >= 2 and self._transport.path_blocked(path, (state.x, state.y)):
                self._replan_observed_blockage(state)
                if self.feedback(self._current).status != "active":
                    return
        distance = math.hypot(state.x - self._goal.x, state.y - self._goal.y)
        if self._fine.applies(self._goal, distance, self._fine_started):
            if self._within_since is None:
                self._within_since = state.time
            heading = 0.0 if self._goal.heading is None else abs(angle_error(self._goal.heading, state.yaw))
            if (
                heading <= self._nav2.handover_heading_tolerance
                or state.time - self._within_since >= self._nav2.handover_wait
            ):
                self._within_since = None
                self._stop(
                    "succeeded",
                    f"Within the fine approach radius ({distance:.2f} m, {heading:.2f} rad); controlled stop",
                )
                self._fine_started = True
                return
        else:
            self._within_since = None
        self._apply_nav2(state)

    def _apply_nav2(self, state: Nav2State) -> None:
        """Nav2's route outcome or its newest velocity command, checked and leased to locomotion."""
        update = self._transport.poll()
        if update.status == "failed" and (
            update.blocked
            or self._nav2.sensor_cameras
            and self._transport.path_blocked(self._records[self._current].path, (state.x, state.y))
        ):
            self._replan_observed_blockage(state)
            return
        if update.status in {"succeeded", "failed"}:
            self._stop(update.status, update.reason)
            return
        if update.status != "active":
            raise ValueError(f"Unexpected Nav2 transport state: {update.status}")
        if update.command is not None and (not math.isfinite(update.age) or update.age < 0):
            raise TimeoutError("Nav2 velocity command is from a future clock")
        if update.command is None or update.age >= self._nav2.command_timeout:
            # No fresh command yet: the last one's lease runs out and the runtime stops the robot.
            if self._runtime.time_seconds - self._fresh_command_time > self._nav2.command_grace:
                raise TimeoutError(f"Nav2 sent no fresh velocity command for {self._nav2.command_grace} s")
            return
        self._fresh_command_time = self._runtime.time_seconds
        vx, vy, yaw = update.command
        linear_speed = math.hypot(vx, vy)
        if (
            not all(map(math.isfinite, update.command))
            or linear_speed > 1.5 * self._config.max_speed
            or abs(yaw) > 1.5 * self._config.max_yaw_speed
        ):
            raise ValueError(
                f"Nav2 command {update.command} violates configured velocity limits "
                f"(linear={self._config.max_speed}, angular={self._config.max_yaw_speed})"
            )
        if linear_speed > self._config.max_speed or abs(yaw) > self._config.max_yaw_speed:
            if not self._saturation_warned:
                logging.warning(
                    "Nav2 command %s exceeded runtime velocity limits; saturating bounded overshoot", update.command
                )
                self._saturation_warned = True
            if linear_speed > self._config.max_speed:
                scale = self._config.max_speed / linear_speed
                vx, vy = vx * scale, vy * scale
            yaw = max(-self._config.max_yaw_speed, min(yaw, self._config.max_yaw_speed))
        lease = math.floor((self._nav2.command_timeout - update.age) / self._runtime.physics_dt)
        if lease < 1:
            raise TimeoutError("Nav2 command expires before the next physics tick")
        self._motion_id = self._motion._command_velocity(vx, vy, yaw, duration=lease * self._runtime.physics_dt)
