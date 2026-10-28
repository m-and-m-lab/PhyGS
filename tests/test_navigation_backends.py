# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend contracts without importing ROS or starting a simulation."""

import dataclasses
import math
from types import SimpleNamespace

import pytest
import torch
from fakes import SPOT_COVERAGE, SPOT_DEAD_BAND, TRANSIT, Motion, Runtime, grid, nav2_transport

from phygs.execution.navigation.backends.nav2 import (
    ExploreEvent,
    Nav2Config,
    Nav2NavigationBackend,
    Nav2PlanningError,
    Nav2Update,
)
from phygs.runtime.episode import EpisodeStatus
from phygs.skills.navigation.api import NavigationClient, NavigationConfig


class Transport:
    """A Nav2 transport recording what the backend sends and replaying a settable command."""

    def __init__(self):
        self.command = Nav2Update("active", (0.4, 0.0, 0.2))
        self.cancelled = self.resets = self.starts = 0
        self.blocked = False
        self.error = None
        self.cameras = []  # (camera, map_from_camera) in order
        self.camera_names = []
        self.explore_events = []  # What each explore_poll returns, in order; then nothing new.
        self.explore_answers = []  # (succeeded, reason) per frontier goal answered
        self.explorers = self.explore_stops = 0

    def publish(self, state, raw_map):
        self.state = state

    def publish_camera(self, name, camera, map_from_camera):
        self.cameras.append((camera, map_from_camera))
        self.camera_names.append(name)

    def start(self, goal):
        self.starts += 1
        if self.error:
            raise self.error
        return ((0.0, 0.0), (goal.x, goal.y))

    def poll(self):
        return self.command

    def path_blocked(self, path, position):
        return self.blocked

    def cancel(self):
        self.cancelled += 1

    def reset(self):
        self.resets += 1
        self.cancel()

    def close(self):
        self.cancel()

    def explore_start(self):
        self.explorers += 1

    def explore_poll(self):
        return self.explore_events.pop(0) if self.explore_events else ExploreEvent()

    def explore_finish(self, succeeded, reason):
        self.explore_answers.append((succeeded, reason))

    def explore_stop(self):
        self.explore_stops += 1


class CameraObservations:
    """The runtime's observations plus depth cameras ``names`` at the map origin."""

    def __init__(self, runtime, names):
        self.runtime, self.names = runtime, tuple(names)

    def observe(self):
        packet = self.runtime.observe()
        packet.cameras = {
            name: SimpleNamespace(frame=f"{name}_optical", stamp=packet.localization.stamp) for name in self.names
        }
        packet.transform = lambda target, source, stamp: torch.eye(4)
        return packet


def make_backend(nav2_config=None):
    """A Nav2 backend over the fake runtime (0.1 s ticks); a sensor camera reads :class:`CameraObservations`."""
    nav2_config = Nav2Config() if nav2_config is None else nav2_config
    runtime = Runtime()
    motion, transport = Motion(runtime), Transport()
    maps = SimpleNamespace(current=grid(), configuration={"source": "test_map"})
    maps.snapshot = maps.raw_snapshot = lambda: maps.current
    cameras = nav2_config.sensor_cameras
    observations = CameraObservations(runtime, cameras) if cameras else None
    nav = Nav2NavigationBackend(
        runtime,
        motion,
        maps,
        NavigationConfig(period=0.1, arm_posture=TRANSIT),
        nav2_config,
        transport,
        observations=observations,
        dead_band=SPOT_DEAD_BAND,
        coverage=SPOT_COVERAGE,
    )
    return nav, runtime, motion, transport, maps


@pytest.fixture
def backend():
    return make_backend()


def test_public_facade_selects_backend_without_changing_navigation_call(backend):
    nav, runtime, motion, transport, maps = backend
    client = NavigationClient(runtime, motion, backend_factory=lambda *args: nav)
    command = client.navigate_to(1.0, 0.0, 0.0)
    client.update()
    assert client.feedback(command).path == ()
    assert client.backend.feedback(command).path == ((0.0, 0.0), (1.0, 0.0))
    assert motion.command == (0.4, 0.0, 0.2)
    assert client.configuration["backend"] == "nav2"
    client.cancel(command)
    assert motion.command == (0, 0, 0)
    assert transport.cancelled >= 2
    runtime.tick = 8
    client.update()
    assert client.feedback(command).status == "cancelled"
    assert client.feedback(command).reason == "Cancelled by agent"


def test_public_facade_enforces_permissions_before_dispatch(backend):
    nav, runtime, motion, transport, maps = backend
    client = NavigationClient(runtime, motion, enabled=False, backend_factory=lambda *args: nav)
    with pytest.raises(PermissionError):
        client.navigate_to(1, 0, 0)
    assert transport.starts == 0


@pytest.mark.parametrize(
    "update,error",
    [
        (Nav2Update("active", (0.1, 0, 0), age=-0.1), TimeoutError),
        (Nav2Update("active", (math.nan, 0, 0)), ValueError),
        (Nav2Update("active", (2, 0, 0)), ValueError),
        (Nav2Update("active", (0, 0, 2.5)), ValueError),
    ],
)
def test_invalid_external_commands_never_reach_locomotion(backend, update, error):
    nav, runtime, motion, transport, maps = backend
    command = nav.navigate_to(1, 0, 0)
    transport.command = update
    with pytest.raises(error):
        nav.update()
    assert not motion.records
    assert nav.feedback(command).status == "failed"


@pytest.mark.parametrize("update", [Nav2Update("active"), Nav2Update("active", (0.1, 0, 0), age=0.2)])
def test_the_simulation_does_not_wait_for_a_late_nav2_command(backend, update):
    """Nav2 runs beside the simulation: without a fresh command the robot gets none (its last lease runs
    out) and the goal goes on; only command_grace without one fails it."""
    nav, runtime, motion, transport, maps = backend
    command = nav.navigate_to(1, 0, 0)
    transport.command = update
    nav.update()
    assert not motion.records and nav.feedback(command).status == "active"
    runtime.tick = 21  # 2.1 s later: past the 2 s grace.
    with pytest.raises(TimeoutError, match="no fresh velocity command"):
        nav.update()
    assert nav.feedback(command).status == "failed"


def test_small_nav2_turn_overshoot_is_limited_before_locomotion(backend):
    nav, runtime, motion, transport, maps = backend
    nav.navigate_to(1, 0, 0)
    transport.command = Nav2Update("active", (0.0, 0.0, 1.6))
    nav.update()
    assert motion.command == (0.0, 0.0, 1.5)


def test_external_success_requires_physical_arrival_and_stopping(backend):
    nav, runtime, motion, transport, maps = backend
    command = nav.navigate_to(1, 0, 0)
    transport.command = Nav2Update("succeeded")
    nav.update()
    assert nav.feedback(command).status == "stopping"
    runtime.tick = 8
    transport.command = Nav2Update("active", (0.4, 0, 0))
    nav.update()
    assert nav.feedback(command).status == "active"
    assert transport.starts == 2
    runtime.x, runtime.tick = 1.0, 9
    transport.command = Nav2Update("succeeded")
    nav.update()
    runtime.tick = 17
    nav.update()
    assert nav.feedback(command).status == "succeeded"


def test_nav2_hands_over_to_fine_moves_once_turned_or_after_the_wait(backend):
    """Inside the fine radius Nav2 keeps driving while the heading is off (it turns while it walks),
    up to handover_wait (2 s; the fake runtime ticks 0.1 s)."""
    nav, runtime, motion, transport, maps = backend
    runtime.x, runtime.yaw = 0.94, 0.5  # Within 0.10 m of the goal, 0.5 rad off its heading.
    command = nav.navigate_to(1.0, 0.0, 0.0)
    nav.update()
    assert nav.feedback(command).status == "active" and motion.command == (0.4, 0.0, 0.2)  # Nav2 drives.
    runtime.yaw, runtime.tick = 0.05, 1  # Turned within 0.1 rad: hand over.
    nav.update()
    assert nav.feedback(command).reason.startswith("Within the fine approach radius")
    nav, runtime, motion, transport, maps = make_backend()
    runtime.x, runtime.yaw = 0.94, 0.5  # 0.5 rad off, and staying so.
    command = nav.navigate_to(1.0, 0.0, 0.0)
    for tick in range(0, 40):
        runtime.tick = tick
        nav.update()
        if nav.feedback(command).reason.startswith("Within the fine approach radius"):
            break
    assert 19 <= tick <= 21  # About 2 s after entering the radius.


def test_an_occupied_goal_within_fine_reach_is_finished_by_fine_moves():
    """Reaching into an open drawer, the goal's footprint touches the drawer in Nav2's costmap
    (GOAL_OCCUPIED); within occupied_goal_reach (0.35 m) the fine moves finish it on the navigation
    map, farther away it is unreachable."""
    occupied = Nav2PlanningError("Nav2 planner failed with code 206", unreachable=True, goal_occupied=True)
    for x, expected in ((0.7, "Fine approach move 1"), (0.0, "Nav2 planner failed")):  # Within 0.35 m, beyond.
        nav, runtime, motion, transport, maps = make_backend()
        runtime.x = x
        transport.error = occupied
        command = nav.navigate_to(1.0, 0.0, 0.0)
        assert nav.feedback(command).reason.startswith(expected), nav.feedback(command).reason
    assert nav.feedback(command).status == "unreachable"


def test_nav2_goal_is_finished_by_single_axis_fine_moves(backend):
    nav, runtime, motion, transport, maps = backend
    runtime.x = 0.94
    command = nav.navigate_to(1.0, 0.0, 0.0)
    nav.update()
    assert nav.feedback(command).reason.startswith("Within the fine approach radius")
    assert transport.cancelled and motion.command == (0, 0, 0)
    for tick in range(1, 40):
        runtime.tick = tick
        nav.update()
        if nav.feedback(command).reason.startswith("Fine approach move 1"):
            break
    assert motion.command[1:] == (0.0, 0.0) and motion.command[0] > 0  # Forward only.
    runtime.x = 1.0
    for tick in range(runtime.tick + 1, runtime.tick + 40):
        runtime.tick = tick
        nav.update()
        if not nav.active:
            break
    assert nav.feedback(command).status == "succeeded"
    assert nav.feedback(command).reason == "Settled pose satisfies goal tolerances"
    assert transport.starts == 1  # No Nav2 replan near the goal.


def test_reset_fences_external_work_and_preserves_command_identity(backend):
    nav, runtime, motion, transport, maps = backend
    old = nav.navigate_to(1, 0, 0)
    nav.reset()
    runtime.episode_id += 1
    new = nav.navigate_to(1, 0, 0)
    assert transport.resets == 1
    assert new != old
    assert nav.feedback(old).status == "cancelled"


def test_terminal_episode_cancels_external_actions(backend):
    nav, runtime, motion, transport, maps = backend
    command = nav.navigate_to(1, 0, 0)
    count = transport.cancelled
    runtime.status = EpisodeStatus.INTERRUPTED
    nav.update()
    assert transport.cancelled == count + 1
    assert nav.feedback(command).status == "interrupted"
    assert not motion.records


def test_changed_map_replans_and_distinguishes_blocked(backend):
    nav, runtime, motion, transport, maps = backend
    command = nav.navigate_to(1, 0, 0)
    nav.update()
    runtime.tick = 1
    maps.current = grid(version=1)
    transport.error = Nav2PlanningError("No path", unreachable=True)
    nav.update()
    assert nav.feedback(command).status == "stopping"
    runtime.tick = 9
    nav.update()
    assert nav.feedback(command).status == "blocked"
    assert nav.feedback(command).reason == "No path"


def test_nav2_scope_is_explicit(backend):
    nav, runtime, motion, transport, maps = backend
    runtime.localized = False
    with pytest.raises(ValueError, match="localization"):
        nav.navigate_to(1, 0, 0)


def test_optional_heading_keeps_public_goal_semantics(backend):
    nav, runtime, motion, transport, maps = backend
    command = nav.navigate_to(1, 0)
    runtime.x, runtime.yaw = 1.0, 2.0
    transport.command = Nav2Update("succeeded")
    nav.update()
    runtime.tick = 8
    nav.update()
    assert nav.feedback(command).status == "succeeded"


def test_ros_command_timestamp_and_frame_fences():
    transport = nav2_transport(_minimum_stamp=100, _stamp=200)
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=99), frame_id="spot_navigation/base_link"),
        twist=SimpleNamespace(linear=SimpleNamespace(x=0.1, y=0, z=0), angular=SimpleNamespace(x=0, y=0, z=0.2)),
    )
    transport._command_received(message)
    assert transport._command is None
    message.header.stamp.nanosec = 200
    transport._command_received(message)
    assert transport._command == (200, (0.1, 0, 0.2))
    message.header.stamp.nanosec = 201
    with pytest.raises(ValueError, match="future"):
        transport._command_received(message)
    message.header.stamp.nanosec = 200
    message.header.frame_id = "world"
    with pytest.raises(ValueError, match="body frame"):
        transport._command_received(message)


def test_ros_timeout_prevents_reuse_after_unknown_action_acceptance():
    transport = nav2_transport(_executor=SimpleNamespace(spin_until_future_complete=lambda *args, **kwargs: None))
    with pytest.raises(TimeoutError):
        transport._wait(SimpleNamespace(done=lambda: False))
    with pytest.raises(RuntimeError, match="restart"):
        transport.reset()


def test_velocity_lease_cannot_outlive_external_command(backend):
    nav, runtime, motion, transport, maps = backend
    nav.navigate_to(1, 0, 0)
    transport.command = Nav2Update("active", (0.1, 0, 0), age=0.15)
    with pytest.raises(TimeoutError, match="physics tick"):
        nav.update()
    assert not motion.records


def test_costmap_acknowledges_physical_cells_without_double_inflation():
    import numpy as np

    from phygs.execution.navigation.backends.ros2 import costmap_matches

    source = grid()
    metadata = SimpleNamespace(
        size_y=40,
        size_x=40,
        resolution=0.2,
        origin=SimpleNamespace(position=SimpleNamespace(x=-2, y=-4), orientation=SimpleNamespace(x=0, y=0, z=0, w=1)),
    )
    data = np.full(1600, 253, dtype=np.uint8)
    costmap = SimpleNamespace(metadata=metadata, data=data)
    assert costmap_matches(costmap, source)
    data[0] = 254
    assert not costmap_matches(costmap, source)
    metadata.origin.position.x = 0
    assert not costmap_matches(costmap, source)


def test_ros_occupancy_preserves_unknown_before_inflation():
    from dataclasses import replace

    import numpy as np

    from phygs.execution.navigation.backends.ros2 import costmap_matches, occupancy_values

    source = grid()
    occupied = source.occupied.copy()
    unknown = np.zeros_like(occupied)
    occupied[1, 1] = occupied[2, 2] = True
    unknown[2, 2] = True
    source = replace(source, occupied=occupied, unknown=unknown)
    values = occupancy_values(source)
    assert values[0, 0] == 0 and values[1, 1] == 100 and values[2, 2] == -1
    # Jazzy inflate_around_unknown turns unknown source cells lethal in the costmap.
    metadata = SimpleNamespace(
        size_y=40,
        size_x=40,
        resolution=0.2,
        origin=SimpleNamespace(position=SimpleNamespace(x=-2, y=-4), orientation=SimpleNamespace(x=0, y=0, z=0, w=1)),
    )
    data = np.where(occupied, 254, 0).astype(np.uint8)
    costmap = SimpleNamespace(metadata=metadata, data=data.ravel())
    assert costmap_matches(costmap, source)
    data[2, 2] = 0
    assert not costmap_matches(costmap, source)


def test_nav2_rejects_previous_episode_observed_map_before_motion(backend):
    from dataclasses import replace

    nav, runtime, motion, transport, maps = backend
    maps.current = replace(maps.current, episode_id=runtime.episode_id + 1)
    with pytest.raises(RuntimeError, match="another episode"):
        nav.navigate_to(1, 0)
    assert transport.starts == 0 and not motion.records
    assert not nav.active


def test_camera_costmap_obstruction_replans_and_stops():
    # The camera is published every 0.1 s tick; the path is checked every 0.1 s.
    camera = Nav2Config(sensor_cameras=("inspection",), sensor_period=0.1, sensor_replan_period=0.1)
    nav, runtime, motion, transport, maps = make_backend(camera)
    command = nav.navigate_to(1, 0, 0)
    nav.update()
    transport.blocked = True
    transport.error = Nav2PlanningError("No route around observed obstacle", unreachable=True)
    runtime.tick = 250
    nav.update()
    assert transport.starts == 2
    assert motion.command == (0, 0, 0)
    assert nav.feedback(command).status == "stopping"
    assert nav.feedback(command).replans == 1
    runtime.tick = 300
    nav.update()
    assert nav.feedback(command).status == "blocked"
    assert transport.cameras  # The camera reached Nav2 with each sensor update.


def test_every_sensor_camera_reaches_nav2_each_sensor_period():
    cameras = Nav2Config(sensor_cameras=("frontleft", "frontright"), sensor_period=0.1, sensor_replan_period=0.1)
    nav, runtime, motion, transport, maps = make_backend(cameras)
    nav.navigate_to(1, 0, 0)
    nav.update()
    assert transport.camera_names[-2:] == ["frontleft", "frontright"]
    with pytest.raises(ValueError, match="repeat"):
        Nav2Config(sensor_cameras=("frontleft", "frontleft"))
    for stride in (0, 1.5, True):
        with pytest.raises(ValueError, match="pixel stride"):
            Nav2Config(sensor_cameras=("frontleft",), sensor_pixel_stride=stride)
    for size in (-0.05, float("nan")):
        with pytest.raises(ValueError, match="voxel size"):
            Nav2Config(sensor_cameras=("frontleft",), sensor_voxel_size=size)


def test_cameras_map_between_goals_only_with_the_arm_folded():
    """Between goals Nav2 keeps mapping, but not while the arm is out (a look at a low handle): the
    front cameras would see the arm and Nav2 would keep it as an obstacle. No posture error then;
    moving the base still needs the folded arm."""
    cameras = Nav2Config(sensor_cameras=("frontleft",), sensor_period=0.1, sensor_replan_period=0.1)
    nav, runtime, motion, transport, maps = make_backend(cameras)
    runtime.tick = 1
    nav.update()
    assert transport.camera_names == ["frontleft"]  # Folded: mapped.
    runtime.joints["arm_sh1"] = 0.0  # The arm is out.
    runtime.tick = 2
    nav.update()
    assert transport.camera_names == ["frontleft"]  # Out: not mapped, and no error.
    with pytest.raises(ValueError, match="transit posture"):
        nav.navigate_to(1, 0, 0)


def test_costmap_path_invalidation_uses_only_remaining_segments():
    from phygs.execution.navigation.backends.ros2 import costmap_blocks_path

    data = [0] * 100
    metadata = SimpleNamespace(
        size_x=10,
        size_y=10,
        resolution=0.1,
        origin=SimpleNamespace(position=SimpleNamespace(x=0, y=0)),
    )
    costmap = SimpleNamespace(metadata=metadata, data=data)
    path = ((0.05, 0.05), (0.45, 0.05), (0.85, 0.05))
    assert not costmap_blocks_path(costmap, path, (0.25, 0.05))
    data[3] = 254
    assert costmap_blocks_path(costmap, path, (0.25, 0.05))
    data[3], data[7] = 0, 253
    assert costmap_blocks_path(costmap, path, (0.25, 0.05))
    data[7], data[0] = 0, 254
    assert not costmap_blocks_path(costmap, path, (0.45, 0.05))
    data[6] = 255  # Unknown: Nav2 plans through it; the cameras mark what is there.
    assert not costmap_blocks_path(costmap, path, (0.45, 0.05))


def test_camera_controller_failure_rechecks_observed_route():
    # The periodic path check (every 1 s) runs at tick 0 and finds the route clear; at tick 1 only
    # the controller's failure can trigger the recheck.
    camera = Nav2Config(sensor_cameras=("inspection",), sensor_period=0.1, sensor_replan_period=1.0)
    nav, runtime, motion, transport, maps = make_backend(camera)
    command = nav.navigate_to(1, 0, 0)
    nav.update()
    assert transport.starts == 1 and motion.command == (0.4, 0.0, 0.2)
    runtime.tick = 1
    transport.command = Nav2Update("failed", reason="Local obstacle")
    transport.blocked = True
    transport.error = Nav2PlanningError("No safe replacement path", unreachable=True)
    nav.update()
    assert transport.starts == 2
    assert nav.feedback(command).replans == 1
    assert nav.feedback(command).status == "stopping"
    assert motion.command == (0, 0, 0)


def test_controller_without_a_collision_free_command_replans_on_any_map():
    """MPPI ends a route it cannot follow without a collision (NO_VALID_CONTROL): replan, even on a
    known map, where no camera check would find the route blocked. Other failures end the command."""
    for blocked, starts, replans, status in ((True, 2, 1, "active"), (False, 1, 0, "stopping")):
        nav, runtime, motion, transport, maps = make_backend()
        command = nav.navigate_to(1, 0, 0)
        nav.update()
        runtime.tick = 1
        transport.command = Nav2Update("failed", reason="Nav2 result 6", blocked=blocked)
        nav.update()
        assert (transport.starts, nav.feedback(command).replans) == (starts, replans)
        assert nav.feedback(command).status == status, blocked


def test_repeated_localization_sample_preserves_yaw_rate(backend):
    nav, runtime, *_ = backend
    nav._state()
    runtime.tick = 1
    runtime.yaw = 0.1
    moving = nav._state()
    assert moving.yaw_rate == pytest.approx(1.0)
    assert nav._state().yaw_rate == pytest.approx(moving.yaw_rate)
    runtime.tick = 2
    assert nav._state().yaw_rate == pytest.approx(0.0)
    nav.reset()
    runtime.episode_id += 1
    runtime.tick = 0
    assert nav._state().yaw_rate == 0.0


def test_ros2_check_before_kit_imports_no_numpy_or_torch():
    """Launchers run this before Kit; NumPy's OpenBLAS threads would crash Kit's telemetry fork."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys; import phygs.execution.navigation.backends.ros2_check; "
        "print(sorted(m for m in ('numpy', 'torch') if m in sys.modules))"
    )
    source = Path(__file__).parents[1]
    loaded = subprocess.run(
        [sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": str(source)}, capture_output=True, text=True
    )
    assert loaded.returncode == 0, loaded.stderr
    assert loaded.stdout.strip() == "[]"


def exploring_backend():
    """A backend whose Nav2 map is unexplored beyond 1 m of the origin (the footprint snapshot stays free)."""
    nav, runtime, motion, transport, maps = make_backend()
    free = grid()
    yy, xx = torch.meshgrid(torch.arange(40), torch.arange(40), indexing="ij")
    unknown = (torch.hypot(-2.0 + (xx + 0.5) * 0.2, -4.0 + (yy + 0.5) * 0.2) > 1.0).numpy()
    unexplored = dataclasses.replace(grid(unknown), unknown=unknown)
    maps.snapshot, maps.raw_snapshot = (lambda: free), (lambda: unexplored)
    return nav, runtime, motion, transport


def finish_leg(nav, runtime, transport, x):
    """Nav2 reports the leg's goal reached at ``x`` [m]; the controlled stop ends 8 ticks later."""
    runtime.x, runtime.tick = x, runtime.tick + 1
    transport.command = Nav2Update("succeeded")
    nav.update()
    runtime.tick += 8
    nav.update()
    transport.command = Nav2Update("active", (0.4, 0.0, 0.2))


def test_exploration_needs_unexplored_space(backend):
    nav, runtime, motion, transport, maps = backend
    with pytest.raises(ValueError, match="unexplored"):
        nav.explore()
    assert transport.explorers == 0


def test_exploration_walks_explore_lites_frontiers_until_none_is_left():
    """Each frontier goal is a leg (any heading, the arrival tolerance); explore_lite hears each leg's
    outcome, a failed leg does not end the exploration, and no frontier left ends it."""
    nav, runtime, motion, transport = exploring_backend()
    client = NavigationClient(runtime, motion, backend_factory=lambda *args: nav)
    command = client.explore()
    assert transport.explorers == 1 and nav.feedback(command).status == "active"
    nav.update()
    assert transport.starts == 0 and motion.sent == []  # No frontier yet: the robot stands.
    transport.explore_events = [ExploreEvent(goal=(1.5, 0.0))]
    runtime.tick = 1
    nav.update()
    assert transport.starts == 1 and motion.command == (0.4, 0.0, 0.2)
    assert "frontier 1 at (1.50, 0.00)" in nav.feedback(command).reason
    finish_leg(nav, runtime, transport, 1.3)  # Within the 0.3 m arrival tolerance.
    assert [succeeded for succeeded, _ in transport.explore_answers] == [True]
    assert nav.feedback(command).status == "active" and nav.active
    transport.error = Nav2PlanningError("No path", unreachable=True)
    transport.explore_events = [ExploreEvent(goal=(3.0, 2.0))]
    runtime.tick += 1
    nav.update()
    assert transport.explore_answers[-1] == (False, "unreachable: No path")
    assert nav.feedback(command).status == "active"
    transport.error = None
    transport.explore_events = [ExploreEvent(complete=True)]
    runtime.tick += 1
    nav.update()
    feedback = client.feedback(command)
    assert (feedback.status, feedback.reason) == ("succeeded", "No frontier left after 2 frontier goals")
    assert transport.explore_stops == 1 and not nav.active


def test_a_newer_frontier_supersedes_the_leg_being_followed():
    nav, runtime, motion, transport = exploring_backend()
    command = nav.explore()
    transport.explore_events = [ExploreEvent(goal=(1.5, 0.0))]
    nav.update()
    transport.explore_events = [ExploreEvent(goal=(0.0, 2.0))]
    runtime.tick = 1
    nav.update()
    assert transport.starts == 2 and nav.feedback(command).path[-1] == (0.0, 2.0)
    assert transport.explore_answers == []  # The transport answers the superseded goal.
    assert "frontier 2" in nav.feedback(command).reason


def test_cancelling_or_the_budget_ends_the_whole_exploration():
    nav, runtime, motion, transport = exploring_backend()
    command = nav.explore()
    transport.explore_events = [ExploreEvent(goal=(1.5, 0.0))]
    nav.update()
    nav.cancel(command)
    runtime.tick = 8
    nav.update()
    assert nav.feedback(command).status == "cancelled" and transport.explore_stops == 1
    assert transport.explore_answers == []  # Not a leg outcome: explore_lite is stopped.
    command = nav.explore(timeout=0.5)
    runtime.tick = 13
    nav.update()
    assert nav.feedback(command).status == "timed_out" and transport.explore_stops == 2
    command = nav.explore()
    transport.explore_events = [ExploreEvent(running=False)]
    runtime.tick = 14
    nav.update()
    assert (nav.feedback(command).status, nav.feedback(command).reason) == ("failed", "explore_lite exited")
