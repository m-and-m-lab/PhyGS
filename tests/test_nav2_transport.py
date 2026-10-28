# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS transport lifecycle checks; requires the optional ROS message packages.

The whole module is skipped, visibly, without them: a test that skips itself halfway hides a
broken setup line on machines without ROS.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

for module in ("nav2_msgs.srv", "nav2_msgs.action", "rosgraph_msgs.msg"):
    pytest.importorskip(module)

from fakes import nav2_transport  # noqa: E402

from phygs.execution.navigation.backends.nav2 import Nav2Config, Nav2State  # noqa: E402
from phygs.execution.navigation.floor import DeclaredFloorMap  # noqa: E402
from phygs.execution.navigation.map import RectangleFootprint  # noqa: E402


def test_costmaps_clear_before_episode_observations_accumulate():
    transport = nav2_transport(_epoch=0, _stamp=0, _await_active=Mock(), _wait=Mock(), cancel=Mock())
    floor = DeclaredFloorMap(
        bounds=(-2.0, -2.0, 2.0, 2.0), resolution=0.1, footprint=RectangleFootprint(-0.7, 0.7, -0.7, 0.7), clearance=0
    )
    state = Nav2State(episode_id=1, time=0, x=0, y=0, yaw=0, vx=0, vy=0, yaw_rate=0)

    # Publish state before the backend publishes its first camera observation.
    transport.publish(state, floor.raw_snapshot())
    assert not transport._clear_pending
    transport._await_active.assert_called_once()
    for client in transport._clear_costmaps:
        client.call_async.assert_called_once()

    transport.publish(state, floor.raw_snapshot())
    for client in transport._clear_costmaps:
        client.call_async.assert_called_once()

    transport.reset()
    assert transport._clear_pending
    transport.publish(state, floor.raw_snapshot())
    assert not transport._clear_pending
    for client in transport._clear_costmaps:
        assert client.call_async.call_count == 2


@pytest.mark.parametrize(
    "error_name,unreachable",
    [
        ("GOAL_OCCUPIED", True),
        ("NO_VALID_PATH", True),
        ("START_OCCUPIED", False),
        ("TF_ERROR", False),
        ("TIMEOUT", False),
    ],
)
def test_planning_failure_distinguishes_blocked_goals(error_name, unreachable):
    from types import SimpleNamespace

    nav2_msgs = pytest.importorskip("nav2_msgs.action")
    from phygs.execution.navigation.backends.nav2 import Nav2PlanningError
    from phygs.execution.navigation.types import NavigationGoal

    result = nav2_msgs.ComputePathToPose.Result()
    result.error_code = getattr(result, error_name)
    waits = [
        SimpleNamespace(result=SimpleNamespace(successful=True)),
        Mock(accepted=True),
        SimpleNamespace(status=6, result=result),
    ]
    transport = nav2_transport(
        _epoch=0,
        _stamp=0,
        _clear_pending=False,
        _state=Mock(),
        _raw_map=Mock(),
        _wait=Mock(side_effect=waits),
        **{name: Mock() for name in ("_check_geometry", "publish", "_await_active", "_await_map")},
    )
    with pytest.raises(Nav2PlanningError) as caught:
        transport.start(NavigationGoal(1.0, 1.0, 0.0))
    assert caught.value.unreachable is unreachable


def test_activation_wait_resends_the_pose_and_survives_a_blocked_lifecycle_node():
    """A costmap activating blocks its node until it sees the bridge's transform; the bridge must
    keep sending its pose and treat an unanswered state request as "not yet", not as a fatal
    timeout (nav2_known_map bringup race, 2026-10-22)."""
    from types import SimpleNamespace

    answers = iter([None, None, "inactive", "active"])  # Two unanswered requests, then the states.

    def spin_until_future_complete(future, timeout_sec):
        answer = next(answers)
        future.done.return_value = answer is not None
        future.result.return_value = SimpleNamespace(current_state=SimpleNamespace(label=answer))

    client = Mock()
    client.wait_for_service.return_value = True
    client.call_async.side_effect = lambda request: Mock()
    transport = nav2_transport(
        Nav2Config(command_timeout=0.01),
        _publish_pose=Mock(),
        _executor=Mock(**{"spin_until_future_complete.side_effect": spin_until_future_complete}),
        _lifecycle=[client],
        _transitions=[Mock()],
    )
    transport._await_active()
    assert transport._publish_pose.call_count == 4 and client.remove_pending_request.call_count == 2
    assert not transport._poisoned


def test_a_second_nav2_stack_on_the_domain_is_refused():
    """A stale stack from a killed run answered the next run's requests ("parameters were not
    declared", commands from another simulation clock, 2026-10-25)."""
    client = Mock()
    client.wait_for_service.return_value = True
    answer = Mock(**{"done.return_value": True})
    answer.result.return_value.current_state.label = "active"
    transport = nav2_transport(
        Nav2Config(command_timeout=0.01),
        _publish_pose=Mock(),
        _executor=Mock(),
        _lifecycle=[client],
        _transitions=[Mock()],
    )
    client.call_async.return_value = answer
    transport._await_active()  # One stack: fine.
    space = "/spot_navigation"
    stale = [("planner_server", space), ("controller_server", space)] * 2
    transport._node.get_node_names_and_namespaces.return_value = stale
    with pytest.raises(RuntimeError, match="another Nav2 stack"):
        transport._await_active()


def test_activation_wait_gives_up_after_the_bringup_timeout():
    client = Mock()
    client.wait_for_service.return_value = True
    transport = nav2_transport(
        Nav2Config(command_timeout=0.01, bringup_timeout=0.05),
        _publish_pose=Mock(),
        _executor=Mock(
            **{
                "spin_until_future_complete.side_effect": lambda future, timeout_sec: setattr(
                    future.done, "return_value", False
                )
            }
        ),
        _lifecycle=[client],
        _transitions=[Mock()],
    )
    with pytest.raises(TimeoutError, match="did not become active within"):
        transport._await_active()


def test_activation_wait_drives_a_transition_the_lifecycle_manager_stalled_on():
    """Nav2's lifecycle manager lost a configure response and never went on (nav2_depth_map,
    2026-10-23): a node unconfigured for stalled_transition_seconds is configured by the bridge,
    then activated, and the wait ends when it reports active."""
    from types import SimpleNamespace

    labels = iter(["unconfigured", "unconfigured", "inactive", "inactive", "active"])

    def spin_until_future_complete(future, timeout_sec):
        future.done.return_value = True
        future.result.return_value = SimpleNamespace(current_state=SimpleNamespace(label=next(labels)))

    client, change = Mock(), Mock()
    client.wait_for_service.return_value = True
    client.srv_name = "planner_server/get_state"
    client.call_async.side_effect = lambda request: Mock()
    transport = nav2_transport(
        Nav2Config(command_timeout=0.01, stalled_transition_seconds=1e-9),
        _publish_pose=Mock(),
        _executor=Mock(**{"spin_until_future_complete.side_effect": spin_until_future_complete}),
        _lifecycle=[client],
        _transitions=[change],
    )
    transport._await_active()
    requested = [call.args[0].transition.label for call in change.call_async.call_args_list]
    assert requested == ["configure", "activate"]  # Once per stalled state, from the second poll on.


def test_voxel_filter_sends_one_centroid_per_occupied_cube():
    """Nav2 raytraces every point it receives: the bridge sends one point per voxel_size cube, marking
    when any point in it marks."""
    torch = pytest.importorskip("torch")
    from phygs.execution.navigation.backends.ros2 import voxel_filter

    points = torch.tensor([[0.01, 0.01, 1.01], [0.03, 0.03, 1.03], [0.07, 0.0, 1.0], [-0.01, 0.0, 1.0]])
    centroids, marked = voxel_filter(points, 0.05, torch.tensor([False, True, False, False]))
    found = {
        tuple(round(value, 4) for value in point): flag for point, flag in zip(centroids.tolist(), marked.tolist())
    }
    assert found == {(-0.01, 0.0, 1.0): False, (0.02, 0.02, 1.02): True, (0.07, 0.0, 1.0): False}
    assert len(voxel_filter(points[:0], 0.05, torch.zeros(0, dtype=torch.bool))[0]) == 0


def test_explore_lites_goals_are_answered_once_and_a_newer_one_supersedes():
    """Each NavigateToPose goal explore_lite sends gets one result: the leg's outcome, superseded by a
    newer goal, or given up when exploration stops (explore_lite skips a frontier given up)."""

    def handle(x, y):
        goal = Mock(is_cancel_requested=False, is_active=True)
        goal.request.pose.pose.position = SimpleNamespace(x=x, y=y)
        goal.goal_id.uuid = bytes([len(goals)]) * 16
        goals.append(goal)
        return goal

    goals = []
    transport = nav2_transport(_explorer=Mock(**{"poll.return_value": None}))
    first = handle(1.0, 2.0)
    transport._explore_goals.append(first)
    event = transport.explore_poll()
    assert event.goal == (1.0, 2.0) and event.running and not event.complete
    assert transport.explore_poll().goal is None  # Handed out once.
    transport._explore_goals += [handle(3.0, 0.0), handle(4.0, 0.0)]
    assert transport.explore_poll().goal == (4.0, 0.0)
    assert first.execute.called and goals[1].execute.called and not goals[2].execute.called
    assert transport._explore_outcomes[bytes(first.goal_id.uuid)] == (False, "Superseded by a newer frontier goal")
    transport.explore_finish(True, "succeeded: arrived")
    assert goals[2].execute.called and transport._explore_outcomes[bytes(goals[2].goal_id.uuid)][0]
    assert transport._explore_goals == [] and transport._explore_current is None
    result = transport._explore_execute(goals[2])
    goals[2].succeed.assert_called_once()
    assert result.error_code == 0
    result = transport._explore_execute(first)
    first.abort.assert_called_once()
    assert result.error_code != 0  # explore_lite blacklists the frontier.


def test_the_bottom_mark_margin_of_each_image_only_clears():
    """Nav2's voxel layer marks from points/<camera> and clears along clearing/<camera>: a voxel seen
    at the bottom edge of the view from a slightly lower camera is crossed by no later ray, so the
    image's bottom rows only clear."""
    import numpy as np
    import torch

    from phygs.execution.navigation.backends.nav2 import Nav2Config, Nav2State

    config = Nav2Config(sensor_cameras=("front",), sensor_mark_margin=0.25)
    transport = nav2_transport(config, _state=Nav2State(1, 0.0, 0, 0, 0, 0, 0, 0))
    camera = SimpleNamespace(  # One valid pixel per row, 8 rows.
        depth=torch.full((8, 1), 2.0),
        depth_valid=torch.ones((8, 1), dtype=torch.bool),
        valid=True,
        frame="front_optical",
        stamp=SimpleNamespace(episode_id=1, time_seconds=0.0),
        points=lambda stride: torch.stack([torch.tensor([0.0, float(row), 2.0]) for row in range(8)]),
    )
    transport.publish_camera("front", camera, np.eye(4))
    (marking,), (clearing,) = (
        transport._points["front"].publish.call_args.args,
        transport._clearing["front"].publish.call_args.args,
    )
    assert (marking.width, clearing.width) == (6, 8)  # The bottom 2 of 8 rows only clear.


def test_poll_and_the_costmap_read_what_nav2_already_sent_without_waiting():
    """The simulation never waits on Nav2: poll returns the newest command with its age, and the route
    check and displays read the newest published costmap; one from before a reset is ignored."""
    import numpy as np

    transport = nav2_transport(_stamp=1_200_000_000, _command=(1_100_000_000, (0.3, 0.0, 0.1)))
    update = transport.poll()
    assert update.command == (0.3, 0.0, 0.1) and abs(update.age - 0.1) < 1e-9
    assert all(call.kwargs == {"timeout_sec": 0.0} for call in transport._executor.spin_once.call_args_list)
    assert transport.costmap() is None
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=500_000_000)),
        metadata=SimpleNamespace(
            size_x=2, size_y=1, resolution=0.1, origin=SimpleNamespace(position=SimpleNamespace(x=-1.0, y=2.0))
        ),
        data=[0, 254],
    )
    transport._costmap_received(message)
    cells, origin, resolution = transport.costmap()
    assert cells.tolist() == [[0, 254]] and origin == (-1.0, 2.0) and resolution == 0.1
    transport._latest_costmap, transport._epoch = None, 2_000_000_000  # A reset moved the epoch.
    transport._costmap_received(message)
    assert transport.costmap() is None and not np.isnan(resolution)
