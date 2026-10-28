# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS 2 Jazzy transport for Nav2 planner/controller actions and stamped velocities.

No ROS modules are imported until construction. The simulation owns /clock;
external processes must use simulation time and a dedicated ROS domain.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time

import numpy as np
import torch

from phygs.execution.navigation.backends.nav2 import (
    ExploreEvent,
    Nav2Config,
    Nav2PlanningError,
    Nav2State,
    Nav2Update,
)
from phygs.execution.navigation.backends.ros2_check import require_ros2 as require_ros2
from phygs.execution.navigation.map import GridMap, RectangleFootprint
from phygs.execution.navigation.types import NavigationGoal
from phygs.observations.types import CameraObservation

LOG = logging.getLogger("phygs.skills.navigation")
# Lifecycle transitions the bridge drives when the manager stalls, by the node's primary state.
STALLED_TRANSITIONS = {"unconfigured": "configure", "inactive": "activate"}
SENSOR_SETTLE_ROUNDS = 10  # Costmap updates with held camera samples before an empty view counts as taken.


def voxel_filter(points: torch.Tensor, size: float, marking: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One point per occupied cube of edge ``size`` [m]: the centroid of ``points`` [m] ([N, 3]) in it,
    and whether any of them is a ``marking`` point ([N] bools); on their device. The cubes are keyed by
    one integer each (a 1D unique: 4 ms for 77k points on the CPU, where a row-wise unique took 110 ms)."""
    if not len(points):
        return points, marking
    keys = torch.floor(points / size).to(torch.int64)
    keys -= keys.min(dim=0).values
    span = keys.max(dim=0).values + 1
    _, cell = torch.unique((keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2], return_inverse=True)
    count = int(cell.max()) + 1
    sums = torch.zeros(count, 3, dtype=points.dtype, device=points.device).index_add_(0, cell, points)
    totals = torch.zeros(count, dtype=points.dtype, device=points.device).index_add_(
        0, cell, torch.ones_like(cell, dtype=points.dtype)
    )
    marked = torch.zeros(count, dtype=torch.bool, device=points.device)
    marked[cell[marking]] = True
    return sums / totals[:, None], marked


def point_cloud(points: np.ndarray, frame: str, stamp):
    """A ``sensor_msgs/PointCloud2`` of xyz ``points`` [m] in ``frame`` at ``stamp``."""
    from sensor_msgs.msg import PointCloud2, PointField

    message = PointCloud2()
    message.header.frame_id, message.header.stamp = frame, stamp
    message.height, message.width = 1, len(points)
    message.fields = [
        PointField(name=axis, offset=4 * index, datatype=PointField.FLOAT32, count=1)
        for index, axis in enumerate(("x", "y", "z"))
    ]
    message.is_bigendian = False
    message.point_step, message.row_step = 12, 12 * len(points)
    message.is_dense = True
    message.data = np.asarray(points, dtype="<f4").tobytes()
    return message


class RclpyNav2Transport:
    """Publish permitted state/maps and bridge Nav2 actions to expiring body commands."""

    def __init__(self, config: Nav2Config, geometry: dict):
        require_ros2()
        import rclpy
        from geometry_msgs.msg import TwistStamped
        from lifecycle_msgs.srv import ChangeState, GetState
        from nav2_msgs.action import ComputePathToPose, FollowPath
        from nav2_msgs.srv import ClearEntireCostmap, GetCostmap
        from nav_msgs.msg import OccupancyGrid, Odometry, Path
        from rcl_interfaces.srv import GetParameters, SetParametersAtomically
        from rclpy.action import ActionClient
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.parameter import Parameter
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import PointCloud2
        from tf2_ros import TransformBroadcaster

        self._rclpy, self._config = rclpy, config
        self._geometry = geometry
        self._poisoned = False
        self._raw_map = None
        self._context = Context()
        rclpy.init(context=self._context)
        self._node = rclpy.create_node(
            "simulation_navigation_bridge",
            namespace=config.namespace,
            context=self._context,
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        reliable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self._clock = self._node.create_publisher(Clock, "/clock", reliable)
        self._odom = self._node.create_publisher(Odometry, "odom", reliable)
        self._map = self._node.create_publisher(OccupancyGrid, "map", latched)
        self._path = self._node.create_publisher(Path, "planned_path", latched)
        # Two point topics per sensor camera, each an observation source of Nav2's voxel layer: the
        # points that mark (points/<camera>) and all points, which clear along their rays from that
        # camera's origin (clearing/<camera>).
        self._points = {
            name: self._node.create_publisher(PointCloud2, f"points/{name}", reliable) for name in config.sensor_cameras
        }
        self._clearing = {
            name: self._node.create_publisher(PointCloud2, f"clearing/{name}", reliable)
            for name in config.sensor_cameras
        }
        self._tf = TransformBroadcaster(self._node)
        # Nav2 publishes its global costmap (raw costs) a few times a second; route checks and displays
        # read the newest one instead of asking for it, so the simulation never waits on Nav2.
        from nav2_msgs.msg import Costmap

        self._latest_costmap = None
        self._costmap_subscription = self._node.create_subscription(
            Costmap, "global_costmap/costmap_raw", self._costmap_received, latched
        )
        self._subscription = self._node.create_subscription(TwistStamped, "cmd_vel", self._command_received, reliable)
        self._planner = ActionClient(self._node, ComputePathToPose, "compute_path_to_pose")
        self._controller = ActionClient(self._node, FollowPath, "follow_path")
        self._parameters = self._node.create_client(
            SetParametersAtomically, "controller_server/set_parameters_atomically"
        )
        self._costmap_parameters = [
            self._node.create_client(GetParameters, name + "/get_parameters")
            for name in ("global_costmap/global_costmap", "local_costmap/local_costmap")
        ]
        self._costmaps = [
            self._node.create_client(GetCostmap, name + "/get_costmap") for name in ("global_costmap", "local_costmap")
        ]
        self._clear_costmaps = [
            self._node.create_client(ClearEntireCostmap, name + "/clear_entirely_" + name)
            for name in ("global_costmap", "local_costmap")
        ]
        self._lifecycle = [
            self._node.create_client(GetState, name + "/get_state") for name in ("planner_server", "controller_server")
        ]
        self._transitions = [
            self._node.create_client(ChangeState, name + "/change_state")
            for name in ("planner_server", "controller_server")
        ]
        self._acknowledged_map = -1
        self._clear_pending = True
        self._epoch = 1_000_000_000
        self._stamp = self._epoch
        self._minimum_stamp = self._epoch
        self._map_version = -1
        self._camera_stamps: dict[str, tuple] = {}  # Last published sample per sensor camera.
        self._last_cameras: dict[str, tuple[CameraObservation, torch.Tensor]] = {}
        self._command = None
        self._planning = self._following = None
        self._plan_result = self._follow_result = None
        self._state: Nav2State | None = None
        self._closed = False
        self._explorer = None  # explore_lite's process while a frontier exploration runs.
        self._navigate_server = self._explore_status = None
        self._explore_goals: list = []  # Open NavigateToPose goal handles, oldest first.
        self._explore_current = None  # The goal handed to navigation by the last explore_poll.
        self._explore_outcomes: dict[bytes, tuple[bool, str]] = {}
        self._explore_complete = False

    def _wait(self, future):
        self._executor.spin_until_future_complete(future, timeout_sec=self._config.service_timeout)
        if not future.done():
            self._poisoned = True
            raise TimeoutError("Nav2 ROS operation timed out; the episode cannot continue")
        return future.result()

    def _time(self):
        from builtin_interfaces.msg import Time

        return Time(sec=self._stamp // 1_000_000_000, nanosec=self._stamp % 1_000_000_000)

    def _publish_pose(self) -> None:
        """The clock, the map->odom->base_link transforms and odometry for the current state."""
        from geometry_msgs.msg import TransformStamped
        from nav_msgs.msg import Odometry
        from rosgraph_msgs.msg import Clock

        state, prefix = self._state, self._config.namespace
        self._clock.publish(Clock(clock=self._time()))
        transforms = []
        for parent, child in (("map", "odom"), ("odom", "base_link")):
            transform = TransformStamped()
            transform.header.stamp = self._time()
            transform.header.frame_id, transform.child_frame_id = f"{prefix}/{parent}", f"{prefix}/{child}"
            transform.transform.rotation.w = 1.0
            if child == "base_link":
                transform.transform.translation.x, transform.transform.translation.y = state.x, state.y
                transform.transform.rotation.z, transform.transform.rotation.w = (
                    math.sin(state.yaw / 2),
                    math.cos(state.yaw / 2),
                )
            transforms.append(transform)
        self._tf.sendTransform(transforms)
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id = self._time(), f"{prefix}/odom"
        odom.child_frame_id = f"{prefix}/base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y = state.x, state.y
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = math.sin(state.yaw / 2), math.cos(state.yaw / 2)
        odom.twist.twist.linear.x, odom.twist.twist.linear.y = state.vx, state.vy
        odom.twist.twist.angular.z = state.yaw_rate
        self._odom.publish(odom)

    def publish(self, state: Nav2State, raw_map) -> None:
        from nav_msgs.msg import OccupancyGrid

        self._state, self._raw_map = state, raw_map
        stamp = self._epoch + round(state.time * 1_000_000_000)
        if stamp < self._stamp:
            raise ValueError("ROS simulation clock moved backwards without reset")
        self._stamp = stamp
        self._publish_pose()
        prefix = self._config.namespace
        if self._map_version != raw_map.version:
            message = OccupancyGrid()
            message.header.stamp, message.header.frame_id = self._time(), f"{prefix}/map"
            message.info.map_load_time = self._time()
            message.info.resolution = raw_map.resolution
            message.info.height, message.info.width = raw_map.occupied.shape
            message.info.origin.position.x, message.info.origin.position.y = raw_map.origin
            message.info.origin.orientation.w = 1.0
            message.data = occupancy_values(raw_map).ravel().tolist()
            self._map.publish(message)
            self._map_version = raw_map.version
        self._executor.spin_once(timeout_sec=0.0)
        # Clear once before this episode's first camera sample, never at its first goal.
        if self._clear_pending:
            from nav2_msgs.srv import ClearEntireCostmap

            self._await_active()
            for client in self._clear_costmaps:
                if not client.wait_for_service(timeout_sec=self._config.service_timeout):
                    raise TimeoutError("Nav2 costmap reset service is unavailable")
                self._wait(client.call_async(ClearEntireCostmap.Request()))
            self._clear_pending = False
            self._acknowledged_map = -1

    def publish_camera(self, name: str, camera: CameraObservation, map_from_camera: torch.Tensor) -> None:
        """Publish sensor camera ``name``'s valid optical-frame points and their capture-time transform."""
        from builtin_interfaces.msg import Time
        from geometry_msgs.msg import TransformStamped
        from scipy.spatial.transform import Rotation

        if name not in self._points:
            raise RuntimeError(f"Nav2 has no point-cloud source for camera {name!r}")
        if camera.depth is None or camera.depth_valid is None or not camera.valid:
            raise RuntimeError("Nav2 sensor mode requires a valid permitted depth image")
        stamp = (camera.stamp.episode_id, camera.stamp.time_seconds)
        if stamp == self._camera_stamps.get(name):
            return
        if self._state is None or camera.stamp.episode_id != self._state.episode_id:
            raise ValueError("Nav2 camera frame belongs to another episode")
        if not np.isfinite(map_from_camera).all() or map_from_camera.shape != (4, 4):
            raise ValueError("Invalid optical-to-map transform")
        # Every sensor_pixel_stride-th row and column, filtered on the device before the copy: Nav2's
        # cells are coarser than neighbouring pixels' points. Every return from 5 cm: Nav2 marks up to its
        # range and clears along rays to farther points up to it. The bottom sensor_mark_margin of the
        # image only clears.
        marking, clearing = self._sensor_clouds(camera)
        if not len(clearing):
            raise RuntimeError("Nav2 camera has no valid points")
        camera_stamp = self._epoch + round(camera.stamp.time_seconds * 1_000_000_000)
        ros_stamp = Time(sec=camera_stamp // 1_000_000_000, nanosec=camera_stamp % 1_000_000_000)
        frame = f"{self._config.namespace}/{camera.frame}"
        transform = TransformStamped()
        transform.header.frame_id = f"{self._config.namespace}/map"
        transform.header.stamp = ros_stamp
        transform.child_frame_id = frame
        transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = (
            float(value) for value in map_from_camera[:3, 3]
        )
        quaternion = Rotation.from_matrix(map_from_camera[:3, :3]).as_quat()
        (
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ) = tuple(map(float, quaternion))
        self._tf.sendTransform(transform)
        self._points[name].publish(point_cloud(marking, frame, ros_stamp))
        self._clearing[name].publish(point_cloud(clearing, frame, ros_stamp))
        self._camera_stamps[name] = stamp
        self._last_cameras[name] = (camera, map_from_camera)

    def _sensor_clouds(self, camera: CameraObservation) -> tuple[np.ndarray, np.ndarray]:
        """A camera's points for Nav2 [m, N,3], those that mark and all (they clear): every
        sensor_pixel_stride-th row and column, from 5 cm, one per voxel. A voxel marks when a point of
        the image above its bottom sensor_mark_margin lies in it."""
        stride = self._config.sensor_pixel_stride
        points = camera.points(stride).detach()
        if not bool(torch.isfinite(points).all()):
            raise ValueError("Valid depth pixels produced nonfinite points")
        # The points come row by row: those of the rows above the margin first.
        height = camera.depth.shape[0]
        rows = height - round(self._config.sensor_mark_margin * height)
        marking = torch.arange(len(points), device=points.device) < int(
            camera.depth_valid[:rows:stride, ::stride].sum()
        )
        near = torch.linalg.vector_norm(points, dim=1) >= 0.05
        points, marking = points[near], marking[near]
        if self._config.sensor_voxel_size > 0:
            points, marking = voxel_filter(points, self._config.sensor_voxel_size, marking)
        points, marking = points.cpu().numpy(), marking.cpu().numpy()
        return points[marking], points

    def _republish_cameras(self) -> None:
        """Send every camera's held sample again (after the bridge epoch moved or the costmaps started)."""
        self._camera_stamps.clear()
        for name, (camera, map_from_camera) in self._last_cameras.items():
            self.publish_camera(name, camera, map_from_camera)

    def _costmap_received(self, message) -> None:
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        if stamp >= self._epoch:  # Not from before the last reset.
            self._latest_costmap = message

    def _command_received(self, message) -> None:
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        if stamp < self._minimum_stamp:
            return
        if stamp > self._stamp:
            raise ValueError("Nav2 received a command from a foreign/future simulation clock")
        if message.header.frame_id not in ("", self._config.namespace + "/base_link"):
            raise ValueError("Nav2 cmd_vel must use the configured body frame")
        twist = message.twist
        if any(abs(value) > 1e-9 for value in (twist.linear.z, twist.angular.x, twist.angular.y)):
            raise ValueError("Nav2 produced a non-planar command")
        self._command = (stamp, (twist.linear.x, twist.linear.y, twist.angular.z))

    def _check_geometry(self) -> None:
        from rcl_interfaces.srv import GetParameters

        footprint = RectangleFootprint(**self._geometry["footprint"])
        x0, x1, y0, y1 = footprint.bounds(padded=False)
        padding = footprint.padding + self._geometry["clearance"]
        radius = math.hypot(max(-x0, x1) + padding, max(-y0, y1) + padding)
        names = [
            "footprint_padding",
            "resolution",
            "footprint",
            "use_sim_time",
            "inflation_layer.inflation_radius",
            "inflation_layer.cost_scaling_factor",
        ]
        for client in self._costmap_parameters:
            if not client.wait_for_service(timeout_sec=self._config.service_timeout):
                raise TimeoutError("Nav2 costmap parameter service is unavailable")
            request = GetParameters.Request(names=names)
            values = self._wait(client.call_async(request)).values
            actual = [value.double_value for value in values[:2]]
            target = [padding, self._geometry["resolution"]]
            if any(not math.isclose(a, b, abs_tol=1e-6) for a, b in zip(actual, target, strict=True)):
                raise ValueError("Nav2 costmap geometry differs from the experiment")
            try:
                polygon = np.asarray(json.loads(values[2].string_value), dtype=float)
            except ValueError:
                polygon = np.empty((0, 2))
            outline = ((x1, y1), (x1, y0), (x0, y0), (x0, y1))
            if polygon.shape != (4, 2) or not np.allclose(polygon, outline, atol=1e-6):
                raise ValueError("Nav2 requires the measured standing outline as its footprint")
            if not values[3].bool_value:
                raise ValueError("Nav2 requires simulation time")
            decay = self._config.inflation_cost_distance
            actual_gradient = (values[4].double_value, values[5].double_value)
            expected_gradient = (radius + 3 * decay, 1 / decay)
            if any(
                not math.isclose(actual, target, abs_tol=1e-9)
                for actual, target in zip(actual_gradient, expected_gradient, strict=True)
            ):
                raise ValueError("Nav2 inflation gradient differs from the experiment")

    def _await_map(self) -> None:
        from nav2_msgs.srv import GetCostmap

        if self._acknowledged_map == self._raw_map.version:
            return
        for client in self._costmaps[:1]:  # The global costmap; the local one is a window on the robot.
            if not client.wait_for_service(timeout_sec=self._config.service_timeout):
                raise TimeoutError("Nav2 costmap inspection service is unavailable")
            deadline = time.monotonic() + self._config.service_timeout
            rounds = 0
            while time.monotonic() < deadline:
                costmap = self._wait(client.call_async(GetCostmap.Request())).map
                if self._config.sensor_cameras:
                    # The obstacle layer has taken the samples once it marks an obstacle; cameras that
                    # see only floor mark nothing, so a few costmap updates with the samples held
                    # (SENSOR_SETTLE_ROUNDS x 20 ms of bridge time) also count.
                    values = np.asarray(costmap.data, dtype=np.uint8)
                    if costmap_matches_geometry(costmap, self._raw_map) and (
                        np.any(values == 254) or rounds >= SENSOR_SETTLE_ROUNDS
                    ):
                        break
                elif costmap_matches(costmap, self._raw_map):
                    break
                if self._config.sensor_cameras:
                    # Nav2's costmap update timers use /clock. During synchronous
                    # planning the physical simulation is paused, so advance only
                    # the ROS bridge's epoch and republish the held sensor samples.
                    self._epoch += 20_000_000
                    self.publish(self._state, self._raw_map)
                    self._republish_cameras()
                    rounds += 1
                self._executor.spin_once(timeout_sec=0.01)
            else:
                if self._config.sensor_cameras:
                    raise TimeoutError("Nav2 costmap geometry differs from the sensor map while waiting for samples")
                raise TimeoutError("Nav2 costmap did not incorporate the current physical obstacle map")
        self._acknowledged_map = self._raw_map.version

    def _await_active(self) -> None:
        from lifecycle_msgs.srv import GetState

        # Activating, a costmap blocks its node until it sees our transform and clock, and a single
        # TF message sent before its listener matched is lost. So the state is re-sent on every
        # poll, and a state request the blocked node cannot answer yet is dropped, not fatal
        # (nav2_known_map, 2026-10-22: the planner waited 31 s for the transform, the one request
        # timed out after 10 s and poisoned the transport).
        # The lifecycle manager itself can stall: its configure request's response was lost once
        # ("failed to send response to .../change_state (timeout)", nav2_depth_map 2026-10-23) and it
        # never went on. A node whose state has not changed for stalled_transition_seconds is
        # driven through the next transition here; a node already mid-transition rejects it.
        for client, transition in zip(self._lifecycle, self._transitions, strict=True):
            if not client.wait_for_service(timeout_sec=self._config.service_timeout):
                raise TimeoutError("Nav2 lifecycle state service is unavailable")
            started = time.monotonic()
            deadline = started + self._config.bringup_timeout
            label, since = None, started
            while time.monotonic() < deadline:
                self._publish_pose()
                future = client.call_async(GetState.Request())
                self._executor.spin_until_future_complete(future, timeout_sec=self._config.command_timeout)
                if not future.done():
                    client.remove_pending_request(future)
                    continue
                current = future.result().current_state.label
                if current == "active":
                    break
                if current != label:
                    label, since = current, time.monotonic()
                elif (
                    current in STALLED_TRANSITIONS
                    and time.monotonic() - since >= self._config.stalled_transition_seconds
                ):
                    self._drive_transition(transition, STALLED_TRANSITIONS[current], client.srv_name)
                    since = time.monotonic()
                self._executor.spin_once(timeout_sec=0.05)
            else:
                raise TimeoutError(
                    f"Nav2 planner/controller did not become active within {self._config.bringup_timeout:g} s"
                )
        self._check_single_stack()

    def _check_single_stack(self) -> None:
        """Fail when another Nav2 stack serves the same namespace on this ROS domain: its servers
        answer our requests too (a stale stack from a killed run gave "parameters were not declared"
        and commands from another simulation clock, 2026-10-25)."""
        namespace = "/" + self._config.namespace
        for name in ("planner_server", "controller_server"):
            count = sum(
                1 for node, space in self._node.get_node_names_and_namespaces() if (node, space) == (name, namespace)
            )
            if count > 1:
                domain = os.environ.get("ROS_DOMAIN_ID", "0")
                raise RuntimeError(
                    f"{count} Nav2 {name} nodes in {namespace} on ROS_DOMAIN_ID {domain}: another Nav2 stack is"
                    " running there; stop it or use another domain"
                )

    def _drive_transition(self, client, transition: str, node: str) -> None:
        """Request a lifecycle transition the stalled manager did not complete; the reply is not awaited
        (the node answers once the transition, which may wait on our transform, has run)."""
        from lifecycle_msgs.srv import ChangeState

        LOG.warning(f"navigate_to: Nav2 bringup stalled; requesting {transition} on {node}")
        from lifecycle_msgs.msg import Transition

        request = ChangeState.Request()
        request.transition.label = transition
        request.transition.id = {
            "configure": Transition.TRANSITION_CONFIGURE,
            "activate": Transition.TRANSITION_ACTIVATE,
        }[transition]
        client.call_async(request)

    def start(self, goal: NavigationGoal) -> tuple[tuple[float, float], ...]:
        from action_msgs.msg import GoalStatus
        from geometry_msgs.msg import PoseStamped
        from nav2_msgs.action import ComputePathToPose, FollowPath
        from rcl_interfaces.srv import SetParametersAtomically
        from rclpy.parameter import Parameter

        if self._poisoned:
            raise RuntimeError("Nav2 connection timed out; restart the external stack and bridge before reuse")
        self._check_geometry()
        if self._following is not None or self._planning is not None:
            raise RuntimeError("Cancel and await old Nav2 actions before starting another goal")
        for client in (self._planner, self._controller):
            if not client.wait_for_server(timeout_sec=self._config.service_timeout):
                raise TimeoutError("Nav2 planner/controller action server is unavailable")
        if not self._parameters.wait_for_service(timeout_sec=self._config.service_timeout):
            raise TimeoutError("Nav2 controller parameter service is unavailable")
        request = SetParametersAtomically.Request()
        request.parameters = [
            Parameter("goal_checker.xy_goal_tolerance", value=goal.position_tolerance).to_parameter_msg(),
            Parameter(
                "goal_checker.yaw_goal_tolerance", value=math.pi if goal.heading is None else goal.heading_tolerance
            ).to_parameter_msg(),
        ]
        response = self._wait(self._parameters.call_async(request))
        if not response.result.successful:
            raise ValueError("Nav2 rejected goal tolerances: " + response.result.reason)
        # Advance the ROS-only epoch by 1 ns to exclude queued commands from a prior action.
        self._epoch += 1
        self._stamp += 1
        self._minimum_stamp = self._stamp
        self._command = None
        self.publish(self._state, self._raw_map)
        self._await_active()
        if self._config.sensor_cameras:
            # Subscriptions may not exist until the costmaps become active.
            self._republish_cameras()
        self._await_map()
        pose = PoseStamped()
        pose.header.frame_id, pose.header.stamp = self._config.namespace + "/map", self._time()
        pose.pose.position.x, pose.pose.position.y = goal.x, goal.y
        heading = self._state.yaw if goal.heading is None else goal.heading
        pose.pose.orientation.z, pose.pose.orientation.w = math.sin(heading / 2), math.cos(heading / 2)
        planning = ComputePathToPose.Goal()
        planning.goal, planning.planner_id, planning.use_start = pose, "GridBased", False
        self._planning = self._wait(self._planner.send_goal_async(planning))
        if not self._planning.accepted:
            self._planning = None
            raise Nav2PlanningError("Nav2 rejected the planning goal")
        self._plan_result = self._planning.get_result_async()
        result = self._wait(self._plan_result)
        self._planning, self._plan_result = None, None
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            code = result.result.error_code
            raise Nav2PlanningError(
                f"Nav2 planner failed with code {code}",
                unreachable=code in (ComputePathToPose.Result.GOAL_OCCUPIED, ComputePathToPose.Result.NO_VALID_PATH),
                goal_occupied=code == ComputePathToPose.Result.GOAL_OCCUPIED,
            )
        path = result.result.path
        # MPPI's route critics look a fixed number of route points ahead; Smac Lattice returns about
        # one pose per motion primitive (~0.15 m apart), so the route is resampled at the costmap
        # resolution first.
        poses = [(p.pose.position.x, p.pose.position.y, _yaw(p.pose.orientation)) for p in path.poses]
        if len(poses) == 1:  # The goal's cell holds the robot: a route from where it stands.
            poses.insert(0, (self._state.x, self._state.y, self._state.yaw))
        dense = dense_route(poses, self._geometry["resolution"])
        template = path.poses[0] if path.poses else None
        path.poses = [_stamped_pose(template, x, y, yaw) for x, y, yaw in dense] if template is not None else []
        self._path.publish(path)
        following = FollowPath.Goal()
        following.path = path
        following.controller_id, following.goal_checker_id = "FollowPath", "goal_checker"
        following.progress_checker_id = "progress_checker"
        self._following = self._wait(self._controller.send_goal_async(following))
        if not self._following.accepted:
            self._following = None
            raise RuntimeError("Nav2 rejected path execution")
        self._follow_result = self._following.get_result_async()
        return tuple((pose.pose.position.x, pose.pose.position.y) for pose in result.result.path.poses)

    def costmap(self) -> tuple[np.ndarray, tuple[float, float], float] | None:
        """Nav2's newest published global costmap: cost per cell [row y, column x] (0-252 free to near,
        253 the robot's centre may not be there, 254 an obstacle, 255 unknown), its origin [m] and cell
        size [m]; None before Nav2 publishes one this episode."""
        self._spin_pending()
        costmap = self._latest_costmap
        if costmap is None:
            return None
        meta = costmap.metadata
        cells = np.asarray(costmap.data, dtype=np.uint8).reshape(meta.size_y, meta.size_x)
        return cells, (float(meta.origin.position.x), float(meta.origin.position.y)), float(meta.resolution)

    def costmap_cost(self, x: float, y: float) -> int:
        """Read the current global costmap cell for trusted diagnostics."""
        from nav2_msgs.srv import GetCostmap

        costmap = self._wait(self._costmaps[0].call_async(GetCostmap.Request())).map
        if not costmap_matches_geometry(costmap, self._raw_map):
            raise ValueError("Nav2 global costmap geometry changed during navigation")
        return costmap_cell_cost(costmap, x, y)

    def path_blocked(self, path: tuple[tuple[float, float], ...], position: tuple[float, float]) -> bool:
        """Check the remaining path against the newest published global costs (False before one)."""
        self._spin_pending()
        costmap = self._latest_costmap
        if costmap is None:
            return False
        if not costmap_matches_geometry(costmap, self._raw_map):
            raise ValueError("Nav2 global costmap geometry changed during navigation")
        return costmap_blocks_path(costmap, path, position)

    def poll(self) -> Nav2Update:
        """Nav2's newest velocity command and its age [s], or the route's outcome. Never waits: Nav2 runs
        beside the simulation, as beside a robot."""
        from action_msgs.msg import GoalStatus
        from nav2_msgs.action import FollowPath

        self._spin_pending()
        if self._follow_result is not None and self._follow_result.done():
            result = self._follow_result.result()
            status = "succeeded" if result.status == GoalStatus.STATUS_SUCCEEDED else "failed"
            return Nav2Update(
                status,
                reason=f"Nav2 result {result.status}, code {result.result.error_code}",
                blocked=result.result.error_code == FollowPath.Result.NO_VALID_CONTROL,
            )
        if self._command is None:
            return Nav2Update("active")
        stamp, command = self._command
        return Nav2Update("active", command, (self._stamp - stamp) / 1e9)

    def _spin_pending(self, limit: int = 20) -> None:
        """Handle the messages and responses already received, without waiting for more."""
        for _ in range(limit):
            self._executor.spin_once(timeout_sec=0.0)

    def cancel(self) -> None:
        for handle, result in ((self._planning, self._plan_result), (self._following, self._follow_result)):
            if handle is not None and result is not None and not result.done():
                self._wait(handle.cancel_goal_async())
                self._wait(result)
        self._planning = self._following = self._plan_result = self._follow_result = None
        self._command = None
        self._minimum_stamp = self._stamp + 1

    def reset(self) -> None:
        if self._poisoned:
            raise RuntimeError("Cannot reset a timed-out Nav2 connection; restart the stack and bridge")
        self.explore_stop()
        self.cancel()
        self._epoch = self._stamp + 1_000_000_000
        self._stamp, self._minimum_stamp = self._epoch, self._epoch
        self._map_version = self._acknowledged_map = -1
        self._latest_costmap = None
        self._clear_pending = True
        self._camera_stamps.clear()
        self._last_cameras.clear()
        from nav_msgs.msg import Path

        empty_path = Path()
        empty_path.header.frame_id = self._config.namespace + "/map"
        empty_path.header.stamp = self._time()
        self._path.publish(empty_path)
        self._state = None

    def close(self) -> None:
        if not self._closed:
            try:
                self.explore_stop()
                self.cancel()
            finally:
                self._executor.remove_node(self._node)
                self._node.destroy_node()
                self._executor.shutdown()
                self._rclpy.shutdown(context=self._context)
                self._closed = True

    def explore_start(self) -> None:
        """Start explore_lite on Nav2's global costmap; its NavigateToPose goals queue here (explore_poll)."""
        import subprocess

        from nav2_msgs.action import NavigateToPose
        from rclpy.action import ActionServer, CancelResponse, GoalResponse
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        if self._explorer is not None:
            raise RuntimeError("A frontier exploration is already running")
        executable = explore_lite_executable()
        try:
            from explore_lite_msgs.msg import ExploreStatus
        except ImportError as error:
            raise RuntimeError(EXPLORE_LITE_MISSING) from error
        if self._navigate_server is None:
            # explore_lite drives the robot through Nav2's NavigateToPose action, which a BT navigator
            # would serve; the bridge serves it instead, so each goal is a leg of the agent's command.
            self._navigate_server = ActionServer(
                self._node,
                NavigateToPose,
                "navigate_to_pose",
                execute_callback=self._explore_execute,
                goal_callback=lambda goal: GoalResponse.ACCEPT,
                handle_accepted_callback=self._explore_goals.append,
                cancel_callback=lambda handle: CancelResponse.ACCEPT,
            )
            latched = QoSProfile(
                depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
            )
            self._explore_status = self._node.create_subscription(
                ExploreStatus, "explore/status", self._explore_status_received, latched
            )
        self._explore_complete = False
        namespace, config = self._config.namespace, self._config
        parameters = dict(
            use_sim_time="true",
            robot_base_frame=f"{namespace}/base_link",
            costmap_topic=f"/{namespace}/global_costmap/costmap",
            costmap_updates_topic=f"/{namespace}/global_costmap/costmap_updates",
            planner_frequency=config.explore_planner_frequency,
            progress_timeout=config.explore_progress_timeout,
            min_frontier_size=config.explore_min_frontier_size,
            potential_scale=3.0,  # explore_lite's defaults (params_costmap.yaml): nearer frontiers first,
            gain_scale=1.0,  # larger ones preferred.
            orientation_scale=0.0,
            transform_tolerance=0.3,
            return_to_init="false",
            visualize="true",  # explore/frontiers markers.
        )
        command = [
            *(["setpriv", "--pdeathsig", "TERM"] if shutil.which("setpriv") else []),
            executable,
            "--ros-args",
            "-r",
            f"__ns:=/{namespace}",
            *(item for key, value in parameters.items() for item in ("-p", f"{key}:={value}")),
        ]
        log = open(config.explore_log, "a") if config.explore_log else subprocess.DEVNULL  # noqa: SIM115
        try:
            self._explorer = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            if log is not subprocess.DEVNULL:
                log.close()

    def _explore_status_received(self, message) -> None:
        if self._explorer is not None and message.status == "exploration_complete":
            self._explore_complete = True

    def explore_poll(self) -> ExploreEvent:
        """explore_lite's newest frontier goal since the last poll, completion, and liveness. A new goal
        supersedes the previous one (explore_lite moved on to a better frontier)."""
        self._executor.spin_once(timeout_sec=0.0)
        running = self._explorer is not None and self._explorer.poll() is None
        fresh = [handle for handle in self._explore_goals if handle is not self._explore_current]
        if not fresh:
            return ExploreEvent(complete=self._explore_complete, running=running)
        for handle in [self._explore_current, *fresh[:-1]]:
            if handle is not None:
                self._explore_answer(handle, False, "Superseded by a newer frontier goal")
        self._explore_current = fresh[-1]
        position = self._explore_current.request.pose.pose.position
        return ExploreEvent(
            goal=(float(position.x), float(position.y)), complete=self._explore_complete, running=running
        )

    def explore_finish(self, succeeded: bool, reason: str) -> None:
        """Answer the current frontier goal: reached, or given up (explore_lite then skips that frontier)."""
        if self._explore_current is not None:
            self._explore_answer(self._explore_current, succeeded, reason)
            self._explore_current = None
        self._executor.spin_once(timeout_sec=0.0)

    def _explore_answer(self, handle, succeeded: bool, reason: str) -> None:
        if handle not in self._explore_goals:
            return
        self._explore_goals.remove(handle)
        if handle.is_cancel_requested:
            handle.canceled()
        elif handle.is_active:
            self._explore_outcomes[bytes(handle.goal_id.uuid)] = (succeeded, reason)
            handle.execute()  # Runs _explore_execute on the executor: the goal's result.

    def _explore_execute(self, handle):
        from nav2_msgs.action import NavigateToPose

        succeeded, reason = self._explore_outcomes.pop(bytes(handle.goal_id.uuid), (False, "Exploration stopped"))
        result = NavigateToPose.Result()
        if succeeded:
            handle.succeed()
        else:
            # A nonzero error code makes explore_lite blacklist the frontier instead of retrying it.
            result.error_code, result.error_msg = 1, reason
            handle.abort()
        return result

    def explore_stop(self) -> None:
        """Stop explore_lite and give up its open goals."""
        from phygs.execution.navigation.backends.nav2_stack import stop_group

        if self._explorer is not None:
            stop_group(self._explorer, timeout=5.0)
            self._explorer = None
        for handle in list(self._explore_goals):
            self._explore_answer(handle, False, "Exploration stopped")
        self._explore_current = None
        if not self._closed:
            self._executor.spin_once(timeout_sec=0.0)
        self._explore_complete = False


EXPLORE_LITE_MISSING = (
    "Frontier exploration needs explore_lite (m-explore-ros2) built and sourced; see the README's"
    " Navigation installation section."
)


def explore_lite_executable() -> str:
    """explore_lite's ``explore`` executable from the sourced ROS environment (the README's overlay)."""
    from ament_index_python.packages import PackageNotFoundError, get_package_prefix

    try:
        prefix = get_package_prefix("explore_lite")
    except PackageNotFoundError as error:
        raise RuntimeError(EXPLORE_LITE_MISSING) from error
    path = os.path.join(prefix, "lib", "explore_lite", "explore")
    if not os.access(path, os.X_OK):
        raise RuntimeError(EXPLORE_LITE_MISSING)
    return path


def costmap_cell_cost(costmap, x: float, y: float) -> int:
    """Return the Nav2 cost at a world point [m]; reject points outside the map."""
    metadata = costmap.metadata
    col = math.floor((x - metadata.origin.position.x) / metadata.resolution)
    row = math.floor((y - metadata.origin.position.y) / metadata.resolution)
    if not 0 <= col < metadata.size_x or not 0 <= row < metadata.size_y:
        raise ValueError("Path point is outside the Nav2 global costmap")
    return int(costmap.data[row * metadata.size_x + col])


def costmap_blocks_path(costmap, path: tuple[tuple[float, float], ...], position: tuple[float, float]) -> bool:
    """Find lethal or inscribed cells on the path ahead of Spot.

    Sampling at half a cell prevents a narrow obstacle from falling between
    sparse Nav2 path poses. The already-traversed portion is ignored. Unknown
    cells (255) do not block: Nav2 plans through them (Smac ``allow_unknown``)
    and the cameras mark what is there as the robot comes closer.
    """
    if len(path) < 2 or costmap.metadata.resolution <= 0:
        raise ValueError("A valid path and costmap resolution are required")
    segments = [(np.asarray(a, dtype=float), np.asarray(b, dtype=float)) for a, b in zip(path, path[1:])]
    start = np.asarray(position, dtype=float)
    if not np.isfinite(start).all() or not all(np.isfinite(a).all() and np.isfinite(b).all() for a, b in segments):
        raise ValueError("Path and current position must be finite")
    projections = []
    for index, (a, b) in enumerate(segments):
        direction = b - a
        length_sq = float(direction @ direction)
        fraction = float(np.clip(((start - a) @ direction) / length_sq, 0, 1)) if length_sq else 0.0
        projections.append((float(np.linalg.norm(start - a - fraction * direction)), index, fraction))
    _, first, fraction = min(projections)
    for index, (a, b) in enumerate(segments[first:], start=first):
        begin = fraction if index == first else 0.0
        count = max(1, math.ceil(float(np.linalg.norm(b - a)) * (1 - begin) / (costmap.metadata.resolution / 2)))
        for sample in range(count + 1):
            xy = a + (b - a) * (begin + (1 - begin) * sample / count)
            if costmap_cell_cost(costmap, float(xy[0]), float(xy[1])) in (253, 254):
                return True
    return False


def costmap_matches_geometry(costmap, raw_map: GridMap) -> bool:
    """Check ROS costmap geometry against the declared planar floor."""
    metadata = costmap.metadata
    if (metadata.size_y, metadata.size_x) != raw_map.occupied.shape:
        return False
    if not math.isclose(metadata.resolution, raw_map.resolution, rel_tol=1e-6):
        return False
    origin = metadata.origin
    if not np.allclose((origin.position.x, origin.position.y), raw_map.origin, rtol=0, atol=1e-6):
        return False
    rotation = origin.orientation
    return bool(np.allclose((rotation.x, rotation.y, rotation.z, rotation.w), (0, 0, 0, 1), atol=1e-6))


def costmap_matches(costmap, raw_map: GridMap) -> bool:
    """Check blocked source cells (a known map's), excluding separate robot inflation costs."""
    metadata = costmap.metadata
    if (metadata.size_y, metadata.size_x) != raw_map.occupied.shape:
        return False
    if not math.isclose(metadata.resolution, raw_map.resolution, rel_tol=1e-6):
        return False
    origin = metadata.origin
    if not np.allclose((origin.position.x, origin.position.y), raw_map.origin, rtol=0, atol=1e-6):
        return False
    rotation = origin.orientation
    if not np.allclose((rotation.x, rotation.y, rotation.z, rotation.w), (0, 0, 0, 1), atol=1e-6):
        return False
    values = np.asarray(costmap.data, dtype=np.uint8).reshape(raw_map.occupied.shape)
    return np.array_equal(values == 254, raw_map.occupied)


def occupancy_values(raw_map: GridMap) -> np.ndarray:
    """Encode uninflated evidence as ROS occupancy: -1 unknown, 0 free, 100 occupied."""
    values = raw_map.occupied.astype(np.int8) * 100
    if raw_map.unknown is not None:
        values[raw_map.unknown] = -1
    return values


def dense_route(poses: list[tuple[float, float, float]], spacing: float) -> list[tuple[float, float, float]]:
    """A route of (x [m], y [m], yaw [rad]) poses resampled so neighbours are at most ``spacing`` [m]
    apart: positions linearly, headings along the shorter turn. Turns in place keep their poses."""
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("Route spacing must be positive and finite.")
    dense = list(poses[:1])
    for (x0, y0, a0), (x1, y1, a1) in zip(poses, poses[1:]):
        steps = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) / spacing - 1e-9))
        turn = math.remainder(a1 - a0, math.tau)
        dense.extend(
            (x0 + (x1 - x0) * k / steps, y0 + (y1 - y0) * k / steps, math.remainder(a0 + turn * k / steps, math.tau))
            for k in range(1, steps + 1)
        )
    return dense


def _yaw(orientation) -> float:
    return math.atan2(2 * (orientation.w * orientation.z + orientation.x * orientation.y),
                      1 - 2 * (orientation.y**2 + orientation.z**2))  # fmt: skip


def _stamped_pose(template, x: float, y: float, yaw: float):
    from geometry_msgs.msg import PoseStamped

    pose = PoseStamped()
    pose.header = template.header
    pose.pose.position.x, pose.pose.position.y = x, y
    pose.pose.orientation.z, pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return pose
