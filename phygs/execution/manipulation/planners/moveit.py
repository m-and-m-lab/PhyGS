# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""MoveItPy/OMPL planning only; execution stays in the shared runtime."""

import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster

from .interface import (
    DepthImage,
    HeldShape,
    IkRequest,
    IkResult,
    PlanRequest,
    PlanResult,
)
from .model import gripper_positions

HELD = "held_object"
CAMERA = "depth_camera"  # TF frame of the depth image being fused.
POINTS = "/spot_planner/depth_points"
FUSED = "/spot_planner/depth_points_fused"  # The Octomap updater publishes each cloud here after fusing it.


def stamped(frame: str, pose: list[float]) -> PoseStamped:
    """PoseStamped from [x, y, z, qx, qy, qz, qw] [m] in frame; ROS messages accept floats only."""
    message = PoseStamped()
    message.header.frame_id = frame
    p, q = message.pose.position, message.pose.orientation
    p.x, p.y, p.z, q.x, q.y, q.z, q.w = map(float, pose)
    return message


def transform(stamp, parent: str, child: str, matrix: np.ndarray) -> TransformStamped:
    """TF message for a 4x4 transform [m] of child in parent."""
    message = TransformStamped(header=Header(stamp=stamp, frame_id=parent), child_frame_id=child)
    t, q = message.transform.translation, message.transform.rotation
    t.x, t.y, t.z = map(float, matrix[:3, 3])
    q.x, q.y, q.z, q.w = map(float, Rotation.from_matrix(matrix[:3, :3]).as_quat())
    return message


def box_object(name: str, box: dict, frame: str) -> CollisionObject:
    """CollisionObject for a box [m] with pose [x, y, z, qx, qy, qz, qw] in frame."""
    obj = CollisionObject(id=name, operation=CollisionObject.ADD)
    obj.header.frame_id = frame
    obj.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(v) for v in box["size"]])]
    obj.primitive_poses = [stamped(frame, box["pose"]).pose]
    return obj


class Planner:
    def __init__(self, config: dict):
        self.config = config
        self.names = config["joints"]
        settings = {
            "robot_description": Path(config["moveit_urdf"]).read_text(),  # Arm links: referee spheres.
            "robot_description_semantic": Path(config["srdf"]).read_text(),
            "robot_description_kinematics": {"arm": {"kinematics_solver": "kdl_kinematics_plugin/KDLKinematicsPlugin"}},
            "totg": {"path_tolerance": 0.0, "resample_dt": 0.01},
            "planning_pipelines": {"pipeline_names": ["ompl"]},
            "ompl": {
                "planning_plugins": ["ompl_interface/OMPLPlanner"],
                "response_adapters": ["default_planning_response_adapters/AddTimeOptimalParameterization"],
                "planner_configs": {"RRTConnect": {"type": "geometric::RRTConnect", "range": 0.1}},
                "arm": {"planner_configs": ["RRTConnect"], "longest_valid_segment_fraction": 0.0005},
            },
            "plan_request_params": {
                "planning_pipeline": "ompl",
                "planner_id": "RRTConnect",
                "planning_attempts": 1,
                "planning_time": config["planning_time_seconds"],
                "max_velocity_scaling_factor": 1.0,
                "max_acceleration_scaling_factor": 1.0,
            },
            "planning_scene_monitor_options": {
                "name": "experiment",
                "robot_description": "robot_description",
                "joint_state_topic": "/manipulation_experiment/joint_states",
                "wait_for_initial_state_timeout": 0.0,
            },
            "octomap_frame": config["base_link"],
            "octomap_resolution": config["depth_map"]["voxel_size"],
            "sensors": ["depth"],
            "depth": {
                "sensor_plugin": "occupancy_map_monitor/PointCloudOctomapUpdater",
                "point_cloud_topic": POINTS,
                "filtered_cloud_topic": FUSED,
                "max_range": 10.0,
                "point_subsample": 1,
                # Self filter: points within this of a robot link or the attached object [m] are dropped.
                "padding_offset": config["depth_map"]["self_filter_padding"],
                "padding_scale": 1.0,
                "max_update_rate": 0.0,
            },
            "robot_description_planning": {
                # The Octomap self filter looks up the link frames in TF; observe() publishes them
                # with each image, so a lookup waits only for the message to arrive.
                "shape_transform_cache_lookup_wait_time": 0.05,
                # Clearance: robot links and the attached object grow for obstacle checks only.
                "default_robot_padding": config["clearance"],
                "default_attached_padding": config["clearance"],
                # Half the arm's acceleration limit, about what cuRobo's jerk-limited profiles reach:
                # at the full limit (Spot: 3.0 rad/s², no jerk limit) a swing from a high view to a
                # drawer handle swayed the legged base past the trajectory guard's 0.03 rad.
                "joint_limits": {
                    n: {
                        "has_velocity_limits": True,
                        "max_velocity": config["max_velocity"],
                        "has_acceleration_limits": True,
                        "max_acceleration": config["max_acceleration"] / 2,
                    }
                    for n in self.names
                },
            },
        }
        self.robot = MoveItPy(node_name="spot_manipulation_experiment", config_dict=settings)
        self.model = self.robot.get_robot_model()
        self.arm = self.robot.get_planning_component("arm")
        self.objects = []
        self.holding = False
        # Robot links with collision shapes, below the root (body): the self filter masks them.
        root = ET.parse(config["moveit_urdf"]).getroot()
        children = {joint.find("child").get("link") for joint in root.findall("joint")}
        self.links = [link.get("name") for link in root.findall("link") if link.find("collision") is not None]
        self.links = [name for name in self.links if name in children]
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("spot_manipulation_depth")
        self.points = self.node.create_publisher(PointCloud2, POINTS, 10)
        self.tf = TransformBroadcaster(self.node)
        self.fused = None
        self.node.create_subscription(
            PointCloud2, FUSED, lambda cloud: setattr(self, "fused", cloud.header.stamp), qos_profile_sensor_data
        )
        # Ready only once the Octomap updater listens: an image sent before is lost and its fusion times out.
        deadline = time.monotonic() + 30.0
        while self.points.get_subscription_count() == 0:
            if time.monotonic() > deadline:
                raise TimeoutError("The Octomap updater did not subscribe to the depth points within 30 s")
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def posture(self, gripper: float | None, arm: list[float]) -> dict[str, float]:
        """All joint positions [m or rad]: locked joints, the gripper's (its primary joint at
        ``gripper``) if given, and the arm."""
        locked = {**self.config["locked_joints"]}
        if gripper is not None:
            locked.update(gripper_positions(self.config, gripper))
        return {**locked, **dict(zip(self.names, arm, strict=True))}

    def state(self, positions: dict[str, float]):
        state = RobotState(self.model)
        state.joint_positions = positions
        state.update()
        return state

    def prepare(self, scene, boxes: list[dict], held: HeldShape | None, posture: dict[str, float]):
        """Put the boxes, the held object and the start posture in the scene; return the start state."""
        for name in self.objects:
            scene.apply_collision_object(CollisionObject(id=name, operation=CollisionObject.REMOVE))
        self.objects = [f"obstacle_{i}" for i in range(len(boxes))]
        for name, box in zip(self.objects, boxes):
            scene.apply_collision_object(box_object(name, box, self.config["base_link"]))
        self.update_held(scene, None if held is None else held.box)
        # The scene state carries the attached body; a fresh RobotState would not.
        start = scene.current_state
        start.joint_positions = posture
        start.update()
        return start

    def samples(self, trajectory) -> dict | None:
        """Timed arm-joint samples [rad, s] from a JointTrajectory message; None if a value is not finite."""
        values = [
            v for p in trajectory.points for v in (*p.positions, p.time_from_start.sec, p.time_from_start.nanosec)
        ]
        if not all(math.isfinite(v) for v in values):
            return None  # Time parameterization rarely returns NaN; the caller reports a failed plan.
        order = [trajectory.joint_names.index(n) for n in self.names]
        return dict(
            joint_names=self.names,
            times=[p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in trajectory.points],
            positions=[[p.positions[i] for i in order] for p in trajectory.points],
        )

    def plan(self, request: PlanRequest) -> PlanResult:
        started = time.perf_counter()
        if request.locked_joints is not None:  # The legs as measured; they stay for later requests.
            others = set(self.config["locked_joints"]) - set(self.config["gripper_ratios"])
            if set(request.locked_joints) != others:
                given = sorted(request.locked_joints)
                raise ValueError(f"Locked joints {given} do not match the model's {sorted(others)}.")
            self.config["locked_joints"].update(request.locked_joints)
        monitor, posture = self.robot.get_planning_scene_monitor(), self.posture(request.gripper, request.start)
        with monitor.read_write() as scene:
            start = self.prepare(scene, request.boxes, request.held, posture)
            # Copies: the scene's state is only valid while the scene is locked (attaching the held
            # object replaces it), and the Octomap updater needs the lock to fuse.
            links = {name: np.array(start.get_global_link_transform(name)) for name in self.links}
        mapped = time.perf_counter()
        monitor.clear_octomap()
        for image in request.images or []:
            self.fuse(image, links)
        mapped = time.perf_counter() - mapped
        with monitor.read_write() as scene:
            start = scene.current_state
            start.joint_positions = posture
            start.update()
            if scene.is_state_colliding(robot_state=start, joint_model_group_name="arm", verbose=True):
                elapsed = time.perf_counter() - started
                return {"success": False, "reason": "start_collision", "plan_seconds": elapsed, "map_seconds": mapped}
        self.arm.set_start_state(robot_state=start)
        if request.goal_pose is not None:
            base, wrist = self.config["base_link"], self.config["wrist_link"]
            self.arm.set_goal_state(pose_stamped_msg=stamped(base, request.goal_pose), pose_link=wrist)
        else:
            self.arm.set_goal_state(robot_state=self.state(self.posture(request.gripper, request.goal)))
        result = self.arm.plan()
        elapsed = time.perf_counter() - started
        if not result:
            return {"success": False, "reason": "no_plan_within_budget", "plan_seconds": elapsed, "map_seconds": mapped}
        samples = self.samples(result.trajectory.get_robot_trajectory_msg().joint_trajectory)
        if samples is None:
            reason = "invalid_time_parameterization"
            return {"success": False, "reason": reason, "plan_seconds": elapsed, "map_seconds": mapped}
        return {"success": True, **samples, "plan_seconds": elapsed, "map_seconds": mapped}

    def ik(self, request: IkRequest) -> IkResult:
        """KDL IK for the wrist pose in the body frame, seeded with the given arm joints."""
        started = time.perf_counter()
        state = self.state(self.posture(None, request.seed))
        if not state.set_from_ik(
            "arm", stamped(self.config["base_link"], request.pose).pose, self.config["wrist_link"], 0.05
        ):
            return {"success": False, "seconds": time.perf_counter() - started}
        joints = state.joint_positions
        return {"success": True, "joints": [joints[n] for n in self.names], "seconds": time.perf_counter() - started}

    def fuse(self, image: DepthImage, links: dict[str, np.ndarray]) -> None:
        """Publish one depth image as a point cloud, with the camera and the robot's link frames
        (``links``: 4x4 in the body frame) in TF; wait until the Octomap has it. The updater's self
        filter drops the points on the links and on the attached held object."""
        depth = np.load(image.path)
        # Every pixel_stride-th pixel per axis: Octomap ray casting takes ~0.9 s per full 640x480 image.
        stride = self.config["depth_map"]["pixel_stride"]
        rows, cols = np.nonzero(depth[::stride, ::stride] > 0)
        rows, cols = rows * stride, cols * stride
        if not len(rows):
            return
        rays = np.linalg.solve(np.array(image.intrinsics), np.stack((cols, rows, np.ones_like(rows))).astype(float))
        points = (rays * depth[rows, cols]).T
        stamp = self.node.get_clock().now().to_msg()
        camera = np.eye(4)
        camera[:3, :3] = Rotation.from_quat(image.pose[3:]).as_matrix()
        camera[:3, 3] = image.pose[:3]
        base = self.config["base_link"]
        frames = [transform(stamp, base, name, matrix) for name, matrix in links.items()]
        self.tf.sendTransform([transform(stamp, base, CAMERA, camera), *frames])
        self.points.publish(point_cloud2.create_cloud_xyz32(Header(stamp=stamp, frame_id=CAMERA), points))
        deadline = time.monotonic() + 5.0
        while self.fused != stamp:
            if time.monotonic() > deadline:
                raise TimeoutError("The Octomap did not fuse a depth image within 5 s")
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def update_held(self, scene, held: dict | None) -> None:
        """Replace the held-object box attached to the wrist; collision model only, not physics."""
        wrist = self.config["wrist_link"]
        if self.holding:
            detach = CollisionObject(id=HELD, operation=CollisionObject.REMOVE)
            scene.process_attached_collision_object(AttachedCollisionObject(link_name=wrist, object=detach))
            scene.apply_collision_object(detach)  # A detached body returns to the world; remove it.
            self.holding = False
        if held is not None:
            attach = AttachedCollisionObject(
                link_name=wrist,
                object=box_object(HELD, held, wrist),
                touch_links=list(self.config["held_contact_links"]),
            )
            scene.process_attached_collision_object(attach)
            self.holding = True

    def close(self):
        self.node.destroy_node()
        self.robot.shutdown()
